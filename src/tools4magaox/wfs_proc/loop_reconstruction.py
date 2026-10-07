# loop_reconstruction.py
# Reconstruct open-loop wavefronts from camwfs xrif frames using cacao calibrations.
#
# Three main stages:
#   1. load_wfs_frames        — read camwfs xrif for a time window
#   2. wfs_to_delta_wavefront — dark/mask/norm/wfsref + modesWFS + DMmodes -> nm OPD
#   3. integrate_open_loop   — sum deltas over a configurable integration window

from __future__ import annotations

import argparse
import ast
import bisect
import logging
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from tools4magaox.constants import WFS_WAVELENGTH

log = logging.getLogger(__name__)

# stream_YYYYmmddHHMMSS + fractional digits + literal trailing 000 before .xrif
_XRIF_TS_RE = re.compile(
    r"^(?P<stream>.+)_(?P<ts>\d{14})(?P<frac>\d+)000\.xrif$"
)
_DARK_TS_RE = re.compile(r"__T(\d{14})(\d*)")
_DATE_FOLDER_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")

CONFIG_SNAPSHOT_NAME = "loop_recon_config.txt"
LOG_NAME = "loop_recon.log"


# ---------------------------------------------------------------------------
# Filename / time helpers
# ---------------------------------------------------------------------------


def parse_utc(value) -> datetime:
    """Parse an ISO-8601 or FITS-style UTC timestamp to an aware datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_xrif_filename_time(name: str) -> datetime | None:
    """Parse ``stream_YYYYmmddHHMMSSffffff000.xrif`` into a UTC datetime."""
    m = _XRIF_TS_RE.match(os.path.basename(name))
    if not m:
        return None
    main = datetime.strptime(m.group("ts"), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    frac = m.group("frac") or ""
    if not frac:
        return main
    usec = int((frac[:6]).ljust(6, "0"))
    return main + timedelta(microseconds=usec)


def filename_time_token(dt: datetime) -> str:
    """Compact ``YYYYmmddHHMMSSffffff`` token for lexicographic filename compare."""
    dt = parse_utc(dt)
    return dt.strftime("%Y%m%d%H%M%S") + f"{dt.microsecond:06d}"


def timing_rows_to_datetimes(timing) -> np.ndarray:
    """
    Convert XRIF timing rows to ``datetime64[us]`` acquisition times (UTC).

    Expected layout after squeeze: ``[N, 5]`` with
    ``[frame_index, acq_sec, acq_nsec, write_sec, write_nsec]``.
    """
    t = np.asarray(timing)
    t = np.squeeze(t)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if t.ndim != 2 or t.shape[1] < 3:
        raise ValueError(f"timing must be [N,5]-like, got {t.shape}")
    out = np.empty(t.shape[0], dtype="datetime64[us]")
    for i, row in enumerate(t):
        acq_sec = int(row[1])
        acq_nsec = int(row[2])
        dt = datetime.fromtimestamp(acq_sec, tz=timezone.utc) + timedelta(
            microseconds=acq_nsec // 1000
        )
        out[i] = np.datetime64(dt.replace(tzinfo=None), "us")
    return out


def resolve_time_window(params) -> tuple[datetime, datetime]:
    """Return ``(t_start, t_end)``; require exactly one of ``t_end`` / ``duration_s``."""
    if "t_start" not in params:
        raise ValueError("config missing t_start")
    t_start = parse_utc(params["t_start"])
    has_end = params.get("t_end") is not None and str(params.get("t_end")).strip() != ""
    has_dur = params.get("duration_s") is not None
    if has_end == has_dur:
        raise ValueError("provide exactly one of t_end or duration_s")
    if has_end:
        t_end = parse_utc(params["t_end"])
    else:
        t_end = t_start + timedelta(seconds=float(params["duration_s"]))
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be after t_start ({t_start})")
    return t_start, t_end


# ---------------------------------------------------------------------------
# File listing / index cache
# ---------------------------------------------------------------------------


def _list_xrif_names_flat(data_dir: Path, stream_name: str) -> list[str]:
    prefix = f"{stream_name}_"
    names = []
    with os.scandir(data_dir) as it:
        for e in it:
            if e.is_file() and e.name.startswith(prefix) and e.name.endswith(".xrif"):
                names.append(e.name)
    names.sort()
    return names


def _list_xrif_names_dated(data_dir: Path, stream_name: str, t_start, t_end) -> list[str]:
    """Scan only ``YYYY_MM_DD`` folders touched by the window (sparkles layout)."""
    days = set()
    d = t_start.astimezone(timezone.utc).date()
    end_day = t_end.astimezone(timezone.utc).date()
    while d <= end_day:
        days.add(d.strftime("%Y_%m_%d"))
        d += timedelta(days=1)
    prev = (t_start.astimezone(timezone.utc) - timedelta(days=1)).strftime("%Y_%m_%d")
    days.add(prev)
    prefix = f"{stream_name}_"
    names = []
    for day in sorted(days):
        folder = data_dir / day
        if not folder.is_dir():
            continue
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file() and e.name.startswith(prefix) and e.name.endswith(".xrif"):
                    names.append(f"{day}/{e.name}")
    names.sort(key=lambda p: os.path.basename(p))
    return names


def _detect_layout(data_dir: Path) -> str:
    """Return ``'flat'`` or ``'dated'`` based on directory contents."""
    with os.scandir(data_dir) as it:
        for e in it:
            if e.is_dir() and _DATE_FOLDER_RE.match(e.name):
                return "dated"
            if e.is_file() and e.name.endswith(".xrif"):
                return "flat"
    return "flat"


def load_or_build_xrif_index(params) -> list[str]:
    """
    Return a sorted list of xrif relative names for ``wfs_data_path``.

    Uses ``wfs_index_cache`` when present and still covers the requested window.
    """
    data_dir = Path(os.path.expanduser(params["wfs_data_path"]))
    stream_name = str(params.get("stream_name", "camwfs"))
    t_start, t_end = resolve_time_window(params)
    cache_path = params.get("wfs_index_cache")
    if cache_path:
        cache_path = Path(os.path.expanduser(cache_path))

    layout = params.get("wfs_layout")
    if layout is None:
        layout = _detect_layout(data_dir)

    end_token = filename_time_token(t_end)

    if cache_path and cache_path.is_file():
        with open(cache_path, encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            last_base = os.path.basename(names[-1])
            last_dt = parse_xrif_filename_time(last_base)
            if last_dt is not None and filename_time_token(last_dt) >= end_token:
                log.info("Using xrif index cache %s (%d files)", cache_path, len(names))
                return names
            log.info(
                "Index cache ends before t_end (%s < %s); rescanning",
                last_base,
                end_token,
            )

    log.info("Scanning %s for %s xrif files (layout=%s) ...", data_dir, stream_name, layout)
    if layout == "dated":
        names = _list_xrif_names_dated(data_dir, stream_name, t_start, t_end)
    else:
        names = _list_xrif_names_flat(data_dir, stream_name)
    log.info("Found %d xrif files", len(names))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names))
            if names:
                fh.write("\n")
        log.info("Wrote index cache %s", cache_path)
    return names


def select_xrif_files(names: list[str], t_start, t_end) -> list[str]:
    """
    Select files whose filename timestamps fall in ``[t_start, t_end)``,
    plus the last file before ``t_start``.
    """
    if not names:
        return []
    start_token = filename_time_token(t_start)
    end_token = filename_time_token(t_end)
    keys = []
    for n in names:
        dt = parse_xrif_filename_time(os.path.basename(n))
        keys.append(filename_time_token(dt) if dt is not None else "")

    i0 = bisect.bisect_left(keys, start_token)
    i1 = bisect.bisect_left(keys, end_token)
    if i0 > 0:
        i0 -= 1
    selected = names[i0:i1]
    if not selected and i0 < len(names):
        selected = names[i0 : i0 + 1]
    return selected


# ---------------------------------------------------------------------------
# XRIF I/O
# ---------------------------------------------------------------------------


def read_xrif_file(path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one xrif file.

    Returns
    -------
    data : float32 ndarray, shape (N, H, W)
    times : datetime64[us] ndarray, shape (N,)
    """
    import fixr

    path = os.fspath(path)
    with open(path, "rb") as fh:
        data = fixr.xrif2numpy(fh)
        timing = fixr.xrif2numpy(fh)
    data = np.asarray(data)
    while data.ndim > 3 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: unexpected data shape {data.shape}")
    times = timing_rows_to_datetimes(timing)
    if times.shape[0] != data.shape[0]:
        n = min(times.shape[0], data.shape[0])
        log.warning(
            "%s: data/timing length mismatch %d vs %d; truncating to %d",
            path,
            data.shape[0],
            times.shape[0],
            n,
        )
        data = data[:n]
        times = times[:n]
    return data.astype(np.float32, copy=False), times


def _window_mask(times: np.ndarray, t_start, t_end) -> np.ndarray:
    t0 = np.datetime64(parse_utc(t_start).replace(tzinfo=None), "us")
    t1 = np.datetime64(parse_utc(t_end).replace(tzinfo=None), "us")
    return (times >= t0) & (times < t1)


def iter_wfs_frames(params):
    """
    Yield ``(data, times, path)`` for each xrif file intersecting the window.

    Frames outside ``[t_start, t_end)`` are dropped per file.
    """
    data_dir = Path(os.path.expanduser(params["wfs_data_path"]))
    t_start, t_end = resolve_time_window(params)
    names = load_or_build_xrif_index(params)
    selected = select_xrif_files(names, t_start, t_end)
    if not selected:
        raise FileNotFoundError(
            f"No xrif files found for window [{t_start.isoformat()}, {t_end.isoformat()})"
        )
    log.info("Selected %d xrif files for window", len(selected))
    for rel in selected:
        path = data_dir / rel
        data, times = read_xrif_file(path)
        keep = _window_mask(times, t_start, t_end)
        if not np.any(keep):
            continue
        yield data[keep], times[keep], str(path)


def load_wfs_frames(params) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all WFS frames in the configured time window.

    Returns
    -------
    cube : float32 (N, H, W)
    times : datetime64[us] (N,)
    """
    chunks = []
    tchunks = []
    for data, times, path in iter_wfs_frames(params):
        log.info("  %s -> %d frames", os.path.basename(path), data.shape[0])
        chunks.append(data)
        tchunks.append(times)
    if not chunks:
        raise RuntimeError("No frames remained after time-window filtering")
    cube = np.concatenate(chunks, axis=0)
    times = np.concatenate(tchunks, axis=0)
    log.info("Loaded %d frames total, shape %s", cube.shape[0], cube.shape[1:])
    return cube, times


# ---------------------------------------------------------------------------
# Dark lookup
# ---------------------------------------------------------------------------


def parse_dark_timestamp(path_like) -> datetime:
    name = Path(path_like).name
    m = _DARK_TS_RE.search(name)
    if not m:
        raise ValueError(f"No __T timestamp in dark filename: {name}")
    main = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    frac = m.group(2) or ""
    if not frac:
        return main
    usec = int((frac[:6]).ljust(6, "0"))
    return main + timedelta(microseconds=usec)


def find_closest_wfs_dark(
    target,
    dark_dir,
    *,
    name_filter: str | None = None,
    expected_shape=(120, 120),
) -> Path:
    """Pick the closest-in-time ``camwfs-dark*__T*.fits`` under ``dark_dir``."""
    dark_dir = Path(os.path.expanduser(dark_dir))
    if not dark_dir.is_dir():
        raise FileNotFoundError(f"Dark directory does not exist: {dark_dir}")
    target_dt = parse_utc(target)
    catalog = []
    with os.scandir(dark_dir) as it:
        for e in it:
            if not e.name.startswith("camwfs-dark") or ".fits" not in e.name:
                continue
            if name_filter and name_filter not in e.name:
                continue
            try:
                dt = parse_dark_timestamp(e.name)
            except ValueError:
                continue
            catalog.append((abs((dt - target_dt).total_seconds()), dt, Path(e.path)))
    if not catalog:
        raise FileNotFoundError(
            f"No matching camwfs darks in {dark_dir}"
            + (f" with filter {name_filter!r}" if name_filter else "")
        )
    catalog.sort(key=lambda x: x[0])
    for delta, dt, path in catalog:
        with fits.open(path, memmap=False) as hdul:
            shape = np.asarray(hdul[0].data).shape
        if expected_shape is not None and tuple(shape) != tuple(expected_shape):
            log.warning(
                "Skipping dark %s: shape %s != %s", path.name, shape, expected_shape
            )
            continue
        log.info(
            "Selected dark %s (Δt = %.1f s from %s)",
            path.name,
            delta,
            target_dt.isoformat(),
        )
        return path
    raise RuntimeError(
        f"No dark with shape {expected_shape} found in {dark_dir}"
        + (f" with filter {name_filter!r}" if name_filter else "")
    )


def resolve_dark(params, target_time) -> tuple[np.ndarray, str]:
    """Return ``(dark_array, dark_path)`` from config."""
    if params.get("wfs_dark_path"):
        path = os.path.expanduser(str(params["wfs_dark_path"]))
        dark = np.asarray(fits.getdata(path), dtype=np.float32)
        log.info("Using explicit dark %s", path)
        return dark, path
    dark_dir = params.get("wfs_dark_dir")
    if not dark_dir:
        raise ValueError("provide wfs_dark_path or wfs_dark_dir")
    path = find_closest_wfs_dark(
        target_time,
        dark_dir,
        name_filter=params.get("dark_name_filter"),
        expected_shape=tuple(params.get("wfs_shape", (120, 120))),
    )
    dark = np.asarray(fits.getdata(path), dtype=np.float32)
    return dark, str(path)


# ---------------------------------------------------------------------------
# Cacao calibration loading
# ---------------------------------------------------------------------------


def _fits_float32(path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=np.float32)


def resolve_calib_paths(params) -> dict:
    """Resolve mask / ref / modesWFS / DMmodes / dmmask paths from config."""
    calib_dir = params.get("cacao_calib_dir")
    aocalib_set = params.get("aocalib_set", "tweeter001")
    base = None
    if calib_dir:
        base = Path(os.path.expanduser(calib_dir)) / aocalib_set

    def _one(key, default_name):
        if params.get(key):
            return Path(os.path.expanduser(str(params[key])))
        if base is None:
            raise ValueError(f"missing {key} and cacao_calib_dir")
        return base / default_name

    paths = {
        "wfs_mask_path": _one("wfs_mask_path", "aol1_wfsmask.fits"),
        "wfs_ref_path": _one("wfs_ref_path", "aol1_wfsref.fits"),
        "modes_wfs_path": _one("modes_wfs_path", "aol1_modesWFS.fits"),
        "dm_mask_path": _one("dm_mask_path", "aol1_dmmask.fits"),
    }
    if params.get("dm_modes_path"):
        paths["dm_modes_path"] = Path(os.path.expanduser(str(params["dm_modes_path"])))
    else:
        if base is None:
            raise ValueError("missing dm_modes_path and cacao_calib_dir")
        modes_wfs = _fits_float32(paths["modes_wfs_path"])
        nmodes = int(modes_wfs.shape[0])
        candidates = [
            base / "aol1_DMmodes.fits",
            base / "aol1_DMmodes_WFSdiag.fits",
        ]
        chosen = None
        for cand in candidates:
            if not cand.is_file():
                continue
            with fits.open(cand, memmap=True) as hdul:
                n = int(hdul[0].header.get("NAXIS3", hdul[0].data.shape[0]))
            if n == nmodes:
                chosen = cand
                break
        if chosen is None:
            raise ValueError(
                f"No DMmodes file with nmodes={nmodes} under {base}; "
                "set dm_modes_path explicitly"
            )
        if chosen.name != "aol1_DMmodes.fits":
            log.warning(
                "aol1_DMmodes mode count != modesWFS (%d); using %s",
                nmodes,
                chosen.name,
            )
        paths["dm_modes_path"] = chosen
    for k, p in paths.items():
        if not Path(p).is_file():
            raise FileNotFoundError(f"{k}: {p}")
    return {k: str(v) for k, v in paths.items()}


def load_reconstructor(params) -> dict:
    """
    Load and preprocess cacao matrices for reconstruction.

    Returns a dict with arrays ready for :func:`wfs_to_delta_wavefront`.
    """
    paths = resolve_calib_paths(params)
    wfs_mask = _fits_float32(paths["wfs_mask_path"])
    wfs_ref = _fits_float32(paths["wfs_ref_path"])
    modes_wfs = _fits_float32(paths["modes_wfs_path"])
    dm_modes = _fits_float32(paths["dm_modes_path"])
    dm_mask = _fits_float32(paths["dm_mask_path"])

    if modes_wfs.ndim != 3:
        raise ValueError(f"modesWFS must be 3D, got {modes_wfs.shape}")
    if dm_modes.ndim != 3:
        raise ValueError(f"DMmodes must be 3D, got {dm_modes.shape}")
    if modes_wfs.shape[0] != dm_modes.shape[0]:
        raise ValueError(
            f"mode count mismatch: modesWFS {modes_wfs.shape[0]} vs "
            f"DMmodes {dm_modes.shape[0]}"
        )
    if modes_wfs.shape[1:] != wfs_mask.shape:
        raise ValueError(
            f"modesWFS spatial shape {modes_wfs.shape[1:]} != mask {wfs_mask.shape}"
        )
    if dm_modes.shape[1:] != dm_mask.shape:
        raise ValueError(
            f"DMmodes spatial shape {dm_modes.shape[1:]} != dmmask {dm_mask.shape}"
        )

    ref_masked = wfs_ref * wfs_mask
    ref_sum = float(np.sum(ref_masked))
    if ref_sum == 0:
        raise ValueError("wfsref has zero masked sum")
    wfs_ref_norm = ref_masked / ref_sum

    nmodes = modes_wfs.shape[0]
    modes_flat = modes_wfs.reshape(nmodes, -1)
    mode_norms2 = np.sum(modes_flat * modes_flat, axis=1).astype(np.float32)
    mode_norms2 = np.where(mode_norms2 == 0, 1.0, mode_norms2)
    dm_flat = dm_modes.reshape(nmodes, -1)

    method = str(params.get("recon_method", "modenorm")).lower()
    pinv_mat = None
    if method == "pinv":
        pix = wfs_mask.ravel() > 0
        pinv_mat = np.linalg.pinv(modes_flat[:, pix]).astype(np.float32)

    scale = float(params.get("dm_units_to_nm", 1000.0)) * float(
        params.get("opd_factor", 2.0)
    )
    recon_sign = float(params.get("recon_sign", 1.0))

    return {
        "paths": paths,
        "wfs_mask": wfs_mask,
        "wfs_ref_norm": wfs_ref_norm.astype(np.float32),
        "modes_flat": modes_flat.astype(np.float32),
        "mode_norms2": mode_norms2,
        "dm_flat": dm_flat.astype(np.float32),
        "dm_mask": dm_mask,
        "dm_shape": tuple(dm_modes.shape[1:]),
        "nmodes": nmodes,
        "method": method,
        "pinv_mat": pinv_mat,
        "scale": scale,
        "recon_sign": recon_sign,
    }


def preprocess_wfs_frames(cube, dark, wfs_mask, wfs_ref_norm) -> np.ndarray:
    """
    Dark-subtract, mask, flux-normalize, and subtract the WFS reference.

    Parameters
    ----------
    cube : (N, H, W)
    dark, wfs_mask, wfs_ref_norm : (H, W); ``wfs_ref_norm`` already masked-sum-1

    Returns
    -------
    residual : float32 (N, H, W)
    """
    cube = np.asarray(cube, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    wfs_mask = np.asarray(wfs_mask, dtype=np.float32)
    wfs_ref_norm = np.asarray(wfs_ref_norm, dtype=np.float32)
    if cube.shape[1:] != dark.shape:
        raise ValueError(f"cube spatial {cube.shape[1:]} != dark {dark.shape}")
    ds = (cube - dark) * wfs_mask
    frame_sums = np.sum(ds, axis=(1, 2), keepdims=True)
    frame_sums = np.where(frame_sums == 0, 1.0, frame_sums)
    normed = ds / frame_sums
    return (normed - wfs_ref_norm).astype(np.float32)


def project_modal_coeffs(residual, recon) -> np.ndarray:
    """Project preprocessed residuals onto modesWFS. Returns ``(N, nmodes)``."""
    resid = np.asarray(residual, dtype=np.float32)
    n = resid.shape[0]
    flat = resid.reshape(n, -1)
    method = recon["method"]
    if method == "modenorm":
        # c_k = <M_k, s> / <M_k, M_k>
        dots = flat @ recon["modes_flat"].T
        coeffs = dots / recon["mode_norms2"][None, :]
    elif method == "pinv":
        pix = recon["wfs_mask"].ravel() > 0
        coeffs = flat[:, pix] @ recon["pinv_mat"].T
    else:
        raise ValueError(f"unknown recon_method {method!r}")
    return coeffs.astype(np.float32)


def coeffs_to_delta_wavefront(coeffs, recon) -> np.ndarray:
    """Map modal coefficients to a DM-grid OPD cube in nm."""
    coeffs = np.asarray(coeffs, dtype=np.float32)
    dm = coeffs @ recon["dm_flat"]
    dm = dm.reshape((-1,) + recon["dm_shape"])
    dm = dm * recon["dm_mask"] * (recon["recon_sign"] * recon["scale"])
    return dm.astype(np.float32)


def wfs_to_delta_wavefront(cube, times, params, *, recon=None, dark=None):
    """
    Convert raw WFS frames to per-frame delta wavefronts in nm OPD.

    Returns
    -------
    delta_wf_nm : float32 (N, dmY, dmX)
    modal_coeffs : float32 (N, nmodes)
    meta : dict with dark path, calib paths, reconstructor
    """
    if recon is None:
        recon = load_reconstructor(params)
    if dark is None:
        target = times[0] if len(times) else params.get("t_start")
        if isinstance(target, np.datetime64):
            target = (
                target.astype("datetime64[us]")
                .astype(datetime)
                .replace(tzinfo=timezone.utc)
            )
        dark, dark_path = resolve_dark(params, target)
    else:
        dark_path = params.get("wfs_dark_path", "")
        dark = np.asarray(dark, dtype=np.float32)

    residual = preprocess_wfs_frames(
        cube, dark, recon["wfs_mask"], recon["wfs_ref_norm"]
    )
    coeffs = project_modal_coeffs(residual, recon)
    delta = coeffs_to_delta_wavefront(coeffs, recon)
    meta = {
        "dark_path": dark_path,
        "calib_paths": recon["paths"],
        "recon": recon,
        "n_frames": int(delta.shape[0]),
        "nmodes": recon["nmodes"],
    }
    return delta, coeffs, meta


def opd_nm_to_phase(opd_nm, wavelength_m=WFS_WAVELENGTH) -> np.ndarray:
    """Convert OPD in nm to phase in radians: ``phi = 2π * OPD / λ``."""
    opd_m = np.asarray(opd_nm, dtype=np.float64) * 1e-9
    return (2.0 * np.pi * opd_m / float(wavelength_m)).astype(np.float32)


# ---------------------------------------------------------------------------
# Open-loop integration
# ---------------------------------------------------------------------------


def integrate_open_loop(delta_wf, times, params):
    """
    Sum consecutive delta wavefronts into windows of ``integration_time_s``.

    Returns
    -------
    full_wf_nm : float32 (M, dmY, dmX)
    window_start_times : datetime64[us] (M,)
    n_per_window : int array (M,)
    """
    delta_wf = np.asarray(delta_wf, dtype=np.float32)
    times = np.asarray(times)
    if times.dtype != np.dtype("datetime64[us]"):
        times = times.astype("datetime64[us]")
    n = delta_wf.shape[0]
    if n == 0:
        raise ValueError("empty delta_wf")
    integ_s = float(params.get("integration_time_s", 10.0))
    gain = float(params.get("integration_gain", 1.0))
    if integ_s <= 0:
        raise ValueError(f"integration_time_s must be positive, got {integ_s}")

    groups = []
    start = 0
    for i in range(1, n):
        dt_sec = (times[i] - times[start]) / np.timedelta64(1, "s")
        if float(dt_sec) >= integ_s:
            groups.append((start, i))
            start = i
    if start < n:
        groups.append((start, n))

    h, w = delta_wf.shape[1], delta_wf.shape[2]
    m = len(groups)
    full = np.zeros((m, h, w), dtype=np.float32)
    t0 = np.empty(m, dtype="datetime64[us]")
    n_per = np.zeros(m, dtype=np.int32)
    for i, (a, b) in enumerate(groups):
        full[i] = gain * np.sum(delta_wf[a:b], axis=0, dtype=np.float64)
        t0[i] = times[a]
        n_per[i] = b - a
    return full, t0, n_per


class OpenLoopAccumulator:
    """Stream-friendly accumulator that emits completed integration windows."""

    def __init__(self, integration_time_s=10.0, integration_gain=1.0, dm_shape=None):
        self.integ_s = float(integration_time_s)
        self.gain = float(integration_gain)
        self.dm_shape = dm_shape
        self._sum = None
        self._t0 = None
        self._n = 0
        self.windows = []
        self.window_times = []
        self.n_per_window = []

    def _ensure(self, shape):
        if self._sum is None:
            self.dm_shape = shape
            self._sum = np.zeros(shape, dtype=np.float64)

    def add(self, delta_chunk, times_chunk):
        """Add a chunk of delta frames; flush completed windows."""
        delta_chunk = np.asarray(delta_chunk, dtype=np.float32)
        times_chunk = np.asarray(times_chunk).astype("datetime64[us]")
        if delta_chunk.shape[0] == 0:
            return
        self._ensure(delta_chunk.shape[1:])
        for i in range(delta_chunk.shape[0]):
            t = times_chunk[i]
            if self._t0 is None:
                self._t0 = t
                self._sum[...] = 0.0
                self._n = 0
            dt_sec = (t - self._t0) / np.timedelta64(1, "s")
            if self._n > 0 and float(dt_sec) >= self.integ_s:
                self.windows.append((self.gain * self._sum).astype(np.float32))
                self.window_times.append(self._t0)
                self.n_per_window.append(self._n)
                self._t0 = t
                self._sum[...] = 0.0
                self._n = 0
            self._sum += delta_chunk[i]
            self._n += 1

    def finish(self):
        """Flush the final partial window if it has any frames."""
        if self._n > 0 and self._sum is not None:
            self.windows.append((self.gain * self._sum).astype(np.float32))
            self.window_times.append(self._t0)
            self.n_per_window.append(self._n)
            self._n = 0
        if not self.windows:
            return (
                np.zeros((0,) + (self.dm_shape or (0, 0)), dtype=np.float32),
                np.array([], dtype="datetime64[us]"),
                np.array([], dtype=np.int32),
            )
        return (
            np.stack(self.windows, axis=0),
            np.asarray(self.window_times, dtype="datetime64[us]"),
            np.asarray(self.n_per_window, dtype=np.int32),
        )


# ---------------------------------------------------------------------------
# Config / driver / CLI
# ---------------------------------------------------------------------------


def read_loop_recon_config(config_path) -> dict:
    """Read a loop-reconstruction config (``name = value``; ``#`` comments)."""
    params = {}
    path = os.fspath(config_path)
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value_str = line.partition("=")
            key = key.strip()
            value_str = value_str.strip()
            if not key:
                continue
            try:
                params[key] = ast.literal_eval(value_str)
            except (SyntaxError, ValueError) as e:
                raise ValueError(
                    f"{path}:{lineno}: cannot parse {key!r} = {value_str!r}"
                ) from e
    return params


def check_loop_recon_config(params) -> list[str]:
    bad = []
    if not isinstance(params.get("wfs_data_path"), str) or not params["wfs_data_path"].strip():
        bad.append("wfs_data_path")
    if "t_start" not in params:
        bad.append("t_start")
    has_end = params.get("t_end") is not None and str(params.get("t_end", "")).strip() != ""
    has_dur = params.get("duration_s") is not None
    if has_end == has_dur:
        bad.append("t_end|duration_s")
    if not params.get("wfs_dark_path") and not params.get("wfs_dark_dir"):
        bad.append("wfs_dark_path|wfs_dark_dir")
    if not params.get("cacao_calib_dir") and not params.get("modes_wfs_path"):
        bad.append("cacao_calib_dir|modes_wfs_path")
    if not isinstance(params.get("output_dir"), str) or not params["output_dir"].strip():
        bad.append("output_dir")
    return bad


def _configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, LOG_NAME)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger("tools4magaox.wfs_proc")
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    root.setLevel(logging.INFO)
    root.propagate = False
    log.propagate = True
    log.info("Log file: %s", log_path)


def _save_fits(data, path, header_items=None):
    hdu = fits.PrimaryHDU(data=np.asarray(data))
    if header_items:
        for k, v in header_items.items():
            try:
                hdu.header[str(k)[:8]] = v
            except Exception:
                hdu.header[str(k)[:8]] = str(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def _dt64_to_iso(t) -> str:
    if isinstance(t, np.datetime64):
        return str(t)
    return parse_utc(t).isoformat()


def run_loop_reconstruction_from_config(params, config_source_path=None):
    """
    Run the full loop-reconstruction pipeline from a config dict.

    Streams frames file-by-file so a long window does not require holding every
    delta wavefront in memory.
    """
    missing = check_loop_recon_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")

    output_dir = os.path.expanduser(params["output_dir"].strip())
    params = dict(params)
    params["output_dir"] = output_dir
    if not params.get("wfs_index_cache"):
        params["wfs_index_cache"] = str(
            Path(output_dir).resolve().parent / "camwfs_index.txt"
        )

    _configure_logging(output_dir)
    if config_source_path:
        src = os.path.abspath(os.fspath(config_source_path))
        if os.path.isfile(src):
            dst = os.path.join(output_dir, CONFIG_SNAPSHOT_NAME)
            shutil.copy2(src, dst)
            log.info("Saved config snapshot: %s", dst)

    recon = load_reconstructor(params)
    log.info(
        "Reconstructor: method=%s nmodes=%d dm_shape=%s scale=%.4g",
        recon["method"],
        recon["nmodes"],
        recon["dm_shape"],
        recon["scale"],
    )

    t_start, t_end = resolve_time_window(params)
    dark, dark_path = resolve_dark(params, t_start)
    if dark.shape != recon["wfs_mask"].shape:
        raise ValueError(
            f"dark shape {dark.shape} incompatible with mask {recon['wfs_mask'].shape}"
        )

    save_delta = bool(params.get("save_delta_cube", False))
    save_coeffs = bool(params.get("save_modal_coeffs", True))
    acc = OpenLoopAccumulator(
        integration_time_s=params.get("integration_time_s", 10.0),
        integration_gain=params.get("integration_gain", 1.0),
        dm_shape=recon["dm_shape"],
    )

    all_times = []
    all_coeffs = []
    all_deltas = []
    n_total = 0

    for data, times, path in iter_wfs_frames(params):
        residual = preprocess_wfs_frames(
            data, dark, recon["wfs_mask"], recon["wfs_ref_norm"]
        )
        coeffs = project_modal_coeffs(residual, recon)
        delta = coeffs_to_delta_wavefront(coeffs, recon)
        acc.add(delta, times)
        n_total += data.shape[0]
        all_times.append(times)
        if save_coeffs:
            all_coeffs.append(coeffs)
        if save_delta:
            all_deltas.append(delta)
        log.info(
            "Processed %s: %d frames (running total %d)",
            os.path.basename(path),
            data.shape[0],
            n_total,
        )

    if n_total == 0:
        raise RuntimeError("No frames processed")

    full_wf, win_times, n_per = acc.finish()
    frame_times = np.concatenate(all_times)

    hdr = {
        "DARK": os.path.basename(dark_path),
        "DARKPATH": dark_path[-68:] if len(dark_path) > 68 else dark_path,
        "MODESWFS": os.path.basename(recon["paths"]["modes_wfs_path"]),
        "DMMODES": os.path.basename(recon["paths"]["dm_modes_path"]),
        "RECONMTH": recon["method"],
        "NMODES": recon["nmodes"],
        "SCALE": recon["scale"],
        "RSIGN": recon["recon_sign"],
        "INTEGS": float(params.get("integration_time_s", 10.0)),
        "NFRAMES": n_total,
        "NWINDOW": int(full_wf.shape[0]),
        "TSTART": t_start.isoformat(),
        "TEND": t_end.isoformat(),
        "BUNIT": "nm",
        "COMMENT": "OPD in nm (surface*dm_units_to_nm*opd_factor)",
    }
    full_path = _save_fits(full_wf, os.path.join(output_dir, "full_wf_nm.fits"), hdr)
    log.info("Wrote %s shape %s", full_path, full_wf.shape)

    if save_coeffs and all_coeffs:
        coeffs = np.concatenate(all_coeffs, axis=0)
        cpath = _save_fits(coeffs, os.path.join(output_dir, "modal_coeffs.fits"), hdr)
        log.info("Wrote %s shape %s", cpath, coeffs.shape)

    if save_delta and all_deltas:
        delta = np.concatenate(all_deltas, axis=0)
        dpath = _save_fits(delta, os.path.join(output_dir, "delta_wf_nm.fits"), hdr)
        log.info("Wrote %s shape %s", dpath, delta.shape)

    times_path = os.path.join(output_dir, "times.txt")
    with open(times_path, "w", encoding="utf-8") as fh:
        fh.write("# frame_index iso_time\n")
        for i, t in enumerate(frame_times):
            fh.write(f"{i} {_dt64_to_iso(t)}\n")
        fh.write("# window_index iso_time n_frames\n")
        for i, (t, n) in enumerate(zip(win_times, n_per)):
            fh.write(f"W {i} {_dt64_to_iso(t)} {int(n)}\n")
    log.info("Wrote %s", times_path)

    prov = os.path.join(output_dir, "provenance.txt")
    with open(prov, "w", encoding="utf-8") as fh:
        fh.write(f"dark_path = {dark_path}\n")
        for k, v in recon["paths"].items():
            fh.write(f"{k} = {v}\n")
        fh.write(f"n_frames = {n_total}\n")
        fh.write(f"n_windows = {full_wf.shape[0]}\n")
        if full_wf.size:
            masked = full_wf * recon["dm_mask"]
            rms = float(np.sqrt(np.mean(masked**2)))
            fh.write(f"full_wf_rms_nm = {rms:.6g}\n")
            log.info("Full-WF RMS (masked) = %.4g nm", rms)

    return {
        "full_wf_nm": full_wf,
        "window_times": win_times,
        "n_per_window": n_per,
        "frame_times": frame_times,
        "n_frames": n_total,
        "dark_path": dark_path,
        "output_dir": output_dir,
        "recon": recon,
    }


def cli_loop_recon(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconstruct open-loop wavefronts from camwfs xrif + cacao calib"
    )
    parser.add_argument("config", help="Path to loop-reconstruction config file")
    args = parser.parse_args(argv)
    params = read_loop_recon_config(args.config)
    return run_loop_reconstruction_from_config(params, config_source_path=args.config)


if __name__ == "__main__":
    cli_loop_recon()
