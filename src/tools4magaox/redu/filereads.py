# filereads.py
# 2026/04/07
# the purpose of this file is to coordinate how files and telemetry is read into the reduction pipeline
import glob
import logging
import os
from collections import Counter
from tqdm import trange, tqdm
from astropy.table import Table
try:
    from . import darks as md
    from .darks import _detect_camera_tag_from_header
except ImportError:
    import darks as md
    from darks import _detect_camera_tag_from_header

from astropy.io import fits
import numpy as np
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_REDU_ASCII_FMT = "ascii.commented_header"


def _load_fits_primary_float32(path):
    """Load primary HDU data as float32 (reduces RAM vs float64 cubes)."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)

def _load_fits_primary_float64(path):
    """Load primary HDU data as float64 (reduces RAM vs float64 cubes)."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)

def _save_fits_primary_float32(data, path):
    """Save primary HDU data as float32 (reduces RAM vs float64 cubes)."""
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32)).writeto(path, overwrite=True)

def _save_fits_primary_float64(data, path):
    """Save primary HDU data as float64 (reduces RAM vs float64 cubes)."""
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float64)).writeto(path, overwrite=True)

def read_redu_table(path):
    """Load a pipeline ASCII metadata table (commented header)."""
    return Table.read(path, format=_REDU_ASCII_FMT)


def write_redu_table(table, path):
    """Write a pipeline ASCII metadata table; creates parent directory if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    table.write(path, format=_REDU_ASCII_FMT, overwrite=True)


def find_camera_files(obs_path, obs_dir, camera="camsci1", max_files=-1):
    """
    Find FITS files in ``{obs_path}{obs_dir}/{camera}/`` and return sorted paths.
    """
    file_paths = glob.glob(f"{obs_path}{obs_dir}/{camera}/*")
    if max_files < 0:
        return sorted(file_paths)
    if max_files > len(file_paths):
        max_files = len(file_paths)
    return sorted(file_paths)[:max_files]


def resolve_masterdark_search_dir(redu_path, masterdark_dir):
    """Directory tree to search for ``*masterdark*.fits*`` (recursive)."""
    if masterdark_dir is not None:
        s = os.path.expanduser(os.fspath(masterdark_dir)).strip()
        if s:
            return s
    return os.path.expanduser(os.fspath(redu_path)).strip()


def attach_masterdarks(file_table, dark_search_dir, camera):
    """
    Add ``masterdark_path`` to each row using darks matched from telemetry.
    """
    unique_configs = md.unique_telemetry_configs_for_dark_lookup(file_table, camera=camera)
    dark_dictionary = md.lookup_masterdarks_from_telemetry_table(
        unique_configs,
        redu_dir=dark_search_dir,
        camera=camera,
    )
    for config in dark_dictionary:
        if len(config["masterdark_paths"]) == 0:
            log.warning("No master dark found for configuration: %s", config)
    return md.merge_file_table_with_darks(file_table, dark_dictionary)


def _coerce_hdr_number(value):
    """Parse a numeric FITS header value; missing or invalid -> NaN."""
    if value is None:
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _config_key_part(value, col):
    """Hashable config value for majority grouping (missing values match)."""
    if col == "camera":
        return str(value)
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def pick_majority_config(file_table):
    """
    Find the detector configuration that appears in the most files.

    Returns a copy of ``file_table`` with a ``to_use`` column: 1 if the file
    matches the majority configuration, else 0.
    """
    cfg_cols = (
        "camera",
        "NAXIS1",
        "NAXIS2",
        "ROI_XCEN",
        "ROI_YCEN",
        "EMGAIN",
        "ADC_SPEED",
        "EXPTIME",
    )
    missing = [c for c in cfg_cols if c not in file_table.colnames]
    if missing:
        raise ValueError(
            f"file_table missing required columns for majority picking: {missing}"
        )

    if len(file_table) == 0:
        out = file_table.copy()
        out["to_use"] = np.array([], dtype=int)
        return out

    keys = [
        tuple(_config_key_part(file_table[col][i], col) for col in cfg_cols)
        for i in range(len(file_table))
    ]
    majority_key = Counter(keys).most_common(1)[0][0]
    mask = np.array([k == majority_key for k in keys], dtype=bool)

    out = file_table.copy()
    out["to_use"] = mask.astype(int)
    return out


def file_table_static_from_full(full_table):
    """Telemetry + masterdark only (no pipeline filter columns)."""
    cols = [c for c in full_table.colnames if c != "to_use"]
    return full_table[cols].copy()


def init_file_table_output(full_table):
    """
    One row per file: majority filter plus placeholders for process pipeline outputs.
    """
    n = len(full_table)
    to_use = np.asarray(full_table["to_use"], dtype=int)
    out = Table()
    out["filename"] = full_table["filename"].copy()
    out["pass_majority_config"] = to_use.copy()
    out["shift_x"] = np.full(n, np.nan, dtype=float)
    out["shift_y"] = np.full(n, np.nan, dtype=float)
    out["recenter_shift_x"] = np.full(n, np.nan, dtype=float)
    out["recenter_shift_y"] = np.full(n, np.nan, dtype=float)
    out["center_stack_id"] = np.full(n, -1, dtype=int)
    out["max_point_radius"] = np.full(n, np.nan, dtype=float)
    out["speckle_intensity"] = np.full(n, np.nan, dtype=float)
    out["rms_deviation"] = np.full(n, np.nan, dtype=float)
    out["used_in_reduction"] = np.zeros(n, dtype=int)
    return out


PROCESS_FILE_TABLE_OUTPUT_COLUMNS = (
    "filename",
    "pass_majority_config",
    "shift_x",
    "shift_y",
    "recenter_shift_x",
    "recenter_shift_y",
    "center_stack_id",
    "max_point_radius",
    "speckle_intensity",
    "rms_deviation",
    "used_in_reduction",
)


_PROCESS_FILTER_COLUMNS = {
    "center_stack_id": ("int", -1),
    "recenter_shift_x": ("float", np.nan),
    "recenter_shift_y": ("float", np.nan),
    "max_point_radius": ("float", np.nan),
    "speckle_intensity": ("float", np.nan),
    "rms_deviation": ("float", np.nan),
    "used_in_reduction": ("int", 0),
}


def ensure_process_filter_columns(file_table_output):
    """Add process step-4 filter columns if loading an older output table."""
    n = len(file_table_output)
    for col, (kind, default) in _PROCESS_FILTER_COLUMNS.items():
        if col in file_table_output.colnames:
            continue
        if kind == "float":
            file_table_output[col] = np.full(n, default, dtype=float)
        else:
            file_table_output[col] = np.full(n, default, dtype=int)
    return file_table_output


def prune_process_output_table(file_table_output):
    """Return a copy with only columns used by the process pipeline."""
    file_table_output = ensure_process_filter_columns(file_table_output)
    keep = [c for c in PROCESS_FILE_TABLE_OUTPUT_COLUMNS if c in file_table_output.colnames]
    return file_table_output[keep].copy()


def ephemeral_file_table_with_to_use(file_table_static, file_table_output):
    """Static roster plus synthetic ``to_use`` from ``pass_majority_config``."""
    t = file_table_static.copy()
    t["to_use"] = np.asarray(file_table_output["pass_majority_config"], dtype=int)
    return t


def update_file_table_output(
    file_table_output,
    row_indices,
    shifts,
    center_stack_id=None,
    shift_cols=("shift_y", "shift_x"),
):
    """
    Write registration shifts for rows in the full output table.

    Parameters
    ----------
    file_table_output : astropy.table.Table
        Table from :func:`init_file_table_output`.
    row_indices : array-like of int
        Row indices in ``file_table_output`` (full roster, not chunk-local).
    shifts : array-like, shape (N, 2)
        Per-frame correction shifts in pixels for ``scipy.ndimage.shift``, columns
        ``[shift_y, shift_x]`` (same order as :func:`center_spark.register_files_fft`).
    center_stack_id : int, optional
        Stack index when frames were coadded for centering (same value for every
        row in ``row_indices``). Only written to ``center_stack_id`` when
        ``shift_cols`` is the default first-pass pair.
    shift_cols : tuple of str
        Column names ``(shift_y_col, shift_x_col)`` to update.
    """
    rows = np.asarray(row_indices, dtype=int).ravel()
    s = np.asarray(shifts, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"shifts must have shape (N, 2), got {s.shape}")
    if s.shape[0] != rows.size:
        raise ValueError(
            f"shifts length {s.shape[0]} != number of row indices {rows.size}"
        )
    sy_col, sx_col = shift_cols
    file_table_output[sy_col][rows] = s[:, 0]
    file_table_output[sx_col][rows] = s[:, 1]
    if center_stack_id is not None and shift_cols == ("shift_y", "shift_x"):
        file_table_output["center_stack_id"][rows] = int(center_stack_id)
    return file_table_output


def pull_hdr_params(hdr, camera, darks=True):
    vals = {
            "DATE_OBS": hdr.get("DATE-OBS"),
            "PARANG": _coerce_hdr_number(hdr.get("PARANG")),
            "NAXIS1": _coerce_hdr_number(_find_hdr_val(hdr, "NAXIS1")),
            "NAXIS2": _coerce_hdr_number(_find_hdr_val(hdr, "NAXIS2")),
            "ROI_XCEN": _coerce_hdr_number(_find_hdr_val(hdr, "ROI_XCEN", camera_tag=camera)),
            "ROI_YCEN": _coerce_hdr_number(_find_hdr_val(hdr, "ROI_YCEN", camera_tag=camera)),
            "EMGAIN": _coerce_hdr_number(_find_hdr_val(hdr, "EMGAIN", camera_tag=camera)),
            "ADC_SPEED": _coerce_hdr_number(_find_hdr_val(hdr, "ADC SPEED", camera_tag=camera)),
            "EXPTIME": _coerce_hdr_number(_find_hdr_val(hdr, "EXPTIME", camera_tag=camera)),
            "SHUTTER": _find_hdr_val(hdr, "SHUTTER", camera_tag=camera),
        }
    return vals

def fits_telemetry_table(file_paths, camera=None):
    """
    Build an Astropy ``Table`` with one row per FITS file: filename, acquisition
    time (``DATE-OBS``), ``PARANG``, and camera/detector keywords from the header.

    Camera-specific keywords (``ROI_*``, ``EMGAIN``, ``ADC SPEED``, ``EXPTIME``,
    ``SHUTTER``) are resolved the same way as in ``darks.pull_hdr_params``.

    Parameters
    ----------
    file_paths : list of str or path-like, or a single path
        FITS files to summarize.
    camera : str, optional
        Detector id, e.g. ``"camsci1"``. If omitted, each file's camera is
        inferred from HIERARCH keys when possible; otherwise this value must be
        set for files without a detectable tag.

    Returns
    -------
    astropy.table.Table
        Columns: ``filename``, ``camera``, ``DATE_OBS``, ``PARANG``, ``NAXIS1``,
        ``NAXIS2``, ``ROI_XCEN``, ``ROI_YCEN``, ``EMGAIN``, ``ADC_SPEED``,
        ``EXPTIME``, ``SHUTTER``. Missing numeric keywords are stored as NaN.
    """

    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    rows = []
    for fp in file_paths:
        fp = os.fspath(fp)
        name = os.path.basename(fp)
        with fits.open(fp, memmap=False) as fh:
            hdr = fh[0].header
        cam = camera
        if cam is None:
            det = _detect_camera_tag_from_header(hdr)
            cam = det.lower() if det else None
        if cam is None:
            raise ValueError(
                f"Could not infer camera for {fp}; pass camera='camsci1' or 'camsci2'."
            )
        vals = pull_hdr_params(hdr, cam, darks=True)
        rows.append(
            {
                "filename": name,
                "camera": cam,
                "DATE_OBS": vals["DATE_OBS"],
                "PARANG": vals["PARANG"],
                "NAXIS1": vals["NAXIS1"],
                "NAXIS2": vals["NAXIS2"],
                "ROI_XCEN": vals["ROI_XCEN"],
                "ROI_YCEN": vals["ROI_YCEN"],
                "EMGAIN": vals["EMGAIN"],
                "ADC_SPEED": vals["ADC_SPEED"],
                "EXPTIME": vals["EXPTIME"],
                "SHUTTER": vals["SHUTTER"],
            }
        )
    return Table(rows)

def write_fits(data, save_path):
    data = np.asarray(data)
    hdu = fits.PrimaryHDU(data=data)
    hdu.writeto(save_path, overwrite=True)
    return save_path

# TODO: Make sure the cube isn't too large
def make_data_avg_cube(file_list, dark_data, n_avg=1, n_files=-1):
    dark_data = np.asarray(dark_data, dtype=np.float32)
    n_files = len(file_list) if n_files == -1 else n_files
    n_avg_data = n_files // n_avg
    avg_data = np.zeros((n_avg_data, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    for i in range(n_avg_data):
        #if i % 100 == 0:
        #    print(f"   Processing file {i*n_avg} to {i*n_avg + n_avg} out of {n_files}")
        for j in range(n_avg):
            with fits.open(file_list[i * n_avg + j], memmap=False) as hdul:
                demo_data = hdul[0].data
            avg_data[i] += np.asarray(demo_data, dtype=np.float32) - dark_data
        avg_data[i] /= n_avg
    return avg_data

def make_data_cube(file_list, dark_data, n_files=-1):
    dark_data = np.asarray(dark_data, dtype=np.float32)
    n_files = len(file_list) if n_files == -1 else n_files
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    for i in range(n_files):
        with fits.open(file_list[i], memmap=False) as hdul:
            demo_data = hdul[0].data
        data_cube[i] = np.asarray(demo_data, dtype=np.float32) - dark_data
    return data_cube

# TODO: Make a cube and keep relevant telemetry 

def make_science_cube(file_list, dark_data, n_files=-1, n_start=0):
    """
    :param file_list: list of the full path to files
    :param dark_data: a cube of the darks associated with these camera parameters
    :param coadd: how many frames we need to add
    :param n_files: how many files total
    :param n_start: where to start in the list of files
    """
    # logic to make sure the cubes aren't too bug
    n_files = len(file_list) if n_files > len(file_list) else n_files
    n_files = len(file_list) if n_files == -1 else n_files
    # make an empty data cube 
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    parang_cube = np.zeros(n_files)
    time_cube = np.zeros(n_files, dtype='datetime64[us]')

    for i in range(n_files):
        with fits.open(file_list[n_start+i]) as fh:
            test_data = fh[0].data
            # get the parang
            try:
                pg_ang = fh[0].header["PARANG"]
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with Parang in file:  {file_list[n_start+i]}")
                pg_ang = -1
            # get the file time 
            try:
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with timestamp in file:  {file_list[n_start+i]}")
                time_stamp = -1
            parang_cube[i] = pg_ang
            time_cube[i] = time_stamp
            data_cube[i] = np.asarray(test_data, dtype=np.float32)
    return data_cube, parang_cube, time_cube

def make_science_cube_coadd(file_list, dark_data, coadd=10, n_files=-1, n_start=0):
    """
    Docstring for make_science_cube
    
    :param file_list: list of the full path to files
    :param dark_data: a cube of the darks associated with these camera parameters
    :param coadd: how many frames we need to add
    :param n_files: how many files total
    :param n_start: where to start in the list of files
    """
    # logic to make sure the cubes aren't too bug
    n_files = len(file_list) if n_files > len(file_list) else n_files
    n_files = len(file_list) if n_files == -1 else n_files
    n_coadd_files = n_files // coadd
    dark_data = np.asarray(dark_data, dtype=np.float32)
    # make an empty data cube 
    data_cube = np.zeros((n_coadd_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    parang_cube = np.zeros((n_coadd_files, coadd))
    time_stamp_cube = np.zeros((n_coadd_files, coadd))

    for i in range(n_coadd_files):
        for j in range(coadd):
            fh = fits.open(file_list[n_start+i*coadd+j])
            test_data = fh[0].data
            data_cube[i] += np.asarray(test_data, dtype=np.float32) - dark_data
            try:
                pg_ang = fh[0].header["PARANG"]
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with Parang in file:  {file_list[i*coadd+j]}")
                pg_ang = -1
            parang_cube[i,j] = pg_ang
            time_stamp_cube[i,j] = time_stamp
            fh.close()
        data_cube[i] /= coadd
    return data_cube, parang_cube


# We don't want to do this by file number anymore, we might have large chunks of time missing
# this is the best way to make sure the images themselves don't get blurred
def coadd_by_frames(data, parang_cube, time_cube, frame_coadd=10):
    """
    Mean-coadd consecutive frames in groups of ``frame_coadd``.

    Parameters
    ----------
    data : ndarray, shape (N, H, W)
    parang_cube : ndarray, shape (N,)
    time_cube : array-like of observation times
    frame_coadd : int
        Number of consecutive frames per output slice.

    Returns
    -------
    data_out, parang_out, time_out
    """
    data = np.asarray(data, dtype=np.float32)
    parang_cube = np.asarray(parang_cube, dtype=float)
    times = _coerce_times_to_datetime64(time_cube)
    n = len(data)
    frame_coadd = max(1, int(frame_coadd))
    n_out = n // frame_coadd
    if n_out == 0:
        raise ValueError(
            f"coadd_by_frames: need at least {frame_coadd} frames, got {n}"
        )
    h, w = data.shape[1], data.shape[2]
    data_out = np.zeros((n_out, h, w), dtype=np.float32)
    parang_out = np.zeros(n_out, dtype=float)
    time_out = np.full(n_out, np.datetime64("NaT"), dtype="datetime64[us]")
    for i in range(n_out):
        sl = slice(i * frame_coadd, (i + 1) * frame_coadd)
        data_out[i] = np.mean(data[sl], axis=0, dtype=np.float64)
        parang_out[i] = np.mean(parang_cube[sl])
        time_out[i] = times[sl.start]
    return data_out, parang_out, time_out


def coadd_by_time(data, time_cube, parang_cube, time_coadd=10):
    """
    Mean-coadd consecutive frames into groups spanning at least ``time_coadd`` seconds.

    Parameters
    ----------
    data : ndarray, shape (N, H, W)
    time_cube : array-like of observation times
    parang_cube : ndarray, shape (N,)
    time_coadd : float
        Minimum elapsed time (seconds) covered by each coadd group.

    Returns
    -------
    data_out, parang_out, time_out
    """
    data = np.asarray(data, dtype=np.float32)
    parang_cube = np.asarray(parang_cube, dtype=float)
    times = _coerce_times_to_datetime64(time_cube)
    n = len(data)
    if n == 0:
        return data, parang_cube, times
    time_coadd = float(time_coadd)
    if time_coadd <= 0:
        raise ValueError(f"time_coadd must be positive, got {time_coadd}")

    groups = []
    start = 0
    for i in range(1, n):
        dt_sec = (times[i] - times[start]) / np.timedelta64(1, "s")
        if float(dt_sec) >= time_coadd:
            groups.append((start, i))
            start = i
    if start < n:
        groups.append((start, n))

    h, w = data.shape[1], data.shape[2]
    n_out = len(groups)
    data_out = np.zeros((n_out, h, w), dtype=np.float32)
    parang_out = np.zeros(n_out, dtype=float)
    time_out = np.full(n_out, np.datetime64("NaT"), dtype="datetime64[us]")
    for i, (a, b) in enumerate(groups):
        data_out[i] = np.mean(data[a:b], axis=0, dtype=np.float64)
        parang_out[i] = np.mean(parang_cube[a:b])
        time_out[i] = times[a]
    return data_out, parang_out, time_out


############## File checking functions ###############

def _norm_key(k):
    return str(k).upper().replace("HIERARCH", "").replace(" ", "").replace("_", "").replace("'", "")

def _find_hdr_val(hdr, key, camera_tag=None):
    wanted = str(key).upper().replace(" ", "").replace("_", "")
    #print(f"Looking for key '{key}' (normalized: '{wanted}') in header with camera tag '{camera_tag}'")
    if camera_tag:
        cam_wanted = camera_tag.upper() + wanted
    else:
        cam_wanted = None
    if cam_wanted:
        # 1) camera-specific
        for k in hdr.keys():
            nk = _norm_key(k)
            if cam_wanted in nk:
                return hdr[k]
    else:
        # 2) generic key
        for k in hdr.keys():
            nk = _norm_key(k)
            if nk == wanted or nk.startswith(wanted) or wanted in nk:
                return hdr[k]
    return None

#################### time functions ####################

def _coerce_times_to_datetime64(times):
    """
    Convert FITS-style DATE-OBS values (usually ISO-8601 strings) into a
    numpy datetime64 array so matplotlib plots a true time axis.
    """
    arr = np.asarray(times)
    if arr.size == 0:
        return arr
    # If already datetime64, keep it.
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr
    # Decode bytes -> str, and normalize common sentinels.
    if arr.dtype.kind in ("S", "a"):
        arr = np.char.decode(arr, "utf-8", errors="ignore")
    if arr.dtype.kind == "O":
        arr = np.array(
            [
                (x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x)
                for x in arr
            ],
            dtype=object,
        )
    # Try vectorized parse into datetime64; replace failures with NaT.
    out = np.full(arr.shape, np.datetime64("NaT"), dtype="datetime64[us]")
    for i, x in np.ndenumerate(arr):
        if x is None:
            continue
        if isinstance(x, (np.datetime64,)):
            out[i] = x.astype("datetime64[us]")
            continue
        s = str(x).strip()
        if s in ("", "None", "nan", "NaN", "-1"):
            continue
        try:
            # numpy.datetime64 has no timezone support. If DATE-OBS includes a timezone
            # (e.g. "Z" or "+00:00"), normalize to UTC and drop tzinfo before casting.
            if s.endswith("Z"):
                s_iso = s[:-1] + "+00:00"
            else:
                s_iso = s
            dt = datetime.fromisoformat(s_iso)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            out[i] = np.datetime64(dt).astype("datetime64[us]")
        except Exception:
            # Leave as NaT if unparseable
            pass
    return out