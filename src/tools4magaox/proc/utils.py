# utils.py — helpers for ADI / VIP post-processing
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    from ..redu.filereads import (
        _coerce_times_to_datetime64,
        _load_fits_primary_float32,
        load_fits_stack,
    )
except ImportError:
    from tools4magaox.redu.filereads import (
        _coerce_times_to_datetime64,
        _load_fits_primary_float32,
        load_fits_stack,
    )


def center_crop_2d(image, crop_size):
    """Center-crop a 2D image to ``(crop_size, crop_size)``."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"center_crop_2d expects 2D input, got shape {image.shape}")
    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y0 = cy - crop_size // 2
    y1 = y0 + crop_size
    x0 = cx - crop_size // 2
    x1 = x0 + crop_size
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        raise ValueError(
            f"crop_size {crop_size} exceeds image shape {(h, w)}"
        )
    return image[y0:y1, x0:x1].copy()


def _distance_from_center(frame_shape):
    """Pixel distance from the frame center for each position."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def outer_disk_keep_mask(frame_shape, radius_px):
    """
    Return a 2D mask that is 1 inside the outer disk and 0 outside.

    Used to zero square-crop corners outside the circular field.
    """
    radius_px = float(radius_px)
    if radius_px < 0:
        raise ValueError(f"radius_px must be non-negative, got {radius_px}")
    return (_distance_from_center(frame_shape) <= radius_px).astype(np.float32)


def inner_disk_keep_mask(frame_shape, radius_px):
    """
    Return a 2D mask that is 1 outside the inner disk and 0 inside.

    Used to zero the central region.
    """
    radius_px = float(radius_px)
    if radius_px < 0:
        raise ValueError(f"radius_px must be non-negative, got {radius_px}")
    return (_distance_from_center(frame_shape) > radius_px).astype(np.float32)


def center_disk_mask(frame_shape, radius_px):
    """Alias for :func:`inner_disk_keep_mask`."""
    return inner_disk_keep_mask(frame_shape, radius_px)


def mask_center_disk(cube_or_frame, radius_px):
    """
    Zero pixels within ``radius_px`` of the frame center.

    Accepts a 2D frame or 3D cube ``(N, H, W)``.
    """
    radius_px = float(radius_px)
    if radius_px <= 0:
        return np.asarray(cube_or_frame, dtype=np.float32)

    arr = np.asarray(cube_or_frame, dtype=np.float32)
    if arr.ndim == 2:
        mask = inner_disk_keep_mask(arr.shape, radius_px)
        return arr * mask
    if arr.ndim == 3:
        mask = inner_disk_keep_mask(arr.shape[1:], radius_px)
        return arr * mask[np.newaxis, :, :]
    raise ValueError(
        f"mask_center_disk expects 2D or 3D input, got shape {arr.shape}"
    )


def apply_crop_radius_masks(cube, radius_outer=None, radius_inner=None):
    """
    Apply outer and/or inner circular radius masks to a datacube.

    ``radius_outer``: crop to ``(2*R, 2*R)`` then zero pixels outside the
    circle of radius ``R`` (corners of the square).

    ``radius_inner``: zero pixels inside the circle of radius ``R``.

    Returns
    -------
    cube, aperture_mask, outer_mask, inner_mask
        ``aperture_mask`` is the combined 2D keep-mask (1 = kept, 0 = zeroed).
    """
    try:
        from ..redu import centering as ct
    except ImportError:
        from tools4magaox.redu import centering as ct

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"apply_crop_radius_masks expects (N, H, W), got {cube.shape}")

    outer_mask = None
    inner_mask = None

    if radius_outer is not None:
        radius_outer = float(radius_outer)
        if radius_outer <= 0:
            raise ValueError(f"crop_radius_outer must be positive, got {radius_outer}")
        side = int(2 * radius_outer)
        cube = ct.crop_cube(cube, new_shape=(side, side))
        outer_mask = outer_disk_keep_mask(cube.shape[1:], radius_outer)
        cube = cube * outer_mask[np.newaxis, :, :]

    if radius_inner is not None:
        radius_inner = float(radius_inner)
        if radius_inner > 0:
            if radius_outer is not None and radius_inner >= radius_outer:
                raise ValueError(
                    f"crop_radius_inner ({radius_inner}) must be < "
                    f"crop_radius_outer ({radius_outer})"
                )
            inner_mask = inner_disk_keep_mask(cube.shape[1:], radius_inner)
            cube = cube * inner_mask[np.newaxis, :, :]

    aperture_mask = np.ones(cube.shape[1:], dtype=np.float32)
    if outer_mask is not None:
        aperture_mask *= outer_mask
    if inner_mask is not None:
        aperture_mask *= inner_mask

    return cube, aperture_mask, outer_mask, inner_mask


def summarize_adi_frame_selection(file_table_static, file_table_output, run_params):
    """
    Count frames at each ADI selection stage.

    Returns a dict with roster / majority / shift / centered-file / used counts.
    """
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, run_params["centered_dir"])
    require_used = bool(run_params.get("require_used_in_reduction", True))
    n = len(file_table_output)

    majority = 0
    finite_shift = 0
    centered_file = 0
    used = 0
    selected = 0

    for i in range(n):
        if int(file_table_output["pass_majority_config"][i]) != 1:
            continue
        majority += 1
        sy = float(file_table_output["shift_y"][i])
        sx = float(file_table_output["shift_x"][i])
        if not np.isfinite(sy) or not np.isfinite(sx):
            continue
        finite_shift += 1
        filename = str(file_table_static["filename"][i])
        if not os.path.isfile(os.path.join(out_dir, filename)):
            continue
        centered_file += 1
        if int(file_table_output["used_in_reduction"][i]) == 1:
            used += 1
        if require_used and int(file_table_output["used_in_reduction"][i]) != 1:
            continue
        selected += 1

    return {
        "roster_rows": n,
        "pass_majority_config": majority,
        "finite_shifts": finite_shift,
        "centered_file_exists": centered_file,
        "used_in_reduction": used if require_used else centered_file,
        "selected_for_adi": selected,
        "require_used_in_reduction": require_used,
    }


def select_adi_frame_rows(file_table_static, file_table_output, run_params):
    """
    Return row indices for frames to include in the ADI cube.

    Uses ``used_in_reduction`` when ``require_used_in_reduction`` is True;
    otherwise all rows with finite shifts and existing centered FITS files.
    """
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, run_params["centered_dir"])
    require_used = bool(run_params.get("require_used_in_reduction", True))
    row_idxs = []
    n = len(file_table_output)
    for i in range(n):
        if int(file_table_output["pass_majority_config"][i]) != 1:
            continue
        if require_used and int(file_table_output["used_in_reduction"][i]) != 1:
            continue
        sy = float(file_table_output["shift_y"][i])
        sx = float(file_table_output["shift_x"][i])
        if not np.isfinite(sy) or not np.isfinite(sx):
            continue
        filename = str(file_table_static["filename"][i])
        if not os.path.isfile(os.path.join(out_dir, filename)):
            continue
        row_idxs.append(i)
    return np.asarray(row_idxs, dtype=int)


def load_centered_cube(file_table_static, row_idxs, run_params):
    """Load centered FITS frames for ``row_idxs`` as a (N, H, W) cube."""
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, run_params["centered_dir"])
    paths = [
        os.path.join(out_dir, str(file_table_static["filename"][i]))
        for i in row_idxs
    ]
    return load_fits_stack(paths, desc="Loading centered ADI cube")


def load_centered_cube_chunked(file_table_static, row_idxs, run_params, chunk_size):
    """Load centered frames in chunks and concatenate into one cube."""
    chunk_size = max(1, int(chunk_size))
    parts = []
    n = len(row_idxs)
    chunk_starts = range(0, n, chunk_size)
    for start in tqdm(chunk_starts, desc="Loading centered cube chunks", unit="chunk", leave=False):
        chunk_rows = row_idxs[start : start + chunk_size]
        parts.append(load_centered_cube(file_table_static, chunk_rows, run_params))
    if not parts:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def parang_and_times_for_rows(file_table_static, row_idxs):
    """Return PARANG and DATE_OBS arrays aligned with ``row_idxs``."""
    parang = np.asarray(file_table_static["PARANG"][row_idxs], dtype=float)
    times = _coerce_times_to_datetime64(file_table_static["DATE_OBS"][row_idxs])
    return parang, times


def save_frame_plot(frame, path, title=None, vmin=None, vmax=None):
    """Save a 2D frame as a PNG using matplotlib."""
    frame = np.asarray(frame, dtype=float)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frame, origin="lower", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def r_guess_from_source_xy(source_xy, frame_shape):
    """Radial separation in pixels from VIP frame center to ``source_xy`` (x, y)."""
    source_xy = tuple(float(v) for v in source_xy)
    if len(source_xy) != 2:
        raise ValueError(f"source_xy must be (x, y), got {source_xy!r}")
    cy, cx = vip_frame_center_xy(frame_shape)
    x, y = source_xy
    return float(np.sqrt((cy - y) ** 2 + (cx - x) ** 2))


def vip_frame_center_xy(frame_shape):
    """Return ``(cy, cx)`` using the same convention as ``vip_hci.var.frame_center``."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    cy = h / 2.0
    cx = w / 2.0
    if h % 2:
        cy -= 0.5
    if w % 2:
        cx -= 0.5
    return float(cy), float(cx)


def brightest_in_radius_annulus(frame, target_r_px, annulus_half_width=3.0):
    """
    Find the brightest pixel in an annulus around ``target_r_px``.

    Returns ``(y, x, radius_px, value)`` using VIP center convention.
    """
    frame = np.asarray(frame, dtype=float)
    cy, cx = vip_frame_center_xy(frame.shape)
    yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    target_r_px = float(target_r_px)
    half_w = float(annulus_half_width)
    ring = (r >= target_r_px - half_w) & (r <= target_r_px + half_w)
    if not np.any(ring):
        raise ValueError(
            f"No pixels in annulus r={target_r_px}±{half_w} for shape {frame.shape}"
        )
    masked = np.where(ring, frame, -np.inf)
    flat_idx = int(np.argmax(masked))
    iy, ix = np.unravel_index(flat_idx, frame.shape)
    rr = float(np.sqrt((iy - cy) ** 2 + (ix - cx) ** 2))
    return int(iy), int(ix), rr, float(frame[iy, ix])


def adi_rotation_alignment_probe(
    cube,
    parang_deg,
    *,
    expected_r_px,
    inner_exclude_px=30.0,
    max_frames=500,
):
    """
    Median-combine VIP-derotated frames and locate the brightest peak near
    ``expected_r_px``.

    VIP ``cube_derotate`` rotates each frame by ``-parang`` internally, so pass
    parallactic angles in degrees as stored (after any config sign/offset).

    Returns a dict of diagnostic scalars, or ``None`` when ``expected_r_px`` is
    unset.
    """
    if expected_r_px is None:
        return None

    from vip_hci.preproc.derotation import cube_derotate

    cube = np.asarray(cube, dtype=np.float32)
    parang_deg = np.asarray(parang_deg, dtype=float)
    if cube.ndim != 3 or parang_deg.ndim != 1 or cube.shape[0] != parang_deg.shape[0]:
        raise ValueError(
            "adi_rotation_alignment_probe expects matching (N, H, W) cube and "
            f"(N,) parang, got {cube.shape} and {parang_deg.shape}"
        )
    if cube.shape[0] == 0:
        return None

    stride = max(1, int(np.ceil(cube.shape[0] / max_frames)))
    sl = slice(None, None, stride)
    cube_s = cube[sl]
    par_s = parang_deg[sl]

    derot = cube_derotate(cube_s, par_s, imlib="vip-fft")
    median_frame = np.median(derot, axis=0)

    iy, ix, rr, peak_val = brightest_in_radius_annulus(
        median_frame, expected_r_px
    )
    cy, cx = vip_frame_center_xy(median_frame.shape)

    # Compare with inner speckle ring for a quick signal proxy.
    _, _, _, inner_val = brightest_in_radius_annulus(
        median_frame, inner_exclude_px
    )

    return {
        "expected_r_px": float(expected_r_px),
        "probe_n_frames": int(cube_s.shape[0]),
        "probe_stride": int(stride),
        "peak_y": iy,
        "peak_x": ix,
        "peak_r_px": rr,
        "peak_value": peak_val,
        "inner_r_px": float(inner_exclude_px),
        "inner_peak_value": inner_val,
        "frame_center_y": cy,
        "frame_center_x": cx,
    }


def save_adi_diagnostics(path, lines):
    """Write key=value diagnostic lines to a text file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def save_text_value(path, value):
    """Write a single scalar to a text file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{value}\n")


def save_dataframe_csv(df, path):
    """Save a pandas DataFrame to CSV when available."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


def save_parang_timeseries_plot(times, parang, path, title="Parallactic angle vs time"):
    """Plot PARANG vs observation time."""
    times = _coerce_times_to_datetime64(times)
    parang = np.asarray(parang, dtype=float)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, parang, "o-", markersize=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("PARANG [deg]")
    fig.autofmt_xdate()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
