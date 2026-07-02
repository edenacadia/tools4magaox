# utils.py — helpers for ADI / VIP post-processing
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from ..redu.filereads import _coerce_times_to_datetime64, _load_fits_primary_float32
except ImportError:
    from tools4magaox.redu.filereads import (
        _coerce_times_to_datetime64,
        _load_fits_primary_float32,
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
    frames = [
        _load_fits_primary_float32(
            os.path.join(out_dir, str(file_table_static["filename"][i]))
        )
        for i in row_idxs
    ]
    return np.stack(frames, axis=0)


def load_centered_cube_chunked(file_table_static, row_idxs, run_params, chunk_size):
    """Load centered frames in chunks and concatenate into one cube."""
    chunk_size = max(1, int(chunk_size))
    parts = []
    for start in range(0, len(row_idxs), chunk_size):
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
    """Radial separation in pixels from frame center to ``source_xy`` (x, y)."""
    source_xy = tuple(float(v) for v in source_xy)
    if len(source_xy) != 2:
        raise ValueError(f"source_xy must be (x, y), got {source_xy!r}")
    h, w = int(frame_shape[0]), int(frame_shape[1])
    cy, cx = h / 2.0, w / 2.0
    x, y = source_xy
    return float(np.sqrt((cy - y) ** 2 + (cx - x) ** 2))


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
