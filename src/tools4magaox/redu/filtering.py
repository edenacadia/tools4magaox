# filtering.py
# 2026/04/06
# these are all the various ways we can filter the data
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .filereads import _coerce_times_to_datetime64
except ImportError:
    from filereads import _coerce_times_to_datetime64

log = logging.getLogger(__name__)

#################### filter functions ####################

def filter_max_value(unsats_data, perc=10):
    # filter out the lowest percentile of max values (3D cube), or of per-frame
    # scalars (1D / column vector from a table).
    arr = np.asarray(unsats_data)
    if arr.ndim == 1:
        max_values = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        max_values = arr.ravel()
    else:
        max_values = np.max(unsats_data, axis=(1, 2))
    finite = np.isfinite(max_values)
    if not np.any(finite):
        log.warning("filter_max_value: no finite values; keeping 0 frames")
        good_empty = np.array([], dtype=int)
        log.info(
            "filter_max_value return: max_values shape=%s dtype=%s good_indexes=%s (empty)",
            getattr(max_values, "shape", ()), max_values.dtype, good_empty,
        )
        return max_values, good_empty
    threshold = np.percentile(max_values[finite], perc)
    good_indexes = np.where(finite & (max_values >= threshold))[0]
    log.info(
        "filter_max_value: perc=%s threshold=%s kept %s/%s frames",
        perc,
        threshold,
        len(good_indexes),
        len(max_values),
    )
    mv_min = float(np.nanmin(max_values[finite]))
    mv_max = float(np.nanmax(max_values[finite]))
    idx_lo = int(np.min(good_indexes)) if len(good_indexes) else None
    idx_hi = int(np.max(good_indexes)) if len(good_indexes) else None
    log.info(
        "filter_max_value return: max_values shape=%s dtype=%s finite_min=%s finite_max=%s | "
        "good_indexes shape=%s dtype=%s count=%s idx_range=[%s,%s]",
        max_values.shape,
        max_values.dtype,
        mv_min,
        mv_max,
        good_indexes.shape,
        good_indexes.dtype,
        len(good_indexes),
        idx_lo,
        idx_hi,
    )
    return max_values, good_indexes

def filter_unstat_shifts(shifts, px_max=10, rolling_avg_window=-1):
    """Keep frames where both shift components are within ``px_max`` (absolute of an average shift)."""
    # TODO: implement a rolling average
    shifts = np.asarray(shifts, dtype=float)
    mean_shift = np.mean(shifts, axis=0)
    shifts_r = np.sqrt((shifts[:,0] - mean_shift[0])**2 + (shifts[:,1] - mean_shift[1])**2)
    # filter based of the mean of the shifts
    if rolling_avg_window > 0:
        # TODO: implement a rolling average
        rolling_avg = np.convolve(shifts_r, np.ones(rolling_avg_window)/rolling_avg_window, mode='valid')
        shifts_from_mean = shifts_r - rolling_avg
    good_indexes = np.where(np.abs(shifts_r) <= px_max)[0]
    log.info(
        "filter_unstat_shifts: px_max=%s kept %s/%s frames mean_shift=%s",
        px_max,
        len(good_indexes),
        len(shifts),
        mean_shift.tolist(),
    )
    idx_lo = int(np.min(good_indexes)) if len(good_indexes) else None
    idx_hi = int(np.max(good_indexes)) if len(good_indexes) else None
    log.info(
        "filter_unstat_shifts return: good_indexes shape=%s dtype=%s count=%s idx_range=[%s,%s]",
        good_indexes.shape,
        good_indexes.dtype,
        len(good_indexes),
        idx_lo,
        idx_hi,
    )
    return good_indexes

#################### process filter functions ####################

def filter_max_point(data_cube, sigma_clip=2.0):
    """
    Keep frames whose brightest-pixel position is within ``sigma_clip`` std of the
    mean radial offset (process ``filter_max_point``).
    """
    from numpy import unravel_index

    data_cube = np.asarray(data_cube)
    n = data_cube.shape[0]
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    peak_idxs = np.array([unravel_index(frame.argmax(), frame.shape) for frame in data_cube])
    return filter_max_point_from_peak_idxs(peak_idxs, sigma_clip=sigma_clip)


def filter_max_point_from_peak_idxs(peak_idxs, sigma_clip=2.0):
    """Apply max-point filter from precomputed ``(N, 2)`` peak pixel indices."""
    peak_idxs = np.asarray(peak_idxs, dtype=float)
    n = peak_idxs.shape[0]
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    means = np.mean(peak_idxs, axis=0)
    radius_sq = (peak_idxs[:, 0] - means[0]) ** 2 + (peak_idxs[:, 1] - means[1]) ** 2
    threshold = np.mean(radius_sq) + sigma_clip * np.std(radius_sq)
    pass_mask = radius_sq <= threshold
    log.info(
        "filter_max_point: sigma_clip=%s kept %s/%s frames",
        sigma_clip,
        int(np.sum(pass_mask)),
        n,
    )
    return pass_mask, np.sqrt(radius_sq, dtype=float)


def filter_max_point_from_radii(radius, sigma_clip=2.0):
    """Reapply max-point filter from stored per-frame radii (no image reload)."""
    radius = np.asarray(radius, dtype=float)
    n = radius.size
    if n == 0:
        return np.array([], dtype=bool)
    radius_sq = radius ** 2
    threshold = np.mean(radius_sq) + sigma_clip * np.std(radius_sq)
    pass_mask = radius_sq <= threshold
    log.info(
        "filter_max_point: sigma_clip=%s kept %s/%s frames (from stored radii)",
        sigma_clip,
        int(np.sum(pass_mask)),
        n,
    )
    return pass_mask


def filter_center_shifts(shifts, sigma_clip=2.0):
    """
    Keep frames with shift_y and shift_x within ``sigma_clip`` std of the mean
    (process ``filter_center_shifts``; shifts are ``[shift_y, shift_x]``).
    """
    shifts = np.asarray(shifts, dtype=float)
    n = shifts.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    sy = shifts[:, 0]
    sx = shifts[:, 1]
    sy_mean, sy_std = np.mean(sy), np.std(sy)
    sx_mean, sx_std = np.mean(sx), np.std(sx)
    pass_y = (sy > sy_mean - sigma_clip * sy_std) & (sy < sy_mean + sigma_clip * sy_std)
    pass_x = (sx > sx_mean - sigma_clip * sx_std) & (sx < sx_mean + sigma_clip * sx_std)
    pass_mask = pass_y & pass_x
    log.info(
        "filter_center_shifts: sigma_clip=%s kept %s/%s frames",
        sigma_clip,
        int(np.sum(pass_mask)),
        n,
    )
    return pass_mask


def filter_speckle_intensity(data_centered, speckle_mask, sigma_clip=2.0):
    """
    Keep frames whose masked speckle sum is within ``sigma_clip`` std of the mean.
    """
    data_centered = np.asarray(data_centered)
    if hasattr(speckle_mask, "shaped"):
        mask = np.asarray(speckle_mask.shaped, dtype=float)
    else:
        mask = np.asarray(speckle_mask, dtype=float)
    intensities = np.sum(data_centered * mask, axis=(1, 2))
    return filter_speckle_intensity_values(intensities, sigma_clip=sigma_clip)


def filter_speckle_intensity_values(intensities, sigma_clip=2.0):
    """Apply speckle-intensity filter from precomputed per-frame sums."""
    intensities = np.asarray(intensities, dtype=float)
    n = intensities.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    i_mean = np.mean(intensities)
    i_std = np.std(intensities)
    pass_mask = (intensities > i_mean - sigma_clip * i_std) & (
        intensities < i_mean + sigma_clip * i_std
    )
    log.info(
        "filter_speckle_intensity: sigma_clip=%s kept %s/%s frames",
        sigma_clip,
        int(np.sum(pass_mask)),
        n,
    )
    return pass_mask, intensities


def filter_rms(data_cube, sigma_clip=2.0, n_iter=3):
    """
    Iteratively reject frames with high RMS residual vs the mean of survivors.
    """
    data_cube = np.asarray(data_cube)
    n = data_cube.shape[0]
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    idx = np.ones(n, dtype=bool)
    rms_deviations = np.full(n, np.nan, dtype=float)
    for _ in range(n_iter):
        if not np.any(idx):
            break
        good_frame = np.mean(data_cube[idx], axis=0)
        rms_deviations = np.array(
            [float(np.std(frame - good_frame)) for frame in data_cube], dtype=float
        )
        ref = np.median(rms_deviations)
        spread = np.std(rms_deviations)
        idx = rms_deviations <= ref + sigma_clip * spread
    log.info(
        "filter_rms: sigma_clip=%s n_iter=%s kept %s/%s frames",
        sigma_clip,
        n_iter,
        int(np.sum(idx)),
        n,
    )
    return idx, rms_deviations


def filter_rms_from_deviations(rms_deviations, sigma_clip=2.0):
    """Reapply RMS filter from stored per-frame deviations (no image reload)."""
    rms_deviations = np.asarray(rms_deviations, dtype=float)
    n = rms_deviations.size
    if n == 0:
        return np.array([], dtype=bool)
    ref = np.median(rms_deviations)
    spread = np.std(rms_deviations)
    pass_mask = rms_deviations <= ref + sigma_clip * spread
    log.info(
        "filter_rms: sigma_clip=%s kept %s/%s frames (from stored deviations)",
        sigma_clip,
        int(np.sum(pass_mask)),
        n,
    )
    return pass_mask

#################### plot functions ####################

def plot_generic_timeseries(values, good_idxs, timeseries_list, plot_path="", plt_title="timeseries", plt_name="generic_filter_timeseries.png"):
    obs_name =  plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    values = np.asarray(values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    t = _coerce_times_to_datetime64(timeseries_list)
    # plot
    plt.title(f"{plt_title} \n {obs_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.plot(t, values, "o", label="discarded frames", alpha=0.1, color="gray", markersize=4)
    plt.plot(t[g], values[g], "o",label="good frames", alpha=0.5, markersize=4)
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    log.info("Saved %s timeseries to %s", plt_title, timeseries_file)
    return

def plot_max_filter_timeseries(max_values, good_idxs, timeseries_list, perc=10, plot_path="", plt_name="2_max_filter_timeseries.png"):
    obs_name =  plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    # TODO: this is repetative
    max_values = np.asarray(max_values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    threshold = np.percentile(max_values, perc)
    t = _coerce_times_to_datetime64(timeseries_list)
    if len(t) != len(max_values):
        raise ValueError(
            f"timeseries_list length {len(t)} must match max_values length {len(max_values)} "
            "(DATE_OBS must be aligned to the same cube indexing)."
        )
    # plot
    plt.title(f"Peak Max filtered timeseries \n {obs_name}")
    plt.xlabel("Time")
    plt.ylabel("Max value")
    plt.plot(t, max_values, "o", label="discarded frames", alpha=0.1, color="gray", markersize=4)
    plt.plot(t[g], max_values[g], "o", label="good frames", alpha=0.5, markersize=4)
    plt.axhline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    log.info("Saved max filter timeseries to %s", timeseries_file)
    return 

def plot_max_filter_hist(max_values, good_idxs, perc=10, plot_path="", plt_name="2_max_filter_hist.png"):
    hist_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    obs_name =  plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    # TODO: fix, a little repetative
    max_values = np.asarray(max_values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    threshold = np.percentile(max_values, perc)
    bins = np.histogram(max_values, bins=50)[1]
    # plot
    plt.title(f"Peak Max filtered histogram \n {obs_name}")
    plt.xlabel("Max value")
    plt.ylabel("Count")
    plt.hist(max_values, bins, alpha=0.5)
    plt.hist(max_values[g], bins, alpha=0.5)
    plt.axvline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    plt.savefig(hist_file)
    plt.close()
    log.info("Saved max filter histogram to %s", hist_file)
    return 

###########################################################################

def plot_shift_filter_timeseries(shifts, good_idxs, timeseries_list, px_max=10, plot_path="", plt_name="3_shift_filter_timeseries.png"):
    obs_name =  plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    shifts = np.asarray(shifts, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    t = _coerce_times_to_datetime64(timeseries_list)
    if len(t) != len(shifts):
        raise ValueError(
            f"timeseries_list length {len(t)} must match shifts length {len(shifts)} "
            "(DATE_OBS must be aligned to the same cube indexing)."
        )
    # plot
    plt.title(f"Unstat shifts filtered timeseries \n {obs_name}")
    plt.xlabel("Time")
    plt.ylabel("Shift")
    plt.plot(t, shifts[:, 0], "o", alpha=0.1, color="gray", markersize=4)
    plt.plot(t, shifts[:, 1], "o", alpha=0.1, color="gray", markersize=4)
    plt.plot(t[g], shifts[g, 0], "o", label="y shift", alpha=0.5, markersize=4)
    plt.plot(t[g], shifts[g, 1], "o", label="x shift", alpha=0.5, markersize=4)
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    log.info("Saved shift filter timeseries to %s", timeseries_file)
    return

def plot_shift_filter_scatter(shifts, good_idxs, px_max=10, plot_path="", plt_name="3_shift_filter_scatter.png"):
    obs_name =  plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    # TODO: fix, a little repetative
    shifts = np.asarray(shifts, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    # plot
    plt.title(f"Peak Max filtered scatter plot \n {obs_name}")
    plt.xlabel("x shift")
    plt.ylabel("y shift")
    plt.scatter(shifts[:, 1], shifts[:, 0], alpha=0.1, color="gray")
    plt.scatter(shifts[g, 1], shifts[g, 0], alpha=0.5, label="good shifts")
    plt.legend()
    # save
    scatter_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(scatter_file)
    plt.close()
    log.info("Saved shift filter scatter plot to %s", scatter_file)
    return

def plot_reference_and_mask(
    reference_image,
    mask_image,
    plot_path="",
    plt_name="1_reference_mask.png",
):
    """Side-by-side reference image and sparkle mask (process step 1)."""
    ny, nx = reference_image.shape
    x0, y0 = nx / 2, ny / 2

    obs_name = plot_path.split("/")[-2] + " " + plot_path.split("/")[-3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(mask_image, origin="lower", cmap="gray")
    axes[0].set_title("Sparkle mask")
    axes[0].axis("off")
    axes[1].imshow(reference_image, origin="lower", cmap="gray")
    axes[1].axvline(x0, color="cyan", lw=0.8, ls=":")
    axes[1].axhline(y0, color="cyan", lw=0.8, ls=":")
    axes[1].set_title("Reference image masked")
    axes[1].axis("off")
    fig.suptitle(f"Process reference and mask\n{obs_name}")
    fig.tight_layout()
    out = os.path.join(plot_path, plt_name) if plot_path else plt_name
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved reference/mask plot to %s", out)