# filtering.py
# 2026/04/06
# these are all the various ways we can filter the data
import os
import numpy as np
import matplotlib.pyplot as plt
from filereads import _coerce_times_to_datetime64

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
        print(f"   Max filter: no finite values; keeping 0 frames")
        return max_values, np.array([], dtype=int)
    threshold = np.percentile(max_values[finite], perc)
    print(f"   Max filter: {perc} percentile value is {threshold}")
    good_indexes = np.where(finite & (max_values >= threshold))[0]
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
    print(f"   Unstat shifts filter: {px_max} px max value is {len(good_indexes)}/{len(shifts)}")
    return good_indexes

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
    plt.plot(t, values, "o", label="discarded frames", alpha=0.1, color="gray")
    plt.plot(t[g], values[g], "o", label="good frames", alpha=0.5)
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved {plt_title} timeseries to {timeseries_file}")
    return

def plot_max_filter_timeseries(max_values, good_idxs, timeseries_list, perc=10, plot_path="", plt_name="max_filter_timeseries.png"):
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
    plt.plot(t, max_values, "o", label="discarded frames", alpha=0.1, color="gray")
    plt.plot(t[g], max_values[g], "o", label="good frames", alpha=0.5)
    plt.axhline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved max filter timeseries to {timeseries_file}")
    return 

def plot_max_filter_hist(max_values, good_idxs, perc=10, plot_path="", plt_name="max_filter_hist.png"):
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
    print(f"    Saved max filter histogram to {hist_file}")
    return 

###########################################################################

def plot_shift_filter_timeseries(shifts, good_idxs, timeseries_list, px_max=10, plot_path="", plt_name="shift_filter_timeseries.png"):
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
    plt.plot(t, shifts[:, 0], "o", alpha=0.1, color="gray")
    plt.plot(t, shifts[:, 1], "o", alpha=0.1, color="gray")
    plt.plot(t[g], shifts[g, 0], "o", label="y shift", alpha=0.5)
    plt.plot(t[g], shifts[g, 1], "o", label="x shift", alpha=0.5)
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved shift filter timeseries to {timeseries_file}")
    return

def plot_shift_filter_scatter(shifts, good_idxs, px_max=10, plot_path="", plt_name="shift_filter_scatter.png"):
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
    print(f"    Saved shift filter scatter plot to {scatter_file}")
    return