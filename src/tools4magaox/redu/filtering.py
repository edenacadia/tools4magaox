# filtering.py
# 2026/04/06
# these are all the various ways we can filter the data
import os
import numpy as np
import matplotlib.pyplot as plt
from filereads import _coerce_times_to_datetime64

#################### filter functions ####################

def filter_max_value(unsats_data, perc=10):
    # filter out the lowes percentile of max values. 
    max_values = np.max(unsats_data, axis=1)
    threshold = np.percentile(max_values, perc)
    print(f"   Max filter: {perc} percentile value is {threshold}")
    good_indexes = np.where(max_values >= threshold)[0]
    return max_values, good_indexes

def filter_unstat_shifts(shifts, px_max=10, rolling_avg_window=-1):
    """Keep frames where both shift components are within ``px_max`` (absolute of an average shift)."""
    # TODO: implement a rolling average
    shifts = np.asarray(shifts, dtype=float)
    # filter based of the mean of the shifts
    if rolling_avg_window > 0:
        rolling_avg = np.convolve(shifts, np.ones(rolling_avg_window)/rolling_avg_window, mode='valid')
        shifts_from_mean = shifts - rolling_avg
    else:
        shifts_from_mean = shifts - np.mean(shifts, axis=0)
    good_indexes = np.where(np.abs(shifts_from_mean) <= px_max and np.abs(shifts_from_mean) <= px_max)[0]
    print(f"   Unstat shifts filter: {px_max} px max value is {len(good_indexes)}/{len(shifts)}")
    return good_indexes

#################### plot functions ####################

def plot_max_filter_timeseries(max_values, good_idxs, timeseries_list, perc=10, plot_path="", plt_name="max_filter_timeseries.png"):
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
    plt.title(f"Peak Max filtered timeseries")
    plt.xlabel("Time")
    plt.ylabel("Max value")
    plt.plot(t, max_values, "o")
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
    obs_name = plot_path.split("/")[-2]
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

def plot_shift_filter_timeseries(shifts, good_idxs, timeseries_list, px_max=10, plot_path="", plt_name="shift_filter_timeseries.png"):
    shifts = np.asarray(shifts, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    t = _coerce_times_to_datetime64(timeseries_list)
    if len(t) != len(shifts):
        raise ValueError(
            f"timeseries_list length {len(t)} must match shifts length {len(shifts)} "
            "(DATE_OBS must be aligned to the same cube indexing)."
        )
    # plot
    plt.title(f"Unstat shifts filtered timeseries")
    plt.xlabel("Time")
    plt.ylabel("Shift")
    plt.plot(t, shifts[:, 0], "o", label="x shift", alpha=0.1, color="gray")
    plt.plot(t, shifts[:, 1], "o", label="y shift", alpha=0.1, color="gray")
    plt.plot(t, shifts[g, 0], "o", label="x shift", alpha=0.5)
    plt.plot(t, shifts[g, 1], "o", label="y shift", alpha=0.5)
    plt.legend()
    # save
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved shift filter timeseries to {timeseries_file}")
    return

def plot_shift_filter_scatter(shifts, good_idxs, plot_path="", plt_name="shift_filter_scatter.png"):
    obs_name = plot_path.split("/")[-2]
    # TODO: fix, a little repetative
    max_values = np.asarray(max_values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    # plot
    plt.title(f"Peak Max filtered scatter plot \n {obs_name}")
    plt.xlabel("Max value")
    plt.ylabel("Count")
    plt.scatter(g[:, 0], g[:, 1], alpha=0.1)
    plt.scatter(g[:, 0], g[:, 1], alpha=0.5, label="good shifts")
    plt.legend()
    # save
    scatter_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.savefig(scatter_file)
    plt.close()
    print(f"    Saved shift filter scatter plot to {scatter_file}")
    return