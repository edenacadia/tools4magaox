# filtering.py
# 2026/04/06
# these are all the various ways we can filter the data
import os

import numpy as np
import matplotlib.pyplot as plt

#################### filter functions ####################

def filter_max_value(unsats_data, perc=10):
    # filter out the lowes percentile of max values. 
    max_values = np.max(unsats_data, axis=1)
    threshold = np.percentile(max_values, perc)
    print(f"   Max filter: {perc} percentile value is {threshold}")
    good_indexes = np.where(max_values >= threshold)[0]
    return max_values, good_indexes

def filter_unstat_shifts(shifts, px_max=10):
    """Keep frames where both shift components are within ``px_max`` (absolute)."""
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim == 2 and shifts.shape[1] == 2:
        ok = np.max(np.abs(shifts), axis=1) <= px_max
        good_indexes = np.nonzero(ok)[0]
    elif shifts.ndim == 1:
        good_indexes = np.where(np.abs(shifts) <= px_max)[0]
    else:
        raise ValueError(f"shifts must be (N,) or (N, 2), got shape {shifts.shape}")
    print(f"   Unstat shifts filter: {px_max} px max value is {len(good_indexes)}/{len(shifts)}")
    return good_indexes

#################### plot functions ####################

def filter_plot(total_idxs, good_idxs, x_values=None, plt_title=None, plt_path="filter_plot.png"):
    # Plot 1: histogram
    hist_file = f"{plot_path}_unsats_maxfilter_hist.png"
    bins = np.histogram(max_values, bins=50)[1]
    plt.hist(max_values, bins, alpha=0.5)
    plt.hist(max_values[good_indexes], bins, alpha=0.5)
    plt.axvline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    plt.savefig(hist_file)
    plt.close()
    print(f"   Saved max filter histogram to {hist_file}")
    timeseries_file = f"{plot_path}_unsats_maxfilter_timeseries.png"
    plt.plot(max_values)
    plt.axhline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved max filter timeseries to {timeseries_file}")
    # plot 2: time series of max values

def plot_max_filter_timeseries(max_values, good_idxs, perc=10, plot_path="", plt_name="max_filter_timeseries.png"):
    max_values = np.asarray(max_values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    threshold = np.percentile(max_values, perc)
    timeseries_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.plot(max_values)
    plt.axhline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    plt.savefig(timeseries_file)
    plt.close()
    print(f"    Saved max filter timeseries to {timeseries_file}")
    return 

def plot_max_filter_hist(max_values, good_idxs, perc=10, plot_path="", plt_name="max_filter_hist.png"):
    max_values = np.asarray(max_values, dtype=float)
    g = np.asarray(good_idxs, dtype=int)
    threshold = np.percentile(max_values, perc)
    bins = np.histogram(max_values, bins=50)[1]
    hist_file = os.path.join(plot_path, plt_name) if plot_path else plt_name
    plt.hist(max_values, bins, alpha=0.5)
    plt.hist(max_values[g], bins, alpha=0.5)
    plt.axvline(threshold, label=f"{threshold}, {perc}pct", c="r")
    plt.legend()
    plt.savefig(hist_file)
    plt.close()
    print(f"    Saved max filter histogram to {hist_file}")
    return 

def plot_shift_filter_timeseries():
    return 