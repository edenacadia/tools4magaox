# filtering.py
# 2026/04/06
# these are all the various ways we can filter the data


def max_filter(unsats_data, perc=10, save_plot=False, plot_path="max_filter_hist.png"):
    # filter out the lowes percentile of max values. 
    max_values = np.max(unsats_data, axis=1)
    threshold = np.percentile(max_values, perc)
    print(f"   Max filter: {perc} percentile value is {threshold}")
    good_indexes = np.where(max_values >= threshold)[0]
    if save_plot:
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
    return unsats_data[good_indexes]