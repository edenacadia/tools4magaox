# preprocess.py
# step 1 in the reduction pipeline
# this makes our unsats
# running this on a directory will make a master unsats in that directory
import argparse
import ast
import glob
import logging
import os
import shutil
import sys
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from scipy import ndimage
import scipy

from tools4magaox.constants import *
import darks as md
import filtering as fl
import filereads as fr
import centering as ct

log = logging.getLogger(__name__)

# files you should expect this script to make
FILE_TABLE_NAME = "file_table.txt"
FILE_TABLE_OUTPUT_NAME = "file_table_output.txt"
CLEAN_CUBE_NAME = "clean_cube.fits"
CENTERED_CUBE_NAME = "centered_cube.fits"
AVERAGE_IMAGE_NAME = "average_image.fits"
# Copy of the preprocess config saved alongside pipeline outputs (CLI supplies path).
PREPROCESS_CONFIG_SNAPSHOT_NAME = "preprocess_config.txt"
PREPROCESS_LOG_NAME = "preprocess.log"

_REDU_ASCII_FMT = "ascii.commented_header"

############# Main Functions #############

# STEP 0
def find_filetable(run_params):
    '''
    Load or build the static file table and ``file_table_output`` (see :func:`_load_file_table`).

    Returns
    -------
    tuple[Table, Table]
        ``(file_table_static, file_table_output)`` — telemetry and ``masterdark_path``
        only in the first table; pipeline filters and Gaussian parameters in the second.

    Reads from ``run_params``: ``redu_dir``, ``obs_path``, ``unsats_dir``, ``camera``,
    ``max_files``, ``redu_path``, ``force_rerun``, ``masterdark_dir``.
    '''
    log.info("0. Finding file table")
    redu_dir = run_params["redu_dir"]
    obs_path = run_params["obs_path"]
    unsats_dir = run_params["unsats_dir"]
    camera = run_params["camera"]
    max_files = run_params["max_files"]
    redu_path = run_params["redu_path"]
    force_rerun = run_params["force_rerun"]
    masterdark_dir = run_params.get("masterdark_dir")

    static_path = os.path.join(redu_dir, FILE_TABLE_NAME)
    output_path = os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)

    if (
        not force_rerun
        and os.path.isfile(static_path)
        and os.path.isfile(output_path)
    ):
        log.info("=> LOADING FILE TABLE")
        file_table_static = _read_redu_table(static_path)
        if "to_use" in file_table_static.colnames:
            raise ValueError(
                f"{FILE_TABLE_NAME} contains deprecated column 'to_use'. "
                "Delete it and file_table_output.txt (or set force_rerun) to regenerate."
            )
        file_table_output = _read_redu_table(output_path)
        return file_table_static, file_table_output

    log.info("=> CREATING FILE TABLE")
    unsat_files = find_files(obs_path, unsats_dir, camera, max_files)
    unsats_table = fr.fits_telemetry_table(unsat_files, camera)
    dark_search_dir = _resolve_masterdark_search_dir(redu_path, masterdark_dir)
    file_table_darks = find_dark_files(dark_search_dir, unsats_table, camera)
    file_table_full = pick_unsat_params(file_table_darks)
    file_table_static = _file_table_static_from_full(file_table_full)
    file_table_output = _init_file_table_output(file_table_full)
    _write_redu_table(file_table_static, static_path)
    _write_redu_table(file_table_output, output_path)
    return file_table_static, file_table_output

# STEP 1
def make_clean_cube(run_params, file_table_static, file_table_output):
    '''
    Makes a clean cube from all unsat files in majority param set

    Reads from ``run_params``: ``redu_dir``, ``obs_path``, ``unsats_dir``, ``camera``,
    ``force_rerun``.
    '''
    redu_dir = run_params["redu_dir"]
    obs_path = run_params["obs_path"]
    unsats_dir = run_params["unsats_dir"]
    camera = run_params["camera"]
    force_rerun = run_params["force_rerun"]

    log.info("1. Making clean cube")
    clean_cube_path = os.path.join(redu_dir, CLEAN_CUBE_NAME)

    if os.path.exists(clean_cube_path) and force_rerun == False:
        log.info("=> CLEAN CUBE EXISTS, SKIPPING")
    else:
        log.info("=> CREATING CLEAN CUBE")
        # 1.0 find files to use (telemetry table stores basenames only)
        file_table_total = _ephemeral_file_table_with_to_use(
            file_table_static, file_table_output
        )
        used = file_table_total[file_table_total["to_use"] == 1]
        prefix = f"{obs_path}{unsats_dir}/{camera}/"
        unsat_files = [prefix + str(fn) for fn in used["filename"]]
        # 1.1 pull dark data
        dark_path = used["masterdark_path"][0]
        dark_data = _load_fits_primary_float32(dark_path)
        # 1.2 pulling all files and making cube
        unsats_data_cube = fr.make_data_cube(unsat_files, dark_data)
        # 1.3 writing the cube (numpy array — use FITS, not ndarray.write)
        fits.PrimaryHDU(data=np.asarray(unsats_data_cube, dtype=np.float32)).writeto(
            clean_cube_path, overwrite=True
        )
    return clean_cube_path

# STEP 2
def make_centered_cube(run_params, clean_cube_path, file_table_static, file_table_output):
    '''
    Input: total cube of unsats
    Process:
        1. filter the cube based on max peaks
        2. center the remaining frames
        3. write the centered cube
        4. update ``file_table_output`` with per-frame filters, shifts, and Gaussian fits

    Reads from ``run_params``: ``redu_dir``, ``pct_cut``, ``force_rerun``, ``plot``,
    ``crop_shape``, ``fit_func``.
    '''
    redu_dir = run_params["redu_dir"]
    pct_cut = run_params["pct_cut"]
    force_rerun = run_params["force_rerun"]
    save_plot = run_params["plot"]
    crop_shape = run_params.get("crop_shape")
    fit_func = run_params["fit_func"]

    log.info("2. Making centered cube")
    centered_cube_path = os.path.join(redu_dir, CENTERED_CUBE_NAME)

    if os.path.exists(centered_cube_path) and force_rerun == False:
        log.info("=> CENTERED CUBE EXISTS, SKIPPING")
        centered_file_table = _filter_file_table_output_for_centering(
            file_table_output
        )
    else:
        log.info("=> CREATING CENTERED CUBE")
        file_table_total = _ephemeral_file_table_with_to_use(
            file_table_static, file_table_output
        )
        unsats_data_cube = _load_fits_primary_float32(clean_cube_path)
        # 2.0 filter the cube based on max peaks
        max_values, good_idxs = fl.filter_max_value(unsats_data_cube, perc=pct_cut)
        if save_plot:
            # Keep DATE_OBS aligned with the clean cube ordering (to_use == 1).
            td_list = _date_obs_for_centering(file_table_static, file_table_output)
            fl.plot_max_filter_timeseries(max_values, good_idxs, td_list, perc=pct_cut, plot_path=redu_dir)
            fl.plot_max_filter_hist(max_values, good_idxs, perc=pct_cut, plot_path=redu_dir)
        # 2.1 find the shifts for the remaining frames
        if fit_func == "gauss_min":
            shifts, info_dict = ct.gaussian_fit_shifts(unsats_data_cube[good_idxs], crop_shape=crop_shape, method="minimize")
        elif fit_func == "gauss_curvefit":
            shifts, info_dict = ct.gaussian_fit_shifts(unsats_data_cube[good_idxs], crop_shape=crop_shape, method="curvefit")
        else:
            raise ValueError(f"Invalid fit function: {fit_func}")
        # 2.3 center frames using the shifts
        centered_data_cube = ct.shift_cube(unsats_data_cube[good_idxs], -shifts)
        # 2.4 update/write file table
        gauss_params = None
        if isinstance(info_dict, dict):
            gauss_params = info_dict.get("gauss_params")
        _update_file_table_output_step2(file_table_output,max_values,
            good_idxs,
            shifts,
            gauss_params,
        )
        _write_redu_table(file_table_output, os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME))
        centered_file_table = _filter_file_table_output_for_centering(
            file_table_output
        )
        # 2.5 write the centered cube
        fits.PrimaryHDU(data=np.asarray(centered_data_cube, dtype=np.float32)).writeto(
            centered_cube_path, overwrite=True
        )
    return centered_file_table

# STEP 3
def make_average_image(run_params, centered_file_table, file_table_static, file_table_output):
    '''
    Input: majority-filtered ``file_table_output`` aligned with the centered cube
    Process:
        1. filter based on the shifts
        2. average the remaining frames
        3. write the average image
        4. update ``file_table_output`` with ``pass_avg_shift``, ``pass_avg_amp``,
           and ``used_in_average``.

    Reads from ``run_params``: ``redu_dir``, ``force_rerun``, ``plot``, ``px_max``,
    ``gauss_amp_pct_cut`` (used for the Gaussian-amplitude percentile filter in this step).
    '''
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    save_plot = run_params["plot"]
    px_max = run_params["px_max"]
    gauss_amp_pct_cut = run_params["gauss_amp_pct_cut"]

    log.info("3. Making average image")
    centered_cube_path = os.path.join(redu_dir, CENTERED_CUBE_NAME)
    average_image_path = os.path.join(redu_dir, AVERAGE_IMAGE_NAME)

    if os.path.exists(average_image_path) and force_rerun == False:
        log.info("=> AVERAGE IMAGE EXISTS, SKIPPING")
        average_image = _load_fits_primary_float32(average_image_path)
    else:
        log.info("=> CREATING AVERAGE IMAGE")
        centered_data_cube = _load_fits_primary_float32(centered_cube_path)
        # 3.0 Find the good indexes from average image
        shifts = load_shifts(centered_file_table)
        good_idxs1 = fl.filter_unstat_shifts(shifts, px_max=px_max)
        # mask on usable centring shifts
        sy = np.asarray(centered_file_table["shift_y"], dtype=float)
        sx = np.asarray(centered_file_table["shift_x"], dtype=float)
        mask = np.isfinite(sy) & np.isfinite(sx)

        g_amps_cube = np.asarray(centered_file_table["gauss_amp"], dtype=float)[mask]
        _, good_idxs2 = fl.filter_max_value(g_amps_cube, perc=gauss_amp_pct_cut)
        good_idxs = np.intersect1d(good_idxs1, good_idxs2)
        log.info(
            "-> %s / %s centered frames kept for average image",
            len(good_idxs),
            len(centered_data_cube),
        )
        # filter the data cube
        centered_data_cube_filtered = centered_data_cube[good_idxs]
        average_stage_subset = _subset_table_with_average_pass_flags(
            centered_file_table, good_idxs1, good_idxs2, good_idxs
        )
        _persist_file_table_output_after_step3(
            file_table_output, redu_dir, average_stage_subset
        )
        # Grab the timeseries aligned with the filtered output rows
        if save_plot:
            td_list = _date_obs_for_centering(file_table_static, file_table_output)[mask]
            fl.plot_shift_filter_timeseries(shifts, good_idxs, td_list, px_max=px_max, plot_path=redu_dir)
            fl.plot_shift_filter_scatter(shifts, good_idxs, px_max=px_max, plot_path=redu_dir)
            # plot the amplitudes 
            fl.plot_generic_timeseries(
                g_amps_cube,
                good_idxs,
                td_list,
                plot_path=redu_dir,
                plt_title="Gaussian fit amplitudes",
                plt_name="3_gauss_amp_filter_timeseries.png",
            )
        # 3.1 Average the remaing frames (accumulate in float32)
        average_image = np.mean(centered_data_cube_filtered, axis=0, dtype=np.float32)
        # 3.1 write the average image
        fits.PrimaryHDU(data=np.asarray(average_image, dtype=np.float32)).writeto(
            average_image_path, overwrite=True
        )
    return average_image


###################### File table helper functions ######################

def _read_redu_table(path):
    """Load a pipeline ASCII metadata table (commented header)."""
    return Table.read(path, format=_REDU_ASCII_FMT)

def _write_redu_table(table, path):
    """Write a pipeline ASCII metadata table; creates parent directory if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    table.write(path, format=_REDU_ASCII_FMT, overwrite=True)

def _file_table_static_from_full(full_table):
    """Telemetry + masterdark only (no pipeline filter columns)."""
    cols = [c for c in full_table.colnames if c != "to_use"]
    return full_table[cols].copy()


def _ephemeral_file_table_with_to_use(file_table_static, file_table_output):
    """Same roster as static + synthetic ``to_use`` for centered/average writers."""
    t = file_table_static.copy()
    t["to_use"] = np.asarray(file_table_output["pass_majority_config"], dtype=int)
    return t


def _filter_file_table_output_for_centering(file_table_output):
    """
    One row per clean-cube frame (``pass_majority_config == 1``), in the same order as
    ``clean_cube.fits`` / ``centered_cube.fits``.
    """
    maj = np.asarray(file_table_output["pass_majority_config"], dtype=int) == 1
    return file_table_output[maj].copy()


def _date_obs_for_centering(file_table_static, file_table_output):
    """DATE_OBS values aligned with the majority-config clean-cube ordering."""
    maj = np.asarray(file_table_output["pass_majority_config"], dtype=int) == 1
    return file_table_static["DATE_OBS"][maj]

###################### STEP 0 helper functions ######################

def find_files(obs_path, unsats_dir, camera="camsci1", max_files=-1):
    '''
    Take the observation path and unsats directory to find as many files as requested
    '''
    log.info("Finding unsat files")
    unsat_files = glob.glob(f"{obs_path}{unsats_dir}/{camera}/*")
    if max_files > len(unsat_files): max_files = len(unsat_files)
    unsat_files = sorted(unsat_files)[:max_files]
    log.info("Found unsat files: %s", len(unsat_files))
    return unsat_files

def find_dark_files(dark_search_dir, file_table, camera):
    '''
    Finds unique configuration parameters for the dataset 
    Finds dark files for each configuration

    ``dark_search_dir`` is the root directory searched for ``*masterdark*.fits*``
    (typically from config ``masterdark_dir``, else ``redu_path``).
    '''
    log.info("Finding dark files")
    # These helpers live in filereads.py and call into darks.py as needed.
    unique_configs = md.unique_telemetry_configs_for_dark_lookup(file_table, camera=camera)
    dark_dictionary = md.lookup_masterdarks_from_telemetry_table(
        unique_configs, redu_dir=dark_search_dir, camera=camera
    )
    # dictionary length will be number of params
    log.info("Found %s unique configurations for this camera", len(dark_dictionary))
    # number of masterdark paths tell us if those 
    for config in dark_dictionary:
        if len(config['masterdark_paths']) == 0:
            log.warning("No master dark found for configuration: %s", config)
    file_table_total = md.merge_file_table_with_darks(file_table, dark_dictionary)
    return file_table_total
    
def pick_unsat_params(file_table_total):
    '''
    Find the detector-configuration that appears in the most files.

    Adds a column ``to_use`` to the table: 1 if the file is in the majority
    configuration, else 0.
    '''
    log.info("Picking unsat parameters")
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
    missing = [c for c in cfg_cols if c not in file_table_total.colnames]
    if missing:
        raise ValueError(f"  file_table_total missing required columns for param picking: {missing}")

    # Count how many files fall into each configuration.
    key_rows = file_table_total[list(cfg_cols)]
    uniq = np.unique(np.array(key_rows), axis=0, return_counts=True)
    uniq_vals, counts = uniq
    if len(counts) == 0:
        out = file_table_total.copy()
        out["to_use"] = np.array([], dtype=int)
        return out

    # Choose the most common config (if tie, numpy picks first in sorted order).
    majority_idx = int(np.argmax(counts))
    majority = uniq_vals[majority_idx]

    # Build mask and write to_use column.
    # Use per-column comparison to avoid dtype surprises from structured arrays.
    mask = np.ones(len(file_table_total), dtype=bool)
    for i, col in enumerate(cfg_cols):
        mask &= (file_table_total[col] == majority[i])

    out = file_table_total.copy()
    out["to_use"] = mask.astype(int)
    return out

def _resolve_masterdark_search_dir(redu_path, masterdark_dir):
    """Directory tree to search for ``*masterdark*.fits*`` (recursive)."""
    if masterdark_dir is not None:
        s = os.path.expanduser(os.fspath(masterdark_dir)).strip()
        if s:
            return s
    return os.path.expanduser(os.fspath(redu_path)).strip()

def _load_fits_primary_float32(path):
    """Load primary HDU data as float32 (reduces RAM vs float64 cubes)."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)

def _init_file_table_output(full_table):
    """One row per file: majority filter + placeholders for later steps."""
    n = len(full_table)
    to_use = np.asarray(full_table["to_use"], dtype=int)
    out = Table()
    out["filename"] = full_table["filename"].copy()
    out["pass_majority_config"] = to_use.copy()
    out["max_value"] = np.full(n, np.nan, dtype=float)
    out["pass_peak_max"] = np.zeros(n, dtype=int)
    out["shift_x"] = np.full(n, np.nan, dtype=float)
    out["shift_y"] = np.full(n, np.nan, dtype=float)
    out["gauss_y"] = np.full(n, np.nan, dtype=float)
    out["gauss_x"] = np.full(n, np.nan, dtype=float)
    out["gauss_sigma_y"] = np.full(n, np.nan, dtype=float)
    out["gauss_sigma_x"] = np.full(n, np.nan, dtype=float)
    out["gauss_amp"] = np.full(n, np.nan, dtype=float)
    out["gauss_offset"] = np.full(n, np.nan, dtype=float)
    out["pass_avg_shift"] = np.zeros(n, dtype=int)
    out["pass_avg_amp"] = np.zeros(n, dtype=int)
    out["used_in_average"] = np.zeros(n, dtype=int)
    return out

###################### STEP 1 helper functions ######################  


###################### STEP 2 helper functions ######################

def _update_file_table_output_step2(
    file_table_output,
    max_values,
    good_idxs,
    shifts,
    gauss_params,
):
    """Fill centering / Gaussian columns for rows that passed majority (step 1)."""
    pass_maj = np.asarray(file_table_output["pass_majority_config"], dtype=int)
    used_rows = np.flatnonzero(pass_maj == 1)
    n_used = len(used_rows)
    mv = np.asarray(max_values).ravel()
    if mv.size != n_used:
        raise ValueError(
            f"max_values length {mv.size} != number of pass_majority rows {n_used}"
        )
    g = np.asarray(good_idxs, dtype=int).ravel()
    gset = set(g.tolist())
    for k in range(n_used):
        gr = int(used_rows[k])
        file_table_output["max_value"][gr] = float(mv[k])
        file_table_output["pass_peak_max"][gr] = 1 if k in gset else 0
    s = np.asarray(shifts, dtype=float)
    gp = None
    if gauss_params is not None:
        gp = np.asarray(gauss_params, dtype=float)
    for k, clean_idx in enumerate(g):
        gr = int(used_rows[int(clean_idx)])
        file_table_output["shift_x"][gr] = float(s[k, 0])
        file_table_output["shift_y"][gr] = float(s[k, 1])
        if gp is not None and gp.ndim == 2 and gp.shape[1] >= 6:
            file_table_output["gauss_y"][gr] = float(gp[k, 0])
            file_table_output["gauss_x"][gr] = float(gp[k, 1])
            file_table_output["gauss_sigma_y"][gr] = float(gp[k, 2])
            file_table_output["gauss_sigma_x"][gr] = float(gp[k, 3])
            file_table_output["gauss_amp"][gr] = float(gp[k, 4])
            file_table_output["gauss_offset"][gr] = float(gp[k, 5])
    return file_table_output

def filter_file_table_to_use(file_table_total, use_col="to_use"):
    """
    Return a copy of ``file_table_total`` containing only rows with ``use_col == 1``.
    """
    if use_col not in file_table_total.colnames:
        raise ValueError(f"table missing column {use_col!r}")
    mask = file_table_total[use_col] == 1
    return file_table_total[mask].copy()


def _subset_table_with_average_pass_flags(
    centered_subset, good_idxs1, good_idxs2, good_idxs
):
    """
    Copy the centered-roster table and add ``pass_avg_shift``, ``pass_avg_amp``,
    ``used_in_average`` (indices along ``centered_cube.fits``).
    """
    sy = np.asarray(centered_subset["shift_y"], dtype=float)
    sx = np.asarray(centered_subset["shift_x"], dtype=float)
    finite = np.isfinite(sy) & np.isfinite(sx)
    n_cube = int(np.sum(finite))
    cube_idx = np.full(len(centered_subset), -1, dtype=np.int64)
    cube_idx[finite] = np.arange(n_cube, dtype=np.int64)

    g1 = set(np.asarray(good_idxs1, dtype=int).ravel().tolist())
    g2 = set(np.asarray(good_idxs2, dtype=int).ravel().tolist())
    gu = set(np.asarray(good_idxs, dtype=int).ravel().tolist())

    pass_shift = np.zeros(len(centered_subset), dtype=int)
    pass_amp = np.zeros(len(centered_subset), dtype=int)
    used_avg = np.zeros(len(centered_subset), dtype=int)
    for i in range(len(centered_subset)):
        ci = int(cube_idx[i])
        if ci < 0:
            continue
        if ci in g1:
            pass_shift[i] = 1
        if ci in g2:
            pass_amp[i] = 1
        if ci in gu:
            used_avg[i] = 1

    out = centered_subset.copy()
    out["pass_avg_shift"] = pass_shift
    out["pass_avg_amp"] = pass_amp
    out["used_in_average"] = used_avg
    return out


###################### STEP 3 helper functions ######################

def _merge_file_table_output_step3(file_table_output, subset_with_avg_flags):
    """Copy step-3 filter flags from the centered-roster subset table back to full roster."""
    fn_to_i = {
        str(file_table_output["filename"][i]): i for i in range(len(file_table_output))
    }
    ct = subset_with_avg_flags
    for i in range(len(ct)):
        fn = str(ct["filename"][i])
        if fn not in fn_to_i:
            continue
        j = fn_to_i[fn]
        file_table_output["pass_avg_shift"][j] = int(ct["pass_avg_shift"][i])
        file_table_output["pass_avg_amp"][j] = int(ct["pass_avg_amp"][i])
        file_table_output["used_in_average"][j] = int(ct["used_in_average"][i])
    return file_table_output

def _persist_file_table_output_after_step3(file_table_output, redu_dir, subset_with_flags):
    """
    Merge step-3 flags from ``subset_with_flags`` (centered roster + ``pass_avg_*`` /
    ``used_in_average``) into the full ``file_table_output`` and save.
    """
    _merge_file_table_output_step3(file_table_output, subset_with_flags)
    _write_redu_table(
        file_table_output, os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)
    )
    return file_table_output

def load_shifts(centered_file_table):
    """
    Rows with finite shift_y and shift_x (frames that passed max filter and were fit).
    Returns an array of shape (N, 2) with columns [shift_y, shift_x].
    """
    sy = np.asarray(centered_file_table["shift_y"], dtype=float)
    sx = np.asarray(centered_file_table["shift_x"], dtype=float)
    mask = np.isfinite(sy) & np.isfinite(sx)
    return np.column_stack([sy[mask], sx[mask]])

def load_table_params(param_col, file_table):
    """
    Values of ``param_col`` for rows with finite ``shift_x`` and ``shift_y`` (same rows as
    :func:`load_shifts`). One entry per centered-cube frame — aligns with ``load_shifts``
    and indexing ``centered_data_cube[i]``.
    """
    param_array = np.asarray(file_table[param_col], dtype=float)
    return param_array


###################### MAIN FUNCTIONS ######################

def build_preprocess_run_params(
    params,
    unsats_dir,
    camera,
    *,
    config_source_path=None,
):
    """
    Shallow copy of config ``params`` plus per-run keys. Each pipeline step reads only
    the keys it needs from the returned dict.

    Required in ``params``: ``obs_path``, ``redu_path`` (and typically ``cameras`` at
    config level; ``camera`` here overrides the per-run value).

    Sets defaults: ``pct_cut``, ``gauss_amp_pct_cut``, ``px_max``, ``plot``,
    ``max_files``, ``force_rerun``, ``fit_func``, ``crop_shape`` (from ``crop_size``
    if needed), ``masterdark_dir``.
    Adds ``unsats_dir``, ``camera``, ``redu_dir``, and optionally ``config_source_path``.
    """
    p = dict(params)
    p["unsats_dir"] = unsats_dir.strip() if isinstance(unsats_dir, str) else unsats_dir
    p["camera"] = camera
    p["redu_dir"] = f"{p['redu_path']}{p['unsats_dir']}/{p['camera']}/"
    p.setdefault("pct_cut", 10)
    p.setdefault("gauss_amp_pct_cut", p["pct_cut"])
    p.setdefault("px_max", 5)
    p.setdefault("plot", False)
    p.setdefault("max_files", -1)
    p.setdefault("force_rerun", False)
    p.setdefault("fit_func", "gauss_min")
    if "crop_shape" not in p and p.get("crop_size") is not None:
        p["crop_shape"] = p["crop_size"]
    p.setdefault("crop_shape", None)
    if config_source_path is not None:
        p["config_source_path"] = config_source_path
    return p


def preprocess_main(run_params):
    """
    Run preprocess steps 0–3.

    Parameters
    ----------
    run_params : dict
        Built with :func:`build_preprocess_run_params`. Must include ``redu_dir``,
        ``obs_path``, ``redu_path``, ``unsats_dir``, ``camera``, and the keys each step
        reads (see ``find_filetable``, ``make_clean_cube``, ``make_centered_cube``,
        ``make_average_image``).

        Step 0 writes ``file_table.txt`` (static telemetry + ``masterdark_path``) and
        ``file_table_output.txt`` (all pipeline filters, fits, and average-stage flags);
        later steps refresh that output table. There are no separate centered/average table files.

    ``masterdark_dir`` in ``run_params`` sets where to search for ``*masterdark*.fits``.
    ``config_source_path`` in ``run_params``: if set, copy to ``redu_dir`` as
    ``preprocess_config.txt``.
    """
    redu_dir = run_params["redu_dir"]
    unsats_dir = run_params["unsats_dir"]
    camera = run_params["camera"]

    if not os.path.exists(redu_dir):
        os.mkdir(redu_dir)

    _configure_preprocess_logging(redu_dir)
    _copy_preprocess_config_to_redu(run_params.get("config_source_path"), redu_dir)
    log.info("=> Processing %s %s (redu_dir=%s)", unsats_dir, camera, redu_dir)

    file_table_static, file_table_output = find_filetable(run_params)
    clean_cube_path = make_clean_cube(run_params, file_table_static, file_table_output)
    centered_file_table = make_centered_cube(
        run_params, clean_cube_path, file_table_static, file_table_output
    )
    average_image = make_average_image(
        run_params, centered_file_table, file_table_static, file_table_output
    )

    return average_image

########################## Pipeline functionality ##############################

########### Logger Setup ##############

def _redu_pkg_logger():
    """Parent of ``preprocess`` / ``filtering`` so both share one set of handlers."""
    p = log.parent
    return p if p.name else log


def _ensure_stderr_logging():
    """Attach a stderr handler on the redu package logger (shared with ``filtering``)."""
    pkg = _redu_pkg_logger()
    if pkg.handlers:
        return
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    pkg.addHandler(sh)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    log.propagate = True


def _configure_preprocess_logging(redu_dir):
    """
    Log to ``{redu_dir}/preprocess.log`` (overwrite each run) and stderr.
    Same directory as ``PREPROCESS_CONFIG_SNAPSHOT_NAME`` (``preprocess_config.txt``).

    Handlers are attached to the ``tools4magaox.redu`` package logger so sibling modules
    (e.g. ``filtering``) also write to the same file.
    """
    os.makedirs(redu_dir, exist_ok=True)
    log_path = os.path.join(redu_dir, PREPROCESS_LOG_NAME)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    pkg = _redu_pkg_logger()
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
    for h in list(log.handlers):
        log.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    pkg.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    pkg.addHandler(sh)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    log.propagate = True
    log.info("Preprocess log file: %s", log_path)


def _copy_preprocess_config_to_redu(config_source_path, redu_dir):
    """Copy the preprocess config file into ``redu_dir`` for provenance."""
    if not config_source_path:
        return
    src = os.path.abspath(os.fspath(config_source_path))
    if not os.path.isfile(src):
        return
    os.makedirs(redu_dir, exist_ok=True)
    dst = os.path.join(redu_dir, PREPROCESS_CONFIG_SNAPSHOT_NAME)
    shutil.copy2(src, dst)
    log.info("Saved config snapshot: %s", dst)


############ Conf File reads ##############
# TODO: maybe make this it's own file

def read_preproc_config(config_path):
    """
    Read a preprocess config file (one ``name = value`` per line; ``#`` starts a comment).

    Values are parsed with ``ast.literal_eval`` (strings, lists, numbers, booleans).

    Parameters
    ----------
    config_path : str or os.PathLike
        Path to the config file (e.g. ``conf_ex/conf_preproc_ex.txt``).

    Returns
    -------
    dict
        All parameters found in the file.
    """
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


def _nonempty_str(x):
    return isinstance(x, str) and bool(x.strip())


def _nonempty_camera_list(x):
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    return all(_nonempty_str(c) for c in x)


def _unsats_dirs_ok(params):
    """True if ``unsats_dirs`` or legacy ``unsats_dir`` is present and usable."""
    if "unsats_dirs" in params:
        u = params["unsats_dirs"]
        if _nonempty_str(u):
            return True
        if isinstance(u, (list, tuple)) and len(u) > 0 and all(_nonempty_str(d) for d in u):
            return True
    if "unsats_dir" in params and _nonempty_str(params["unsats_dir"]):
        return True
    return False


def check_preproc_config(params):
    """
    Verify required preprocess parameters: ``obs_path``, ``redu_path``,
    ``unsats_dirs`` (or ``unsats_dir``), and ``cameras``.

    Parameters
    ----------
    params : dict
        Typically from :func:`read_preproc_config`.

    Returns
    -------
    list of str
        Names of missing or invalid keys. Empty if all required fields are OK.
    """
    bad = []
    if not _nonempty_str(params.get("obs_path")):
        bad.append("obs_path")
    if not _nonempty_str(params.get("redu_path")):
        bad.append("redu_path")
    if not _unsats_dirs_ok(params):
        bad.append("unsats_dirs")
    if not _nonempty_camera_list(params.get("cameras")):
        bad.append("cameras")
    return bad

def _unsats_dir_list(params):
    """Return a list of unsats directory names from config (``unsats_dirs`` or ``unsats_dir``)."""
    if "unsats_dirs" in params:
        u = params["unsats_dirs"]
        if _nonempty_str(u):
            return [u.strip()]
        if isinstance(u, (list, tuple)):
            return [d.strip() for d in u if _nonempty_str(d)]
    if "unsats_dir" in params and _nonempty_str(params["unsats_dir"]):
        return [params["unsats_dir"].strip()]
    return []

def run_preprocess_from_config(params, config_source_path=None):
    """
    Validate ``params`` and run :func:`preprocess_main` for each unsats directory and camera.

    For each run, builds a per-camera dict with :func:`build_preprocess_run_params` and passes
    it to :func:`preprocess_main`; step functions read the keys they need from that dict.

    Optional keys in ``params`` (defaults in :func:`build_preprocess_run_params`):
    ``pct_cut``, ``gauss_amp_pct_cut``, ``px_max``, ``plot``, ``max_files``,
    ``force_rerun``, ``fit_func``.
    ``masterdark_dir`` (optional): root to search for master dark FITS; defaults to ``redu_path``.
    ``crop_shape`` or ``crop_size`` (optional): 2-tuple like ``(64, 64)`` for step-2 cropping.

    ``config_source_path`` (optional): path to the config file on disk; copied into each
    ``{redu_path}{unsats_dir}/{camera}/`` as ``preprocess_config.txt``.
    """
    missing = check_preproc_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")
    unsats_dirs = _unsats_dir_list(params)
    cameras = list(params["cameras"])

    for unsats_dir in unsats_dirs:
        for camera in cameras:
            try:
                run = build_preprocess_run_params(
                    params,
                    unsats_dir,
                    camera,
                    config_source_path=config_source_path,
                )
                preprocess_main(run)
            except Exception:
                log.exception("Error processing %s %s", unsats_dir, camera)


def cli_preprocess(argv=None):
    """CLI entry: one or more preprocess config paths."""
    parser = argparse.ArgumentParser(
        description="Run the preprocess pipeline from config file(s)."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        metavar="CONF",
        help="Preprocess config file(s), e.g. conf_ex/conf_preproc_ex.txt",
    )
    args = parser.parse_args(argv)
    _ensure_stderr_logging()
    for cfg_path in args.configs:
        params = read_preproc_config(cfg_path)
        log.info("=> Config: %s", cfg_path)
        try:
            run_preprocess_from_config(params, config_source_path=cfg_path)
        except ValueError as e:
            log.exception("%s: %s", cfg_path, e)
            raise SystemExit(1) from e

#####################################################################

if __name__ == "__main__":
    # collects args from command line and runs the preprocess pipeline
    cli_preprocess()