# preprocess.py
# step 1 in the reduction pipeline
# this makes our unsats
# running this on a directory will make a master unsats in that directory
import argparse
import ast
import glob
import os
import shutil
import sys
import traceback
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

# files you should expect this script to make
FILE_TABLE_NAME = "file_table.txt"
CENTERED_FILE_TABLE_NAME = "file_table_centered.txt"
FILE_TABLE_AVERAGE_NAME = "file_table_average.txt"
CLEAN_CUBE_NAME = "clean_cube.fits"
CENTERED_CUBE_NAME = "centered_cube.fits"
AVERAGE_IMAGE_NAME = "average_image.fits"
# Copy of the preprocess config saved alongside pipeline outputs (CLI supplies path).
PREPROCESS_CONFIG_SNAPSHOT_NAME = "preprocess_config.txt"


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
    print(f"      => Saved config snapshot: {dst}")


############# Main Functions #############

# STEP 0
def find_filetable(redu_dir, obs_path, unsats_dir, camera, max_files, redu_path, force_rerun=False, masterdark_dir=None):
    '''
    Checks to see if the table exists in redu_path
    If it doesn't, it's created
    '''
    print("   0. Finding file table")
    file_table_path = f"{redu_dir}/{FILE_TABLE_NAME}"

    if os.path.exists(file_table_path) and force_rerun == False:
        print("      => LOADING FILE TABLE")
        file_table = Table.read(file_table_path, format="ascii.commented_header")
    else:
        print("      => CREATING FILE TABLE")
        # 0.0 - file table
        unsat_files = find_files(obs_path, unsats_dir, camera, max_files)
        unsats_table = fr.fits_telemetry_table(unsat_files, camera)
        # 0.1 - dark files needed merged with file table
        dark_search_dir = _resolve_masterdark_search_dir(redu_path, masterdark_dir)
        file_table_darks = find_dark_files(dark_search_dir, unsats_table, camera)
        # 0.2 - pick the parameters that are in the largest number of files
        file_table = pick_unsat_params(file_table_darks)
        # 0.3 - write the file table
        file_table.write(file_table_path, format="ascii.commented_header", overwrite=True)
    return file_table

# STEP 1
def make_clean_cube(file_table_total, redu_dir, obs_path, unsats_dir, camera, force_rerun=False):
    '''
    Makes a clean cube from all unsat files in majority param set
    '''
    print("   1. Making clean cube")
    clean_cube_path = f"{redu_dir}/{CLEAN_CUBE_NAME}"

    if os.path.exists(clean_cube_path) and force_rerun == False:
        print("      => CLEAN CUBE EXISTS, SKIPPING")
    else:
        print("      => CREATING CLEAN CUBE")
        # 1.0 find files to use (telemetry table stores basenames only)
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
def make_centered_cube(clean_cube_path, file_table_total, redu_dir, pct_cut, force_rerun=False, save_plot=True, crop_shape=None, fit_func="gauss_min"):
    '''
    Input: total cube of unsats
    Process:
        1. filter the cube based on max peaks
        2. center the remaining frames
        3. write the centered cube, 
        4. write which frames were filtered out, shifts for the remaining frames
    '''
    print("   2. Making centered cube")
    centered_cube_path = f"{redu_dir}/{CENTERED_CUBE_NAME}"
    centered_file_table_path = f"{redu_dir}/{CENTERED_FILE_TABLE_NAME}"

    if os.path.exists(centered_cube_path) and force_rerun == False:
        print("      => CENTERED CUBE EXISTS, SKIPPING")
        centered_file_table = Table.read(centered_file_table_path, format="ascii.commented_header")
    else:
        print("      => CREATING CENTERED CUBE")
        unsats_data_cube = _load_fits_primary_float32(clean_cube_path)
        # 2.0 filter the cube based on max peaks
        max_values, good_idxs = fl.filter_max_value(unsats_data_cube, perc=pct_cut)
        if save_plot:
            # Keep DATE_OBS aligned with the clean cube ordering (to_use == 1).
            td_list = filter_file_table_to_use(file_table_total, use_col="to_use")["DATE_OBS"]
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
        centered_file_table = write_centered_file_table(
            file_table_total,
            redu_dir,
            max_values,
            good_idxs,
            shifts,
            gauss_params_good=gauss_params,
        )
        # 2.5 write the centered cube
        fits.PrimaryHDU(data=np.asarray(centered_data_cube, dtype=np.float32)).writeto(
            centered_cube_path, overwrite=True
        )
    return centered_file_table

# STEP 3
def make_average_image(
    centered_file_table,
    redu_dir,
    force_rerun=False,
    save_plot=True,
    px_max=10,
    pct_cut=20,
):
    '''
    Input: centered cube of unsats 
    Process:
        1. filter based on the shifts
        2. average the remaining frames
        3. write the average image, which frames were filtered out
        4. write ``file_table_average.txt`` — centered table plus ``pass_avg_shift``,
           ``pass_avg_amp``, ``used_in_average`` (see :func:`write_average_file_table`).
    '''
    print("   3. Making average image")
    centered_cube_path = f"{redu_dir}/{CENTERED_CUBE_NAME}"
    average_image_path = f"{redu_dir}/{AVERAGE_IMAGE_NAME}"

    if os.path.exists(average_image_path) and force_rerun == False:
        print("      => AVERAGE IMAGE EXISTS, SKIPPING")
        average_image = _load_fits_primary_float32(average_image_path)
    else:
        print("      => CREATING AVERAGE IMAGE")
        centered_data_cube = _load_fits_primary_float32(centered_cube_path)
        # 3.0 filter based on the shifts (indices are centered-cube rows 0..n-1)
        shifts = load_shifts(centered_file_table)
        good_idxs1 = fl.filter_unstat_shifts(shifts, px_max=px_max)
        # mask on usable centring shifts
        sy = np.asarray(centered_file_table["shift_y"], dtype=float)
        sx = np.asarray(centered_file_table["shift_x"], dtype=float)
        mask = np.isfinite(sy) & np.isfinite(sx)

        g_amps_cube = np.asarray(centered_file_table["gauss_amp"], dtype=float)[mask]
        _, good_idxs2 = fl.filter_max_value(g_amps_cube, perc=pct_cut)
        good_idxs = np.intersect1d(good_idxs1, good_idxs2)
        print(f"        -> {len(good_idxs)} / {len(centered_data_cube)} centered frames kept for average image")
        centered_data_cube_filtered = centered_data_cube[good_idxs]
        write_average_file_table(
            centered_file_table,
            redu_dir,
            good_idxs1,
            good_idxs2,
            good_idxs,
        )
        # Grab the timeseries from the centered file table
        if save_plot:
            td_list = centered_file_table["DATE_OBS"][mask]
            fl.plot_shift_filter_timeseries(shifts, good_idxs, td_list, px_max=10, plot_path=redu_dir)
            fl.plot_shift_filter_scatter(shifts, good_idxs, px_max=10, plot_path=redu_dir)
            # plot the amplitudes 
            fl.plot_generic_timeseries(
                g_amps_cube,
                good_idxs,
                td_list,
                plot_path=redu_dir,
                plt_title="Gaussian fit amplitudes",
                plt_name="gauss_amp_filter_timeseries.png",
            )
        # 3.1 Average the remaing frames (accumulate in float32)
        average_image = np.mean(centered_data_cube_filtered, axis=0, dtype=np.float32)
        # 3.1 write the average image
        fits.PrimaryHDU(data=np.asarray(average_image, dtype=np.float32)).writeto(
            average_image_path, overwrite=True
        )
    return average_image

###################### STEP 0 helper functions ######################

def find_files(obs_path, unsats_dir, camera="camsci1", max_files=-1):
    '''
    Take the observation path and unsats directory to find as many files as requested
    '''
    print("   Finding unsat files")
    unsat_files = glob.glob(f"{obs_path}{unsats_dir}/{camera}/*")
    if max_files > len(unsat_files): max_files = len(unsat_files)
    unsat_files = sorted(unsat_files)[:max_files]
    print("   Found unsat files: ", len(unsat_files))
    return unsat_files

def find_dark_files(dark_search_dir, file_table, camera):
    '''
    Finds unique configuration parameters for the dataset 
    Finds dark files for each configuration

    ``dark_search_dir`` is the root directory searched for ``*masterdark*.fits*``
    (typically from config ``masterdark_dir``, else ``redu_path``).
    '''
    print("   Finding dark files")
    # These helpers live in filereads.py and call into darks.py as needed.
    unique_configs = md.unique_telemetry_configs_for_dark_lookup(file_table, camera=camera)
    dark_dictionary = md.lookup_masterdarks_from_telemetry_table(
        unique_configs, redu_dir=dark_search_dir, camera=camera
    )
    # dictionary length will be number of params
    print("   Found ", len(dark_dictionary), " unique configurations for this camera")
    # number of masterdark paths tell us if those 
    for config in dark_dictionary:
        if len(config['masterdark_paths']) == 0:
            print("     WARNING: No master dark found for configuration: ", config)
    file_table_total = md.merge_file_table_with_darks(file_table, dark_dictionary)
    return file_table_total
    
def pick_unsat_params(file_table_total):
    '''
    Find the detector-configuration that appears in the most files.

    Adds a column ``to_use`` to the table: 1 if the file is in the majority
    configuration, else 0.
    '''
    print("   Picking unsat parameters")
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

###################### STEP 1 helper functions ######################  


###################### STEP 2 helper functions ######################

def filter_file_table_to_use(file_table_total, use_col="to_use"):
    """
    Return a copy of ``file_table_total`` containing only rows with ``use_col == 1``.
    """
    if use_col not in file_table_total.colnames:
        raise ValueError(f"table missing column {use_col!r}")
    mask = file_table_total[use_col] == 1
    return file_table_total[mask].copy()

def write_centered_file_table(
    file_table_total,
    redu_dir,
    max_values,
    good_idxs,
    shifts_yx_good,
    gauss_params_good=None,
    use_col="to_use"):
    """
    Build the centered metadata table: copy the full file table, keep only rows
    with ``use_col == 1`` (same order as the clean cube), then add

    - ``max_value`` — per-frame max used by :func:`filtering.max_filter`
    - ``in_good_idxs`` — 1 if that cube index is in ``good_idxs``, else 0
    - ``shift_x``, ``shift_y`` — centroid shifts for frames in ``good_idxs``
      (from :func:`centering.gaussian_fit_shifts`: row ``k`` is ``(dx, dy)`` as
      x then y); NaN for frames that failed the max filter
    - Gaussian fit parameters for frames in ``good_idxs`` (NaN otherwise):
      ``gauss_x``, ``gauss_y``, ``gauss_sigma_x``, ``gauss_sigma_y``,
      ``gauss_amp``, ``gauss_offset``.

    ``shifts_yx_good`` must have shape ``(len(good_idxs), 2)``; row ``k`` pairs
    with cube index ``good_idxs[k]`` (components are x, y).

    Writes ``{redu_dir}/{CENTERED_FILE_TABLE_NAME}`` (commented_header ASCII).
    Returns the table.
    """
    ft = filter_file_table_to_use(file_table_total, use_col=use_col)
    n = len(ft)
    mv = np.asarray(max_values, dtype=float).ravel()
    if mv.shape[0] != n:
        raise ValueError(
            f"max_values length {mv.shape[0]} != number of used rows {n}"
        )
    g = np.asarray(good_idxs, dtype=int).ravel()
    if np.any(g < 0) or np.any(g >= n):
        raise ValueError(f"good_idxs must be in [0, {n - 1}], got {g}")
    s = np.asarray(shifts_yx_good, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError("shifts_yx_good must have shape (N, 2) with y, x per row")
    if s.shape[0] != len(g):
        raise ValueError(
            f"shifts_yx_good has {s.shape[0]} rows but good_idxs has {len(g)}"
        )

    gp = None
    if gauss_params_good is not None:
        gp = np.asarray(gauss_params_good, dtype=float)
        if gp.ndim != 2 or gp.shape[1] != 6:
            raise ValueError(
                "gauss_params_good must have shape (N, 6): (y0, x0, sigma_y, sigma_x, amp, offset)"
            )
        if gp.shape[0] != len(g):
            raise ValueError(
                f"gauss_params_good has {gp.shape[0]} rows but good_idxs has {len(g)}"
            )

    in_good = np.zeros(n, dtype=int)
    in_good[g] = 1
    shift_y = np.full(n, np.nan, dtype=float)
    shift_x = np.full(n, np.nan, dtype=float)
    gauss_y = np.full(n, np.nan, dtype=float)
    gauss_x = np.full(n, np.nan, dtype=float)
    gauss_sigma_y = np.full(n, np.nan, dtype=float)
    gauss_sigma_x = np.full(n, np.nan, dtype=float)
    gauss_amp = np.full(n, np.nan, dtype=float)
    gauss_offset = np.full(n, np.nan, dtype=float)
    # ``gaussian_fit_shifts`` returns (dx, dy) as (x, y) — see centering._gaussian_xy_shifts
    for k, idx in enumerate(g):
        shift_x[idx] = s[k, 0]
        shift_y[idx] = s[k, 1]
        if gp is not None:
            # params: (y0, x0, sigma_y, sigma_x, amplitude, offset)
            gauss_y[idx] = gp[k, 0]
            gauss_x[idx] = gp[k, 1]
            gauss_sigma_y[idx] = gp[k, 2]
            gauss_sigma_x[idx] = gp[k, 3]
            gauss_amp[idx] = gp[k, 4]
            gauss_offset[idx] = gp[k, 5]

    out = ft.copy()
    out["max_value"] = mv
    out["in_good_idxs"] = in_good
    out["shift_y"] = shift_y
    out["shift_x"] = shift_x
    out["gauss_y"] = gauss_y
    out["gauss_x"] = gauss_x
    out["gauss_sigma_y"] = gauss_sigma_y
    out["gauss_sigma_x"] = gauss_sigma_x
    out["gauss_amp"] = gauss_amp
    out["gauss_offset"] = gauss_offset

    os.makedirs(redu_dir, exist_ok=True)
    out_path = os.path.join(redu_dir, CENTERED_FILE_TABLE_NAME)
    out.write(out_path, format="ascii.commented_header", overwrite=True)
    return out


###################### STEP 3 helper functions ######################

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


def write_average_file_table(
    centered_file_table,
    redu_dir,
    good_idxs1,
    good_idxs2,
    good_idxs,
):
    """
    Copy the centered metadata table and add per-row flags for step-3 filters (indices are
    **centered-cube** frame indices ``0 .. n_cube-1``, same as ``centered_cube.fits``).

    New columns (0/1; rows without finite shifts are 0):

    - ``pass_avg_shift`` — frame index in ``good_idxs1`` (unstable-shift filter)
    - ``pass_avg_amp`` — frame index in ``good_idxs2`` (amplitude / max-value percentile filter)
    - ``used_in_average`` — frame index in ``good_idxs`` (intersection; frames averaged)

    Writes ``{redu_dir}/{FILE_TABLE_AVERAGE_NAME}``.
    """
    sy = np.asarray(centered_file_table["shift_y"], dtype=float)
    sx = np.asarray(centered_file_table["shift_x"], dtype=float)
    finite = np.isfinite(sy) & np.isfinite(sx)
    n_cube = int(np.sum(finite))
    cube_idx = np.full(len(centered_file_table), -1, dtype=np.int64)
    cube_idx[finite] = np.arange(n_cube, dtype=np.int64)

    g1 = set(np.asarray(good_idxs1, dtype=int).ravel().tolist())
    g2 = set(np.asarray(good_idxs2, dtype=int).ravel().tolist())
    gu = set(np.asarray(good_idxs, dtype=int).ravel().tolist())

    pass_shift = np.zeros(len(centered_file_table), dtype=int)
    pass_amp = np.zeros(len(centered_file_table), dtype=int)
    used_avg = np.zeros(len(centered_file_table), dtype=int)
    for i in range(len(centered_file_table)):
        ci = int(cube_idx[i])
        if ci < 0:
            continue
        if ci in g1:
            pass_shift[i] = 1
        if ci in g2:
            pass_amp[i] = 1
        if ci in gu:
            used_avg[i] = 1

    out = centered_file_table.copy()
    out["pass_avg_shift"] = pass_shift
    out["pass_avg_amp"] = pass_amp
    out["used_in_average"] = used_avg

    os.makedirs(redu_dir, exist_ok=True)
    out_path = os.path.join(redu_dir, FILE_TABLE_AVERAGE_NAME)
    out.write(out_path, format="ascii.commented_header", overwrite=True)
    print(f"      => Wrote average-stage table: {out_path}")
    return out


###################### MAIN FUNCTIONS ######################

def preprocess_main(
    obs_path,
    unsats_dir,
    redu_path,
    camera="camsci1",
    pct_cut=10,
    px_max=5,
    plot=False,
    max_files=-1,
    force_rerun=False,
    masterdark_dir=None,
    crop_shape=None,
    fit_func="gauss_min",
    config_source_path=None,
):
    """
    Run preprocess steps 0–3.

    ``masterdark_dir`` sets where to search for ``*masterdark*.fits`` (recursive).
    If omitted, ``redu_path`` is used.

    ``config_source_path``: if set, copy this file to ``redu_dir`` as
    ``preprocess_config.txt`` (overwrites on each run).
    """
    # specific folder for ther redu dir
    redu_dir = f"{redu_path}{unsats_dir}/{camera}/"
    # if it doesn't already exisct, make it
    if not os.path.exists(redu_dir):
        os.mkdir(redu_dir)

    # STEP 0
    file_table = find_filetable(
        redu_dir,
        obs_path,
        unsats_dir,
        camera,
        max_files,
        redu_path,
        force_rerun,
        masterdark_dir=masterdark_dir,
    )
    
    # STEP 1
    clean_cube_path = make_clean_cube(file_table, redu_dir, obs_path, unsats_dir, camera, force_rerun)

    # STEP 2
    centered_file_table = make_centered_cube(
        clean_cube_path,
        file_table,
        redu_dir,
        pct_cut,
        force_rerun,
        save_plot=plot,
        crop_shape=crop_shape,
        fit_func=fit_func,
    )

    # STEP 3
    average_image = make_average_image(centered_file_table, redu_dir, px_max=px_max, pct_cut=pct_cut, force_rerun=force_rerun, save_plot=plot)

    # END OF PIPELINE
    return average_image

########################## Pipeline functionality ##############################

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

    Optional keys (defaults match :func:`preprocess_main`): ``pct_cut``, ``plot``, ``max_files``.
    ``masterdark_dir`` (optional): root to search for master dark FITS; defaults to ``redu_path``.
    ``crop_shape`` or ``crop_size`` (optional): 2-tuple like ``(64, 64)`` used to crop frames
    before Gaussian fitting in step 2.

    ``config_source_path`` (optional): path to the config file on disk; copied into each
    ``{redu_path}{unsats_dir}/{camera}/`` as ``preprocess_config.txt``.
    """
    missing = check_preproc_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")
    #TODO: maybe just pass all the params into preprocess_main instead of having individuals
    obs_path = params["obs_path"]
    redu_path = params["redu_path"]
    unsats_dirs = _unsats_dir_list(params)
    cameras = list(params["cameras"])
    pct_cut = params.get("pct_cut", 10)
    px_max = params.get("px_max", 5)
    plot = params.get("plot", False)
    fit_func = params.get("fit_func", "gauss_min")
    force_rerun = params.get("force_rerun", False)
    max_files = params.get("max_files", -1)
    masterdark_dir = params.get("masterdark_dir")
    crop_shape = params.get("crop_shape", params.get("crop_size"))

    config_copy_path = f"{redu_path}/{unsats_dirs[0]}"
    _copy_preprocess_config_to_redu(config_source_path, config_copy_path)
    for unsats_dir in unsats_dirs:
        for camera in cameras:
            print(f"=> Processing {unsats_dir} {camera}")
            try:
                preprocess_main(
                    obs_path,
                    unsats_dir,
                    redu_path,
                    camera=camera,
                    pct_cut=pct_cut,
                    px_max=px_max,
                    plot=plot,
                    max_files=max_files,
                    masterdark_dir=masterdark_dir,
                    crop_shape=crop_shape,
                    fit_func=fit_func,
                    force_rerun=force_rerun,
                    config_source_path=config_source_path,
                )
            except Exception as e:
                print(
                    f"Error processing {unsats_dir} {camera}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exception(
                    type(e), e, e.__traceback__, chain=True, file=sys.stderr
                )


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
    for cfg_path in args.configs:
        params = read_preproc_config(cfg_path)
        print(f"=> Config: {cfg_path}")
        try:
            run_preprocess_from_config(params, config_source_path=cfg_path)
        except ValueError as e:
            print(f"{cfg_path}: {e}", file=sys.stderr)
            traceback.print_exception(
                type(e), e, e.__traceback__, chain=True, file=sys.stderr
            )
            raise SystemExit(1) from e

if __name__ == "__main__":
    # collects args from command line and runs the preprocess pipeline
    cli_preprocess()