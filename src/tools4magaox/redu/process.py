import argparse
import ast
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy
import scipy.signal
from astropy.io import fits
from scipy import ndimage
from hcipy import *

from tools4magaox.constants import *
try:
    from . import filereads as fr
    from . import center_spark as cs
    from . import filtering as fl
except ImportError:
    import filereads as fr
    import center_spark as cs
    import filtering as fl

log = logging.getLogger(__name__)

PROCESS_CONFIG_SNAPSHOT_NAME = "process_config.txt"
PROCESS_LOG_NAME = "process.log"
FILE_TABLE_NAME = "file_table.txt"
FILE_TABLE_OUTPUT_NAME = "file_table_output.txt"
REFERENCE_IMAGE_NAME = "reference_sparkles.fits"
MASKED_REFERENCE_IMAGE_NAME = "reference_sparkles_masked.fits"
CENTERED_CUBE_NAME = "average_image.fits"
MASK_IMAGE_NAME = "mask.fits"
CENTERED_FRAMES_DIR = "centered"

########################################################
######################### Main #########################
########################################################

def process_main(run_params):
    """
    Run the coronagraphic processing pipeline for one ``data_dir`` / ``camera`` pair.

    Parameters
    ----------
    run_params : dict
        Built with :func:`build_process_run_params`. Must include ``redu_dir``,
        ``data_dir``, and ``camera``. If ``config_source_path`` is set, a copy of the
        config is saved in ``redu_dir`` as ``process_config.txt``.
    """
    redu_dir = run_params["redu_dir"]
    data_dir = run_params["data_dir"]
    camera = run_params["camera"]

    os.makedirs(redu_dir, exist_ok=True)
    _configure_process_logging(redu_dir)
    _copy_process_config_to_redu(run_params.get("config_source_path"), redu_dir)

    log.info("=> Processing %s %s (redu_dir=%s)", data_dir, camera, redu_dir)

    # STEP 0
    file_table_static, file_table_output = s0_create_filetable(run_params)
    # STEP 1
    masked_reference_image, mask_image = s1_create_reference(run_params)
    # STEP 2
    file_table_output = s2_image_center(
        run_params, file_table_static, file_table_output, masked_reference_image, mask_image
    )
    # STEP 3
    s3_save_centered_images(run_params, file_table_static, file_table_output)
    # STEP 2b (optional second-pass centering on centered frames)
    file_table_output = s2b_recenter(
        run_params, file_table_static, file_table_output, masked_reference_image, mask_image
    )
    # OPTIONAL STEP 3b
    s3b_save_first_centered_cube(run_params, file_table_static, file_table_output)
    # STEP 4
    file_table_output = s4_save_statistics(
        run_params, file_table_static, file_table_output, mask_image
    )
    # OPTIONAL STEP 4b
    s4b_save_filtered_centered_cube(run_params, file_table_static, file_table_output)
    return True

# STEP 0 - File tables
def s0_create_filetable(run_params):
    """
    Load or build the static process file table and the process output manifest.

    Returns
    -------
    tuple[Table, Table]
        ``(file_table_static, file_table_output)`` where the static table contains one
        row per science file with telemetry plus ``masterdark_path``. The output table
        has one row per file with ``pass_majority_config``, shift columns, process
        filter metrics, and ``used_in_reduction``.

    Reads from ``run_params``: ``redu_dir``, ``obs_path``, ``data_dir``, ``camera``,
    ``max_files``, ``redu_path``, ``force_rerun``, ``masterdark_dir``.
    """
    log.info("0. Finding file tables")
    redu_dir = run_params["redu_dir"]
    obs_path = run_params["obs_path"]
    data_dir = run_params["data_dir"]
    camera = run_params["camera"]
    max_files = run_params["max_files"]
    redu_path = run_params["redu_path"]
    force_rerun = run_params["force_rerun"]
    masterdark_dir = run_params.get("masterdark_dir")

    static_path = os.path.join(redu_dir, FILE_TABLE_NAME)
    output_path = os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)

    if (not force_rerun and os.path.isfile(static_path) and os.path.isfile(output_path)):
        log.info("=> LOADING FILE TABLES")
        file_table_static = fr.read_redu_table(static_path)
        file_table_output = fr.read_redu_table(output_path)
        file_table_output = fr.prune_process_output_table(file_table_output)
    else:
        log.info("=> CREATING FILE TABLES")
        # find all files, error if none found
        data_files = fr.find_camera_files(obs_path, data_dir, camera, max_files)
        if len(data_files) == 0:
            raise ValueError(f"No files found for {data_dir} {camera}")
        # collect telemetry from all files
        file_table_telemetry = fr.fits_telemetry_table(data_files, camera)
        dark_search_dir = fr.resolve_masterdark_search_dir(redu_path, masterdark_dir)
        file_table_with_darks = fr.attach_masterdarks(
            file_table_telemetry, dark_search_dir, camera
        )
        file_table_full = fr.pick_majority_config(file_table_with_darks)
        file_table_static = fr.file_table_static_from_full(file_table_full)
        file_table_output = fr.init_file_table_output(file_table_full)
        # save both tables
        fr.write_redu_table(file_table_static, static_path)
        fr.write_redu_table(fr.prune_process_output_table(file_table_output), output_path)
    return file_table_static, file_table_output

# STEP 1 - Reference Image
def s1_create_reference(run_params):
    '''
    Take the unsat, and subtract the unsat nospark if available, then mask the sparkles.
    Save full reference image, masked reference image, and mask image.
    '''
    # load needed parameters
    log.info("1. Making reference image")
    redu_dir = run_params["redu_dir"]
    redu_path = run_params["redu_path"]
    force_rerun = run_params["force_rerun"]
    save_plot = _process_plot_enabled(run_params, "plot_reference")
    unsats_dir = run_params["unsats_dir"]
    unsats_nospark_dir = run_params["unsats_nospark_dir"]
    camera = run_params["camera"]
    spark_ang = run_params["spark_ang"]
    spark_sep = run_params["spark_sep"]
    wavelength = run_params["wavelength"]
    # create paths
    reference_image_path = os.path.join(redu_dir, REFERENCE_IMAGE_NAME)
    masked_reference_image_path = os.path.join(redu_dir, MASKED_REFERENCE_IMAGE_NAME)
    mask_image_path = os.path.join(redu_dir, MASK_IMAGE_NAME)
    unsats_path = os.path.join(redu_path, unsats_dir, camera)
    unsats_nospark_path = os.path.join(redu_path, unsats_nospark_dir, camera)

    if (not force_rerun and os.path.isfile(masked_reference_image_path)):
        log.info("=> LOADING REF IMAGE")
        masked_reference_image = fr._load_fits_primary_float32(masked_reference_image_path)
        mask_image = fr._load_fits_primary_float32(mask_image_path)
        # Backward compatibility: older runs may have saved the mask as a 1D
        # flattened array. Reshape it to 2D so downstream broadcasting works.
        if getattr(mask_image, "ndim", 0) == 1:
            target_shape = masked_reference_image.shape
            if int(np.prod(target_shape)) == int(mask_image.size):
                mask_image = np.asarray(mask_image, dtype=np.float32).reshape(target_shape)
            else:
                raise ValueError(
                    f"Loaded mask is 1D (len={mask_image.size}) but cannot reshape to "
                    f"masked_reference_image shape {target_shape}"
                )
    else:
        log.info("=> CREATING REF IMAGE")
        # check to see if average image has been made already
        reference_image = create_reference_image(unsats_path, unsats_nospark_path)
        mask_field = cs.make_sparkle_mask(
            spark_ang, spark_sep, reference_image, wavelength=wavelength
        )
        # Persist mask as a plain 2D array for downstream compatibility.
        mask_image = (
            np.asarray(mask_field.shaped, dtype=np.float32)
            if hasattr(mask_field, "shaped")
            else np.asarray(mask_field, dtype=np.float32)
        )
        masked_reference_image = reference_image * mask_image
        # save the reduced images
        fr._save_fits_primary_float32(reference_image, reference_image_path)
        fr._save_fits_primary_float32(mask_image, mask_image_path)
        fr._save_fits_primary_float32(masked_reference_image, masked_reference_image_path)
        if save_plot:
            fl.plot_reference_and_mask(masked_reference_image, mask_image, plot_path=redu_dir)

    return masked_reference_image, mask_image


# STEP 2 - Dark subtract and center
def s2_image_center(run_params, file_table_static, file_table_output, masked_reference_image, mask_image):
    '''
    Dark subtract, optionally coadd stacks, and find image centers.
    Writes ``shift_x``, ``shift_y``, and ``center_stack_id`` into the output table.
    '''
    log.info("2. Finding image centers")
    # load needed parameters
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    # get the paths for the different files: 
    mask_image_path = os.path.join(redu_dir, MASK_IMAGE_NAME)
    reference_image_path = os.path.join(redu_dir, REFERENCE_IMAGE_NAME)
    output_path = os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)

    if not force_rerun and _centering_complete(file_table_output):
        log.info("=> NO ACTION NEEDED, CENTERS ALREADY FOUND")
        return file_table_output
    
    log.info("=> FINDING IMAGE CENTERS")
    grid = cs.make_camsci_grid(masked_reference_image)
    mask = Field(np.asarray(mask_image, dtype=float).ravel(), grid)
    
    # centering function
    # TODO: should be able to start halfway if interrupted
    file_table_output = center_pool(
        file_table_static,
        file_table_output,
        masked_reference_image,
        mask,
        grid,
        run_params,
        max_files=-1,
        chunk_size=run_params.get("chunk_size", 100),
    )

    fr.write_redu_table(fr.prune_process_output_table(file_table_output), output_path)
    return file_table_output

# STEP 3 - shift and save centered frames
def s3_save_centered_images(run_params, file_table_static, file_table_output):
    """
    Dark-subtract, apply registration shifts, and write one FITS per frame under
    ``{redu_dir}/centered/``. Skips when outputs already exist unless
    ``force_rerun`` is set.
    """
    log.info("3. Saving centered frames")
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    n_workers = run_params.get("n_workers")

    jobs = _centered_frame_jobs(file_table_static, file_table_output, run_params)
    if not jobs:
        log.warning("3. No frames with finite shifts to save")
        return

    out_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    if not force_rerun and _centered_frames_complete(jobs):
        log.info("=> CENTERED FRAMES EXIST, SKIPPING")
        return

    log.info("3. Saving %s centered frames to %s", len(jobs), out_dir)
    save_centered_images_parallel(jobs, n_workers=n_workers)
    log.info("=> SAVED %s centered frames", len(jobs))


def s2b_recenter(
    run_params, file_table_static, file_table_output, masked_reference_image, mask_image
):
    """
    Second registration pass on frames already in ``centered/``.

    Estimates residual shifts, writes ``recenter_shift_x`` / ``recenter_shift_y``,
    and overwrites the centered FITS files with the additional correction applied.
    """
    if not run_params.get("recenter", False):
        return file_table_output

    log.info("2b. Second-pass centering on centered frames")
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    output_path = os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)
    file_table_output = fr.prune_process_output_table(file_table_output)

    jobs = _centered_frame_jobs(file_table_static, file_table_output, run_params)
    if not jobs:
        log.warning("2b. No first-pass centered frames available for recentering")
        return file_table_output
    if not _centered_frames_complete(jobs):
        log.warning("2b. Centered FITS files missing; complete step 3 before recentering")
        return file_table_output

    if not force_rerun and _recenter_complete(file_table_output):
        log.info("=> RECENTERING ALREADY DONE, SKIPPING")
        return file_table_output

    log.info("=> FINDING RESIDUAL IMAGE CENTERS")
    grid = cs.make_camsci_grid(masked_reference_image)
    mask = Field(np.asarray(mask_image, dtype=float).ravel(), grid)
    recenter_params = dict(run_params)
    recenter_params["center_from_centered"] = True

    file_table_output = center_pool(
        file_table_static,
        file_table_output,
        masked_reference_image,
        mask,
        grid,
        recenter_params,
        max_files=-1,
        chunk_size=run_params.get("chunk_size", 100),
        shift_cols=("recenter_shift_y", "recenter_shift_x"),
    )

    recenter_jobs = _recenter_apply_jobs(file_table_static, file_table_output, run_params)
    if recenter_jobs:
        log.info("2b. Applying residual shifts to %s centered frames", len(recenter_jobs))
        save_centered_images_parallel(recenter_jobs, n_workers=run_params.get("n_workers"))

    fr.write_redu_table(fr.prune_process_output_table(file_table_output), output_path)
    return file_table_output


def s3b_save_first_centered_cube(run_params, file_table_static, file_table_output):
    """
    Optional: write a FITS cube of the first N centered frames.

    Controlled by config key ``save_centered_cube_first_n``. When set to a
    positive integer, this loads the first N rows (in table order) that have
    finite shifts and an existing centered FITS file under ``centered/``, and
    writes a stacked cube to ``{redu_dir}/centered_firstN_cube.fits``.

    This is primarily for quick-look / debugging and is independent of step-4
    filtering (``used_in_reduction`` is not yet computed at this point).
    """
    n_first = run_params.get("save_centered_cube_first_n")
    if n_first is None:
        return
    n_first = int(n_first)
    if n_first <= 0:
        return

    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    out_path = os.path.join(redu_dir, f"centered_first{n_first}_cube.fits")
    if not force_rerun and os.path.isfile(out_path):
        log.info("3b. Centered cube exists, skipping (%s)", out_path)
        return

    row_idxs = _process_filter_row_idxs(file_table_static, file_table_output, run_params)
    if len(row_idxs) == 0:
        log.warning("3b. No centered frames available to cube")
        return

    use = row_idxs[: min(n_first, len(row_idxs))]
    log.info("3b. Saving first %s centered frames as cube: %s", len(use), out_path)
    cube = _load_centered_chunk(file_table_static, use, run_params)
    fits.PrimaryHDU(data=np.asarray(cube, dtype=np.float32)).writeto(out_path, overwrite=True)

# STEP 4 - filtering the new files - save plots, update the table
def s4_save_statistics(run_params, file_table_static, file_table_output, mask_image):
    """
    Apply process filters to all centered frames and write per-frame metrics plus
    ``used_in_reduction`` (1 only when every filter passes).
    """
    log.info("4. Applying process filters")
    # needed configuration parameters
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["rerun_filtering"]
    sigma_mp = run_params.get("max_point_sigma_clip", 2.0)
    sigma_shift = run_params.get("shift_sigma_clip", 2.0)
    sigma_int = run_params.get("speckle_intensity_sigma_clip", 2.0)
    sigma_rms = run_params.get("rms_sigma_clip", 2.0)
    rms_iter = run_params.get("rms_iterations", 3)
    output_path = os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)

    file_table_output = fr.prune_process_output_table(file_table_output)

    if not force_rerun:
        log.info("=> PROCESS FILTERS ALREADY APPLIED, SKIPPING")
        return file_table_output

    # Check for centered files
    row_idxs = _process_filter_row_idxs(file_table_static, file_table_output, run_params)
    if len(row_idxs) == 0:
        log.warning("4. No centered frames available for filtering")
        return file_table_output

    chunk_size = run_params.get("chunk_size", 100)
    speckle_mask = _mask_as_2d(mask_image)
    n = len(row_idxs)

    log.info("=> COLLECTING PEAK IDXS")
    peak_idxs = _collect_peak_idxs_chunked(
        file_table_static, row_idxs, run_params, chunk_size
    )
    pass_mp, radius = fl.filter_max_point_from_peak_idxs(peak_idxs, sigma_clip=sigma_mp)
    file_table_output["max_point_radius"][row_idxs] = radius

    log.info("=> COLLECTING SHIFTS")
    shifts = _shifts_for_row_idxs(file_table_output, row_idxs)
    pass_cs = fl.filter_center_shifts(shifts, sigma_clip=sigma_shift)

    log.info("=> COLLECTING SPECKLE INTENSITIES")
    intensities = _collect_speckle_intensities_chunked(
        file_table_static, row_idxs, run_params, speckle_mask, chunk_size
    )
    pass_si, _ = fl.filter_speckle_intensity_values(intensities, sigma_clip=sigma_int)
    file_table_output["speckle_intensity"][row_idxs] = intensities

    log.info("=> FILTERING RMS")
    pass_rms, rms_dev = _filter_rms_chunked(
        file_table_static,
        row_idxs,
        run_params,
        sigma_clip=sigma_rms,
        n_iter=rms_iter,
        chunk_size=chunk_size,
    )
    file_table_output["rms_deviation"][row_idxs] = rms_dev

    # check against all the filters for which to keep
    log.info("=> CHECKING AGAINST ALL FILTERS")
    used = pass_mp & pass_cs & pass_si & pass_rms
    file_table_output["used_in_reduction"][row_idxs] = used.astype(int)

    log.info(
        "4. used_in_reduction: %s/%s frames",
        int(np.sum(used)),
        n,
    )

    td_list = file_table_static["DATE_OBS"][row_idxs]
    if _process_plot_enabled(run_params, "plot_max_point"):
        fl.plot_generic_timeseries(
            radius,
            np.where(pass_mp)[0],
            td_list,
            plot_path=redu_dir,
            plt_title="Max point radius filter",
            plt_name="4_max_point_filter_timeseries.png",
        )
    if _process_plot_enabled(run_params, "plot_center_shift"):
        fl.plot_shift_filter_timeseries(
            shifts,
            np.where(pass_cs)[0],
            td_list,
            plot_path=redu_dir,
            plt_name="4_shift_filter_timeseries.png",
        )
        fl.plot_shift_filter_scatter(
            shifts,
            np.where(pass_cs)[0],
            plot_path=redu_dir,
            plt_name="4_shift_filter_scatter.png",
        )
    if _process_plot_enabled(run_params, "plot_speckle_intensity"):
        fl.plot_generic_timeseries(
            intensities,
            np.where(pass_si)[0],
            td_list,
            plot_path=redu_dir,
            plt_title="Speckle intensity filter",
            plt_name="4_speckle_intensity_filter_timeseries.png",
        )
    if _process_plot_enabled(run_params, "plot_rms"):
        fl.plot_generic_timeseries(
            rms_dev,
            np.where(pass_rms)[0],
            td_list,
            plot_path=redu_dir,
            plt_title="RMS deviation filter",
            plt_name="4_rms_filter_timeseries.png",
        )

    fr.write_redu_table(fr.prune_process_output_table(file_table_output), output_path)
    return file_table_output


def s4b_save_filtered_centered_cube(run_params, file_table_static, file_table_output):
    """
    Optional: write a FITS cube of the first N *kept* centered frames.

    Controlled by config key ``save_filtered_centered_cube_first_n``. When set to a
    positive integer, this selects the first N rows (in table order) with
    ``used_in_reduction == 1`` and an existing centered FITS file under
    ``{redu_dir}/centered/``. It writes the stacked cube to
    ``{redu_dir}/centered_kept_firstN_cube.fits``.

    This runs after step 4 so it respects the filter decisions.
    """
    n_first = run_params.get("save_filtered_centered_cube_first_n")
    if n_first is None:
        return
    n_first = int(n_first)
    if n_first <= 0:
        return

    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    out_path = os.path.join(redu_dir, f"centered_kept_first{n_first}_cube.fits")
    if not force_rerun and os.path.isfile(out_path):
        log.info("4b. Filtered centered cube exists, skipping (%s)", out_path)
        return

    # Use the same criteria as step 4 for centered-file existence, plus the final keep flag.
    row_idxs = _process_filter_row_idxs(file_table_static, file_table_output, run_params)
    if len(row_idxs) == 0:
        log.warning("4b. No centered frames available to cube")
        return

    used = np.asarray(file_table_output["used_in_reduction"][row_idxs], dtype=int) == 1
    kept_rows = row_idxs[used]
    if len(kept_rows) == 0:
        log.warning("4b. No kept frames (used_in_reduction=1); not writing cube")
        return

    use = kept_rows[: min(n_first, len(kept_rows))]
    log.info(
        "4b. Saving first %s kept centered frames as cube: %s",
        len(use),
        out_path,
    )
    cube = _load_centered_chunk(file_table_static, use, run_params)
    fits.PrimaryHDU(data=np.asarray(cube, dtype=np.float32)).writeto(out_path, overwrite=True)

    
########################################################
################### Helper functions ###################
########################################################


################### STEP 1 ###################

def create_reference_image(unsats_dir, unsats_nospark_dir):
    '''
    Create a reference image from the unsats and unsats_nospark directories
    '''
    unsats_image_path = os.path.join(unsats_dir, CENTERED_CUBE_NAME)
    unsats_nospark_image_path = os.path.join(unsats_nospark_dir, CENTERED_CUBE_NAME)
    # if we have both unsats and unsats_nospark, subtract the nospark from the unsats
    if os.path.isfile(unsats_image_path) and os.path.isfile(unsats_nospark_image_path):
        unsats = fr._load_fits_primary_float32(unsats_image_path)
        unsats_nospark = fr._load_fits_primary_float32(unsats_nospark_image_path)
        # TODO: do I need to normalize before subtracting
        reference_image = unsats - unsats_nospark
    # if we only have unsats, use that
    elif os.path.isfile(unsats_image_path):
        reference_image = fr._load_fits_primary_float32(unsats_image_path)
        #TODO: I think I need to highpass filter if I'm only using the spark unsat
    # if we don't have any files, error
    else:
        raise ValueError(f"No unsats or unsats_nospark files found in {unsats_image_path} and {unsats_nospark_image_path}")
    # normalize the reference image
    reference_image = reference_image / np.mean(reference_image)
    return reference_image

################### STEP 2 ###################

def center_pool(
    file_table,
    file_table_output,
    ref_image,
    mask,
    grid,
    params,
    max_files=-1,
    chunk_size=100,
    shift_cols=("shift_y", "shift_x"),
):
    """
    Find registration shifts in stacks of ``center_coadd_n`` frames (mean coadd
    before FFT registration when ``center_coadd_n`` > 1).
    """
    file_table_total = fr.ephemeral_file_table_with_to_use(file_table, file_table_output)
    all_idxs = np.arange(len(file_table_total))
    used_idxs = all_idxs[file_table_total["to_use"] == 1]
    no_files = len(used_idxs)
    if max_files == -1:
        max_files = no_files
    elif max_files > no_files:
        max_files = no_files
    used_idxs = used_idxs[:max_files]

    coadd_n = max(1, int(params.get("center_coadd_n", 1)))
    pass_label = "2b" if shift_cols != ("shift_y", "shift_x") else "2"
    stacks = _stack_row_indices(used_idxs, coadd_n)
    log.info(
        "%s. Centering %s frames in %s stacks (center_coadd_n=%s)",
        pass_label,
        len(used_idxs),
        len(stacks),
        coadd_n,
    )

    stack_i = 0
    while stack_i < len(stacks):
        batch = []
        frame_count = 0
        while stack_i < len(stacks):
            stack_rows = stacks[stack_i]
            n_frames = len(stack_rows)
            if batch and frame_count + n_frames > chunk_size:
                break
            batch.append((stack_i, stack_rows))
            frame_count += n_frames
            stack_i += 1
        for stack_id, stack_rows in batch:
            shifts = center_stack(file_table, stack_rows, ref_image, mask, grid, params)
            file_table_output = fr.update_file_table_output(
                file_table_output,
                stack_rows,
                shifts,
                center_stack_id=stack_id,
                shift_cols=shift_cols,
            )
    return file_table_output


def _stack_row_indices(used_idxs, coadd_n):
    """Split table row indices into consecutive stacks of up to ``coadd_n`` frames."""
    used_idxs = np.asarray(used_idxs, dtype=int).ravel()
    coadd_n = max(1, int(coadd_n))
    return [
        used_idxs[start : start + coadd_n]
        for start in range(0, len(used_idxs), coadd_n)
    ]


def center_stack(file_table, idxs, ref_image, mask, grid, params):
    """Load frames for one stack, optionally coadd, and estimate registration shifts."""
    if params.get("center_from_centered", False):
        data_cube = _load_centered_chunk(file_table, idxs, params)
    else:
        data_cube = data_cube_fom_idxs(file_table, idxs, params)
    coadd_n = max(1, int(params.get("center_coadd_n", 1)))
    if coadd_n > 1 and len(idxs) > 1:
        coadded = np.mean(data_cube, axis=0, keepdims=True)
        shifts = cs.register_files_fft(coadded, ref_image, mask, grid)
        shifts = np.repeat(shifts, len(idxs), axis=0)
    else:
        shifts = cs.register_files_fft(data_cube, ref_image, mask, grid)
    return shifts


def _centering_complete(file_table_output):
    """True when every majority-config row has finite shifts and a stack id."""
    maj = np.asarray(file_table_output["pass_majority_config"], dtype=int) == 1
    if not np.any(maj):
        return False
    sy = np.asarray(file_table_output["shift_y"], dtype=float)[maj]
    sx = np.asarray(file_table_output["shift_x"], dtype=float)[maj]
    if not bool(np.all(np.isfinite(sy) & np.isfinite(sx))):
        return False
    if "center_stack_id" in file_table_output.colnames:
        csi = np.asarray(file_table_output["center_stack_id"], dtype=int)[maj]
        return bool(np.all(csi >= 0))
    return True


def _recenter_complete(file_table_output):
    """True when every majority-config row has finite second-pass shifts."""
    if "recenter_shift_y" not in file_table_output.colnames:
        return False
    maj = np.asarray(file_table_output["pass_majority_config"], dtype=int) == 1
    if not np.any(maj):
        return False
    sy = np.asarray(file_table_output["recenter_shift_y"], dtype=float)[maj]
    sx = np.asarray(file_table_output["recenter_shift_x"], dtype=float)[maj]
    return bool(np.all(np.isfinite(sy) & np.isfinite(sx)))


def data_cube_fom_idxs(file_table_static, idxs, params):
    obs_path = params["obs_path"]
    data_dir = params["data_dir"]
    camera = params["camera"]
    used = file_table_static[idxs]
    prefix = f"{obs_path}{data_dir}/{camera}/"
    unsat_files = [prefix + str(fn) for fn in used["filename"]]
    # 1.1 pull dark data
    dark_path = used["masterdark_path"][0]
    dark_data = fr._load_fits_primary_float32(dark_path)
    # 1.2 pulling all files and making cube
    unsats_data_cube = fr.make_data_cube(unsat_files, dark_data)
    return unsats_data_cube


################### STEP 3 ###################

def shift_and_save_frame(src_path, dst_path, shift_y, shift_x, dark_path=None):
    """
    Load a science frame, optionally dark-subtract, shift by ``(shift_y, shift_x)`` pixels,
    and write a FITS with the original primary header.

    Returns the output path.
    """
    with fits.open(src_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    if dark_path:
        dark_data = fr._load_fits_primary_float32(dark_path)
        data = data - dark_data
    shifted = ndimage.shift(
        data, shift=(float(shift_y), float(shift_x)), mode="constant", cval=0.0
    )
    parent = os.path.dirname(os.path.abspath(dst_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fits.PrimaryHDU(
        data=np.asarray(shifted, dtype=np.float32), header=header
    ).writeto(dst_path, overwrite=True)
    return dst_path


def _shift_and_save_frame_worker(job):
    """Unpack a job tuple for :class:`ProcessPoolExecutor` workers."""
    src_path, dst_path, shift_y, shift_x, dark_path = job
    return shift_and_save_frame(src_path, dst_path, shift_y, shift_x, dark_path)


def save_centered_images_parallel(jobs, n_workers=None):
    """
    Shift and save many frames. Each job is
    ``(src_path, dst_path, shift_y, shift_x, dark_path)``.
    """
    if not jobs:
        return []
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), len(jobs)))

    if n_workers == 1:
        return [_shift_and_save_frame_worker(job) for job in jobs]

    saved = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_shift_and_save_frame_worker, job) for job in jobs]
        for fut in as_completed(futures):
            saved.append(fut.result())
    return saved


def _centered_frame_jobs(file_table_static, file_table_output, run_params):
    """Build per-frame shift/save jobs for rows with finite registration shifts."""
    obs_path = run_params["obs_path"]
    data_dir = run_params["data_dir"]
    camera = run_params["camera"]
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    prefix = f"{obs_path}{data_dir}/{camera}/"

    jobs = []
    n = len(file_table_output)
    for i in range(n):
        if int(file_table_output["pass_majority_config"][i]) != 1:
            continue
        sy = float(file_table_output["shift_y"][i])
        sx = float(file_table_output["shift_x"][i])
        if not np.isfinite(sy) or not np.isfinite(sx):
            continue
        filename = str(file_table_static["filename"][i])
        src_path = prefix + filename
        dst_path = os.path.join(out_dir, filename)
        dark_path = str(file_table_static["masterdark_path"][i])
        jobs.append((src_path, dst_path, sy, sx, dark_path))
    return jobs


def _recenter_apply_jobs(file_table_static, file_table_output, run_params):
    """Build in-place shift jobs for second-pass correction on centered FITS files."""
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    jobs = []
    n = len(file_table_output)
    for i in range(n):
        if int(file_table_output["pass_majority_config"][i]) != 1:
            continue
        sy = float(file_table_output["recenter_shift_y"][i])
        sx = float(file_table_output["recenter_shift_x"][i])
        if not np.isfinite(sy) or not np.isfinite(sx):
            continue
        path = os.path.join(out_dir, str(file_table_static["filename"][i]))
        jobs.append((path, path, sy, sx, None))
    return jobs


def _centered_frames_complete(jobs):
    """True when every centered output file in ``jobs`` already exists."""
    return bool(jobs) and all(os.path.isfile(dst) for _, dst, _, _, _ in jobs)


################### STEP 4 ###################

def _mask_as_2d(mask_image):
    if hasattr(mask_image, "shaped"):
        return np.asarray(mask_image.shaped, dtype=float)
    arr = np.asarray(mask_image, dtype=float)
    if arr.ndim == 1:
        n = arr.size
        side = int(np.sqrt(n))
        if side * side == n:
            return arr.reshape((side, side))
    return arr


def _chunk_slices(n, chunk_size):
    """Yield ``(start, end)`` slices covering ``range(n)``."""
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def _shifts_for_row_idxs(file_table_output, row_idxs):
    sy = np.asarray(file_table_output["shift_y"][row_idxs], dtype=float)
    sx = np.asarray(file_table_output["shift_x"][row_idxs], dtype=float)
    if "recenter_shift_y" in file_table_output.colnames:
        ry = np.asarray(file_table_output["recenter_shift_y"][row_idxs], dtype=float)
        rx = np.asarray(file_table_output["recenter_shift_x"][row_idxs], dtype=float)
        sy = sy + np.where(np.isfinite(ry), ry, 0.0)
        sx = sx + np.where(np.isfinite(rx), rx, 0.0)
    return np.column_stack([sy, sx])


def _collect_peak_idxs_chunked(file_table_static, row_idxs, run_params, chunk_size):
    """Load raw frames in chunks and return brightest-pixel indices for each row."""
    from numpy import unravel_index

    n = len(row_idxs)
    peak_idxs = np.zeros((n, 2), dtype=float)
    for start, end in _chunk_slices(n, chunk_size):
        chunk_rows = row_idxs[start:end]
        raw_cube = data_cube_fom_idxs(file_table_static, chunk_rows, run_params)
        for k, frame in enumerate(raw_cube):
            peak_idxs[start + k] = unravel_index(frame.argmax(), frame.shape)
    return peak_idxs


def _load_centered_chunk(file_table_static, chunk_row_idxs, run_params):
    """Load a batch of centered FITS frames as a cube."""
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    centered = [
        fr._load_fits_primary_float32(
            os.path.join(out_dir, str(file_table_static["filename"][i]))
        )
        for i in chunk_row_idxs
    ]
    return np.stack(centered, axis=0)


def _collect_speckle_intensities_chunked(
    file_table_static, row_idxs, run_params, speckle_mask, chunk_size
):
    """Load centered frames in chunks and return masked speckle sums."""
    n = len(row_idxs)
    intensities = np.full(n, np.nan, dtype=float)
    for start, end in _chunk_slices(n, chunk_size):
        chunk_rows = row_idxs[start:end]
        centered_cube = _load_centered_chunk(file_table_static, chunk_rows, run_params)
        intensities[start:end] = np.sum(centered_cube * speckle_mask, axis=(1, 2))
    return intensities


def _filter_rms_chunked(
    file_table_static,
    row_idxs,
    run_params,
    *,
    sigma_clip,
    n_iter,
    chunk_size,
):
    """
    Iterative RMS filter loading centered frames in chunks (same logic as
    :func:`filtering.filter_rms`).
    """
    n = len(row_idxs)
    keep = np.ones(n, dtype=bool)
    rms_dev = np.full(n, np.nan, dtype=float)
    for _ in range(n_iter):
        if not np.any(keep):
            break
        good_frame = None
        count = 0
        for start, end in _chunk_slices(n, chunk_size):
            chunk_rows = row_idxs[start:end]
            local_keep = keep[start:end]
            if not np.any(local_keep):
                continue
            centered_cube = _load_centered_chunk(
                file_table_static, chunk_rows, run_params
            )
            chunk_sum = np.sum(centered_cube[local_keep], axis=0, dtype=np.float64)
            good_frame = chunk_sum if good_frame is None else good_frame + chunk_sum
            count += int(np.sum(local_keep))
        good_frame = good_frame / count

        for start, end in _chunk_slices(n, chunk_size):
            chunk_rows = row_idxs[start:end]
            centered_cube = _load_centered_chunk(
                file_table_static, chunk_rows, run_params
            )
            for k, frame in enumerate(centered_cube):
                rms_dev[start + k] = float(np.std(frame - good_frame))

        ref = np.median(rms_dev)
        spread = np.std(rms_dev)
        keep = rms_dev <= ref + sigma_clip * spread

    log.info(
        "filter_rms: sigma_clip=%s n_iter=%s kept %s/%s frames",
        sigma_clip,
        n_iter,
        int(np.sum(keep)),
        n,
    )
    return keep, rms_dev


def _process_filter_row_idxs(file_table_static, file_table_output, run_params):
    """Table row indices with finite shifts and existing centered FITS files."""
    redu_dir = run_params["redu_dir"]
    out_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    row_idxs = []
    n = len(file_table_output)
    for i in range(n):
        if int(file_table_output["pass_majority_config"][i]) != 1:
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


def _process_filters_complete(file_table_static, file_table_output, run_params):
    """True when every filterable row has a computed max-point radius."""
    row_idxs = _process_filter_row_idxs(file_table_static, file_table_output, run_params)
    if len(row_idxs) == 0:
        return False
    radii = np.asarray(file_table_output["max_point_radius"][row_idxs], dtype=float)
    return bool(np.all(np.isfinite(radii)))


def _process_plot_enabled(run_params, plot_key):
    """
    Return whether to save plots for ``plot_key``.

    Per-filter keys (e.g. ``plot_max_point``) override the master ``plot`` flag
    when set explicitly in the config; otherwise ``plot`` is used as the default.
    """
    if plot_key in run_params:
        return bool(run_params[plot_key])
    return bool(run_params.get("plot", False))


#############################################################
################### Pipeline functionality ###################
##############################################################


########### Read from configuration file ############

def read_process_config(config_path):
    """
    Read a process config file (one ``name = value`` per line; ``#`` starts a comment).

    Values are parsed with ``ast.literal_eval`` (strings, lists, numbers, booleans).
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


def check_process_config(params):
    """
    Verify required process parameters: ``obs_path``, ``redu_path``, ``data_dir``,
    ``unsats_dir``, and ``cameras``.
    """
    bad = []
    if not _nonempty_str(params.get("obs_path")):
        bad.append("obs_path")
    if not _nonempty_str(params.get("redu_path")):
        bad.append("redu_path")
    if not _nonempty_str(params.get("data_dir")):
        bad.append("data_dir")
    if not _nonempty_str(params.get("unsats_dir")):
        bad.append("unsats_dir")
    if not _nonempty_camera_list(params.get("cameras")):
        bad.append("cameras")
    return bad


def _nonempty_str(x):
    return isinstance(x, str) and bool(x.strip())


def _nonempty_camera_list(x):
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    return all(_nonempty_str(c) for c in x)


def run_process_from_config(params, config_source_path=None):
    """
    Validate ``params`` and run :func:`process_main` for each camera listed in config.
    """
    missing = check_process_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")
    data_dir = params["data_dir"].strip()
    cameras = list(params["cameras"])

    for camera in cameras:
        try:
            run = build_process_run_params(
                params,
                camera,
                config_source_path=config_source_path,
            )
            process_main(run)
        except Exception:
            log.exception("Error processing %s %s", data_dir, camera)


def _resolve_wavelength_for_camera(params, camera):
    """
    Return wavelength in metres for ``camera``.

    ``wavelength`` may be a scalar (all cameras) or a list/tuple in the same
    order as ``cameras`` (e.g. ``[camsci1_wl, camsci2_wl]``).
    """
    wl = params.get("wavelength", 908e-9)
    if isinstance(wl, (list, tuple)):
        cameras = list(params.get("cameras", []))
        if camera not in cameras:
            raise ValueError(
                f"wavelength is a list but {camera!r} is not listed in cameras={cameras}"
            )
        idx = cameras.index(camera)
        if idx >= len(wl):
            raise ValueError(
                f"wavelength list length {len(wl)} is too short for {camera!r} "
                f"(index {idx} in cameras={cameras})"
            )
        return float(wl[idx])
    return float(wl)


def build_process_run_params(
    params,
    camera,
    *,
    config_source_path=None,
):
    """
    Shallow copy of config ``params`` plus per-run keys for one camera.

    Adds ``camera`` and ``redu_dir``, and sets defaults for ``max_files``,
    ``plot``, ``force_rerun``, and ``masterdark_dir``.
    """
    p = dict(params)
    p["camera"] = camera
    p["data_dir"] = p["data_dir"].strip()
    if _nonempty_str(p.get("unsats_dir")):
        p["unsats_dir"] = p["unsats_dir"].strip()
    if _nonempty_str(p.get("unsats_nospark_dir")):
        p["unsats_nospark_dir"] = p["unsats_nospark_dir"].strip()
    p["redu_dir"] = f"{p['redu_path']}{p['data_dir']}/{p['camera']}/"
    p.setdefault("max_files", -1)
    p.setdefault("plot", False)
    p.setdefault("plot_reference", p["plot"])
    p.setdefault("plot_max_point", p["plot"])
    p.setdefault("plot_center_shift", p["plot"])
    p.setdefault("plot_speckle_intensity", p["plot"])
    p.setdefault("plot_rms", p["plot"])
    p.setdefault("force_rerun", False)
    p.setdefault("rerun_filtering", False)
    p.setdefault("n_workers", os.cpu_count() or 1)
    p.setdefault("chunk_size", 100)
    p.setdefault("center_coadd_n", 1)
    p.setdefault("recenter", False)
    # Optional quick-look product: save first N centered frames as a cube in step 3b.
    # Set to an int > 0 to enable.
    p.setdefault("save_centered_cube_first_n", None)
    # Optional quick-look product: save first N *kept* centered frames (after step 4 filtering).
    # Set to an int > 0 to enable.
    p.setdefault("save_filtered_centered_cube_first_n", None)
    p.setdefault("masterdark_dir", p["redu_path"])
    p.setdefault("max_point_sigma_clip", 2.0)
    p.setdefault("shift_sigma_clip", 2.0)
    p.setdefault("speckle_intensity_sigma_clip", 2.0)
    p.setdefault("rms_sigma_clip", 2.0)
    p.setdefault("rms_iterations", 3)
    p["wavelength"] = _resolve_wavelength_for_camera(p, camera)
    if config_source_path is not None:
        p["config_source_path"] = config_source_path
    return p

def cli_process(argv=None):
    """CLI entry: one or more process config paths."""
    parser = argparse.ArgumentParser(
        description="Run the process pipeline from config file(s)."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        metavar="CONF",
        help="Process config file(s), e.g. conf_ex/conf_process_ex.txt",
    )
    args = parser.parse_args(argv)
    _ensure_stderr_logging()
    for cfg_path in args.configs:
        params = read_process_config(cfg_path)
        log.info("=> Config: %s", cfg_path)
        try:
            run_process_from_config(params, config_source_path=cfg_path)
        except ValueError as e:
            log.exception("%s: %s", cfg_path, e)
            raise SystemExit(1) from e

########### Logger Setup ##############

def _redu_pkg_logger():
    """Parent of ``preprocess`` / ``filtering`` so both share one set of handlers."""
    p = log.parent
    return p if p.name else log


def _ensure_stderr_logging():
    """Attach a stderr handler on the redu package logger."""
    pkg = _redu_pkg_logger()
    if pkg.handlers:
        return
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    pkg.addHandler(sh)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    log.propagate = True


def _configure_process_logging(redu_dir):
    """
    Log to ``{redu_dir}/process.log`` (overwrite each run) and stderr.
    Same directory as ``PROCESS_CONFIG_SNAPSHOT_NAME`` (``process_config.txt``).

    Handlers are attached to the ``tools4magaox.redu`` package logger so sibling modules
    can also write to the same file.
    """
    os.makedirs(redu_dir, exist_ok=True)
    log_path = os.path.join(redu_dir, PROCESS_LOG_NAME)
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
    log.info("Process log file: %s", log_path)


def _copy_process_config_to_redu(config_source_path, redu_dir):
    """Copy the process config file into ``redu_dir`` for provenance."""
    if not config_source_path:
        return
    src = os.path.abspath(os.fspath(config_source_path))
    if not os.path.isfile(src):
        return
    os.makedirs(redu_dir, exist_ok=True)
    dst = os.path.join(redu_dir, PROCESS_CONFIG_SNAPSHOT_NAME)
    shutil.copy2(src, dst)
    log.info("Saved config snapshot: %s", dst)

####################################################

if __name__ == "__main__":
    cli_process()

