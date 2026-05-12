import argparse
import ast
import logging
import os
import shutil
import sys

import numpy as np
import scipy
import scipy.signal
from hcipy import *

from tools4magaox.constants import *
import filereads as fr

log = logging.getLogger(__name__)

PROCESS_CONFIG_SNAPSHOT_NAME = "process_config.txt"
PROCESS_LOG_NAME = "process.log"
FILE_TABLE_NAME = "file_table.txt"
FILE_TABLE_OUTPUT_NAME = "file_table_output.txt"
REFERENCE_IMAGE_NAME = "reference_sparkles.fits"
MASKED_REFERENCE_IMAGE_NAME = "reference_sparkles_masked.fits"
MASK_IMAGE_NAME = "mask.fits"

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
    # STEP 3 - main function, will take the most time
    s2_image_center(run_params, file_table_static, file_table_output, masked_reference_image, mask_image)
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
        starts empty and is populated with rows as later pipeline steps generate outputs.

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
    else:
        log.info("=> CREATING FILE TABLES")
        # find all files, error if none found
        data_files = fr.find_camera_files(obs_path, data_dir, camera, max_files)
        if len(data_files) == 0:
            raise ValueError(f"No files found for {data_dir} {camera}")
        # collect telemetry from all files
        file_table_static = fr.fits_telemetry_table(data_files, camera)
        # find a dark for each camera config
        dark_search_dir = fr.resolve_masterdark_search_dir(redu_path, masterdark_dir)
        # build the param table
        file_table_static = fr.attach_masterdarks(file_table_static, dark_search_dir,camera,)
        # initialize the utput table
        file_table_output = fr.init_process_output_table()
        # save noth tables
        fr.write_redu_table(file_table_static, static_path)
        fr.write_redu_table(file_table_output, output_path)
    return file_table_static, file_table_output

# STEP 1 - Reference Image
def s1_create_reference(run_params):
    '''
    Take the unsat, and subtract the unsat nospark if available, then mask the sparkles.
    Save full reference image, masked reference image, and mask image.
    '''
    redu_dir = run_params["redu_dir"]
    force_rerun = run_params["force_rerun"]
    unsats_dir = run_params["unsats_dir"]
    unsats_nospark_dir = run_params["unsats_nospark_dir"]
    reference_image_path = os.path.join(redu_dir, REFERENCE_IMAGE_NAME)
    masked_reference_image_path = os.path.join(redu_dir, MASKED_REFERENCE_IMAGE_NAME)
    mask_image_path = os.path.join(redu_dir, MASK_IMAGE_NAME)

    if (not force_rerun and os.path.isfile(masked_reference_image_path)):
        log.info("=> LOADING REF IMAGE")
        masked_reference_image = fr._load_fits_primary_float32(masked_reference_image_path)
        mask_image = fr._load_fits_primary_float32(mask_image)
    else:
        log.info("=> CREATING REF IMAGE")
        reference_image = create_reference_image(unsats_dir, unsats_nospark_dir)
        mask_image = make_sparkle_mask(reference_image)
        masked_reference_image = reference_image * mask_image
        # save the reduced images
        fr._save_fits_primary_float32(reference_image, reference_image_path)
        fr._save_fits_primary_float32(mask_image, mask_image_path)
        fr._save_fits_primary_float32(masked_reference_image, masked_reference_image_path)
    return masked_reference_image, mask_image


# STEP 2 - Dark subtract and center
def s2_image_center(run_params, file_table_static, file_table_output):
    '''
    Dark subtract the cube, and center images
    Save each individual image back to it's fits file
    this should be done intelligently to keep memory happy
    '''
    # TODO: all of this yeah 
    return True


########################################################
################### Helper functions ###################
########################################################


################### STEP 1 ###################

def create_reference_image(unsats_dir, unsats_nospark_dir):
    '''
    Create a reference image from the unsats and unsats_nospark directories
    '''
    # if we have both unsats and unsats_nospark, subtract the nospark from the unsats
    if os.path.isfile(unsats_dir) and os.path.isfile(unsats_nospark_dir):
        unsats = fr._load_fits_primary_float32(unsats_dir)
        unsats_nospark = fr._load_fits_primary_float32(unsats_nospark_dir)
        # TODO: do I need to normalize before subtracting
        reference_image = unsats - unsats_nospark
    # if we only have unsats, use that
    elif os.path.isfile(unsats_dir):
        reference_image = fr._load_fits_primary_float32(unsats_dir)
        #TODO: I think I need to highpass filter if I'm only using the spark unsat
    # if we don't have any files, error
    else:
        raise ValueError(f"No unsats or unsats_nospark files found in {unsats_dir} and {unsats_nospark_dir}")
    return reference_image

def make_sparkle_mask(spark_ang, spark_sep, ex_data, camsci_grid, width_r_ld=8, width_phi_rld=3):
    '''
    Making a mask based on expected location of the sparkles
    '''
    # building up the elipises 
    spark_sep_as = 0.92 * rc.spark_to_dist(spark_sep, wavelength=908e-9) #as
    angs = np.deg2rad(np.array([spark_ang-90*i for i in range(4)]))
    centers_x = np.array([spark_sep_as * np.cos(ang) for ang in angs])
    centers_y = np.array([spark_sep_as * np.sin(ang) for ang in angs])
    centers = list(zip(centers_y, centers_x))
    width_r = LAM_D_AS * width_r_ld
    width_phi = LAM_D_AS * width_phi_rld
    diameters = [(width_r, width_phi) for i in range(4)]
    # iteratively build up the mask
    ap_total = np.zeros_like(ex_data.ravel())
    for i in range(4):
        ap_temp = evaluate_supersampled(make_elliptical_aperture(diameters[i], centers[i], angs[i]+np.pi/2), camsci_grid, 4)
        ap_total += ap_temp
    # soften these with the tukey window
    taper = TukeyWindow(alpha=0.4)
    window_data = taper((27, 27))
    sof_mask = scipy.signal.convolve2d(ap_total.reshape(camsci_grid.dims), window_data, mode = 'same')
    sof_mask/= np.max(sof_mask)
    soft_mask_field = Field(sof_mask.ravel(), camsci_grid)
    return soft_mask_field

################### STEP 2 ###################


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
    p.setdefault("force_rerun", False)
    p.setdefault("masterdark_dir", p["redu_path"])
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

