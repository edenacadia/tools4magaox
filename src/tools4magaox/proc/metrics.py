# metrics.py — VIP throughput, contrast curves, and SNR source peak finding
import argparse
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from vip_hci.fm import normalize_psf
from vip_hci.metrics.contrcurve import contrast_curve, throughput
from vip_hci.metrics.detection import peak_coordinates
from vip_hci.psfsub import pca

from tools4magaox.proc import ADI
from tools4magaox.proc import utils as pu
from tools4magaox.redu import filereads as fr

# Allow ``python path/to/metrics.py conf.txt`` without installing the package.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

log = logging.getLogger(__name__)

METRICS_OUTPUT_DIR = "metrics"
METRICS_CONFIG_SNAPSHOT_NAME = "metrics_config.txt"
METRICS_LOG_NAME = "metrics.log"

THROUGHPUT_CSV_NAME = "metrics_throughput.csv"
THROUGHPUT_PLOT_NAME = "metrics_throughput.png"
CONTRAST_CSV_NAME = "metrics_contrast.csv"
CONTRAST_PLOT_NAME = "metrics_contrast.png"
SOURCE_PEAK_TXT_NAME = "metrics_source_peak.txt"
SOURCE_PEAK_PLOT_NAME = "metrics_source_peak.png"


########################################################
######################### Main #########################
########################################################


def metrics_main(
    run_params,
    *,
    run_throughput=None,
    run_contrast=None,
    run_source_peak=None,
):
    """
    Run selected metrics for one ``data_dir`` / ``camera`` pair.

    Parameters
    ----------
    run_throughput, run_contrast, run_source_peak : bool or None
        When set, override config ``run_*`` flags. When all are ``None``, use
        config defaults.
    """
    redu_dir = run_params["redu_dir"]
    metrics_dir = run_params["metrics_dir"]
    data_dir = run_params["data_dir"]
    camera = run_params["camera"]

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(run_params["adi_dir"], exist_ok=True)
    _configure_metrics_logging(metrics_dir)
    _copy_metrics_config_to_redu(run_params.get("config_source_path"), metrics_dir)
    ADI._validate_adi_inputs(run_params)

    do_throughput = _metric_enabled(run_params, "run_throughput", run_throughput)
    do_contrast = _metric_enabled(run_params, "run_contrast", run_contrast)
    do_source_peak = _metric_enabled(run_params, "run_source_peak", run_source_peak)

    log.info(
        "=> Metrics %s %s (redu_dir=%s, metrics_dir=%s) "
        "throughput=%s contrast=%s source_peak=%s",
        data_dir,
        camera,
        redu_dir,
        metrics_dir,
        do_throughput,
        do_contrast,
        do_source_peak,
    )

    file_table_static = fr.read_redu_table(
        os.path.join(redu_dir, ADI.FILE_TABLE_NAME)
    )
    file_table_output = fr.read_redu_table(
        os.path.join(redu_dir, ADI.FILE_TABLE_OUTPUT_NAME)
    )
    file_table_output = fr.prune_process_output_table(file_table_output)

    need_pca_snr = do_source_peak
    cube, parang, psfn, fwhm, fr_pca, snr_map = load_metrics_inputs(
        run_params,
        file_table_static,
        file_table_output,
        need_pca=need_pca_snr,
        need_snr=need_pca_snr,
    )
    run_params["fwhm"] = fwhm

    if do_throughput:
        out_path = os.path.join(metrics_dir, THROUGHPUT_CSV_NAME)
        if run_params["force_rerun"] or not os.path.isfile(out_path):
            measure_throughput(cube, parang, psfn, fwhm, run_params)
        else:
            log.info("=> THROUGHPUT EXISTS, SKIPPING")

    if do_contrast:
        out_path = os.path.join(metrics_dir, CONTRAST_CSV_NAME)
        if run_params["force_rerun"] or not os.path.isfile(out_path):
            measure_contrast_curve(
                cube,
                parang,
                psfn,
                fwhm,
                run_params,
                file_table_static,
                file_table_output,
            )
        else:
            log.info("=> CONTRAST EXISTS, SKIPPING")

    if do_source_peak:
        out_path = os.path.join(metrics_dir, SOURCE_PEAK_TXT_NAME)
        if run_params["force_rerun"] or not os.path.isfile(out_path):
            find_significant_source(snr_map, fwhm, run_params)
        else:
            log.info("=> SOURCE PEAK EXISTS, SKIPPING")

    return True


def load_metrics_inputs(
    run_params,
    file_table_static,
    file_table_output,
    *,
    need_pca=False,
    need_snr=False,
):
    """
    Load ADI products when present; otherwise build them via ADI step functions.

    Returns
    -------
    cube, parang, psfn, fwhm, fr_pca, snr_map
        ``fr_pca`` and ``snr_map`` are ``None`` when not requested.
    """
    adi_dir = run_params["adi_dir"]
    force_rerun = run_params["force_rerun"]

    psf_norm_path = os.path.join(adi_dir, ADI.PSF_NORMALIZED_NAME)
    fwhm_path = os.path.join(adi_dir, ADI.PSF_FWHM_NAME)
    cube_path = os.path.join(adi_dir, ADI.ADI_CUBE_NAME)
    parang_path = os.path.join(adi_dir, ADI.ADI_PARANG_NAME)
    pca_path = os.path.join(adi_dir, ADI.ADI_PCA_NAME)
    snr_path = os.path.join(adi_dir, ADI.ADI_SNRMAP_NAME)

    if (
        not force_rerun
        and os.path.isfile(psf_norm_path)
        and os.path.isfile(fwhm_path)
    ):
        log.info("=> LOADING PSF from ADI")
        psfn = fr._load_fits_primary_float32(psf_norm_path)
        if run_params.get("fwhm_override") is not None:
            fwhm = float(run_params["fwhm_override"])
        else:
            with open(fwhm_path, encoding="utf-8") as fh:
                fwhm = float(fh.read().strip())
    else:
        log.info("=>  => BUILDING PSF via ADI step 1")
        psfn, fwhm = ADI.s1_build_psf(run_params)

    if (
        not force_rerun
        and os.path.isfile(cube_path)
        and os.path.isfile(parang_path)
    ):
        log.info("=> LOADING ADI CUBE")
        cube = fr._load_fits_primary_float32(cube_path)
        parang = fr._load_fits_primary_float32(parang_path)
    else:
        log.info("=> BUILDING ADI CUBE via ADI step 2")
        cube, parang, _times = ADI.s2_build_adi_cube(
            run_params, file_table_static, file_table_output
        )

    fr_pca = None
    snr_map = None

    if need_pca or need_snr:
        if not force_rerun and os.path.isfile(pca_path):
            log.info("=> LOADING PCA FRAME from ADI")
            fr_pca = fr._load_fits_primary_float32(pca_path)
        else:
            log.info("=> BUILDING PCA FRAME via ADI step 3")
            fr_pca = ADI.s3_run_pca(run_params, cube, parang)

    if need_snr:
        if not force_rerun and os.path.isfile(snr_path):
            log.info("=> LOADING SNR MAP from ADI")
            snr_map = fr._load_fits_primary_float32(snr_path)
        else:
            log.info("=> BUILDING SNR MAP via ADI step 4")
            snr_map = ADI.s4_snr_map(run_params, fr_pca, fwhm)

    return cube, parang, psfn, float(fwhm), fr_pca, snr_map


def measure_throughput(cube, parang, psfn, fwhm, run_params):
    """
    Measure VIP algorithmic throughput vs separation.

    Returns
    -------
    dict
        ``{"table": DataFrame or dict, "csv_path": str}``
    """
    metrics_dir = run_params["metrics_dir"]
    log.info("=> RUNNING VIP throughput()")
    algo_dict = _pca_algo_kwargs(run_params)
    inner_rad = _metrics_inner_rad(run_params)
    nbranch = int(run_params.get("throughput_nbranch", 1))

    res = throughput(
        cube,
        parang,
        psfn,
        fwhm,
        algo=pca,
        nbranch=nbranch,
        inner_rad=inner_rad,
        fc_rad_sep=int(run_params.get("fc_rad_sep", 3)),
        noise_sep=int(run_params.get("noise_sep", 1)),
        wedge=tuple(run_params.get("wedge", (0, 360))),
        full_output=True,
        verbose=True,
        **algo_dict,
    )

    table = _vip_result_table(res)
    csv_path = os.path.join(metrics_dir, THROUGHPUT_CSV_NAME)
    pu.save_dataframe_csv(table, csv_path)
    log.info("Saved throughput: %s", csv_path)

    if _metrics_plot_enabled(run_params, "plot_throughput"):
        _plot_throughput(table, run_params, os.path.join(metrics_dir, THROUGHPUT_PLOT_NAME))

    return {"table": table, "csv_path": csv_path}


def measure_contrast_curve(
    cube,
    parang,
    psfn,
    fwhm,
    run_params,
    file_table_static,
    file_table_output,
):
    """
    Measure VIP contrast curve vs separation.

    Returns
    -------
    dict
        ``{"table": DataFrame or dict, "csv_path": str, "starphot": float}``
    """
    metrics_dir = run_params["metrics_dir"]
    pxscale = run_params.get("pxscale")
    if pxscale is None:
        raise ValueError(
            "contrast curve requires pxscale (arcsec/px) in metrics config"
        )
    pxscale = float(pxscale)

    starphot = run_params.get("starphot")
    if starphot is None:
        starphot = _derive_starphot(
            run_params, file_table_static, file_table_output
        )
    else:
        starphot = float(starphot)
    log.info("Using starphot=%.3g for contrast curve", starphot)

    log.info("=> RUNNING VIP contrast_curve()")
    algo_dict = _pca_algo_kwargs(run_params)
    inner_rad = _metrics_inner_rad(run_params)
    nbranch = int(run_params.get("contrast_nbranch", 1))
    sigma = float(run_params.get("contrast_sigma", 5))

    res = contrast_curve(
        cube,
        parang,
        psfn,
        fwhm=fwhm,
        pxscale=pxscale,
        starphot=starphot,
        algo=pca,
        sigma=sigma,
        nbranch=nbranch,
        inner_rad=inner_rad,
        fc_rad_sep=int(run_params.get("fc_rad_sep", 3)),
        noise_sep=int(run_params.get("noise_sep", 1)),
        wedge=tuple(run_params.get("wedge", (0, 360))),
        full_output=True,
        plot=False,
        verbose=True,
        **algo_dict,
    )

    table = _vip_result_table(res)
    csv_path = os.path.join(metrics_dir, CONTRAST_CSV_NAME)
    pu.save_dataframe_csv(table, csv_path)
    log.info("Saved contrast curve: %s", csv_path)

    if _metrics_plot_enabled(run_params, "plot_contrast"):
        _plot_contrast(table, run_params, os.path.join(metrics_dir, CONTRAST_PLOT_NAME))

    return {"table": table, "csv_path": csv_path, "starphot": starphot}


def find_significant_source(snr_map, fwhm, run_params):
    """
    Locate the most significant source in an SNR map.

    Returns
    -------
    dict
        Keys: ``peak_y``, ``peak_x``, ``peak_snr``, ``peak_r_px``,
        ``frame_center_y``, ``frame_center_x``.
    """
    metrics_dir = run_params["metrics_dir"]
    snr_map = np.asarray(snr_map, dtype=float)
    fwhm = float(fwhm)

    masked = _mask_inner_for_peak(snr_map, run_params)
    expected_r = run_params.get("expected_source_r_px")
    half_w = float(run_params.get("source_peak_annulus_half_width", 3.0))

    if expected_r is not None:
        peak_y, peak_x, peak_r_px, peak_snr = pu.brightest_in_radius_annulus(
            masked,
            float(expected_r),
            annulus_half_width=half_w,
        )
    else:
        peak_y, peak_x = peak_coordinates(masked, fwhm)
        peak_snr = float(masked[peak_y, peak_x])
        cy, cx = pu.vip_frame_center_xy(snr_map.shape)
        peak_r_px = float(np.sqrt((peak_y - cy) ** 2 + (peak_x - cx) ** 2))

    cy, cx = pu.vip_frame_center_xy(snr_map.shape)
    result = {
        "peak_y": int(peak_y),
        "peak_x": int(peak_x),
        "peak_snr": float(peak_snr),
        "peak_r_px": float(peak_r_px),
        "frame_center_y": cy,
        "frame_center_x": cx,
    }

    txt_path = os.path.join(metrics_dir, SOURCE_PEAK_TXT_NAME)
    lines = [f"{key}={value}" for key, value in result.items()]
    pu.save_adi_diagnostics(txt_path, lines)
    log.info(
        "Source peak at (y,x)=(%s,%s) SNR=%.2f r=%.1f px",
        result["peak_y"],
        result["peak_x"],
        result["peak_snr"],
        result["peak_r_px"],
    )

    if _metrics_plot_enabled(run_params, "plot_source_peak"):
        plot_path = os.path.join(metrics_dir, SOURCE_PEAK_PLOT_NAME)
        _plot_source_peak(snr_map, result, plot_path)

    return result


#############################################################
################### Pipeline functionality ###################
#############################################################


def read_metrics_config(config_path):
    """Read a metrics config file (``name = value`` per line; ``#`` comments)."""
    return ADI.read_adi_config(config_path)


def check_metrics_config(params):
    """Verify required metrics parameters."""
    return ADI.check_adi_config(params)


def build_metrics_run_params(params, camera, *, config_source_path=None):
    """Shallow copy of config plus per-camera paths and metrics defaults."""
    p = ADI.build_adi_run_params(params, camera, config_source_path=config_source_path)
    p.setdefault("metrics_output_dir", METRICS_OUTPUT_DIR)
    p["metrics_dir"] = os.path.join(p["redu_dir"], p["metrics_output_dir"])
    p.setdefault("run_throughput", True)
    p.setdefault("run_contrast", True)
    p.setdefault("run_source_peak", True)
    p.setdefault("pxscale", None)
    p.setdefault("starphot", None)
    p.setdefault("starphot_psf_exptime", None)
    p.setdefault("contrast_sigma", 5)
    p.setdefault("throughput_nbranch", 1)
    p.setdefault("contrast_nbranch", 1)
    p.setdefault("metrics_inner_rad", None)
    p.setdefault("fc_rad_sep", 3)
    p.setdefault("noise_sep", 1)
    p.setdefault("wedge", (0, 360))
    p.setdefault("source_peak_inner_exclude_px", None)
    p.setdefault("source_peak_annulus_half_width", 3.0)
    p.setdefault("plot_throughput", p.get("plot", False))
    p.setdefault("plot_contrast", p.get("plot", False))
    p.setdefault("plot_source_peak", p.get("plot", False))
    return p


def run_metrics_from_config(
    params,
    config_source_path=None,
    *,
    run_throughput=None,
    run_contrast=None,
    run_source_peak=None,
):
    """Validate config and run :func:`metrics_main` for each camera."""
    missing = check_metrics_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")
    data_dir = params["data_dir"].strip()
    cameras = list(params["cameras"])

    for camera in cameras:
        try:
            run = build_metrics_run_params(
                params,
                camera,
                config_source_path=config_source_path,
            )
            metrics_main(
                run,
                run_throughput=run_throughput,
                run_contrast=run_contrast,
                run_source_peak=run_source_peak,
            )
        except Exception:
            log.exception("Error in metrics for %s %s", data_dir, camera)


def cli_metrics(argv=None):
    """CLI entry: one or more metrics config paths with optional metric flags."""
    parser = argparse.ArgumentParser(
        description="Run VIP metrics (throughput, contrast, SNR source peak) from config."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        metavar="CONF",
        help="Metrics config file(s), e.g. proc/conf_ex/conf_metrics_ex.txt",
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Run throughput measurement only (overrides config run_* flags)",
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="Run contrast curve only (overrides config run_* flags)",
    )
    parser.add_argument(
        "--source-peak",
        action="store_true",
        help="Find SNR map source peak only (overrides config run_* flags)",
    )
    args = parser.parse_args(argv)
    _ensure_stderr_logging()

    flag_override = args.throughput or args.contrast or args.source_peak
    run_throughput = True if args.throughput else None
    run_contrast = True if args.contrast else None
    run_source_peak = True if args.source_peak else None

    if flag_override:
        run_throughput = True if args.throughput else False
        run_contrast = True if args.contrast else False
        run_source_peak = True if args.source_peak else False

    for cfg_path in args.configs:
        params = read_metrics_config(cfg_path)
        log.info("=> Config: %s", cfg_path)
        try:
            run_metrics_from_config(
                params,
                config_source_path=cfg_path,
                run_throughput=run_throughput,
                run_contrast=run_contrast,
                run_source_peak=run_source_peak,
            )
        except ValueError as e:
            log.exception("%s: %s", cfg_path, e)
            raise SystemExit(1) from e


def _pca_algo_kwargs(run_params):
    """PCA kwargs passed to VIP throughput / contrast_curve (mirrors ADI.s3_run_pca)."""
    kwargs = dict(
        ncomp=run_params["ncomp"],
        mask_center_px=run_params.get("mask_center_px"),
        imlib=run_params.get("imlib", "vip-fft"),
        interpolation=run_params.get("interpolation"),
        svd_mode=run_params.get("svd_mode", "arpack"),
    )
    batch = run_params.get("batch")
    if batch is not None:
        kwargs["batch"] = batch
    fwhm = run_params.get("fwhm")
    if run_params.get("fwhm_override") is not None:
        fwhm = run_params["fwhm_override"]
    if fwhm is not None:
        kwargs["fwhm"] = fwhm
    nproc = run_params.get("nproc")
    if nproc is not None:
        kwargs["nproc"] = nproc
    return kwargs


def _metrics_inner_rad(run_params):
    inner = run_params.get("metrics_inner_rad")
    if inner is not None:
        return float(inner)
    inner = run_params.get("crop_radius_inner")
    if inner is not None and float(inner) > 0:
        return float(inner)
    inner = run_params.get("mask_center_px")
    if inner is not None and float(inner) > 0:
        return float(inner)
    return 30.0


def _derive_starphot(run_params, file_table_static, file_table_output):
    """
    Estimate star flux in coronagraphic frames from reference PSF normalization.

    Scales the PSF integrated flux by the median science EXPTIME divided by the
    reference PSF EXPTIME.
    """
    redu_dir = run_params["redu_dir"]
    psf_crop_size = int(run_params.get("psf_crop_size", 30))
    psf_norm_size = int(run_params.get("psf_norm_size", 19))

    ref_path = os.path.join(redu_dir, ADI.REFERENCE_IMAGE_NAME)
    reference = fr._load_fits_primary_float32(ref_path)
    psf_crop = pu.center_crop_2d(reference, psf_crop_size)
    _psfn, flux, _fwhm = normalize_psf(
        psf_crop,
        size=psf_norm_size,
        debug=False,
        full_output=True,
    )
    flux = float(flux)

    row_idxs = pu.select_adi_frame_rows(
        file_table_static, file_table_output, run_params
    )
    if len(row_idxs) == 0:
        raise ValueError("No frames available for starphot EXPTIME median")

    exptimes = np.asarray(
        file_table_static["EXPTIME"][row_idxs], dtype=float
    )
    finite = np.isfinite(exptimes) & (exptimes > 0)
    if not np.any(finite):
        raise ValueError(
            "Cannot derive starphot: no finite positive EXPTIME in selected frames"
        )
    sci_exptime = float(np.median(exptimes[finite]))

    psf_exptime = run_params.get("starphot_psf_exptime")
    if psf_exptime is not None:
        psf_exptime = float(psf_exptime)
    else:
        psf_exptime = sci_exptime
        log.warning(
            "starphot_psf_exptime not set; assuming PSF and science share "
            "median EXPTIME=%.6g s",
            sci_exptime,
        )

    starphot = flux * (sci_exptime / psf_exptime)
    log.info(
        "Derived starphot=%.3g (PSF flux=%.3g, sci_exptime=%.6g, psf_exptime=%.6g)",
        starphot,
        flux,
        sci_exptime,
        psf_exptime,
    )
    return starphot


def _mask_inner_for_peak(snr_map, run_params):
    """Return SNR map with inner disk zeroed for peak search."""
    inner = run_params.get("source_peak_inner_exclude_px")
    if inner is None:
        inner = _metrics_inner_rad(run_params)
    inner = float(inner)
    if inner <= 0:
        return np.asarray(snr_map, dtype=float)
    masked = np.asarray(snr_map, dtype=float).copy()
    mask = pu.inner_disk_keep_mask(masked.shape, inner)
    masked = np.where(mask > 0, masked, -np.inf)
    return masked


def _vip_result_table(res):
    """Normalize VIP throughput/contrast return value to a pandas DataFrame."""
    import pandas as pd

    if isinstance(res, pd.DataFrame):
        return res
    if isinstance(res, dict):
        return pd.DataFrame(res)
    raise TypeError(f"Unexpected VIP metrics return type: {type(res)!r}")


def _metric_enabled(run_params, config_key, cli_override):
    if cli_override is not None:
        return bool(cli_override)
    return bool(run_params.get(config_key, False))


def _metrics_plot_enabled(run_params, plot_key):
    if plot_key in run_params:
        return bool(run_params[plot_key])
    return bool(run_params.get("plot", False))


def _plot_throughput(table, run_params, path):
    pxscale = run_params.get("pxscale")
    if "distance" in table.columns:
        x = table["distance"].to_numpy(dtype=float)
        xlabel = "Separation [px]"
        if pxscale is not None:
            x = x * float(pxscale)
            xlabel = "Separation [arcsec]"
    elif "separation" in table.columns:
        x = table["separation"].to_numpy(dtype=float)
        xlabel = "Separation"
    else:
        x = np.arange(len(table), dtype=float)
        xlabel = "Index"

    if "throughput" in table.columns:
        y = table["throughput"].to_numpy(dtype=float)
    else:
        raise ValueError(f"throughput table missing 'throughput' column: {list(table.columns)}")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, "o-", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Throughput")
    ax.set_title("Algorithm throughput")
    ax.set_ylim(0, 1.05)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_contrast(table, run_params, path):
    pxscale = float(run_params["pxscale"])
    if "distance" in table.columns:
        x = table["distance"].to_numpy(dtype=float) * pxscale
    elif "separation" in table.columns:
        x = table["separation"].to_numpy(dtype=float) * pxscale
    else:
        x = np.arange(len(table), dtype=float)

    if "contrast" in table.columns:
        y = table["contrast"].to_numpy(dtype=float)
    elif "contrast_curve" in table.columns:
        y = table["contrast_curve"].to_numpy(dtype=float)
    else:
        raise ValueError(f"contrast table missing contrast column: {list(table.columns)}")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(x, y, "o-", markersize=3)
    ax.set_xlabel("Separation [arcsec]")
    ax.set_ylabel("Contrast")
    ax.set_title(f"Contrast curve ({run_params.get('contrast_sigma', 5)} sigma)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_source_peak(snr_map, result, path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(snr_map, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(
        result["peak_x"],
        result["peak_y"],
        "r+",
        markersize=14,
        markeredgewidth=2,
        label=f"SNR={result['peak_snr']:.1f}",
    )
    ax.legend(loc="upper right")
    ax.set_title("SNR map — significant source peak")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


########### Logger Setup ##############


def _ensure_stderr_logging():
    pkg = _metrics_pkg_logger()
    if pkg.handlers:
        return
    import logging
    import sys

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    pkg.addHandler(sh)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    log.propagate = True


def _metrics_pkg_logger():
    p = log.parent
    return p if p.name else log


def _configure_metrics_logging(metrics_dir):
    """Log to ``{metrics_dir}/metrics.log`` and stderr."""
    import logging
    import sys

    os.makedirs(metrics_dir, exist_ok=True)
    log_path = os.path.join(metrics_dir, METRICS_LOG_NAME)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    pkg = _metrics_pkg_logger()
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
    log.info("Metrics log file: %s", log_path)


def _copy_metrics_config_to_redu(config_source_path, metrics_dir):
    if not config_source_path:
        return
    src = os.path.abspath(os.fspath(config_source_path))
    if not os.path.isfile(src):
        return
    os.makedirs(metrics_dir, exist_ok=True)
    dst = os.path.join(metrics_dir, METRICS_CONFIG_SNAPSHOT_NAME)
    shutil.copy2(src, dst)
    log.info("Saved config snapshot: %s", dst)


if __name__ == "__main__":
    cli_metrics()
