# ADI.py — VIP PCA-ADI post-processing on process.py outputs
import argparse
import ast
import logging
import os
import shutil
import sys

import numpy as np
from vip_hci.fm import normalize_psf
from vip_hci.psfsub import pca, pca_annular, pca_grid
from vip_hci.metrics import snrmap

from tools4magaox.redu import filereads as fr
from tools4magaox.proc import utils as pu

# Allow ``python path/to/ADI.py conf.txt`` without installing the package.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

log = logging.getLogger(__name__)

ADI_OUTPUT_DIR = "adi"
ADI_CONFIG_SNAPSHOT_NAME = "adi_config.txt"
ADI_LOG_NAME = "adi.log"
PSF_FWHM_NAME = "psf_fwhm.txt"
REFERENCE_IMAGE_NAME = "reference_sparkles.fits"
FILE_TABLE_NAME = "file_table.txt"
FILE_TABLE_OUTPUT_NAME = "file_table_output.txt"
CENTERED_FRAMES_DIR = "centered"

PSF_30X30_NAME = "psf_30x30.fits"
PSF_NORMALIZED_NAME = "psf_normalized.fits"
ADI_CUBE_NAME = "adi_cube.fits"
ADI_PARANG_NAME = "adi_parang.fits"
ADI_APERTURE_MASK_NAME = "adi_aperture_mask.fits"
ADI_OUTER_MASK_NAME = "adi_outer_mask.fits"
ADI_INNER_MASK_NAME = "adi_inner_mask.fits"
ADI_PCA_NAME = "adi_pca.fits"
ADI_SNRMAP_NAME = "adi_snrmap.fits"
ADI_PCA_ANNULUS_NAME = "adi_pca_annulus.fits"
ADI_PCA_GRID_FULLFR_NAME = "adi_pca_grid_fullfr.fits"
ADI_PCA_GRID_ANNULAR_NAME = "adi_pca_grid_annular.fits"
ADI_PCA_GRID_FULLFR_NPC_NAME = "adi_pca_grid_fullfr_opt_npc.txt"
ADI_PCA_GRID_ANNULAR_NPC_NAME = "adi_pca_grid_annular_opt_npc.txt"
ADI_PCA_GRID_FULLFR_CSV_NAME = "adi_pca_grid_fullfr.csv"
ADI_PCA_GRID_ANNULAR_CSV_NAME = "adi_pca_grid_annular.csv"
ADI_FRAME_SELECTION_NAME = "adi_frame_selection.txt"
ADI_ROTATION_PROBE_NAME = "adi_rotation_probe.txt"

PLOT_PSF_NAME = "adi_psf_normalized.png"
PLOT_PARANG_NAME = "adi_parang_timeseries.png"
PLOT_PCA_NAME = "adi_pca.png"
PLOT_SNRMAP_NAME = "adi_snrmap.png"
PLOT_PCA_ANNULUS_NAME = "adi_pca_annulus.png"
PLOT_PCA_GRID_FULLFR_NAME = "adi_pca_grid_fullfr.png"
PLOT_PCA_GRID_ANNULAR_NAME = "adi_pca_grid_annular.png"
PLOT_PCA_GRID_FULLFR_OPT_NAME = "adi_pca_grid_fullfr_opt.png"
PLOT_PCA_GRID_ANNULAR_OPT_NAME = "adi_pca_grid_annular_opt.png"


########################################################
######################### Main #########################
########################################################


def adi_main(run_params):
    """
    Run the ADI / PCA pipeline for one ``data_dir`` / ``camera`` pair.

    Reads process outputs from ``run_params["redu_dir"]`` and writes ADI
    products (PSF, cube, PCA frame, SNR map) under ``run_params["adi_dir"]``.
    """
    redu_dir = run_params["redu_dir"]
    adi_dir = run_params["adi_dir"]
    data_dir = run_params["data_dir"]
    camera = run_params["camera"]

    os.makedirs(adi_dir, exist_ok=True)
    _configure_adi_logging(adi_dir)
    _copy_adi_config_to_redu(run_params.get("config_source_path"), adi_dir)
    _validate_adi_inputs(run_params)

    log.info(
        "=> ADI processing %s %s (redu_dir=%s, adi_dir=%s)",
        data_dir,
        camera,
        redu_dir,
        adi_dir,
    )

    file_table_static = fr.read_redu_table(
        os.path.join(redu_dir, FILE_TABLE_NAME)
    )
    file_table_output = fr.read_redu_table(
        os.path.join(redu_dir, FILE_TABLE_OUTPUT_NAME)
    )
    file_table_output = fr.prune_process_output_table(file_table_output)

    psfn, fwhm = s1_build_psf(run_params)
    run_params["fwhm"] = fwhm

    cube, angs, times = s2_build_adi_cube(
        run_params, file_table_static, file_table_output
    )
    fr_pca = s3_run_pca(run_params, cube, angs)
    s4_snr_map(run_params, fr_pca, fwhm)
    if run_params.get("run_pca_annulus_grid"):
        s5_pca_annulus_grid(run_params, cube, angs, fwhm)

    return True


def s1_build_psf(run_params):
    """
    Crop ``reference_sparkles.fits`` to 30×30 and VIP-normalize the PSF.

    Returns
    -------
    psfn : ndarray
        Flux-normalized PSF.
    fwhm : float
        Mean FWHM from ``normalize_psf``.
    """
    log.info("1. Building PSF from reference_sparkles")
    redu_dir = run_params["redu_dir"]
    adi_dir = run_params["adi_dir"]
    force_rerun = run_params["force_rerun"]
    psf_crop_size = int(run_params["psf_crop_size"])
    psf_norm_size = int(run_params["psf_norm_size"])

    ref_path = os.path.join(redu_dir, REFERENCE_IMAGE_NAME)
    psf_crop_path = os.path.join(adi_dir, PSF_30X30_NAME)
    psf_norm_path = os.path.join(adi_dir, PSF_NORMALIZED_NAME)

    if not force_rerun and os.path.isfile(psf_norm_path):
        log.info("=> LOADING NORMALIZED PSF")
        psfn = fr._load_fits_primary_float32(psf_norm_path)
        fwhm_path = os.path.join(adi_dir, PSF_FWHM_NAME)
        if run_params.get("fwhm_override") is not None:
            fwhm = float(run_params["fwhm_override"])
        elif os.path.isfile(fwhm_path):
            with open(fwhm_path, encoding="utf-8") as fh:
                fwhm = float(fh.read().strip())
        else:
            psf_crop = fr._load_fits_primary_float32(psf_crop_path)
            _, _, fwhm = normalize_psf(
                psf_crop, size=psf_norm_size, debug=False, full_output=True
            )
        return psfn, float(fwhm)

    log.info("=> CREATING PSF")
    reference = fr._load_fits_primary_float32(ref_path)
    psf_crop = pu.center_crop_2d(reference, psf_crop_size)
    fr._save_fits_primary_float32(psf_crop, psf_crop_path)

    plot_psf = _adi_plot_enabled(run_params, "plot_psf")
    result = normalize_psf(
        psf_crop,
        size=psf_norm_size,
        debug=False,
        full_output=True,
    )
    psfn, flux, fwhm = result
    log.info("PSF normalize: FWHM=%.3f flux=%.3f", fwhm, flux)

    fr._save_fits_primary_float32(psfn, psf_norm_path)
    fwhm_path = os.path.join(adi_dir, PSF_FWHM_NAME)
    with open(fwhm_path, "w", encoding="utf-8") as fh:
        fh.write(f"{fwhm}\n")
    if plot_psf:
        pu.save_frame_plot(
            psfn,
            os.path.join(adi_dir, PLOT_PSF_NAME),
            title=f"Normalized PSF (FWHM={fwhm:.2f} px)",
        )
    return psfn, float(fwhm)


def s2_build_adi_cube(run_params, file_table_static, file_table_output):
    """
    Load centered frames, optionally crop and coadd, save ADI cube + PARANG.

    Returns
    -------
    cube, parang, times
    """
    log.info("2. Building ADI cube")
    adi_dir = run_params["adi_dir"]
    force_rerun = run_params["force_rerun"]
    cube_path = os.path.join(adi_dir, ADI_CUBE_NAME)
    parang_path = os.path.join(adi_dir, ADI_PARANG_NAME)

    row_idxs = pu.select_adi_frame_rows(file_table_static, file_table_output, run_params)
    selection = pu.summarize_adi_frame_selection(
        file_table_static, file_table_output, run_params
    )
    _write_adi_frame_selection_summary(adi_dir, selection, row_idxs, run_params)

    if not force_rerun and os.path.isfile(cube_path) and os.path.isfile(parang_path):
        log.info("=> LOADING ADI CUBE")
        cube = fr._load_fits_primary_float32(cube_path)
        parang = fr._load_fits_primary_float32(parang_path)
        times = None
        _run_adi_rotation_probe(adi_dir, cube, parang, run_params)
        return cube, parang, times

    if selection["selected_for_adi"] == 0:
        raise ValueError(
            "No frames available for ADI cube (check centered/ and used_in_reduction)"
        )
    if len(row_idxs) == 0:
        raise ValueError(
            "No frames available for ADI cube (check centered/ and used_in_reduction)"
        )

    n_before_subsample = len(row_idxs)
    if bool(run_params.get("fast_test", False)):
        stride = int(run_params.get("fast_test_stride", 100))
        if stride <= 0:
            raise ValueError(f"fast_test_stride must be positive, got {stride}")
        row_idxs = row_idxs[::stride]
        log.warning(
            "=> fast_test enabled: using %s/%s approved frames (every %sth frame). "
            "Disable fast_test for production ADI.",
            len(row_idxs),
            n_before_subsample,
            stride,
        )
    else:
        log.info(
            "=> Loading %s approved frames (used_in_reduction=%s)",
            len(row_idxs),
            bool(run_params.get("require_used_in_reduction", True)),
        )

    cube = pu.load_centered_cube_chunked(
        file_table_static,
        row_idxs,
        run_params,
        run_params.get("chunk_size", 100),
    )
    parang, times = pu.parang_and_times_for_rows(file_table_static, row_idxs)

    # Parallactic angles handling: unit conversion + sign/offset.
    parang = np.asarray(parang, dtype=float)
    parang_units = str(run_params.get("parang_units", "deg")).lower()
    if parang_units == "auto":
        # Heuristic: radians are typically within [-2π, 2π] while degrees span tens/hundreds.
        parang_units = "rad" if np.nanmax(np.abs(parang)) <= 7.0 else "deg"
        log.info("=> parang_units auto-detected as %s", parang_units)
    if parang_units == "rad":
        parang = np.rad2deg(parang)
    elif parang_units != "deg":
        raise ValueError(f"parang_units must be 'deg', 'rad', or 'auto', got {parang_units!r}")

    parang_sign = float(run_params.get("parang_sign", 1.0))
    parang_offset_deg = float(run_params.get("parang_offset_deg", 0.0))
    parang = parang_sign * (parang + parang_offset_deg)

    # Write a quick diagnostic summary to help catch unit/sign mistakes.
    summary_path = os.path.join(adi_dir, "adi_parang_summary.txt")
    finite = np.isfinite(parang)
    if np.any(finite):
        pmin = float(np.min(parang[finite]))
        pmax = float(np.max(parang[finite]))
        pspan = float(pmax - pmin)
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"parang_units={parang_units}\n")
            fh.write(f"parang_sign={parang_sign}\n")
            fh.write(f"parang_offset_deg={parang_offset_deg}\n")
            fh.write(
                "vip_derotation_note=VIP cube_derotate applies -angle internally; "
                "use parang_sign=1.0 with header PARANG unless debugging\n"
            )
            fh.write(f"min_deg={pmin}\nmax_deg={pmax}\nspan_deg={pspan}\n")

    radius_outer = run_params.get("crop_radius_outer")
    radius_inner = run_params.get("crop_radius_inner")
    if radius_outer is not None or radius_inner is not None:
        log.info(
            "=> Applying crop radii outer=%s inner=%s",
            radius_outer,
            radius_inner,
        )
        cube, aperture_mask, outer_mask, inner_mask = pu.apply_crop_radius_masks(
            cube,
            radius_outer=radius_outer,
            radius_inner=radius_inner,
        )
        fr._save_fits_primary_float32(
            aperture_mask, os.path.join(adi_dir, ADI_APERTURE_MASK_NAME)
        )
        if outer_mask is not None:
            fr._save_fits_primary_float32(
                outer_mask, os.path.join(adi_dir, ADI_OUTER_MASK_NAME)
            )
        if inner_mask is not None:
            fr._save_fits_primary_float32(
                inner_mask, os.path.join(adi_dir, ADI_INNER_MASK_NAME)
            )

    coadd_mode = str(run_params.get("coadd_mode", "none")).lower()
    if coadd_mode == "frames":
        n_coadd = int(run_params["frame_coadd_n"])
        log.info("=> Coadding by frame groups of %s", n_coadd)
        cube, parang, times = fr.coadd_by_frames(
            cube, parang, times, frame_coadd=n_coadd
        )
    elif coadd_mode == "time":
        t_coadd = float(run_params["time_coadd_sec"])
        log.info("=> Coadding by time windows of %.1f s", t_coadd)
        cube, parang, times = fr.coadd_by_time(
            cube, times, parang, time_coadd=t_coadd
        )
    elif coadd_mode != "none":
        raise ValueError(
            f"coadd_mode must be 'none', 'frames', or 'time', got {coadd_mode!r}"
        )

    log.info("ADI cube shape: %s, %s parang values", cube.shape, len(parang))
    fr._save_fits_primary_float32(cube, cube_path)
    fr._save_fits_primary_float32(np.asarray(parang, dtype=np.float32), parang_path)

    _run_adi_rotation_probe(adi_dir, cube, parang, run_params)

    if _adi_plot_enabled(run_params, "plot_parang") and times is not None:
        pu.save_parang_timeseries_plot(
            times,
            parang,
            os.path.join(adi_dir, PLOT_PARANG_NAME),
        )

    return cube, parang, times


def s3_run_pca(run_params, cube, angs):
    """Run VIP full-frame PCA-ADI (subtraction + derotation + combine)."""
    batch = run_params.get("batch")
    log.info(
        "3. Running PCA-ADI (ncomp=%s, batch=%s)",
        run_params["ncomp"],
        batch,
    )
    adi_dir = run_params["adi_dir"]
    force_rerun = run_params["force_rerun"]
    pca_path = os.path.join(adi_dir, ADI_PCA_NAME)

    if not force_rerun and os.path.isfile(pca_path):
        log.info("=> LOADING PCA FRAME")
        return fr._load_fits_primary_float32(pca_path)

    fwhm = run_params.get("fwhm")
    if run_params.get("fwhm_override") is not None:
        fwhm = run_params["fwhm_override"]

    kwargs = dict(
        ncomp=run_params["ncomp"],
        mask_center_px=run_params.get("mask_center_px", None),
        imlib=run_params.get("imlib", "vip-fft"),
        interpolation=run_params.get("interpolation"),
        svd_mode=run_params.get("svd_mode", "arpack"),
    )
    if batch is not None:
        kwargs["batch"] = batch
    if fwhm is not None:
        kwargs["fwhm"] = fwhm
    nproc = run_params.get("nproc")
    if nproc is not None:
        kwargs["nproc"] = nproc

    log.info("=> RUNNING VIP pca()")
    fr_pca = pca(cube, angs, **kwargs)
    fr_pca = np.asarray(fr_pca, dtype=np.float32)
    fr._save_fits_primary_float32(fr_pca, pca_path)

    if _adi_plot_enabled(run_params, "plot_pca"):
        pu.save_frame_plot(
            fr_pca,
            os.path.join(adi_dir, PLOT_PCA_NAME),
            title=f"PCA-ADI (ncomp={run_params['ncomp']})",
        )
    return fr_pca


def s4_snr_map(run_params, fr_pca, fwhm):
    """Compute and save VIP SNR map."""
    log.info("4. Computing SNR map")
    adi_dir = run_params["adi_dir"]
    force_rerun = run_params["force_rerun"]
    snr_path = os.path.join(adi_dir, ADI_SNRMAP_NAME)

    if not force_rerun and os.path.isfile(snr_path):
        log.info("=> LOADING SNR MAP")
        return fr._load_fits_primary_float32(snr_path)

    if run_params.get("fwhm_override") is not None:
        fwhm = run_params["fwhm_override"]

    log.info("=> RUNNING VIP snrmap()")
    snr_map = snrmap(fr_pca, fwhm, plot=False)
    snr_map = np.asarray(snr_map, dtype=np.float32)
    fr._save_fits_primary_float32(snr_map, snr_path)

    if _adi_plot_enabled(run_params, "plot_snrmap"):
        pu.save_frame_plot(
            snr_map,
            os.path.join(adi_dir, PLOT_SNRMAP_NAME),
            title="SNR map",
        )
    return snr_map


def s5_pca_annulus_grid(run_params, cube, angs, fwhm):
    """
    VIP tutorial 3.5.6: PCA in a single annulus and PCA grid optimization.

    Runs whenever ``run_pca_annulus_grid`` is True, independent of
    ``force_rerun``.
    """
    log.info("5. Running PCA annulus + PCA grid (VIP tutorial 3.5.6)")
    adi_dir = run_params["adi_dir"]
    os.makedirs(adi_dir, exist_ok=True)

    source_xy = run_params.get("source_xy")
    if source_xy is None:
        raise ValueError(
            "run_pca_annulus_grid=True requires source_xy = (x, y) in config"
        )
    source_xy = tuple(float(v) for v in source_xy)

    if run_params.get("fwhm_override") is not None:
        fwhm = float(run_params["fwhm_override"])
    fwhm = float(fwhm)

    annulus_width = run_params.get("annulus_width")
    if annulus_width is None:
        annulus_width = 3.0 * fwhm
    else:
        annulus_width = float(annulus_width)

    r_guess = run_params.get("r_guess")
    if r_guess is None:
        r_guess = pu.r_guess_from_source_xy(source_xy, cube.shape[1:])
    else:
        r_guess = float(r_guess)

    rot_kwargs = _pca_rot_kwargs(run_params)
    plot_step = _adi_plot_enabled(run_params, "plot_pca_annulus_grid")

    ncomp_ann = int(run_params.get("pca_annulus_ncomp", run_params["ncomp"]))
    log.info(
        "=> pca_annulus (ncomp=%s, r_guess=%.2f, annulus_width=%.2f)",
        ncomp_ann,
        r_guess,
        annulus_width,
    )
    pca_ann = pca_annular(
        cube,
        angs,
        ncomp=ncomp_ann,
        annulus_width=annulus_width,
        r_guess=r_guess,
        svd_mode=run_params.get("svd_mode", "arpack"),
        **rot_kwargs,
    )
    pca_ann = np.asarray(pca_ann, dtype=np.float32)
    fr._save_fits_primary_float32(
        pca_ann, os.path.join(adi_dir, ADI_PCA_ANNULUS_NAME)
    )
    if plot_step:
        pu.save_frame_plot(
            pca_ann,
            os.path.join(adi_dir, PLOT_PCA_ANNULUS_NAME),
            title=f"PCA annulus (ncomp={ncomp_ann}, r={r_guess:.1f} px)",
        )

    range_pcs = run_params.get("pca_grid_range_pcs", (1, 31, 1))
    modes = run_params.get("pca_grid_modes", ["fullfr", "annular"])
    if isinstance(modes, str):
        modes = [modes]
    modes = [str(m).lower() for m in modes]

    grid_outputs = {
        "fullfr": {
            "fits": ADI_PCA_GRID_FULLFR_NAME,
            "npc": ADI_PCA_GRID_FULLFR_NPC_NAME,
            "csv": ADI_PCA_GRID_FULLFR_CSV_NAME,
            "plot": PLOT_PCA_GRID_FULLFR_NAME,
            "vip_plot": PLOT_PCA_GRID_FULLFR_OPT_NAME,
        },
        "annular": {
            "fits": ADI_PCA_GRID_ANNULAR_NAME,
            "npc": ADI_PCA_GRID_ANNULAR_NPC_NAME,
            "csv": ADI_PCA_GRID_ANNULAR_CSV_NAME,
            "plot": PLOT_PCA_GRID_ANNULAR_NAME,
            "vip_plot": PLOT_PCA_GRID_ANNULAR_OPT_NAME,
        },
    }

    for mode in modes:
        if mode not in grid_outputs:
            raise ValueError(
                f"pca_grid_modes entries must be 'fullfr' or 'annular', got {mode!r}"
            )
        out = grid_outputs[mode]
        log.info(
            "=> pca_grid mode=%s range_pcs=%s source_xy=%s",
            mode,
            range_pcs,
            source_xy,
        )
        grid_kwargs = dict(
            fwhm=fwhm,
            range_pcs=range_pcs,
            source_xy=source_xy,
            mode=mode,
            svd_mode=run_params.get("svd_mode", "arpack"),
            full_output=True,
            plot=plot_step,
            save_plot=(
                os.path.join(adi_dir, out["vip_plot"]) if plot_step else None
            ),
            **rot_kwargs,
        )
        if mode == "annular":
            grid_kwargs["annulus_width"] = annulus_width

        _, final_frame, df, opt_npc = pca_grid(cube, angs, **grid_kwargs)
        final_frame = np.asarray(final_frame, dtype=np.float32)
        fr._save_fits_primary_float32(
            final_frame, os.path.join(adi_dir, out["fits"])
        )
        pu.save_text_value(os.path.join(adi_dir, out["npc"]), opt_npc)
        pu.save_dataframe_csv(df, os.path.join(adi_dir, out["csv"]))
        log.info("pca_grid %s: optimal ncomp=%s", mode, opt_npc)
        if plot_step:
            pu.save_frame_plot(
                final_frame,
                os.path.join(adi_dir, out["plot"]),
                title=f"PCA grid {mode} (opt npc={opt_npc})",
            )


def _run_adi_rotation_probe(adi_dir, cube, parang, run_params):
    probe = pu.adi_rotation_alignment_probe(
        cube,
        parang,
        expected_r_px=run_params.get("expected_source_r_px"),
        inner_exclude_px=run_params.get("crop_radius_inner")
        or run_params.get("mask_center_px")
        or 30.0,
        max_frames=int(run_params.get("rotation_probe_max_frames", 500)),
    )
    _write_adi_rotation_probe(adi_dir, probe, run_params)


def _write_adi_frame_selection_summary(adi_dir, selection, row_idxs, run_params):
    """Log and save how many frames pass each ADI selection gate."""
    lines = [f"{key}={value}" for key, value in selection.items()]
    lines.append(f"cube_frames={len(row_idxs)}")
    if bool(run_params.get("fast_test", False)):
        lines.append(f"fast_test=True")
        lines.append(f"fast_test_stride={run_params.get('fast_test_stride')}")
    pu.save_adi_diagnostics(os.path.join(adi_dir, ADI_FRAME_SELECTION_NAME), lines)
    log.info(
        "ADI frame selection: %s/%s roster rows selected "
        "(majority=%s, centered=%s, used_in_reduction=%s)",
        selection["selected_for_adi"],
        selection["roster_rows"],
        selection["pass_majority_config"],
        selection["centered_file_exists"],
        selection["used_in_reduction"],
    )


def _write_adi_rotation_probe(adi_dir, probe, run_params):
    """Log median-derotation peak radius vs expected companion separation."""
    if probe is None:
        return
    lines = [f"{key}={value}" for key, value in probe.items()]
    pu.save_adi_diagnostics(os.path.join(adi_dir, ADI_ROTATION_PROBE_NAME), lines)
    expected = probe["expected_r_px"]
    peak_r = probe["peak_r_px"]
    log.info(
        "Rotation probe (median VIP derotate): peak at (y,x)=(%s,%s) r=%.1f px "
        "(expected %.1f px); peak=%.3g inner_ring=%.3g",
        probe["peak_y"],
        probe["peak_x"],
        peak_r,
        expected,
        probe["peak_value"],
        probe["inner_peak_value"],
    )
    if abs(peak_r - expected) > 5.0:
        log.warning(
            "Rotation probe peak radius %.1f px differs from expected_source_r_px "
            "%.1f px — check parang_sign (default 1.0; VIP negates angles internally) "
            "and disable fast_test for production runs",
            peak_r,
            expected,
        )


def _pca_rot_kwargs(run_params):
    """Rotation / multiprocessing kwargs shared by VIP PCA routines."""
    kwargs = {
        "imlib": run_params.get("imlib", "vip-fft"),
        "interpolation": run_params.get("interpolation"),
    }
    nproc = run_params.get("nproc")
    if nproc is not None:
        kwargs["nproc"] = nproc
    return kwargs


def _validate_adi_inputs(run_params):
    """Ensure process outputs exist before running ADI."""
    redu_dir = run_params["redu_dir"]
    missing = []
    for name in (REFERENCE_IMAGE_NAME, FILE_TABLE_NAME, FILE_TABLE_OUTPUT_NAME):
        if not os.path.isfile(os.path.join(redu_dir, name)):
            missing.append(name)
    centered_dir = os.path.join(redu_dir, CENTERED_FRAMES_DIR)
    if not os.path.isdir(centered_dir):
        missing.append(f"{CENTERED_FRAMES_DIR}/")
    if missing:
        raise FileNotFoundError(
            f"ADI requires process outputs in {redu_dir}; missing: {missing}"
        )


def _adi_plot_enabled(run_params, plot_key):
    if plot_key in run_params:
        return bool(run_params[plot_key])
    return bool(run_params.get("plot", False))


#############################################################
################### Pipeline functionality ###################
#############################################################


def read_adi_config(config_path):
    """Read an ADI config file (``name = value`` per line; ``#`` comments)."""
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


def check_adi_config(params):
    """Verify required ADI parameters."""
    bad = []
    if not _nonempty_str(params.get("redu_path")):
        bad.append("redu_path")
    if not _nonempty_str(params.get("data_dir")):
        bad.append("data_dir")
    if not _nonempty_camera_list(params.get("cameras")):
        bad.append("cameras")
    return bad


def _nonempty_str(x):
    return isinstance(x, str) and bool(x.strip())


def _nonempty_camera_list(x):
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        return False
    return all(_nonempty_str(c) for c in x)


def run_adi_from_config(params, config_source_path=None):
    """Validate config and run :func:`adi_main` for each camera."""
    missing = check_adi_config(params)
    if missing:
        raise ValueError(f"config missing or invalid keys: {missing}")
    data_dir = params["data_dir"].strip()
    cameras = list(params["cameras"])

    for camera in cameras:
        try:
            run = build_adi_run_params(
                params,
                camera,
                config_source_path=config_source_path,
            )
            adi_main(run)
        except Exception:
            log.exception("Error in ADI for %s %s", data_dir, camera)


def _resolve_crop_radius_params(params):
    """
    Normalize crop radius config keys.

    Primary keys: ``crop_radius_outer``, ``crop_radius_inner``.
    Legacy aliases: ``crop_r``, ``cube_mask_center_px``.
    """
    outer = params.get("crop_radius_outer")
    if outer is None and params.get("crop_r") is not None:
        outer = params["crop_r"]
    inner = params.get("crop_radius_inner")
    if inner is None and params.get("cube_mask_center_px") is not None:
        inner = params["cube_mask_center_px"]

    if outer is not None:
        outer = float(outer)
        if outer <= 0:
            raise ValueError(f"crop_radius_outer must be positive, got {outer}")
        params["crop_radius_outer"] = outer
    else:
        params["crop_radius_outer"] = None

    if inner is not None:
        inner = float(inner)
        if inner < 0:
            raise ValueError(f"crop_radius_inner must be non-negative, got {inner}")
        params["crop_radius_inner"] = inner if inner > 0 else None
    else:
        params["crop_radius_inner"] = None

    if (
        params["crop_radius_outer"] is not None
        and params["crop_radius_inner"] is not None
        and params["crop_radius_inner"] >= params["crop_radius_outer"]
    ):
        raise ValueError(
            "crop_radius_inner must be < crop_radius_outer when both are set"
        )


def build_adi_run_params(params, camera, *, config_source_path=None):
    """Shallow copy of config plus per-camera ``redu_dir`` and defaults."""
    p = dict(params)
    p["camera"] = camera
    p["data_dir"] = p["data_dir"].strip()
    p["redu_dir"] = f"{p['redu_path']}{p['data_dir']}/{p['camera']}/"
    p.setdefault("adi_output_dir", ADI_OUTPUT_DIR)
    p["adi_dir"] = os.path.join(p["redu_dir"], p["adi_output_dir"])
    p["centered_dir"] = CENTERED_FRAMES_DIR
    p.setdefault("require_used_in_reduction", True)
    p.setdefault("psf_crop_size", 30)
    p.setdefault("psf_norm_size", 19)
    p.setdefault("coadd_mode", "none")
    p.setdefault("frame_coadd_n", 10)
    p.setdefault("time_coadd_sec", 10.0)
    p.setdefault("ncomp", 5)
    p.setdefault("imlib", "vip-fft")
    p.setdefault("interpolation", None)
    p.setdefault("svd_mode", "arpack")
    p.setdefault("nproc", None)
    p.setdefault("batch", None)
    p.setdefault("run_pca_annulus_grid", False)
    p.setdefault("pca_annulus_ncomp", p.get("ncomp", 5))
    p.setdefault("annulus_width", None)
    p.setdefault("r_guess", None)
    p.setdefault("source_xy", None)
    p.setdefault("pca_grid_range_pcs", (1, 31, 1))
    p.setdefault("pca_grid_modes", ["fullfr", "annular"])
    p.setdefault("plot", False)
    p.setdefault("plot_psf", p["plot"])
    p.setdefault("plot_pca", p["plot"])
    p.setdefault("plot_snrmap", p["plot"])
    p.setdefault("plot_parang", p["plot"])
    p.setdefault("plot_pca_annulus_grid", p["plot"])
    p.setdefault("force_rerun", False)
    p.setdefault("chunk_size", 100)
    # Quick mode: use a sparse subset of frames for the ADI cube build.
    p.setdefault("fast_test", False)
    p.setdefault("fast_test_stride", 100)
    p.setdefault("crop_radius_outer", None)
    p.setdefault("crop_radius_inner", None)
    # Parallactic angle handling.
    p.setdefault("parang_units", "deg")  # 'deg', 'rad', or 'auto'
    p.setdefault("parang_sign", 1.0)     # VIP derotates by -angle internally
    p.setdefault("parang_offset_deg", 0.0)
    p.setdefault("expected_source_r_px", None)
    p.setdefault("rotation_probe_max_frames", 500)
    if p.get("fwhm") is not None:
        p["fwhm_override"] = float(p["fwhm"])
    else:
        p["fwhm_override"] = None
    _resolve_crop_radius_params(p)
    if p.get("fast_test"):
        p["fast_test_stride"] = int(p.get("fast_test_stride", 100))
        if p["fast_test_stride"] <= 0:
            raise ValueError(
                f"fast_test_stride must be positive, got {p['fast_test_stride']}"
            )
    if p.get("batch") is not None:
        batch = p["batch"]
        if isinstance(batch, bool) or not isinstance(batch, (int, float)):
            raise ValueError(
                f"batch must be None, an int (frames per mini-batch), or a float "
                f"in (0, 1] (fraction of available memory), got {batch!r}"
            )
        if isinstance(batch, float) and not (0.0 < batch <= 1.0):
            raise ValueError(
                f"batch float must be in (0, 1], got {batch}"
            )
        if isinstance(batch, int) and batch <= 0:
            raise ValueError(f"batch int must be positive, got {batch}")
    if p.get("run_pca_annulus_grid"):
        if p.get("source_xy") is None:
            raise ValueError(
                "run_pca_annulus_grid=True requires source_xy = (x, y)"
            )
        p["source_xy"] = tuple(float(v) for v in p["source_xy"])
        if p.get("r_guess") is not None:
            p["r_guess"] = float(p["r_guess"])
        if p.get("annulus_width") is not None:
            p["annulus_width"] = float(p["annulus_width"])
        p["pca_annulus_ncomp"] = int(p.get("pca_annulus_ncomp", p["ncomp"]))
        modes = p.get("pca_grid_modes", ["fullfr", "annular"])
        if isinstance(modes, str):
            modes = [modes]
        p["pca_grid_modes"] = [str(m).lower() for m in modes]
        for mode in p["pca_grid_modes"]:
            if mode not in ("fullfr", "annular"):
                raise ValueError(
                    f"pca_grid_modes entries must be 'fullfr' or 'annular', got {mode!r}"
                )
        range_pcs = p.get("pca_grid_range_pcs", (1, 31, 1))
        if not isinstance(range_pcs, (list, tuple)) or len(range_pcs) not in (2, 3):
            raise ValueError(
                f"pca_grid_range_pcs must be (start, stop) or (start, stop, step), "
                f"got {range_pcs!r}"
            )
        p["pca_grid_range_pcs"] = tuple(int(v) for v in range_pcs)
    if config_source_path is not None:
        p["config_source_path"] = config_source_path
    return p


def cli_adi(argv=None):
    """CLI entry: one or more ADI config paths."""
    parser = argparse.ArgumentParser(
        description="Run the ADI / PCA pipeline from config file(s)."
    )
    parser.add_argument(
        "configs",
        nargs="+",
        metavar="CONF",
        help="ADI config file(s), e.g. redu/conf_ex/conf_adi_ex.txt",
    )
    args = parser.parse_args(argv)
    _ensure_stderr_logging()
    for cfg_path in args.configs:
        params = read_adi_config(cfg_path)
        log.info("=> Config: %s", cfg_path)
        try:
            run_adi_from_config(params, config_source_path=cfg_path)
        except ValueError as e:
            log.exception("%s: %s", cfg_path, e)
            raise SystemExit(1) from e


########### Logger Setup ##############


def _ensure_stderr_logging():
    pkg = _adi_pkg_logger()
    if pkg.handlers:
        return
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    pkg.addHandler(sh)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    log.propagate = True


def _adi_pkg_logger():
    p = log.parent
    return p if p.name else log


def _configure_adi_logging(adi_dir):
    """Log to ``{adi_dir}/adi.log`` and stderr."""
    os.makedirs(adi_dir, exist_ok=True)
    log_path = os.path.join(adi_dir, ADI_LOG_NAME)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    pkg = _adi_pkg_logger()
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
    log.info("ADI log file: %s", log_path)


def _copy_adi_config_to_redu(config_source_path, adi_dir):
    if not config_source_path:
        return
    src = os.path.abspath(os.fspath(config_source_path))
    if not os.path.isfile(src):
        return
    os.makedirs(adi_dir, exist_ok=True)
    dst = os.path.join(adi_dir, ADI_CONFIG_SNAPSHOT_NAME)
    shutil.copy2(src, dst)
    log.info("Saved config snapshot: %s", dst)


if __name__ == "__main__":
    cli_adi()
