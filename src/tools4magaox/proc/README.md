# PROC

Post-processing routines for MagAO-X reduced data. These run **after** the coronagraphic `process.py` pipeline and use [VIP](https://vip.readthedocs.io/) for ADI PSF subtraction.

## ADI / PCA

[`ADI.py`](ADI.py) implements full-frame PCA-ADI following the [VIP tutorial (3A)](https://vip.readthedocs.io/en/latest/tutorials/03A_psfsub_ADI.html). It reads centered science frames and a reference PSF from each camera's redu folder, builds an ADI datacube, runs PCA subtraction with derotation, and writes an SNR map.

Run after `process.py` on the same `data_dir` and `cameras`.

```
python ADI.py ../redu/conf_ex/conf_adi_ex.txt
```

From the repo root (with the package on `PYTHONPATH` or installed):

```
PYTHONPATH=src python src/tools4magaox/proc/ADI.py src/tools4magaox/redu/conf_ex/conf_adi_ex.txt
```

### Prerequisites

- `process.py` completed for the target `data_dir` / `camera`
- Python dependencies installed (`pip install -e .` pulls in `vip-hci`, `hciplot`, etc.)

Required inputs in `{redu_path}{data_dir}/{camera}/`:

| File | Source |
|---|---|
| `reference_sparkles.fits` | process step 1 |
| `file_table.txt` | process step 0 (`PARANG`, `DATE_OBS`) |
| `file_table_output.txt` | process step 4 (`used_in_reduction`) |
| `centered/*.fits` | process step 3 |

### Pipeline steps

1. **PSF** — Center-crop `reference_sparkles.fits` to `psf_crop_size` (default 30 px), VIP `normalize_psf`, save normalized PSF and FWHM.
2. **ADI cube** — Load centered frames that pass frame selection, optionally crop and coadd, save cube + parallactic angles.
3. **PCA-ADI** — VIP `pca()` (subtraction, derotation, and median combine in one call).
4. **SNR map** — VIP `snrmap()` on the final PCA frame.
5. **PCA annulus + grid** (optional) — VIP `pca_annulus()` and `pca_grid()` when `run_pca_annulus_grid = True`. Always runs when enabled, regardless of `force_rerun`.

Steps 1–4 skip if their outputs already exist unless `force_rerun = True`.

### conf variables

Required:

- `redu_path` — root redu directory (same as process config)
- `data_dir` — coronagraphic observation directory name
- `cameras` — list of cameras, e.g. `["camsci1", "camsci2"]`

Frame selection:

- `require_used_in_reduction` — if `True` (default), only frames with `used_in_reduction = 1` in `file_table_output.txt` are used

Step 1 (PSF):

- `psf_crop_size` — center crop size in pixels before normalization (default `30`)
- `psf_norm_size` — VIP `normalize_psf` fitting aperture (default `19`)

Step 2 (cube):

- `crop_radius_outer` — crop to square `(2*R, 2*R)` then zero pixels outside the circle of radius `R` (square corners masked). Legacy alias: `crop_r`
- `crop_radius_inner` — zero pixels inside the central disk of radius `R`. Legacy alias: `cube_mask_center_px`
- `fast_test` — if `True`, only every `fast_test_stride` **approved** frame is used (quick-look only; leave `False` for production ADI)
- `fast_test_stride` — stride for `fast_test` (default `100`)
- `coadd_mode` — `"none"`, `"frames"`, or `"time"`
- `frame_coadd_n` — consecutive frames per coadd when `coadd_mode = "frames"`
- `time_coadd_sec` — minimum time span per coadd group (seconds) when `coadd_mode = "time"`
- `chunk_size` — batch size when loading centered FITS files (default `100`)
- `parang_units` — `"deg"` (default), `"rad"`, or `"auto"` (heuristic)
- `parang_sign` — `1.0` (default). Pass header `PARANG` as-is: VIP `cube_derotate` already rotates each frame by `-angle`. Only flip to `-1.0` when debugging rotation direction
- `parang_offset_deg` — additional offset applied to angles (degrees), default `0.0`
- `expected_source_r_px` — optional companion separation (px) for `adi_rotation_probe.txt` (median VIP derotation sanity check)
- `rotation_probe_max_frames` — max frames subsampled for the rotation probe (default `500`)

Step 3 (PCA):

- `ncomp` — number of principal components for VIP PCA-ADI
- `imlib` — rotation library (default `"vip-fft"`)
- `interpolation` — interpolation method for rotation (default `None`)
- `svd_mode` — SVD backend (default `"arpack"`)
- `nproc` — VIP multiprocessing CPU count (`None` = VIP default)
- `batch` — VIP incremental PCA batching (`None` = standard single-pass PCA). Set an `int` for frames per mini-batch, or a `float` in `(0, 1]` to size batches from available system memory (see VIP tutorial 3.5.4)

Step 4 (SNR):

- `fwhm` — PSF FWHM in pixels for the SNR map; `None` uses the value from step 1

Step 5 (PCA annulus + grid, optional):

- `run_pca_annulus_grid` — if `True`, run VIP tutorial 3.5.6 after step 4 (always runs when enabled, ignoring `force_rerun`)
- `source_xy` — `(x, y)` coordinates of the companion candidate (required when step 5 is enabled)
- `pca_annulus_ncomp` — PCs for `pca_annulus` (defaults to `ncomp`)
- `annulus_width` — annulus width in pixels for annular PCA/grid; `None` uses `3 * fwhm`
- `r_guess` — annulus radius in pixels; `None` is computed from `source_xy`
- `pca_grid_range_pcs` — PC search range as `(start, stop)` or `(start, stop, step)` for `pca_grid`
- `pca_grid_modes` — list of modes to run: `"fullfr"` and/or `"annular"`

General:

- `adi_output_dir` — subdirectory under the camera redu folder for ADI products (default `"adi"`)
- `plot` — master switch for diagnostic PNGs (default `False`)
- `plot_psf`, `plot_pca`, `plot_snrmap`, `plot_parang`, `plot_pca_annulus_grid` — per-step plot overrides (default to `plot`)
- `force_rerun` — recompute steps even when outputs exist (default `False`)

### pipeline outputs

Written to `{redu_path}{data_dir}/{camera}/adi/` (override folder name with `adi_output_dir` in config):

| File | Description |
|---|---|
| `adi_config.txt` | copy of the config used for this run |
| `adi.log` | run log |
| `psf_30x30.fits` | center-cropped reference PSF |
| `psf_normalized.fits` | VIP flux-normalized PSF |
| `psf_fwhm.txt` | FWHM from `normalize_psf` |
| `adi_cube.fits` | ADI datacube `(N, H, W)` |
| `adi_aperture_mask.fits` | combined outer/inner keep-mask (1 = valid pixels) |
| `adi_outer_mask.fits` | outer circular mask when `crop_radius_outer` is set |
| `adi_inner_mask.fits` | inner mask when `crop_radius_inner` is set |
| `adi_parang.fits` | parallactic angles, one per cube frame |
| `adi_parang_summary.txt` | PARANG range and VIP derotation note |
| `adi_frame_selection.txt` | frame counts at each selection gate |
| `adi_rotation_probe.txt` | median-derotation peak radius vs `expected_source_r_px` |
| `adi_pca.fits` | derotated PCA-ADI combined frame |
| `adi_snrmap.fits` | SNR map |
| `adi_psf_normalized.png` | normalized PSF plot |
| `adi_parang_timeseries.png` | PARANG vs time |
| `adi_pca.png` | PCA result |
| `adi_snrmap.png` | SNR map plot |
| `adi_pca_annulus.fits` | single-annulus PCA frame (step 5) |
| `adi_pca_grid_fullfr.fits` | optimal full-frame PCA grid result (step 5) |
| `adi_pca_grid_annular.fits` | optimal annular PCA grid result (step 5) |
| `adi_pca_grid_*_opt_npc.txt` | optimal PC count from `pca_grid` |
| `adi_pca_grid_*.csv` | S/N and flux vs PCs from `pca_grid` |
| `adi_pca_annulus.png` | single-annulus PCA plot (step 5) |
| `adi_pca_grid_*.png` | PCA grid result plots (step 5) |

### Supporting modules

- [`utils.py`](utils.py) — frame selection, centered-cube loading, center crop, matplotlib plots
- Example config: [`conf_ex/conf_adi_ex.txt`](conf_ex/conf_adi_ex.txt)

## Metrics

[`metrics.py`](metrics.py) runs VIP throughput, contrast curve, and SNR source-peak metrics on centered/ADI data. It prefers existing ADI products under `adi/` and rebuilds them from `centered/` via ADI step functions when missing.

Run after `process.py` (and ideally `ADI.py`) on the same `data_dir` and `cameras`.

```
python metrics.py conf_ex/conf_metrics_ex.txt --throughput --contrast --source-peak
```

From the repo root:

```
PYTHONPATH=src python src/tools4magaox/proc/metrics.py \
  src/tools4magaox/proc/conf_ex/conf_metrics_ex.txt \
  --throughput --contrast --source-peak
```

CLI flags (`--throughput`, `--contrast`, `--source-peak`) select which metrics to run. When no flags are given, the config keys `run_throughput`, `run_contrast`, and `run_source_peak` control selection.

### Prerequisites

- `process.py` completed for the target `data_dir` / `camera`
- ADI products recommended (`adi/adi_cube.fits`, `adi_pca.fits`, `adi_snrmap.fits`, etc.); metrics will call ADI steps to create missing products

### conf variables

Inherits ADI/reduction keys (`redu_path`, `data_dir`, `cameras`, frame selection, crop radii, PCA params) plus:

- `run_throughput`, `run_contrast`, `run_source_peak` — default on/off when no CLI flags
- `metrics_output_dir` — output subdirectory (default `"metrics"`)
- `pxscale` — arcsec/px (required for contrast curve)
- `starphot` — star flux in coronagraphic frames; `None` auto-derives from PSF flux and EXPTIME ratio
- `starphot_psf_exptime` — reference PSF exposure time for starphot scaling (optional)
- `contrast_sigma`, `throughput_nbranch`, `contrast_nbranch` — VIP sampling controls
- `metrics_inner_rad`, `fc_rad_sep`, `noise_sep`, `wedge` — fake-companion injection geometry
- `expected_source_r_px`, `source_peak_inner_exclude_px`, `source_peak_annulus_half_width` — SNR peak search
- `plot_throughput`, `plot_contrast`, `plot_source_peak` — diagnostic PNGs
- `force_rerun` — recompute even when outputs exist

Example config: [`conf_ex/conf_metrics_ex.txt`](conf_ex/conf_metrics_ex.txt)

### pipeline outputs

Written to `{redu_path}{data_dir}/{camera}/metrics/`:

| File | Description |
|---|---|
| `metrics_config.txt` | copy of the config used for this run |
| `metrics.log` | run log |
| `metrics_throughput.csv` | throughput vs separation |
| `metrics_throughput.png` | throughput plot (optional) |
| `metrics_contrast.csv` | contrast curve table |
| `metrics_contrast.png` | contrast plot (optional) |
| `metrics_source_peak.txt` | peak `(y, x)`, SNR, and radius |
| `metrics_source_peak.png` | SNR map with peak marked (optional) |
