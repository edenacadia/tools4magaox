# REDU

These functions are made to process MagAO-X data files


## Preprocess

Preprocessing steps are applicable to non-coronagraphic unsats. Flexibility to use with non-corongraphic data has not yet been implemented.  

Running the pipeline from a conf file: 

```
python preprocess.py conf_ex/conf_preproc_ex.txt
```


### conf variables
- `obs_path` where to find the observation directory
- `unsats_dir` observation directory name only
- `camera` which camera folders to iter through
- `plot` whether or not to plot the filter plots
- `force_rerun` redo created portions instead of loading in
- `fit_func` PSF core fitter for centering: `"airy"` (default), `"gauss_min"`, or `"gauss_curvefit"`
- `wavelength` initial lambda/D guess for the Airy fit (scalar, or list in `cameras` order); lambda/D is fit freely
- `airy_fit_radius_ld` only pixels within this many lambda/D of the core are fit (default `1.5`)
- `airy_obscuration` fractional central obscuration of the Airy model (default `0.0`)
- `jitter_match_exptime` the sats' EXPTIME in seconds; consecutive unsat frames are coadded without registration to this exposure before the core fit, so the average image has the same jitter as one sat frame (default unset, no coadding)
- `max_files` how many files to load in, defaults to 1
- `pct_cut` which percentile of peak intensity to use in centering
- `gauss_amp_pct_cut` which percentile cut to use for the core-fit amplitude filter in the averaging step


### pipeline outputs
- `file_table.txt` per-file telemetry and `masterdark_path` (no pipeline filter columns)
- `file_table_output.txt` one row per file: all filters (majority, peak max, average), core fit columns, shifts `(shift_y, shift_x)`, and average-stage flags (`pass_avg_shift`, `pass_avg_amp`, `used_in_average`). The `gauss_*` columns hold the core fit for either fitter; for the Airy fit `gauss_sigma_*` is the Gaussian-equivalent width (FWHM / 2.355). With jitter matching, all frames in a coadd share its shift and fit.
- `clean_cube.fits` all files stacked in a cube
- `centered_cube.fits` unsats, filtered, centered, saved as a cube
- `average_image.fits` unsats, filted on shift amount, avereaged into a cube

## Process
These functions are appropriate for coronagraphic observations, where there is not a central PSF. 

These use the outputs from the process step and should be done in sequence. 

```
python process.py conf_ex/conf_process_ex.txt
```