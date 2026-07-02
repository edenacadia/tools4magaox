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
- `fit_function` what psf fitter to use
- `max_files` how many files to load in, defaults to 1
- `pct_cut` which percentile of peak intensity to use in centering
- `gauss_amp_pct_cut` which percentile cut to use for the Gaussian-amplitude filter in the averaging step


### pipeline outputs
- `file_table.txt` per-file telemetry and `masterdark_path` (no pipeline filter columns)
- `file_table_output.txt` one row per file: all filters (majority, peak max, average), Gaussian fit columns, shifts, and average-stage flags (`pass_avg_shift`, `pass_avg_amp`, `used_in_average`)
- `clean_cube.fits` all files stacked in a cube
- `centered_cube.fits` unsats, filtered, centered, saved as a cube
- `average_image.fits` unsats, filted on shift amount, avereaged into a cube

## Process
These functions are appropriate for coronagraphic observations, where there is not a central PSF. 

These use the outputs from the process step and should be done in sequence. 

```
python process.py conf_ex/conf_process_ex.txt
```