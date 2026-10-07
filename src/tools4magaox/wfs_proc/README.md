# wfs_proc

Tools for processing MagAO-X wavefront-sensor telemetry.

## loop_reconstruction

Reconstruct open-loop wavefronts (nm OPD on the tweeter DM grid) from `camwfs` xrif frames and a cacao AO calibration set.

### Pipeline

1. **Load WFS frames** — select xrif files for a UTC time window (`t_start` + `duration_s` or `t_end`), read with `fixr`, keep frames whose acquisition times fall in the window.
2. **WFS → delta wavefront** — dark subtract, apply `aol1_wfsmask`, flux-normalize, subtract `aol1_wfsref`, project onto `aol1_modesWFS` (`modenorm` = cacao `MODENORM`), map coefficients through `aol1_DMmodes` (or `DMmodes_WFSdiag` when mode counts match), scale to nm OPD.
3. **Integrate** — sum consecutive delta wavefronts over `integration_time_s` (default 10 s) to form the full wavefront per window.

### Run

```bash
python -m tools4magaox.wfs_proc.loop_reconstruction path/to/conf.txt
```

See [conf_ex/conf_loop_recon_ex.txt](conf_ex/conf_loop_recon_ex.txt) for a filled example using the NAS paths:

- WFS: `/srv/nas/magaox_rawdata/rtc/rawimages/camwfs/`
- cacao: `/srv/nas/magaox_rawdata/rtc/cacao_pre20230823/tweeter/tweeter001/AOcalibs/<aocalib_set>/`
- darks: `/srv/nas/magaox_rawdata/rtc/calib/camwfs-dark/`

### Outputs

Written under `output_dir`:

| file | contents |
|------|----------|
| `full_wf_nm.fits` | Integrated wavefront cubes `(M, 50, 50)` in nm OPD |
| `modal_coeffs.fits` | Per-frame modal coefficients `(N, nmodes)` |
| `delta_wf_nm.fits` | Optional per-frame deltas |
| `times.txt` | Frame and window timestamps |
| `provenance.txt` | Dark + calib paths used |
| `loop_recon_config.txt` | Snapshot of the input config |
| `loop_recon.log` | Run log |

### Notes

- The camwfs archive is a flat directory (~1.5e6 files). Set `wfs_index_cache` so the first scan is reused.
- For `tweeter001`, `aol1_DMmodes` has 1830 modes while `aol1_modesWFS` has 1791; the loader automatically selects `aol1_DMmodes_WFSdiag` when counts match.
- `opd_nm_to_phase(opd_nm, wavelength_m)` converts nm OPD to radians.
