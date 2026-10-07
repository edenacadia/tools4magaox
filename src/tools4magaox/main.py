"""Package map for tools4magaox.

Most modules are imported from notebooks or other scripts. Pipelines that take a
config file can also be run as ``python -m tools4magaox.<module> CONF``.
"""

from __future__ import annotations

MAP = """
tools4magaox — MagAO-X data processing helpers
==============================================

Install / path
  pip install -e .
  # or: PYTHONPATH=src

Run this map
  python -m tools4magaox


Subpackages
-----------
  redu       Reduce raw science data (dark, center, filter, cubes)
  proc       Post-process reduced cubes (ADI/PCA, VIP metrics)
  wfs_proc   Wavefront-sensor telemetry → open-loop wavefronts
  sims       Camera simulators (import-only)
  constants  Shared platescale / wavelength constants (import-only)


CLI pipelines (config file required)
------------------------------------
  redu — unsaturated preprocess
    python -m tools4magaox.redu.preprocess CONF [CONF ...]
    example: src/tools4magaox/redu/conf_ex/conf_preproc_ex.txt

  redu — coronagraphic process
    python -m tools4magaox.redu.process CONF [CONF ...]
    example: src/tools4magaox/redu/conf_ex/conf_process_ex.txt

  proc — ADI / PCA
    python -m tools4magaox.proc.ADI CONF [CONF ...]
    example: src/tools4magaox/proc/conf_ex/conf_adi_ex.txt

  proc — VIP metrics (throughput / contrast / SNR peak)
    python -m tools4magaox.proc.metrics CONF [CONF ...]
             [--throughput] [--contrast] [--source-peak]
    example: src/tools4magaox/proc/conf_ex/conf_metrics_ex.txt

  wfs_proc — open-loop wavefront reconstruction
    python -m tools4magaox.wfs_proc.loop_reconstruction CONF
    example: src/tools4magaox/wfs_proc/conf_ex/conf_loop_recon_ex.txt


Library modules (import; no CLI)
--------------------------------
  redu.centering, redu.center_spark, redu.darks,
  redu.filereads, redu.filtering
  proc.utils
  sims.camsci, sims.camtip
  constants

Typical import
  from tools4magaox.constants import CS_PLATESCALE
  from tools4magaox.redu.filereads import load_fits_stack
"""


def main(argv=None) -> int:
    """Print the package map. ``argv`` is accepted for CLI symmetry and ignored."""
    del argv  # map has no subcommands yet
    print(MAP.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
