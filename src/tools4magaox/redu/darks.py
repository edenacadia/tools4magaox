# darks.py
# 01/27/2026
# adapted from preprocess_data.py, used on AOC to make new darks
# This file is purely to make and filter dark files
# Based on MagAO-X telemetry

import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
import scipy

def pull_hdr_params(hdr, camera, darks=True):
    # we are pulling all the same values many times over
    if darks:
        vals = {
                "NAXIS1": _find_hdr_val(hdr, "NAXIS1"),
                "NAXIS2": _find_hdr_val(hdr, "NAXIS2"),
                "ROI_XCEN": _find_hdr_val(hdr, "ROI_XCEN", camera_tag=camera),
                "ROI_YCEN": _find_hdr_val(hdr, "ROI_YCEN", camera_tag=camera),
                "EMGAIN": _find_hdr_val(hdr, "EMGAIN", camera_tag=camera),
                "ADC_SPEED": _find_hdr_val(hdr, "ADC SPEED", camera_tag=camera),
                "EXPTIME": _find_hdr_val(hdr, "EXPTIME", camera_tag=camera),
                "SHUTTER": _find_hdr_val(hdr, "SHUTTER", camera_tag=camera),
            }
    else:   
        vals = {
                "NAXIS1": _find_hdr_val(hdr, "NAXIS1"),
                "NAXIS2": _find_hdr_val(hdr, "NAXIS2"),
                "ROI_XCEN": _find_hdr_val(hdr, "ROI_XCEN", camera_tag=camera),
                "ROI_YCEN": _find_hdr_val(hdr, "ROI_YCEN", camera_tag=camera),
                "EMGAIN": _find_hdr_val(hdr, "EMGAIN", camera_tag=camera),
                "ADC_SPEED": _find_hdr_val(hdr, "ADC SPEED", camera_tag=camera),
                "EXPTIME": _find_hdr_val(hdr, "EXPTIME", camera_tag=camera),
            }
    return vals


def gen_masterdark(dark_dir, data_dir, redu_dir="./", camera="camsci1", max_clip=1000, mean_clip=900):
    """
    Create a master dark by averaging all FITS files in the folder
    {data_dir}/{dark_dir}/{camera}/ provided they share the same header
    values for the keys:
      - NAXIS1
      - NAXIS2
      - ROI_XCEN and ROI_YCEN
      - (CAMSCIx) EMGAIN
      - HIERARCH CAMSCIx ADC SPEED
      - HIERARCH CAMSCIx EXPTIME
    The camera argument (e.g. "camsci1" or "camsci2") is used to
    distinguish the HIERARCH keys for each detector.
    Returns: (output_path, n_used_files)
    """
    folder = f"{data_dir}{dark_dir}/{camera}"
    fits_files = sorted(glob.glob(os.path.join(folder, "*.fits*")))
    if len(fits_files) == 0:
        raise FileNotFoundError(f"No FITS files found in {folder}")

    cam_tag = camera.upper()  # e.g. CAMSCI1 or CAMSCI2

    # reference header values from the first file
    with fits.open(fits_files[0], memmap=False) as fh0:
        ref_hdr = fh0[0].header
        # pulling reference values
        ref_vals = pull_hdr_params(ref_hdr, camera)
        demo_data = fh0[0].data.copy()
        demo_header = ref_hdr.copy()

    missing = [k for k, v in ref_vals.items() if v is None]
    if missing:
        raise KeyError(f"Could not locate required header keywords (camera={camera}): {missing} in {fits_files[0]}")
    
    # check that we start with a closing shutter
    if ref_vals['SHUTTER'] is not None:
        if str(ref_vals['SHUTTER']).strip().upper() != 'SHUT':
            print(f"Warning: first file {fits_files[0]} has SHUTTER={ref_vals['SHUTTER']} (expected 'SHUT')")

    #sum_data = np.zeros_like(demo_data, dtype=float)
    data_stack = []
    count = 0
    skipped = []

    for fn in fits_files:
        with fits.open(fn, memmap=False) as fh:
            hdr = fh[0].header
            # pulling reference values
            vals = pull_hdr_params(hdr, camera)
            # Check to make sure all values match, add to dak stack 
            if all(np.equal(vals[k], ref_vals[k]) for k in ref_vals):
                #sum_data += fh[0].data.astype(float)
                data_stack.append(fh[0].data.astype(float))
                count += 1
            else:
                skipped.append(fn)

    if count == 0:
        raise RuntimeError("No files matched the reference header parameters; master dark not created.")
    
    # calculating the master dark 
    # master_dark = sum_data / count

    # filtering before averaging the data stack
    px_median = np.mean(data_stack, axis=(1,2))
    idx_px_use = np.where(px_median < mean_clip)[0]
    print(f"Mean Value test: {np.mean(px_median)} using {len(idx_px_use)} / {len(data_stack)} files")
    
    # filtering in to see if there is a weirdly high max pixel
    px_max = np.max(data_stack, axis=(1,2))
    idx_px_use_mx = np.where(px_max < max_clip )[0]
    print(f"Max pixel test: {np.mean(px_max)} using {len(idx_px_use_mx)} / {len(data_stack)} files")

    # good indexes
    common_elements = np.intersect1d(idx_px_use_mx, idx_px_use)
    master_dark = np.mean(np.array(data_stack)[common_elements], axis=0)

    def _clean(x):
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, float) and float(x).is_integer():
            return str(int(x))
        return str(x).replace(" ", "_")
    
    # making the filename 
    fname = (
        f"masterdark_{dark_dir[:17]}_{camera}_{_clean(ref_vals['NAXIS1'])}_"
        f"{_clean(ref_vals['NAXIS2'])}_"
        f"ROI_{_clean(ref_vals['ROI_XCEN'])}_"
        f"{_clean(ref_vals['ROI_YCEN'])}_"
        f"EMGAIN_{_clean(ref_vals['EMGAIN'])}_"
        f"ADC_{_clean(ref_vals['ADC_SPEED'])}_"
        f"EXPTIME_{_clean(ref_vals['EXPTIME'])}.fits"
    )
    out_path = os.path.join(redu_dir, fname)

    hdr_out = demo_header.copy()
    hdr_out['HISTORY'] = f"Master dark created from {len(common_elements)} files (camera={camera})"
    hdr_out['DATADIR'] = f"{dark_dir}"
    if skipped:
        hdr_out['HISTORY'] = f"{hdr_out.get('HISTORY','')}  Skipped {len(skipped)} files that had different header params."

    hdu = fits.PrimaryHDU(data=master_dark.astype(demo_data.dtype), header=hdr_out)
    hdu.writeto(out_path, overwrite=True)

    return out_path, count

def _norm_key(k):
    return str(k).upper().replace("HIERARCH", "").replace(" ", "").replace("_", "").replace("'", "")

def _find_hdr_val(hdr, key, camera_tag=None):
    wanted = str(key).upper().replace(" ", "").replace("_", "")
    #print(f"Looking for key '{key}' (normalized: '{wanted}') in header with camera tag '{camera_tag}'")
    if camera_tag:
        cam_wanted = camera_tag.upper() + wanted
    else:
        cam_wanted = None
    if cam_wanted:
        # 1) camera-specific
        for k in hdr.keys():
            nk = _norm_key(k)
            if cam_wanted in nk:
                return hdr[k]
    else:
        # 2) generic key
        for k in hdr.keys():
            nk = _norm_key(k)
            if nk == wanted or nk.startswith(wanted) or wanted in nk:
                return hdr[k]
    return None

def _detect_camera_tag_from_header(hdr):
    # Look for keys that contain CAMSCI1 or CAMSCI2 (or similar)
    for k in hdr.keys():
        nk = _norm_key(k)
        if "CAMSCI1" in nk:
            return "CAMSCI1"
        if "CAMSCI2" in nk:
            return "CAMSCI2"
    return None

def _masterdark_params_match(ref, cand_vals):
    """Compare dicts from ``pull_hdr_params(..., darks=False)`` (same keys)."""
    for k in ref:
        a = ref[k]
        b = cand_vals[k]
        if a is None or b is None:
            return False
        try:
            a_f = float(a)
            b_f = float(b)
            if not np.isclose(a_f, b_f, atol=1e-6):
                return False
        except Exception:
            if str(a) != str(b):
                return False
    return True


# Columns that identify a detector setup for masterdark matching (same as
# ``darks.pull_hdr_params(..., darks=False)``). ``SHUTTER`` is omitted because
# science frames differ from darks only by shutter state.
_UNIQ_DARK_LOOKUP_COLS = (
    "camera",
    "NAXIS1",
    "NAXIS2",
    "ROI_XCEN",
    "ROI_YCEN",
    "EMGAIN",
    "ADC_SPEED",
    "EXPTIME",
)

def unique_telemetry_configs_for_dark_lookup(table, camera=None):
    """
    Collapse a per-file telemetry table to unique detector configurations.

    Drops per-exposure fields (``filename``, ``DATE_OBS``, ``PARANG``) and
    ignores ``SHUTTER`` when uniquifying, since :func:`darks.find_masterdark_for_params`
    matches the same seven numeric / ROI parameters as
    :func:`darks.find_masterdark_for_file` (shutter is not part of that match).

    Parameters
    ----------
    table : astropy.table.Table
        E.g. from :func:`fits_telemetry_table` (must contain the columns in
        ``_UNIQ_DARK_LOOKUP_COLS`` except ``camera`` may be filled via ``camera``).
    camera : str, optional
        If the table has no ``camera`` column, this value is applied to every row.

    Returns
    -------
    astropy.table.Table
        One row per unique combination of camera + NAXIS / ROI / gain / ADC / exptime.
    """
    from astropy.table import unique as table_unique

    t = table.copy()
    if "camera" not in t.colnames:
        if camera is None:
            raise ValueError(
                "telemetry table has no 'camera' column; pass camera='camsci1' (or similar)."
            )
        t["camera"] = camera
    keys = [k for k in _UNIQ_DARK_LOOKUP_COLS if k in t.colnames]
    missing = [k for k in _UNIQ_DARK_LOOKUP_COLS if k not in keys]
    if missing:
        raise ValueError(f"telemetry table missing columns required for dark lookup: {missing}")
    return table_unique(t[keys], keys=keys)


def _require_masterdark_search_dir(redu_dir):
    """Non-empty path root for recursive masterdark glob; no built-in default."""
    if redu_dir is None:
        raise ValueError(
            "redu_dir must be set (root directory to search for *masterdark*.fits*). "
            "Pass masterdark_dir from the preprocess config or explicit redu_dir=..."
        )
    s = os.path.expanduser(os.fspath(redu_dir)).strip()
    if not s:
        raise ValueError("redu_dir must be a non-empty path.")
    return s


def lookup_masterdarks_from_telemetry_table(
    table, redu_dir=None, camera=None
):
    """
    For each unique detector configuration in ``table``, call
    :func:`darks.find_masterdark_for_params`.

    Parameters
    ----------
    table : astropy.table.Table
        From :func:`fits_telemetry_table` or compatible.
    redu_dir : str
        Root directory passed to :func:`darks.find_masterdark_for_params` (search for
        ``*masterdark*.fits*`` under this path). Required; use e.g. ``masterdark_dir``
        from preprocess config.
    camera : str, optional
        Used only if ``table`` has no ``camera`` column.

    Returns
    -------
    list of dict
        One entry per unique config, with keys matching the masterdark match
        dict plus ``masterdark_paths`` (list of str).
    """
    try:
        from .darks import find_masterdark_for_params
    except ImportError:
        from darks import find_masterdark_for_params

    redu_dir = _require_masterdark_search_dir(redu_dir)
    uniq = unique_telemetry_configs_for_dark_lookup(table, camera=camera)
    results = []
    for row in uniq:
        cam = row["camera"]
        params = {
            "NAXIS1": row["NAXIS1"],
            "NAXIS2": row["NAXIS2"],
            "ROI_XCEN": row["ROI_XCEN"],
            "ROI_YCEN": row["ROI_YCEN"],
            "EMGAIN": row["EMGAIN"],
            "ADC_SPEED": row["ADC_SPEED"],
            "EXPTIME": row["EXPTIME"],
        }
        paths = find_masterdark_for_params(params, cam, redu_dir=redu_dir)
        results.append(
            {
                "camera": cam,
                **params,
                "masterdark_paths": paths,
            }
        )
    return results

def merge_file_table_with_darks(
    file_table,
    dark_lookup_results,
    dark_col="masterdark_path",
):
    """
    Add a masterdark path column to a per-file telemetry table.

    Parameters
    ----------
    file_table : astropy.table.Table
        Table with (at minimum) the columns:
        ``camera``, ``NAXIS1``, ``NAXIS2``, ``ROI_XCEN``, ``ROI_YCEN``,
        ``EMGAIN``, ``ADC_SPEED``, ``EXPTIME``.
    dark_lookup_results : list[dict]
        Output from a dark lookup step, with one dict per *unique* configuration.
        Each dict must contain the same identifying keys as above plus
        ``masterdark_paths`` (list of str). The *first* path is used.
    dark_col : str
        Name of the new column to add to ``file_table``.

    Returns
    -------
    astropy.table.Table
        Copy of ``file_table`` with an added ``dark_col`` column containing the
        first matching masterdark path (or ``""`` if no match).
    """
    required = (
        "camera",
        "NAXIS1",
        "NAXIS2",
        "ROI_XCEN",
        "ROI_YCEN",
        "EMGAIN",
        "ADC_SPEED",
        "EXPTIME",
    )
    missing = [c for c in required if c not in file_table.colnames]
    if missing:
        raise ValueError(f"file_table missing required columns: {missing}")

    # Map (camera + config params) -> first masterdark path
    def _key(d):
        return (
            str(d["camera"]),
            d["NAXIS1"],
            d["NAXIS2"],
            d["ROI_XCEN"],
            d["ROI_YCEN"],
            d["EMGAIN"],
            d["ADC_SPEED"],
            d["EXPTIME"],
        )

    cfg_to_dark = {}
    for d in dark_lookup_results:
        if any(k not in d for k in required) or "masterdark_paths" not in d:
            continue
        paths = d.get("masterdark_paths") or []
        cfg_to_dark[_key(d)] = paths[0] if len(paths) > 0 else ""

    out = file_table.copy()
    out[dark_col] = [
        cfg_to_dark.get(
            (
                str(row["camera"]),
                row["NAXIS1"],
                row["NAXIS2"],
                row["ROI_XCEN"],
                row["ROI_YCEN"],
                row["EMGAIN"],
                row["ADC_SPEED"],
                row["EXPTIME"],
            ),
            "",
        )
        for row in out
    ]
    return out

def find_masterdark_for_params(params, camera, redu_dir=None):
    """
    Search ``redu_dir`` for masterdark FITS whose headers match ``params``.

    Parameters
    ----------
    params : dict
        Same keys as ``pull_hdr_params(hdr, camera, darks=False)``: NAXIS1,
        NAXIS2, ROI_XCEN, ROI_YCEN, EMGAIN, ADC_SPEED, EXPTIME.
    camera : str
        Detector id, e.g. ``"camsci1"``.
    redu_dir : str
        Root directory to search (recursive glob for ``*masterdark*.fits*``).
        Required (no default); set via preprocess ``masterdark_dir`` or pass explicitly.

    Returns
    -------
    list of str
        Paths to matching masterdark files (possibly empty).
    """
    redu_dir = _require_masterdark_search_dir(redu_dir)
    params_pretty = ["{}={}".format(k, params[k]) for k in params]
    print("   ", params_pretty)
    pattern = os.path.join(redu_dir, "**", "*masterdark*.fits*")
    candidates = sorted(glob.glob(pattern, recursive=True))

    matches = []
    cam = camera.lower()
    for cand in candidates:
        try:
            with fits.open(cand, memmap=False) as fhc:
                hdrc = fhc[0].header
            cam_cand = _detect_camera_tag_from_header(hdrc)
            if cam_cand is None:
                print(f"   Could not detect camera tag from candidate {cand}")
                continue
            if cam_cand.lower() != cam:
                continue
            cand_vals = pull_hdr_params(hdrc, cam_cand, darks=False)
        except Exception:
            continue

        if _masterdark_params_match(params, cand_vals):
            matches.append(cand)

    return matches


def find_masterdark_for_file(file_path, camera="camsci1", redu_dir=None):
    """
    Given a FITS file (raw dark or otherwise), search redu_dir for a masterdark
    whose encoded header/filename parameters match this file.
    Returns: list of matching masterdark paths (possibly empty).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with fits.open(file_path, memmap=False) as fh:
        hdr = fh[0].header.copy()

    params = pull_hdr_params(hdr, camera, darks=False)
    return find_masterdark_for_params(params, camera, redu_dir=redu_dir)

def find_masterdark_by_params(naxis1, naxis2, emgain, adc_speed, exptime, redu_dir=None):
    """
    Search redu_dir for masterdark files whose filename or header contain the supplied parameters.
    Returns list of matching file paths.
    """
    redu_dir = _require_masterdark_search_dir(redu_dir)
    pattern = os.path.join(redu_dir, "**", "*masterdark*.fits*")
    candidates = sorted(glob.glob(pattern, recursive=True))
    matches = []
    for cand in candidates:
        try:
            with fits.open(cand, memmap=False) as fh:
                hdr = fh[0].header
        except Exception:
            continue

        vals = {
            "NAXIS1": _find_hdr_val(hdr, "NAXIS1"),
            "NAXIS2": _find_hdr_val(hdr, "NAXIS2"),
            "EMGAIN": _find_hdr_val(hdr, "EMGAIN"),
            "ADC":    _find_hdr_val(hdr, "ADC SPEED"),
            "EXPTIME":_find_hdr_val(hdr, "EXPTIME"),
        }

        ok = True
        for (a, b) in zip([naxis1, naxis2, emgain, adc_speed, exptime],
                          [vals['NAXIS1'], vals['NAXIS2'], vals['EMGAIN'], vals['ADC'], vals['EXPTIME']]):
            if a is None or b is None:
                ok = False; break
            try:
                if not np.isclose(float(a), float(b), atol=1e-6):
                    ok = False; break
            except Exception:
                if str(a) != str(b):
                    ok = False; break
        if ok:
            matches.append(cand)
    return matches
# cross check that every file has a dark file associated, so sort based on 

def validate_dark_configs(config_df, output_report=None):
    """
    Cross-check that for every configuration with SHUTTER='OPEN', 
    there exists an identical configuration with SHUTTER='SHUT'.
    
    Parameters
    ----------
    config_df : pd.DataFrame
        DataFrame from scan_data_dir_for_configs()
    output_report : str or None
        If provided, save validation report to this file
    
    Returns
    -------
    dict with keys:
        'missing_darks': pd.DataFrame (configs missing a SHUT counterpart)
        'all_valid': bool (True if all OPEN configs have SHUT pairs)
        'summary': str (human-readable summary)
    """
    # Identify OPEN and SHUT configs
    open_configs = config_df[config_df['SHUTTER'] == 'OPEN'].copy()
    shut_configs = config_df[config_df['SHUTTER'] == 'SHUT'].copy()
    
    # Parameters to match (exclude 'folder', 'camera', 'SHUTTER', 'n_files')
    match_cols = ['NAXIS1', 'NAXIS2', 'EMGAIN', 'ADC_SPEED', 'EXPTIME', 'ROI_XCEN', 'ROI_YCEN']
    
    missing_darks = []
    
    for idx, open_row in open_configs.iterrows():
        # Look for matching SHUT config
        mask = True
        for col in match_cols:
            mask = mask & (shut_configs[col] == open_row[col])
        
        matching_shut = shut_configs[mask]
        
        if len(matching_shut) == 0:
            missing_darks.append({
                'Camera': open_row['camera'],
                'NAXIS1': open_row['NAXIS1'],
                'NAXIS2': open_row['NAXIS2'],
                'EMGAIN': open_row['EMGAIN'],
                'ADC_SPEED': open_row['ADC_SPEED'],
                'EXPTIME': open_row['EXPTIME'],
                'ROI_XCEN': open_row['ROI_XCEN'],
                'ROI_YCEN': open_row['ROI_YCEN'],
                'Folder': open_row['folder']
            })
    
    all_valid = len(missing_darks) == 0
    missing_df = pd.DataFrame(missing_darks)
    
    # Build summary
    summary = f"Validation Report:\n"
    summary += f"{'='*80}\n"
    summary += f"Total OPEN configs:   {len(open_configs)}\n"
    summary += f"Total SHUT configs:   {len(shut_configs)}\n"
    summary += f"Missing SHUT darks:   {len(missing_darks)}\n"
    summary += f"{'='*80}\n\n"
    
    if all_valid:
        summary += "✓ All OPEN configurations have matching SHUT darks!\n"
    else:
        summary += f"✗ {len(missing_darks)} OPEN config(s) missing matching SHUT dark(s):\n\n"
        summary += missing_df.to_string(index=False)
        summary += "\n"
    
    if output_report:
        with open(output_report, 'w') as f:
            f.write(summary)
        print(f"Report saved to {output_report}")
    
    print(summary)
    
    return {'missing_darks': missing_df, 'all_valid': all_valid, 'summary': summary}


    
