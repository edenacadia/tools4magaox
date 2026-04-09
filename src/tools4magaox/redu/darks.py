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

def find_masterdark_for_file(file_path, camera="camsci1", redu_dir="/Volumes/magaox_bpic/redu"):
    """
    Given a FITS file (raw dark or otherwise), search redu_dir for a masterdark
    whose encoded header/filename parameters match this file.
    Returns: list of matching masterdark paths (possibly empty).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with fits.open(file_path, memmap=False) as fh:
        hdr = fh[0].header.copy()

    # extract key params from the provided file
    params = pull_hdr_params(hdr, camera, darks=False) # we don't care about SHUTTER
    params_pretty = ["{}={}".format(k, params[k]) for k in params]
    print("   ", params_pretty)
    # build glob pattern to find candidate masterdark files that include the tokens
    # look for any files under redu_dir that contain 'masterdark' and these numeric tokens
    pattern = os.path.join(redu_dir, "**", "*masterdark*.fits*")
    candidates = sorted(glob.glob(pattern, recursive=True))

    matches = []
    for cand in candidates:
        try:
            with fits.open(cand, memmap=False) as fhc:
                hdrc = fhc[0].header
            # pull same set of params from candidate header (try camera tag from candidate if available)
            cam_cand = _detect_camera_tag_from_header(hdrc)
            if cam_cand is None:
                print(f"   Could not detect camera tag from candidate {cand}")
                continue
            # extract key params from the provided file
            if cam_cand.lower() != camera.lower():
                continue
            cand_vals = pull_hdr_params(hdrc, cam_cand, darks=False)
        except Exception:
            continue

        # compare params (use isclose for floats)
        ok = True
        for k in params:
            a = params[k]
            b = cand_vals[k]
            if a is None or b is None:
                ok = False
                break
            try:
                a_f = float(a); b_f = float(b)
                if not np.isclose(a_f, b_f, atol=1e-6):
                    ok = False; break
            except Exception:
                if str(a) != str(b):
                    ok = False; break

        if ok:
            matches.append(cand)

    return matches

def find_masterdark_by_params(naxis1, naxis2, emgain, adc_speed, exptime, redu_dir="/Volumes/magaox_bpic/redu"):
    """
    Search redu_dir for masterdark files whose filename or header contain the supplied parameters.
    Returns list of matching file paths.
    """
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


    
