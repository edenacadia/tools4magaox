# filereads.py
# 2026/04/07
# the purpose of this file is to coordinate how files and telemetry is read into the reduction pipeline
import os
from tqdm import trange, tqdm
from astropy.table import Table
from darks import _detect_camera_tag_from_header

from astropy.io import fits
import numpy as np


def pull_hdr_params(hdr, camera, darks=True):
    vals = {
            "DATE_OBS": hdr.get("DATE-OBS"),
            "PARANG": hdr.get("PARANG"),
            "NAXIS1": _find_hdr_val(hdr, "NAXIS1"),
            "NAXIS2": _find_hdr_val(hdr, "NAXIS2"),
            "ROI_XCEN": _find_hdr_val(hdr, "ROI_XCEN", camera_tag=camera),
            "ROI_YCEN": _find_hdr_val(hdr, "ROI_YCEN", camera_tag=camera),
            "EMGAIN": _find_hdr_val(hdr, "EMGAIN", camera_tag=camera),
            "ADC_SPEED": _find_hdr_val(hdr, "ADC SPEED", camera_tag=camera),
            "EXPTIME": _find_hdr_val(hdr, "EXPTIME", camera_tag=camera),
            "SHUTTER": _find_hdr_val(hdr, "SHUTTER", camera_tag=camera),
        }
    return vals

def fits_telemetry_table(file_paths, camera=None):
    """
    Build an Astropy ``Table`` with one row per FITS file: filename, acquisition
    time (``DATE-OBS``), ``PARANG``, and camera/detector keywords from the header.

    Camera-specific keywords (``ROI_*``, ``EMGAIN``, ``ADC SPEED``, ``EXPTIME``,
    ``SHUTTER``) are resolved the same way as in ``darks.pull_hdr_params``.

    Parameters
    ----------
    file_paths : list of str or path-like, or a single path
        FITS files to summarize.
    camera : str, optional
        Detector id, e.g. ``"camsci1"``. If omitted, each file's camera is
        inferred from HIERARCH keys when possible; otherwise this value must be
        set for files without a detectable tag.

    Returns
    -------
    astropy.table.Table
        Columns: ``filename``, ``camera``, ``DATE_OBS``, ``PARANG``, ``NAXIS1``,
        ``NAXIS2``, ``ROI_XCEN``, ``ROI_YCEN``, ``EMGAIN``, ``ADC_SPEED``,
        ``EXPTIME``, ``SHUTTER``. Missing ``PARANG`` is stored as NaN; missing
        optional keywords are None.
    """

    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    rows = []
    for fp in file_paths:
        fp = os.fspath(fp)
        name = os.path.basename(fp)
        with fits.open(fp, memmap=False) as fh:
            hdr = fh[0].header
        cam = camera
        if cam is None:
            det = _detect_camera_tag_from_header(hdr)
            cam = det.lower() if det else None
        if cam is None:
            raise ValueError(
                f"Could not infer camera for {fp}; pass camera='camsci1' or 'camsci2'."
            )
        vals = pull_hdr_params(hdr, cam, darks=True)
        rows.append(
            {
                "filename": name,
                "camera": cam,
                "DATE_OBS": vals["DATE_OBS"],
                "PARANG": vals["PARANG"],
                "NAXIS1": vals["NAXIS1"],
                "NAXIS2": vals["NAXIS2"],
                "ROI_XCEN": vals["ROI_XCEN"],
                "ROI_YCEN": vals["ROI_YCEN"],
                "EMGAIN": vals["EMGAIN"],
                "ADC_SPEED": vals["ADC_SPEED"],
                "EXPTIME": vals["EXPTIME"],
                "SHUTTER": vals["SHUTTER"],
            }
        )
    return Table(rows)

def write_fits(data, save_path):
    data = np.asarray(data)
    hdu = fits.PrimaryHDU(data=data)
    hdu.writeto(save_path, overwrite=True)
    return save_path

# TODO: Make sure the cube isn't too large
def make_data_avg_cube(file_list, dark_data, n_avg=1, n_files=-1):
    n_files = len(file_list) if n_files == -1 else n_files
    n_avg_data = n_files // n_avg
    avg_data = np.zeros((n_avg_data, dark_data.shape[0], dark_data.shape[1]))
    for i in range(n_avg_data):
        #if i % 100 == 0:
        #    print(f"   Processing file {i*n_avg} to {i*n_avg + n_avg} out of {n_files}")
        for j in range(n_avg):
            demo_data = fits.open(file_list[i+j])[0].data
            avg_data[i] += demo_data.astype(float) - dark_data
        avg_data[i] /=n_avg
    return avg_data

def make_data_cube(file_list, dark_data, n_files=-1):
    n_files = len(file_list) if n_files == -1 else n_files
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]))
    for i in range(n_files):
        demo_data = fits.open(file_list[i])[0].data
        data_cube[i] = demo_data.astype(float) - dark_data
    return data_cube

# TODO: Make a cube and keep relevant telemetry 

def make_science_cube(file_list, dark_data, n_files=-1, n_start=0):
    """
    :param file_list: list of the full path to files
    :param dark_data: a cube of the darks associated with these camera parameters
    :param coadd: how many frames we need to add
    :param n_files: how many files total
    :param n_start: where to start in the list of files
    """
    # logic to make sure the cubes aren't too bug
    n_files = len(file_list) if n_files > len(file_list) else n_files
    n_files = len(file_list) if n_files == -1 else n_files
    # make an empty data cube 
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]))
    parang_cube = np.zeros(n_files)
    time_cube = np.zeros(n_files, dtype='datetime64[us]')

    for i in range(n_files):
        with fits.open(file_list[n_start+i]) as fh:
            test_data = fh[0].data
            # get the parang
            try:
                pg_ang = fh[0].header["PARANG"]
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with Parang in file:  {file_list[n_start+i]}")
                pg_ang = -1
            # get the file time 
            try:
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with timestamp in file:  {file_list[n_start+i]}")
                time_stamp = -1
            parang_cube[i] = pg_ang
            time_cube[i] = time_stamp
            data_cube[i] = test_data
    return data_cube, parang_cube, time_cube

def make_science_cube_coadd(file_list, dark_data, coadd=10, n_files=-1, n_start=0):
    """
    Docstring for make_science_cube
    
    :param file_list: list of the full path to files
    :param dark_data: a cube of the darks associated with these camera parameters
    :param coadd: how many frames we need to add
    :param n_files: how many files total
    :param n_start: where to start in the list of files
    """
    # logic to make sure the cubes aren't too bug
    n_files = len(file_list) if n_files > len(file_list) else n_files
    n_files = len(file_list) if n_files == -1 else n_files
    n_coadd_files = n_files // coadd
    # make an empty data cube 
    data_cube = np.zeros((n_coadd_files, dark_data.shape[0], dark_data.shape[1]))
    parang_cube = np.zeros((n_coadd_files, coadd))
    time_stamp_cube = np.zeros((n_coadd_files, coadd))

    for i in range(n_coadd_files):
        for j in range(coadd):
            fh = fits.open(file_list[n_start+i*coadd+j])
            test_data = fh[0].data
            data_cube[i] += test_data.astype(float) - dark_data
            try:
                pg_ang = fh[0].header["PARANG"]
                time_stamp = fh[0].header["DATE-OBS"]
            except:
                print(f"Issue with Parang in file:  {file_list[i*coadd+j]}")
                pg_ang = -1
            parang_cube[i,j] = pg_ang
            time_stamp_cube[i,j] = time_stamp
            fh.close()
        data_cube[i] /= coadd
    return data_cube, parang_cube


# We don't want to do this by file number anymore, we might have large chunks of time missing
# this is the best way to make sure the images themselves don't get blurred
def coadd_by_time(data, time_cube, parang_cube, time_coadd=10):
    """
    We want to stack cubes by time, suggestion is 10s
    :param data: Description
    :param time_cube: Description
    :param parang_cube: Description
    """
    # TODO
    pass


############## File checking functions ###############

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