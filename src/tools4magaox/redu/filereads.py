# filereads.py
# 2026/04/07
# the purpose of this file is to coordinate how files and telemetry is read into the reduction pipeline
import glob
import logging
import os
from tqdm import trange, tqdm
from astropy.table import Table
import darks as md
from darks import _detect_camera_tag_from_header

from astropy.io import fits
import numpy as np
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_REDU_ASCII_FMT = "ascii.commented_header"


def _load_fits_primary_float32(path):
    """Load primary HDU data as float32 (reduces RAM vs float64 cubes)."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)

def _load_fits_primary_float64(path):
    """Load primary HDU data as float64 (reduces RAM vs float64 cubes)."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)

def _save_fits_primary_float32(data, path):
    """Save primary HDU data as float32 (reduces RAM vs float64 cubes)."""
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32)).writeto(path, overwrite=True)

def _save_fits_primary_float64(data, path):
    """Save primary HDU data as float64 (reduces RAM vs float64 cubes)."""
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float64)).writeto(path, overwrite=True)

def read_redu_table(path):
    """Load a pipeline ASCII metadata table (commented header)."""
    return Table.read(path, format=_REDU_ASCII_FMT)


def write_redu_table(table, path):
    """Write a pipeline ASCII metadata table; creates parent directory if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    table.write(path, format=_REDU_ASCII_FMT, overwrite=True)


def find_camera_files(obs_path, obs_dir, camera="camsci1", max_files=-1):
    """
    Find FITS files in ``{obs_path}{obs_dir}/{camera}/`` and return sorted paths.
    """
    file_paths = glob.glob(f"{obs_path}{obs_dir}/{camera}/*")
    if max_files < 0:
        return sorted(file_paths)
    if max_files > len(file_paths):
        max_files = len(file_paths)
    return sorted(file_paths)[:max_files]


def resolve_masterdark_search_dir(redu_path, masterdark_dir):
    """Directory tree to search for ``*masterdark*.fits*`` (recursive)."""
    if masterdark_dir is not None:
        s = os.path.expanduser(os.fspath(masterdark_dir)).strip()
        if s:
            return s
    return os.path.expanduser(os.fspath(redu_path)).strip()


def attach_masterdarks(file_table, dark_search_dir, camera):
    """
    Add ``masterdark_path`` to each row using darks matched from telemetry.
    """
    unique_configs = md.unique_telemetry_configs_for_dark_lookup(file_table, camera=camera)
    dark_dictionary = md.lookup_masterdarks_from_telemetry_table(
        unique_configs,
        redu_dir=dark_search_dir,
        camera=camera,
    )
    for config in dark_dictionary:
        if len(config["masterdark_paths"]) == 0:
            log.warning("No master dark found for configuration: %s", config)
    return md.merge_file_table_with_darks(file_table, dark_dictionary)


def init_process_output_table():
    """
    Zero-row manifest of process-pipeline outputs to populate as steps run.
    """
    out = Table()
    out["step"] = np.array([], dtype=int)
    out["product"] = np.array([], dtype="U64")
    out["source_filename"] = np.array([], dtype="U256")
    out["path"] = np.array([], dtype="U1024")
    out["status"] = np.array([], dtype="U32")
    out["notes"] = np.array([], dtype="U256")
    return out


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
    dark_data = np.asarray(dark_data, dtype=np.float32)
    n_files = len(file_list) if n_files == -1 else n_files
    n_avg_data = n_files // n_avg
    avg_data = np.zeros((n_avg_data, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    for i in range(n_avg_data):
        #if i % 100 == 0:
        #    print(f"   Processing file {i*n_avg} to {i*n_avg + n_avg} out of {n_files}")
        for j in range(n_avg):
            with fits.open(file_list[i * n_avg + j], memmap=False) as hdul:
                demo_data = hdul[0].data
            avg_data[i] += np.asarray(demo_data, dtype=np.float32) - dark_data
        avg_data[i] /= n_avg
    return avg_data

def make_data_cube(file_list, dark_data, n_files=-1):
    dark_data = np.asarray(dark_data, dtype=np.float32)
    n_files = len(file_list) if n_files == -1 else n_files
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    for i in range(n_files):
        with fits.open(file_list[i], memmap=False) as hdul:
            demo_data = hdul[0].data
        data_cube[i] = np.asarray(demo_data, dtype=np.float32) - dark_data
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
    data_cube = np.zeros((n_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
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
            data_cube[i] = np.asarray(test_data, dtype=np.float32)
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
    dark_data = np.asarray(dark_data, dtype=np.float32)
    # make an empty data cube 
    data_cube = np.zeros((n_coadd_files, dark_data.shape[0], dark_data.shape[1]), dtype=np.float32)
    parang_cube = np.zeros((n_coadd_files, coadd))
    time_stamp_cube = np.zeros((n_coadd_files, coadd))

    for i in range(n_coadd_files):
        for j in range(coadd):
            fh = fits.open(file_list[n_start+i*coadd+j])
            test_data = fh[0].data
            data_cube[i] += np.asarray(test_data, dtype=np.float32) - dark_data
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

#################### time functions ####################

def _coerce_times_to_datetime64(times):
    """
    Convert FITS-style DATE-OBS values (usually ISO-8601 strings) into a
    numpy datetime64 array so matplotlib plots a true time axis.
    """
    arr = np.asarray(times)
    if arr.size == 0:
        return arr
    # If already datetime64, keep it.
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr
    # Decode bytes -> str, and normalize common sentinels.
    if arr.dtype.kind in ("S", "a"):
        arr = np.char.decode(arr, "utf-8", errors="ignore")
    if arr.dtype.kind == "O":
        arr = np.array(
            [
                (x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x)
                for x in arr
            ],
            dtype=object,
        )
    # Try vectorized parse into datetime64; replace failures with NaT.
    out = np.full(arr.shape, np.datetime64("NaT"), dtype="datetime64[us]")
    for i, x in np.ndenumerate(arr):
        if x is None:
            continue
        if isinstance(x, (np.datetime64,)):
            out[i] = x.astype("datetime64[us]")
            continue
        s = str(x).strip()
        if s in ("", "None", "nan", "NaN", "-1"):
            continue
        try:
            # numpy.datetime64 has no timezone support. If DATE-OBS includes a timezone
            # (e.g. "Z" or "+00:00"), normalize to UTC and drop tzinfo before casting.
            if s.endswith("Z"):
                s_iso = s[:-1] + "+00:00"
            else:
                s_iso = s
            dt = datetime.fromisoformat(s_iso)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            out[i] = np.datetime64(dt).astype("datetime64[us]")
        except Exception:
            # Leave as NaT if unparseable
            pass
    return out