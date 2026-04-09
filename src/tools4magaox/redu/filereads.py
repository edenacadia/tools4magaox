# filereads.py
# 2026/04/07
# the purpose of this file is to coordinate how files and telemetry is read into the reduction pipeline
from astropy.io import fits
import numpy as np

# TODO: Make sure the cube isn't too large
def make_data_cube(file_list, dark_data, n_avg=1, n_files=-1):
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

def write_fits(data, save_path):
    data = np.asarray(data)
    hdu = fits.PrimaryHDU(data=data)
    hdu.writeto(save_path, overwrite=True)
    return save_path


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