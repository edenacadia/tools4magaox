# preprocess.py
# step 1 in the reduction pipeline
# this makes our unsats
# running this on a directory will make a master unsats in that directory 
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from scipy import ndimage
import scipy

from constants import *
import darks as md
import filtering as fl
import filereads as fr
import centering as ct

# philosophy going forward is to have all the funtions in a tool script, this one is just using them

# STEP 1
def make_clean_cube(unsat_files, dark_data, camsci_grid, pct_cut, plt_path):
    # This function pulls in ALL unsat files and makes a clean cube
    # cleaning a cube is only filtering out the lowest % max values
    unsats_data_cube = fr.make_data_cube(unsat_files, dark_data)
    unsats_data_flat = unsats_data_cube.reshape(unsats_data_cube.shape[0], unsats_data_cube.shape[1]**2)
    unsats_data  = Field(unsats_data_flat, camsci_grid)
    unsats_data_filtered = fl.max_filter(unsats_data, perc=pct_cut, save_plot=True, plot_path=plt_path)
    return unsats_data_filtered

# STEP 2
# TODO: get this all into the centering.py file.... all the stuff that matters
def make_centered_cube(unsats_data_filtered, camsci_grid, plot_path, save_plot=True, DAO=True):

    if DAO == True:
        unsats_c_cube, shifts = _center_DAO(unsats_data_filtered)
    else:
        unsats_c_cube, shifts = _center_gauss(unsats_data_filtered)
    
    if save_plot:
        shifts_arr = np.array(shifts)
        plt.plot(shifts_arr[:, 0], label="y shift")
        plt.plot(shifts_arr[:, 1], label="x shift")
        plt.legend()
        plt.savefig(f"{plot_path}_unsats_shifts.png")
        plt.close()
        print(f"   Saved unsats shifts to {plot_path}_unsats_shifts.png")
    unsats_c_flat = unsats_c_cube.reshape(unsats_c_cube.shape[0], unsats_c_cube.shape[1]**2)
    unsats_centerd = Field(unsats_c_flat, camsci_grid)
    return unsats_centerd

def _center_DAO(unsats_data_filtered):
    sources_dict = rc.check_cube_sources(unsats_data_filtered.shaped)
    unsats_c_cube, idx, shifts = rc.shift_cube_soures(unsats_data_filtered.shaped, sources_dict['sources'], sources_dict['bad_idx'], sources_dict['multi_idx'])
    return unsats_c_cube, shifts

def _center_gauss(unsats_data_filtered):
    # TODO: build a guassian fitter
    unsats_c_cube = None
    shifts = None
    return unsats_c_cube, shifts


# main file for running through the preprocessing pipeline
def main(obs_path, unsats_dir, redu_path, camera="camsci1", pct_cut=10, plot=False, max_files=-1):
    # specific folder for ther redu dir
    redu_dir = f"{redu_path}{unsats_dir}"
    plt_path = f"{redu_dir}/{camera}"
    # if it doesn't already exist, make it
    if not os.path.exists(redu_dir):
        os.mkdir(redu_dir)

    #### Part 1
    # find all unsat files
    unsat_files = glob.glob(f"{obs_path}{unsats_dir}/{camera}/*")
    if max_files > len(unsat_files): max_files = len(unsat_files)
    unsat_files = sorted(unsat_files)[:max_files]
    print("   Found unsat files: ", len(unsat_files))
    # fild the dark associated with these files 
    dark_file = md.find_masterdark_for_file(unsat_files[1], camera)
    if len(dark_file) == 0:
        print("   WARNING: No master dark found for these unsats, exiting")
        return False
    dark_data = fits.open(dark_file[0])[0].data.astype(float)
    dw, dh = dark_data.shape
    camsci_grid = make_pupil_grid(dw, diameter=dw*plate_scale)
    
    # 1. unsats cube
    filtered_path = f"{redu_dir}/{camera}_unsats_cube.fits"
    if os.path.exists(filtered_path):
        print(f"   1. unsats cube: reading {filtered_path}")
        unsats_filtered = read_field(filtered_path, fmt="fits")
    else:
        # 1/28/2026 addition: filtering on the top 90% of frames
        unsats_filtered = make_clean_cube(unsat_files, dark_data, camsci_grid, pct_cut, plt_path)
        print("   1. unsats cube: writing")
        write_field(unsats_filtered, filtered_path, fmt="fits")

    # 2. center unsats
    unsats_c_path = f"{redu_dir}/{camera}_unsats_centered.fits"
    print(unsats_c_path)
    if os.path.exists(unsats_c_path):
        print(f"   2. centered cube: reading {unsats_c_path}")
        unsats_centerd = read_field(unsats_c_path, fmt="fits")
    else:
        # TODO: make type of centering configurable
        unsats_centered = make_centered_cube(unsats_filtered, camsci_grid, plt_path)
        print(f"   2. centered cube: writing {unsats_c_path}")
        write_field(unsats_centered, unsats_c_path, fmt="fits")
    
    # Frame select unsats 
    # unsats_filtered = clean_unsats(unsats_centerd, camsci_grid)
    # we've decided this is not needed for the unsats
    
    # 3. save the reference image 
    ref_img_path = f"{redu_dir}/{camera}_average_image.fits"
    print(ref_img_path)
    if os.path.exists(ref_img_path):
        # yay! all done
        print("   3. reference image: already exists, skipping")
        return True
    else:
        unsats_centered_avg = np.mean(unsats_centered, axis=0)
        print("   3. reference image: writing")
        write_field(unsats_centered_avg, ref_img_path, fmt="fits")
    return True


##############################################################

def inter_dir():
    obs_path = "/Volumes/magaox_bpic/edenmcewen@arizona.edu/"
    redu_path = "/Volumes/magaox_bpic/redu/"
    unsats_dir = "2025-12-02_084046_beta_pic_coron_lg_unsats"
    camera = "camsci1"
    main(obs_path, unsats_dir, redu_path, camera=camera, pct_cut=20, plot=True)

def iter_unsats():
    obs_path = "/Volumes/magaox_bpic/edenmcewen@arizona.edu/"
    redu_path = "/Volumes/magaox_bpic/redu/"
    unsats_dirs = ["2025-12-01_081206_beta_pic_piaa_unsats",
               "2025-12-02_065035_beta_pic_coron_lg_unsats",
               "2025-12-02_084046_beta_pic_coron_lg_unsats",
               "2025-12-03_071901_beta_pic_piaa_unsats",
               "2025-12-03_081533_beta_pic_piaa_unsats"]
    cameras = ["camsci1", "camsci2"]
    for unsats_dir in unsats_dirs:
        for camera in cameras:
            print(f"=> Processing {unsats_dir} {camera}")
            try:
                #unsat_files = glob.glob(f"{obs_path}{unsats_dir}/{camera}/*")
                #print("   Found unsat files: ", len(unsat_files))
                main(obs_path, unsats_dir, redu_path, camera=camera, pct_cut=20, plot=True)
            except Exception as e:
                print(f"Error processing {unsats_dir} {camera}: {e}")

def iter_unsats_nospark():
    obs_path = "/Volumes/magaox_bpic/edenmcewen@arizona.edu/"
    redu_path = "/Volumes/magaox_bpic/redu/"
    unsats_nospark_dirs = [
                "2025-12-01_081737_beta_pic_piaa_unsats_nospark",
                "2025-12-02_065549_beta_pic_coron_lg_unsats_nospark",
                "2025-12-03_072425_beta_pic_piaa_unsats_nospark",
                "2025-12-03_082054_beta_pic_piaa_unsats_nospark"]
    cameras = ["camsci1", "camsci2"]
    for unsats_dir in unsats_nospark_dirs:
        for camera in cameras:
            print(f"Processing {unsats_dir} {camera}")
            try:
                main(obs_path, unsats_dir, redu_path, camera=camera, pct_cut=20, plot=True)
            except Exception as e:
                print(f"Error processing {unsats_dir} {camera}: {e}")

########################################################

if __name__ == "__main__":
    inter_dir()
    #iter_unsats()
    #iter_unsats_nospark()