# process.py
# this file processes the larger format data cubes
# TODO: assumed to be coronagraphic, for now
import numpy as np
import scipy
import scipy.signal

from constants import *
from hcipy import *


# make a mask for the sparkle cube
# guassian like masks to crop out the sparkles
def make_sparkle_mask(spark_ang, spark_sep, dark_data, camsci_grid, width_r_ld=8, width_phi_rld=3):
    # building up the elipises 
    spark_sep_as = 0.92 * rc.spark_to_dist(spark_sep, wavelength=908e-9) #as
    angs = np.deg2rad(np.array([spark_ang-90*i for i in range(4)]))
    centers_x = np.array([spark_sep_as * np.cos(ang) for ang in angs])
    centers_y = np.array([spark_sep_as * np.sin(ang) for ang in angs])
    centers = list(zip(centers_y, centers_x))
    width_r = LAM_D_AS * width_r_ld
    width_phi = LAM_D_AS * width_phi_rld
    diameters = [(width_r, width_phi) for i in range(4)]
    # iteratively build up the mask
    ap_total = np.zeros_like(dark_data.ravel())
    for i in range(4):
        ap_temp = evaluate_supersampled(make_elliptical_aperture(diameters[i], centers[i], angs[i]+np.pi/2), camsci_grid, 4)
        ap_total += ap_temp
    # soften these with the tukey window
    taper = TukeyWindow(alpha=0.4)
    window_data = taper((27, 27))
    sof_mask = scipy.signal.convolve2d(ap_total.reshape(camsci_grid.dims), window_data, mode = 'same')
    sof_mask/= np.max(sof_mask)
    soft_mask_field = Field(sof_mask.ravel(), camsci_grid)
    return soft_mask_field