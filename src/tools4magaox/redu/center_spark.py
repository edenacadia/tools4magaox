# May 12th 2026
# This file is specifically for functions for centering coron data
# It has functions for:
#   - simulating the sparkles given parameters
#   - making a mask based on the expected location of the sparkles
#   - cross correlating a reference image with data for shifts

from hcipy import *
from scipy import ndimage
import numpy as np
import scipy.signal
from tools4magaox.constants import *

########## Dang I guess I'm using HCIPy #############

def hpf(data, sigma):
    '''
    data here is a Field object
    '''
    # SYH
    return Field((data.shaped - ndimage.gaussian_filter(data.shaped,sigma)).ravel(), data.grid)

def lpf(data,sigma):
    # SYH
    return Field((ndimage.gaussian_filter(data.shaped,sigma)).ravel(),data.grid)

def make_camsci_grid(ex_data):
    '''
    Take the shape of data from camsci
    Make an HCIPy compatible grid using the known platescale
    '''
    dw, dh = ex_data.shape
    camsci_grid = make_pupil_grid(dw, diameter=dw*CS_PLATESCALE)
    return camsci_grid

################# Create Mask ######################

def spark_to_dist(n_airy, wavelength=900e-9, D=6.5, fudge_factor=0.92):
    '''
    Given a lamda/D distance, calculate to an arcsecond distance
    A fudge factor is needed because these distances aren't actually 1:1
    '''
    as_factor = 206265 
    sp_res = wavelength / D
    dist_lambdaD = n_airy * as_factor * sp_res 
    return dist_lambdaD * fudge_factor


def make_sparkle_mask(spark_ang, spark_sep, ex_data, width_r_ld=8, width_phi_rld=3, wavelength=908e-9):
    '''
    Making a mask based on expected location of the sparkles
    '''
    camsci_grid = make_camsci_grid(ex_data)
    # building up the elipises 
    spark_sep_as = spark_to_dist(spark_sep, wavelength=wavelength) #as
    angs = np.deg2rad(np.array([spark_ang-90*i for i in range(4)]))
    centers_x = np.array([spark_sep_as * np.cos(ang) for ang in angs])
    centers_y = np.array([spark_sep_as * np.sin(ang) for ang in angs])
    centers = list(zip(centers_y, centers_x))
    width_r = LAM_D_AS * width_r_ld
    width_phi = LAM_D_AS * width_phi_rld
    diameters = [(width_r, width_phi) for i in range(4)]
    # iteratively build up the mask
    ap_total = np.zeros_like(ex_data.ravel())
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


############ Cross correlate masked unsat with masked ref image ############

def hpf_array(data, sigma):
    '''
    data here is a 2D array
    '''
    return data - ndimage.gaussian_filter(data,sigma)

    # want to convolve the data cube with the masked sparkles... 
def register_files_fft(science_cube, ref_image, mask, grid):
    '''
    Science_cube is a 3D array
    Ref_image is a 2D array
    Mask is a Field object
    Grid is a Grid object
    '''
    #reference_image = np.median(science_cube, axis=0)
    # set up fourier transforms 
    ft_in = FastFourierTransform(grid)
    fft_grid = ft_in.output_grid
    sample_grid = make_pupil_grid(480, 20.0) # dims, diameter | subsampling, pixels 
    ft_out = MatrixFourierTransform(sample_grid, ft_in.output_grid)
    # high pass filter the data cube
    data_cube_hpf = Field(np.array([hpf_array(dc, sigma=10) for dc in science_cube]), grid)

    # Frames shifted
    ref_kernel = np.conj(ft_in.forward(ref_image + 0j))
    fourier_sci = ft_in.forward(data_cube_hpf*mask + 0j)
    # calculate shifts with max of fourier space convolution
    sci_xcorr = np.real(ft_out.backward(fourier_sci * ref_kernel))
    sci_grid_shifts = np.array([sample_grid.points[ai] for ai in np.argmax(sci_xcorr, axis=-1)])
    # generate and use the shift kernel 

    # Return shifts instead of the shifted data
    #c_y = sci_grid_shifts[:, 0]
    #c_x = sci_grid_shifts[:, 1]
    #shifted_sci = Field([ndimage.shift(science_cube[i], (c_y[i], c_x[i])).ravel()  for i in range(science_cube.shape[0])], grid)

    return sci_grid_shifts