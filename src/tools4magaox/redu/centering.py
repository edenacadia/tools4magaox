# centering.py
# 14/03/2026
# This file uses the central peak for fitting and centering

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from hcipy import *
from scipy.special import j1

############ Center ID Functions #############

def gaussian_fit_center(data):
    '''
    This function will fit a gaussian to the unsats data and return the center of the gaussian
    Eventually should also handle cropping the data for the fitting. 
    '''
    # iteratively fir a gaussian to the data
    # return the coordinates of the center of the gaussian

    cube = np.asarray(data)
    return 

def DAO_fit_centers(data):
    '''
    This function finds the center of the PSF using DAOstarfinder routine
    '''
    # this finds the sources in a numpy cube
    cube = np.asarray(data)

    # Expect a data cube of shape (N, H, W)
    if cube.ndim != 3 or cube.shape[0] == 0:
        return (None, None)
    
    sources_info = _DAO_check_sources(cube)
    sources_list = sources_info['sources']
    centers = _DAO_xy_centers(sources_list)
    #shifts = _DAO_xy_shifts(sources_list, cube.shape[1:])

    # TODO: handle the bad indexes elegantly 
    return centers

def weighted_sum_center(data):
    '''
    This function will return the center of the unsats data using a weighted sum of the coordinates
    '''
    pass


############ Helper Functions ##################

def _DAO_check_sources(cube, fwhm=5.0, threshold_sigma=1e3, max_allowed=1):
    """
    Scan each frame in a data cube and:
      - collect frames with no detected source (bad_idx)
      - warn (print) when more than max_allowed sources are found (multi_idx)
    Parameters
    ----------
    cube : array-like, shape (N, H, W)
    fwhm : float
        FWHM for DAOStarFinder
    threshold_sigma : float
        Detection threshold = threshold_sigma * frame_std
    max_allowed : int
        Maximum allowed sources per frame (default 1)
    Returns
    -------
    dict with keys: 'bad_idx', 'multi_idx', 'sources' (list of photutils tables or None)
    """
    cube = np.asarray(cube)
    bad_idx = []
    multi_idx = []
    sources_list = []

    for i, frame in enumerate(cube):
        mean, median, std = sigma_clipped_stats(frame, sigma=3.0, maxiters=5)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
        sources = daofind(frame - median) #TODO: Check if this makes sense
        sources_list.append(sources)

        if sources is None or len(sources) == 0:
            bad_idx.append(i)
        elif len(sources) > max_allowed:
            multi_idx.append(i)
            print(f"Warning: frame {i} has {len(sources)} sources (allowed {max_allowed})")

    return {'bad_idx': bad_idx, 'multi_idx': multi_idx, 'sources': sources_list}

def _DAO_xy_centers(sources_list):
    """
    This function will return the x and y center for a list of sources
    """
    # TODO: check if this errors with a non-detection, return None for that frame
    centers = np.array([[float(sources['xcentroid'][0]), float(sources['ycentroid'][0])] for sources in sources_list])
    return centers

def _DAO_xy_shifts(sources_list, frame_shape):
    """
    This function will return the x and y shifts for a list of sources
    """
    # TODO: check if this errors with a non-detection, return None for that frame
    frame_center = (frame_shape[0] // 2, frame_shape[1] // 2)
    centers = np.array([[float(sources['xcentroid'][0]), float(sources['ycentroid'][0])] for sources in sources_list])
    shifts = centers - frame_center
    return shifts


############ Shifting Functions ################

def shift_frame(data, shift):
    '''
    This function will shift a single frame by a given shift
    '''
    shifted = ndimage.shift(data, shift=(shift[0], shift[1]), mode='constant')
    return shifted

def shift_field(data, shift):
	'''Shifts a Field class object by shift.
    Data is a single frame, not a cube
	'''
	if data.is_scalar_field:
		return Field(ndimage.shift(data.shaped, np.array([shift[1], shift[0]]) / data.grid.delta[0]).ravel(), data.grid)
	else:
		return Field(ndimage.shift(data.shaped, np.array([0, shift[1], shift[0]]) / data.grid.delta[0]).reshape((data.shape[0], -1)), data.grid)


def shift_cube(data, shift):
    '''
    This function will shift a cube by a given shift
    '''
    pass

########### Simulated Data Functions ################

@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray

@dataclass(frozen=True)
class GaussParams:
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    offset: float

def gaussian_2d(grid: Grid, params: GaussParams):
    '''
    This function generates a 2D gaussian function
    '''
    return params.amplitude * np.exp(-(grid.x - params.center_x)**2 / (2 * params.sigma_x**2) - (grid.y - params.center_y)**2 / (2 * params.sigma_y**2)) + params.offset

@dataclass(frozen=True)
class AiryDiskParams:
    center_x: float
    center_y: float
    fwhm: float
    amp: float

def airyfnc_2d(grid: Grid, params: AiryDiskParams):
    '''
    This function generates a PSF with given amounts of noise.
    There is an error at the zero point because of a zero pole...
    Not sure how to analytically handle this, feels like a fft moment
    '''
    r = np.sqrt((grid.x - params.center_x)**2 + (grid.y - params.center_y)**2)
    psf = (2*j1(r/params.fwhm) / r / params.fwhm)**2
    return psf * params.amp

def airydisk_2d(grid: Grid, params: AiryDiskParams):
    '''
    Doing this with an FFT from the  
    '''
     
def simulate_psf(grid, params: AiryDiskParams, read_noise=0.2):
    ''' 
    This function simulates a PSF with given amounts of noise
    '''
    num_pixels = grid.shape[0] * grid.shape[1]
    psf = airydisk_2d(grid, params)
    phot_noise = np.random.poisson(psf, (num_pixels,num_pixels))
    readout_noise = np.random.normal(read_noise, size=psf.shape)

    return psf + phot_noise + readout_noise

####################################################
############### Helper Functions ###################
####################################################

# more centering tests

def DAO_fit_center_singleframe(data, n=0):
    '''
    A simple center finding function for testing 
    This function finds DAO sources in the first frame of a data cube
    and returns the first detected source location as (x, y).
    If no source is found, returns (None, None).
    '''
    cube = np.asarray(data)
    frame = cube[n]
    _, median, std = sigma_clipped_stats(frame, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=5.0, threshold=1e3 * std)
    sources = daofind(frame - median)
    if sources is None or len(sources) == 0:
        return (None, None)
    first_source = sources[0]
    return (float(first_source['xcentroid']), float(first_source['ycentroid']))

def check_center_DAO(unsats_c_cube, save_plot=True):
    sources_dict = _DAO_check_sources(unsats_c_cube.shaped)
    # plot the x and y cen returns 
    n_frames = unsats_c_cube.shape[0]
    old_centers = np.array([[float(sources_dict[i]['xcentroid'][0]), float(sources_dict[i]['ycentroid'][0])] for i in range(n_frames)])
    shifts = old_centers - unsats_c_cube.shape[1]
    return shifts