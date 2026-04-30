# centering.py
# 14/03/2026
# This file uses the central peak for fitting and centering
# Functional with both cube and field data

import numpy as np
import scipy
from scipy import ndimage
from dataclasses import dataclass
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from hcipy import *
from scipy.special import j1

#region agent log
def _agent_log(*, runId: str, hypothesisId: str, location: str, message: str, data: dict):
    import json, time
    payload = {
        "sessionId": "05c9b8",
        "runId": runId,
        "hypothesisId": hypothesisId,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/Users/clever/code/proj_research/tools4magaox/.cursor/debug-05c9b8.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
#endregion

############ Center ID Functions #############

def gaussian_fit_shifts(data, crop_shape=None, method="minimize"):
    '''
    This function will fit a gaussian to the unsats data and return the center of the gaussian
    Eventually should also handle cropping the data for the fitting. 
    '''
    # iteratively fir a gaussian to the data
    # return the coordinates of the center of the gaussian
    cube = check_cube(data)
    # if desired, crop to a smaller shape
    if crop_shape is not None:
        cube = crop_cube(cube, crop_shape)

    # call gaussian fitter code here
    if method == "minimize":
        sources_info = _gaussian_fit_minimize(cube)
    elif method == "curvefit":
        sources_info = _gaussian_fit_curvefit(cube)
    else:
        raise ValueError(f"Invalid method: {method}")
    sources_list = sources_info['sources']
    shifts = _gaussian_xy_shifts(sources_list, cube.shape[1:])
    return shifts

def DAO_fit_shifts(data, crop_shape=None):
    '''
    This function finds the center of the PSF using DAOstarfinder routine
    '''
    # this finds the sources in a numpy cube
    cube = check_cube(data)

    # if desired, crop to a smaller shape
    if crop_shape is not None:
        cube = crop_cube(cube, crop_shape)

    #region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="H1",
        location="centering.py:DAO_fit_shifts",
        message="Entering DAO_fit_shifts",
        data={"cube_shape": list(np.shape(cube)), "crop_shape": list(crop_shape) if crop_shape is not None else None},
    )
    #endregion

    sources_info = _DAO_check_sources(cube)
    sources_list = sources_info['sources']

    #region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="H1",
        location="centering.py:DAO_fit_shifts",
        message="DAO sources summary",
        data={
            "n_frames": int(np.shape(cube)[0]),
            "bad_idx": list(sources_info.get("bad_idx", [])),
            "multi_idx": list(sources_info.get("multi_idx", [])),
            "sources_none_count": int(sum(1 for s in sources_list if s is None)),
        },
    )
    #endregion

    shifts = _DAO_xy_shifts(sources_list, cube.shape[1:])
    # if looking for centers, need to convert to the original size centers
    #centers = _DAO_xy_centers(sources_list)

    # TODO: handle the bad indexes elegantly 
    return shifts

def weighted_sum_fit_shifts(data, crop_shape=None):
    '''
    This function will return the center of the unsats data using a weighted sum of the coordinates
    '''
    # this finds the sources in a numpy cube
    cube = check_cube(data)

    # if desired, crop to a smaller shape
    if crop_shape is not None:
        cube = crop_cube(cube, crop_shape)
    
    # The center of mass is pretty simple so we'll just put it here.
    # ndimage.center_of_mass returns (y, x); convert to (x, y) for consistency.
    centers_yx = np.array([ndimage.center_of_mass(frame) for frame in cube], dtype=float)
    centers_xy = centers_yx[:, ::-1]
    shifts_xy = centers_xy - (cube.shape[2] // 2, cube.shape[1] // 2)
    return shifts_xy

def check_cube(cube):
    '''
    This function will check if the data is a cube
    If cube is a Field, it will be converted to a cube
    '''
    if isinstance(cube, Field):
        cube = cube.shaped
    if cube.ndim != 3 or cube.shape[0] == 0:
        raise ValueError("Data must be a cube of shape (N, H, W)")
    return np.asarray(cube)

def crop_cube(cube, new_shape=(64,64), center_shift=(0,0)):
    '''
    This function will crop a cube to a new shape
    from the center of the cube with a given center shift
    '''
    shape = cube.shape
    center = (shape[1] // 2, shape[2] // 2)
    new_center = (center[0] + center_shift[0], center[1] + center_shift[1])
    # TODO: this doesn't work for odd shapes
    x_i, x_f = new_center[0] - new_shape[0] // 2, new_center[0] + new_shape[0] // 2
    y_i, y_f = new_center[1] - new_shape[1] // 2, new_center[1] + new_shape[1] // 2
    new_cube = cube[:, x_i:x_f, y_i:y_f]
    return new_cube

############ DAO Functions ##################

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
    # Return an (N, 2) array where missing detections are [None, None].
    # Use dtype=object so None can be represented.
    n = len(sources_list)
    centers = np.full((n, 2), None, dtype=object)
    for i, sources in enumerate(sources_list):
        if sources is None or len(sources) == 0:
            continue
        centers[i, 0] = float(sources["xcentroid"][0])
        centers[i, 1] = float(sources["ycentroid"][0])
    return centers

def _DAO_xy_shifts(sources_list, frame_shape):
    """
    This function will return the x and y shifts for a list of sources
    """
    # If DAO finds no sources for a frame, return [None, None] for that frame
    # instead of erroring. This keeps the output length aligned with N frames.
    #
    # sources are reported in (x, y), while frame_shape is (H, W)
    n = len(sources_list)
    shifts = np.full((n, 2), None, dtype=object)
    if n == 0:
        return shifts

    frame_center_x = frame_shape[1] // 2
    frame_center_y = frame_shape[0] // 2

    for i, sources in enumerate(sources_list):
        if sources is None or len(sources) == 0:
            continue
        x = float(sources["xcentroid"][0])
        y = float(sources["ycentroid"][0])
        shifts[i, 0] = x - frame_center_x
        shifts[i, 1] = y - frame_center_y
    return shifts

############## Gaussian Functions ##################

def _gaussian_fit_curvefit(cube):
    cube = np.asarray(cube)
    grid = Grid(cube.shape[1:])
    bad_idx = []
    sources_list = []

    for i, frame in enumerate(cube):
        # set up guesses per frame
        offset_guess = np.median(frame)
        amp_guess = np.max(frame) - offset_guess
        params = (
            grid.x_center, 
            grid.y_center, 
            1, 
            1, 
            amp_guess, 
            offset_guess)
        # optimizing the fit
        params_opt, _ = scipy.optimize.curve_fit(gaussian_2d, grid, frame.ravel(), p0=params)
        sources_list.append(params_opt)
    return {'sources': sources_list}

def _gaussian_fit_minimize(cube):
    cube = np.asarray(cube)
    grid = Grid(cube.shape[1:])
    bad_idx = []
    sources_list = []

    for i, frame in enumerate(cube):
        # set up guesses per frame
        offset_guess = np.median(frame)
        amp_guess = np.max(frame) - offset_guess
        params_0 = (
            grid.x_center, 
            grid.y_center, 
            1,  # sigma_x 
            1,  # sigma_y
            amp_guess, 
            offset_guess)
        # making the fit function
        cost_func_params = _gaussian_fit_function(frame.ravel(), grid)
        # optimizing the fit
        params_opt = scipy.optimize.minimize(cost_func_params, params_0)
        sources_list.append(params_opt.x)
        # barebones, could probably do better
        if not params_opt['success']:
            bad_idx.append(i)
    return {'bad_idx': bad_idx, 'sources': sources_list}

def _gaussian_fit_function(frame: np.ndarray, grid: Grid):
    # define a fit function as a cost function
    # specific to the image
    def fit_func(params: tuple):
        gaus_sim = gaussian_2d(grid, *params)
        return np.sum((frame - gaus_sim)**2)
    return fit_func

def _gaussian_xy_centers(sources_list):
    centers = np.array([[float(sources[0]), float(sources[1])] for sources in sources_list])
    return centers

def _gaussian_xy_shifts(sources_list, frame_shape):
    # Gaussian fitters return GaussParams; report shifts in (x, y).
    frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
    if len(sources_list) == 0:
        return np.empty((0, 2), dtype=float)
    centers = np.array([[p[0], p[1]] for p in sources_list], dtype=float)
    return centers - frame_center

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

class Grid:
    def __init__(self, shape):
        self.nx = shape[0]
        self.ny = shape[1]
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.xy = np.vstack((self.xx.ravel(), self.yy.ravel()))
        self.x_center = self.x[self.nx // 2]
        self.y_center = self.y[self.ny // 2]

def gaussian_2d(grid: Grid, x0, y0, sigma_x, sigma_y, amplitude, offset):
    '''
    This function generates a 2D gaussian function
    '''
    # Use meshgrid coordinates; Grid.x and Grid.y are 1D vectors.
    xx, yy = grid.xy
    return (
        amplitude
        * np.exp(
            -((xx - x0) ** 2) / (2 * sigma_x**2)
            -((yy - y0) ** 2) / (2 * sigma_y**2)
        )
        + offset
    )

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