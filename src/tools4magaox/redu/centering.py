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
from tqdm import tqdm


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
    sources_list = sources_info.get("sources", [])
    shifts = _gaussian_xy_shifts(sources_list, cube.shape[1:])

    # Normalize gaussian fit outputs into a numeric array for downstream logging.
    # Parameter order is defined by gaussian_2d: (y0, x0, sigma_y, sigma_x, amplitude, offset).
    params = np.full((len(sources_list), 6), np.nan, dtype=float)
    for i, p in enumerate(sources_list):
        try:
            params[i, :] = np.asarray(p, dtype=float).ravel()[:6]
        except Exception:
            pass

    bad = set(sources_info.get("bad_idx") or [])
    for i in bad:
        if 0 <= i < params.shape[0]:
            params[i, :] = np.nan

    sources_info["gauss_params"] = params
    return shifts, sources_info

def airy_fit_shifts(data, lam_d_px, crop_shape=None, fit_radius_ld=1.5, obscuration=0.0):
    '''
    Fit an Airy pattern to the PSF core of each frame and return the shifts
    ``(dy, dx)`` of the core from the (cropped) frame center ``shape // 2``.

    Only pixels within ``fit_radius_ld`` (in lambda/D) of the brightest pixel of
    the lightly smoothed frame are fit, so the fit is driven by the core and the
    first dark ring rather than the aberrated outer rings. ``lam_d_px`` is the
    initial guess for lambda/D in pixels; it is a free parameter of the fit.

    ``sources_info["gauss_params"]`` holds ``(y0, x0, sigma_y, sigma_x, amplitude, offset)``
    in the same layout as :func:`gaussian_fit_shifts`, where sigma is the
    Gaussian-equivalent width of the fitted core (FWHM / 2.355), and
    ``sources_info["airy_params"]`` holds ``(y0, x0, lam_d_px, amplitude, offset)``.
    '''
    cube = check_cube(data)
    if crop_shape is not None:
        cube = crop_cube(cube, crop_shape)
    sources_info = _airy_fit_least_squares(cube, lam_d_px, fit_radius_ld, obscuration)
    airy = sources_info["airy_params"]
    h, w = cube.shape[1:]
    shifts = np.column_stack([airy[:, 0] - h // 2, airy[:, 1] - w // 2])
    sigma = airy_fwhm_ld(obscuration) * airy[:, 2] / (2 * np.sqrt(2 * np.log(2)))
    params = np.column_stack([airy[:, 0], airy[:, 1], sigma, sigma, airy[:, 3], airy[:, 4]])
    params[list(sources_info["bad_idx"])] = np.nan
    sources_info["gauss_params"] = params
    return shifts, sources_info

def DAO_fit_shifts(data, crop_shape=None):
    '''
    This function finds the center of the PSF using DAOstarfinder routine
    '''
    # this finds the sources in a numpy cube
    cube = check_cube(data)

    # if desired, crop to a smaller shape
    if crop_shape is not None:
        cube = crop_cube(cube, crop_shape)

    sources_info = _DAO_check_sources(cube)
    sources_list = sources_info['sources']

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
            grid.y_center, 
            grid.x_center, 
            1, 
            1, 
            amp_guess, 
            offset_guess)
        # optimizing the fit
        try:
            params_opt, _ = scipy.optimize.curve_fit(
                gaussian_2d,
                grid,
                frame.ravel(),
                p0=params,
                maxfev=10_000,
                bounds = ((grid.ny*0.2, grid.nx*0.2, 1, 1, 0.5*amp_guess, -0.1*offset_guess),
                        (grid.ny*0.8, grid.nx*0.8, grid.ny/2, grid.nx/2, 1.5*amp_guess, 2*offset_guess))
            )
            sources_list.append(params_opt)
        except Exception:
            # Common failure: "Optimal parameters not found: Number of calls to function
            # has reached maxfev". Mark as bad and keep placeholder params.
            bad_idx.append(i)
            sources_list.append(np.full(6, np.nan, dtype=float))
    return {"bad_idx": bad_idx, "sources": sources_list}

def _gaussian_fit_minimize(cube):
    cube = np.asarray(cube)
    grid = Grid(cube.shape[1:])
    bad_idx = []
    sources_list = []

    for i, frame in enumerate(tqdm(cube)):
        # set up guesses per frame
        offset_guess = np.median(frame)
        amp_guess = np.max(frame) - offset_guess
        params_0 = (
            grid.y_center, 
            grid.x_center, 
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
    # Gaussian params start with (y0, x0); shifts are reported as (dy, dx).
    frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
    if len(sources_list) == 0:
        return np.empty((0, 2), dtype=float)
    centers = np.array([[p[0], p[1]] for p in sources_list], dtype=float)
    return centers - frame_center

############## Airy Functions ##################

def airy_2d(yy, xx, y0, x0, lam_d_px, amplitude, offset, obscuration=0.0):
    '''
    Airy pattern of a circular pupil with fractional central obscuration
    ``obscuration``, centered at (y0, x0) with lambda/D = ``lam_d_px`` pixels.
    '''
    eps = float(obscuration)
    v = np.pi * np.hypot(yy - y0, xx - x0) / lam_d_px
    field = _airy_amplitude(v)
    if eps > 0:
        field = (field - eps**2 * _airy_amplitude(eps * v)) / (1 - eps**2)
    return amplitude * field**2 + offset

def _airy_amplitude(v):
    '''2 J1(v) / v, equal to 1 at v = 0.'''
    v = np.asarray(v, dtype=float)
    out = np.ones_like(v)
    nz = v > 1e-8
    out[nz] = 2 * scipy.special.j1(v[nz]) / v[nz]
    return out

def airy_fwhm_ld(obscuration=0.0):
    '''FWHM of the Airy core in lambda/D (1.029 for an unobscured pupil).'''
    v = np.linspace(0, 2 * np.pi, 20001)
    prof = airy_2d(0.0, v / np.pi, 0.0, 0.0, 1.0, 1.0, 0.0, obscuration)
    return 2 * v[np.argmax(prof < 0.5)] / np.pi

def _airy_fit_least_squares(cube, lam_d_px, fit_radius_ld, obscuration):
    cube = np.asarray(cube, dtype=float)
    yy, xx = np.indices(cube.shape[1:])
    radius_px = fit_radius_ld * lam_d_px
    bad_idx = []
    params = np.full((len(cube), 5), np.nan, dtype=float)

    for i, frame in enumerate(tqdm(cube)):
        y_g, x_g = np.unravel_index(np.argmax(ndimage.gaussian_filter(frame, 1)), frame.shape)
        sel = np.hypot(yy - y_g, xx - x_g) <= radius_px
        ys, xs, vals = yy[sel], xx[sel], frame[sel]
        offset_guess = np.median(frame)
        p0 = (y_g, x_g, lam_d_px, frame[y_g, x_g] - offset_guess, offset_guess)
        lower = (y_g - lam_d_px, x_g - lam_d_px, 0.5 * lam_d_px, 0.0, -np.inf)
        upper = (y_g + lam_d_px, x_g + lam_d_px, 2.0 * lam_d_px, np.inf, np.inf)

        def resid(p):
            return airy_2d(ys, xs, *p, obscuration=obscuration) - vals

        try:
            fit = scipy.optimize.least_squares(resid, p0, bounds=(lower, upper), x_scale="jac")
            if not fit.success:
                bad_idx.append(i)
            params[i] = fit.x
        except Exception:
            bad_idx.append(i)
    return {"bad_idx": bad_idx, "airy_params": params}

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

def shift_cube(data_cube, shifts):
    '''
    This function will shift a cube by a given shift
    '''
    for i, frame in enumerate(data_cube):
        data_cube[i] = shift_frame(frame, shifts[i])
    return data_cube

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

def gaussian_2d(grid: Grid, y0, x0, sigma_y, sigma_x, amplitude, offset):
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