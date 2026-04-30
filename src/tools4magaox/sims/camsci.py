# camsci.py
# April 8th 2026
# this simuates the camsci PSF with HCIPY
# integrating stellar magnitudes and detector noise

from hcipy import *
import numpy as np
import astropy.modeling as apm
import scipy.integrate as integrate
from hcipy import atmosphere

from tools4magaox.constants import *

# TODO: get these dictionaries in a better form, for both filters and beamsplitters
filt_dict = {
    "ri_ip": {
        "filter": "ip",
        "beamsplitter": "ri",
        "width": 120e-9,
        "center": 910e-9,
        "QE": 0.11,
        "F_0": 2.1e10, # e-/s
    },
    "ri_zp": {
        "filter": "zp",
        "beamsplitter": "ri",
        "width": 130e-9,
        "center": 908e-9,
        "QE": 0.08,
        "F_0": 1.3e10, # e-/s
    },
}

class CamsciSim(object):
    # relatively constant
    read_noise = 0.2 # e / exposure
    dark_noise = 35 # e / exposure #GUESS
    D = 6.5 #Magellan diameter in meters 
    px_size = 13.7e-6 # m/px
    px_ang_size = 0.00579 # as/px
    bitdepth = 16

    def __init__(self, n_px_w=512, bin=1, filter_name="ri_ip"):
        self.n_px_w = n_px_w
        self.n_px = n_px_w**2
        self.filter_name = filter_name
        self.wavelength = filt_dict[filter_name]["center"] # m
        self.spatial_resolution = self.wavelength / self.D # rad / lamD
        self.px_resolution = self.px_ang_size / (self.spatial_resolution * RAD_TO_AS) # lamD / px
        self.n_lD_w = (n_px_w * bin) * self.px_resolution # lamD

    def setup_camera(self):
        '''
        Setting up all the grids and things
        '''
        self.make_pupilgrid()
        self.make_focalgrid()
        self.make_prop()
        return self.pupil_grid, self.focal_grid, self.prop

    def make_pupilgrid(self):
        '''
        this pupil grid is trying to be a goldilocks
        not too small to avoid ailiasing
        not too big so that we mess up runtimes
        '''
        self.n_pix_pupil = 256
        self.pupil_grid = make_pupil_grid(self.n_pix_pupil, diameter=self.D)
        self.magellan_aperture = evaluate_supersampled(make_magellan_aperture(), self.pupil_grid, 6)
        return self.pupil_grid

    def make_focalgrid(self):
        '''
        This function creates the focal grid for the camsci simulation
        '''
        q = 1 / (self.px_resolution * self.spatial_resolution) # px / rad # double check this
        n_airy = self.n_lD_w * self.spatial_resolution / 2 # rumber of cycles, radius
        self.focal_grid = make_focal_grid(q, n_airy)
        return self.focal_grid

    def make_prop(self):
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        return self.prop

    def camsci_psf(self):
        wf = Wavefront(self.magellan_aperture, self.wavelength)
        wf.total_power = 1 # TODO: might scale this by photons, we'll see
        psf = self.prop(wf).intensity
        return psf

    def camsci_image(self, s_mag, t_int, EMGAIN=1.0, saturation=False, verbose=False):
        psf = self.camsci_psf()
        phot_flux_ct = self.camsci_mag_to_phot_flux(s_mag, t_int) # total e- on the detector in time frame
        # scale the PSF by photon_flux
        psf = psf * ( phot_flux_ct / np.sum(psf) )
        shot_noise = np.random.poisson(psf)
        photons = psf + shot_noise
        if verbose: print(f"Max photon count:   {np.max(photons)}")
        # convert to detector
        QE = filt_dict[self.filter_name]["QE"] # quantum efficiency
        electrons = photons * QE * EMGAIN
        # TODO: clean up how I'm handling electron based noise:
        dark_e = np.random.normal(scale=self.dark_noise, size=electrons.shape)
        electrons_out  = electrons + dark_e
        if verbose: print(f"Max electron count: {np.max(electrons_out)}")
        # Digitization of the signal
        bias = 600
        ADU = (electrons_out/2.32).astype(int)
        ADU += bias
        if saturation:
            max_adu = int(2**self.bitdepth - 1)
            ADU = np.clip(ADU, 0, max_adu)
        if verbose:print(f"Max ADU count:       {np.max(ADU)}")
        return ADU

    def camsci_mag_to_phot_flux(self, s_mag, t_int):
        F_0 = filt_dict[self.filter_name]["F_0"] # zero mag flux
        phot_flux = F_0 * 10**(-s_mag/2.5) # rescaling to source flux
        phot_flux_ct = phot_flux * t_int # converting to photons per exposure
        return phot_flux_ct