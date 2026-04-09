# camtip.py
# Originally from camtipSR/camtipSim.py
# May 15th 2025
# simulating camtip's psf with HCIPY

from hcipy import *
import numpy as np 
import astropy.modeling as apm
import scipy.integrate as integrate
from hcipy import atmosphere


# system settings
F0_65 = 4.2e10 # from website per aperture (6.5)
wfs_split = 0.35 # photons for wfs
ct_throughput = 0.08 # from the thorlabs spec sheet
# just some numbers I like
rads_per_as = 0.00000485 # rads / as

# these usually don't change, but we can make variables later
wavelength_wfs = 800.0E-9
D_tel = 6.5
spatial_resolution = wavelength_wfs / D_tel

class CamtipSim(object):
    # This class is both to store camera variables, and to create a HCIPY image of the camera

    def __init__(self, mod=3, n_px_w=128,  QE=1.0, read_noise=0.0, dark_noise=0, px_resolution=1, px_size=1, bin = 2):
        self.read_noise = read_noise # e / exposure
        #self.gain = gain # e / photon
        self.QE = QE # percentage
        self.px_resolution = px_resolution # pixel per lamda/D 
        self.grid_d_lD = n_px_w / px_resolution # lD across grid
        self.mod_r = mod
        self.mod = mod * spatial_resolution
        self.n_px_w = n_px_w
        self.EE = self.ee_ring()
        # set up sim stuff
        self.n_pix_pwfs = 60
        self.pupil_grid_D = 60/56 * D_tel
        self.pupil_grid = make_pupil_grid(self.n_pix_pwfs*bin, self.pupil_grid_D)
        self.pwfs_grid = make_pupil_grid(2*self.n_pix_pwfs*bin, 2 * self.pupil_grid_D)
        self.magellan_aperture = evaluate_supersampled(make_magellan_aperture(), self.pupil_grid, 6)
        self.pwfs = PyramidWavefrontSensorOptics(self.pupil_grid, self.pwfs_grid, 
                                    separation=self.pupil_grid_D, 
                                    pupil_diameter=D_tel, 
                                    wavelength_0=wavelength_wfs, 
                                    q=4)
        self.n_steps = 36
        self.mpwfs = ModulatedPyramidWavefrontSensorOptics(self.pwfs, self.mod , num_steps=self.n_steps)
        # camtip specific
        self.focal_camtip_grid = make_focal_grid(self.px_resolution/spatial_resolution, 
                                                 self.grid_d_lD*spatial_resolution/2)
        self.pupil_to_camtip = FraunhoferPropagator(self.pupil_grid, self.focal_camtip_grid)
        #TODO: make this not noiseless
        self.camera = NoiselessDetector(self.focal_camtip_grid)
        self.camera_n = NoisyDetector(self.focal_camtip_grid, dark_current_rate=dark_noise)

    ############### HCIPY Functions #############################

    def camtip_forward(self, wavefront):
        wf_modulated = []
        dm_wf = []
        dm_command = []
        for point in self.mpwfs.modulation_positions.points:
            self.mpwfs.tip_tilt_mirror.flatten()
            self.mpwfs.tip_tilt_mirror.actuators = point
            mod_pupil = self.mpwfs.tip_tilt_mirror.forward(wavefront)
            #mod_focal = self.pyramid_wavefront_sensor.pupil_to_focal.forward(mod_pupil)
            mod_focal = self.pupil_to_camtip.forward(mod_pupil)
            dm_command.append(self.mpwfs.tip_tilt_mirror.surface)
            dm_wf.append(mod_pupil)
            wf_modulated.append(mod_focal)
        return wf_modulated, dm_wf, dm_command
    
    ################ HCIPY Sim Image #########################

    def camtip_image(self, s_mag, t_int, gain = 1, noisy=True):
        # calculate flux
        phot_flux_ct = self.phot_flux_ct(t_int, s_mag)
        # scale wavefront
        wf = Wavefront(self.magellan_aperture, wavelength_wfs)
        wf.total_power = phot_flux_ct * gain
        # forward to the focal plane
        wfs_pywfs, dm_wfs, dm_c = self.camtip_forward(wf)
        # average the focal plane images
        camtip_image = mod_forward_int(wfs_pywfs)
        # pick camera
        cam = self.camera
        if noisy:
            cam = self.camera_n
        # integrate the image 
        cam.integrate(camtip_image, t_int)
        img_detector = cam.read_out()
        return img_detector
    
    # get the atmospher involved
    def gen_layer_r0(self, r0, L0=25):
        """
        Generate a layer with a given r0 and L0
        """
        # forward the wavefront through turbulence
        cn2 = atmosphere.Cn_squared_from_fried_parameter(r0, wavelength=5e-07) # r0 is at 500nm here
        layer = atmosphere.InfiniteAtmosphericLayer(self.pupil_grid, cn2, L0=L0)
        return layer
    
    def camtip_image_r0(self, layer, s_mag, t_int, gain = 1, noisy=True):
        # calculate flux
        phot_flux_ct = self.phot_flux_ct(t_int, s_mag)
        # scale wavefront
        wf = Wavefront(self.magellan_aperture, wavelength_wfs)
        wf.total_power = phot_flux_ct * gain
        # forward to the focal plane
        wfs_pywfs, dm_wfs, dm_c = self.camtip_forward(layer(wf))
        # average the focal plane images
        camtip_image = mod_forward_int(wfs_pywfs)
        # pick camera
        cam = self.camera
        if noisy:
            cam = self.camera_n
        # integrate the image 
        cam.integrate(camtip_image, t_int)
        img_detector = cam.read_out()
        return img_detector
    
    def camtip_SRest_r0(self, layer, s_mag, t_int, gain=1, noisy=True):
        # calculate flux
        phot_flux_ct = self.phot_flux_ct(t_int, s_mag)
        # scale wavefront
        wf = Wavefront(self.magellan_aperture, wavelength_wfs)
        wf.total_power = phot_flux_ct * gain
        # forward to the focal plane
        pupil_phase = layer(wf).phase_for(wavelength)
        return np.exp(-np.std(pupil_phase[self.magellan_aperture>0])**2) 
    
    ############### Self RM calculations ########### 


    ############### Error Budget Equations ######################

    def phot_flux_ct(self, t_int, s_mag):
        phot_flux = F0_65*t_int*10**(-s_mag/2.5)
        phot_flux_ct =  phot_flux * self.EE * wfs_split * ct_throughput * self.QE
        return phot_flux_ct

    def signal_to_noise_ratio(self, phot_flux, t_int, n_px):
        #TODO: scaling readnoise by ROI
        signal = phot_flux * t_int  # * self.gain # I don't THINK we use gain here
        noise = np.sqrt(signal + n_px*self.read_noise**2)
        return signal / noise

    def SNR_full(self, t_int, s_mag):
        n_pix = self.n_px_w**2
        #TODO: use the number of pixels in a useful way
        phot_flux = self.phot_flux_ct(t_int, s_mag)
        snr = self.signal_to_noise_ratio(phot_flux, t_int, n_pix)
        return snr
    
    def SNR_max_px(self, t_int, s_mag, frac_max):
        n_pix = self.n_px_w**2
        phot_flux = self.phot_flux_ct(t_int, s_mag)
        phot_flux_px = phot_flux * frac_max
        snr = self.signal_to_noise_ratio(phot_flux_px, t_int, n_pix)
        return snr
    
    def contrast_max_px(self, t_int, s_mag, frac_max):
        n_pix = self.n_px_w**2
        phot_flux = self.phot_flux_ct(t_int, s_mag)
        phot_flux_px = phot_flux * frac_max
        contrast = self.read_noise / phot_flux_px
        return contrast

    def ee_ring(self):
        # use the lam/D to calculate energy capture by a pixel
        airy_disk = apm.functional_models.AiryDisk2D()
        lamD  = self.n_px_w / 2 / self.px_resolution 
        mod = self.mod

        full_int = integrate.quad(lambda x: airy_disk(x, 0),  0,  np.inf)
        FOV_int = integrate.quad(lambda x: airy_disk(x, 0),  -(lamD+mod), lamD-mod)

        EE = FOV_int[0] / (full_int[0]*2)
        return EE
    
    def calc_SR_SNR(self, t_int, s_mag, frac_max):
        phot_flux = self.phot_flux_ct(t_int, s_mag)
        n_pix = self.n_px_w**2
        # determine peak pixel
        P = phot_flux * frac_max # the peak pixel flux, can infer from the ratio of the perfect image
        # determine volume flux
        V = phot_flux # all flux in the frame, so like 99% of the incoming flux
        # determine peak nosie
        simga_P  = P + self.read_noise**2 # this is the photon + read noise + dark current maybe?
        # determine volume noise
        sigma_V =  V + n_pix*self.read_noise**2
        # the SR
        SR = P / V
        # the noise or variance on the SR
        N = (simga_P / V)**2 + (P*sigma_V / V**2)**2
        # Signal over noise
        SNR = SR / np.sqrt(N)
        return SR, SNR
    
##### Helper functions

def mod_forward_int(wfs_pywfs):
    """This iterates on wfs and returns an average of the power
    """
    image_final = 0
    for e, wfs_i in enumerate(wfs_pywfs):
        image_final += wfs_i.power 
    return image_final / ( e + 1) #/num_mod_STEPS