"""This module contains routines to add moving object to simulated data products.
"""
import numpy as np
import galsim
import webbpsf as wpsf
from romanisim import psf
from romanisim import bandpass

from . import parameters
from . import log

class MovingBody():

    def __init__(self, catalog_row, filter_name, detector_ind):
        self.magnitude = catalog_row['magnitude']
        self.initial_position = catalog_row['initial_position']
        self.angular_radius = catalog_row['angular_radius'] / 1e3 # convert mas / s to 
        self.angular_speed = catalog_row['angular_speed'] / 1e3 # convert mas / s to arcseconds / s
        self.direction = catalog_row['direction']

        if self.angular_radius < 0:
            self.height = 0.0001
        else:
            self.height = 2*self.angular_radius

        abflux = bandpass.get_abflux(filter_name, detector_ind)
        self.extra_flux_factor = abflux * 4.7 # I'm not sure why this is needed, but it is here...
        self.photon_flux = 10.**(self.magnitude / -2.5) 

        cos_dir = np.cos(self.direction * np.pi / 180.0)
        sin_dir = np.sin(self.direction * np.pi / 180.0)

        pixel_scale = parameters.pixel_scale
        self.x_vel = self.angular_speed * cos_dir / pixel_scale
        self.y_vel = self.angular_speed * sin_dir / pixel_scale
        
        if not isinstance(self.initial_position, np.ndarray):
            self.read_start_position = np.array(self.initial_position)
        else:
            self.read_start_position = self.initial_position

    def calculate_read_end_position(self, delta_t):
        read_movement = np.array([self.x_vel, self.y_vel]) * delta_t
        self.read_end_position = self.read_start_position + read_movement

    def get_galsim_profile(self, delta_t):
        if self.angular_speed <= 0:
            return galsim.DeltaFunction().withFlux(self.photon_flux) * self.extra_flux_factor * delta_t
        width = self.angular_speed * delta_t
        profile = galsim.Box(width, self.height).withFlux(self.photon_flux).rotate(galsim.Angle(self.direction, unit=galsim.degrees))
        return profile * self.extra_flux_factor * delta_t


def simulate_body(
    resultants,
    times,
    moving_bodies_catalog,
    wcs=None,
    rng=None,
    seed=47,
    oversample=4,
):
    """Adds a moving body to an existing image.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    times : list[list[float]]
        list of list of readout times for each read entering a resultant
    moving_bodies_catalog : astropy.table.Table
        must contain the following columns
            magnitude : float
                Moving body flux, units of mag. 
            initial_position : tuple of 2 floats
                Pixel starting position of the moving body, with science frame (X,Y).
            angular_radius : float
                Radius of the body in milliarcsec. If <= 0, assumes point source.
            angular_speed : float
                Speed of the body moving across the sky in milliarcsec s^-1.
            direction : float
                Angle relative to the science frame in degrees.
    wcs : galsim.image.wcs
        WCS associated with the galsim image
    rng : np.random.Generator
        Random number generator to use
    seed : int
        seed to use for random number generator
    oversample : int
        oversampling with which to sample WebbPSF PSF

    Returns
    -------
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    """

    # TODO: add inverse linearity
    # TODO: try to pass PSF object so it is not created twice (this is the 
    #       longest step of the moving body calculations)

    pixel_scale = parameters.pixel_scale
    if wcs is None:
        wcs = galsim.PixelScale(pixel_scale)
    if rng is None:
        rng = galsim.BaseDeviate(seed)

    filter_name = parameters.default_parameters_dictionary['instrument']['optical_element']
    detector = parameters.default_parameters_dictionary['instrument']['detector']
    detector_ind = int(detector[-2:])

    moving_body_list = []
    for row in moving_bodies_catalog:
        moving_body_list.append(MovingBody(row, filter_name, detector_ind))
    
    moving_psf = psf.make_psf(detector_ind, filter_name, wcs=wcs, variable=False, oversample=oversample) ## TODO: should variable=True?

    body_full_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)

    last_time = 0
    
    for i_res in range(resultants.shape[0]):
        # make blank reads with PSF at that location
        body_resultant_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)

        for read_time in times[i_res]:
            # calculate new position
            elapsed_time = read_time - last_time

            for i,mb in enumerate(moving_body_list):
                mb.calculate_read_end_position(elapsed_time)
                psf_position = (mb.read_start_position + mb.read_end_position) / 2. #adjust to make the boxes not overlap

                # add new psf at the read position
                if hasattr(moving_psf, 'at_position'):
                    psf0 = moving_psf.at_position(psf_position[1], psf_position[0])
                else:
                    psf0 = moving_psf

                profile = mb.get_galsim_profile(elapsed_time)
                body_conv = galsim.Convolve(profile, psf0)
                image_pos = galsim.PositionD(psf_position[0], psf_position[1])
                pwcs = wcs.local(image_pos)
                stamp = body_conv.drawImage(center=image_pos, wcs=pwcs)
                stamp.addNoise(galsim.PoissonNoise(rng))

                overlapping_bounds = stamp.bounds & body_resultant_image.bounds
                if overlapping_bounds.area() > 0:
                    body_resultant_image[overlapping_bounds] += stamp[overlapping_bounds]

                # Update the position for the next step
                moving_body_list[i].read_start_position = mb.read_end_position

            # Update time for the next step
            last_time = read_time

        # average the reads into a single resultant
        #body_resultant_image /= len(times[i_res])
        body_full_image += body_resultant_image

        # add the new PSF to the resultant
        resultants[i_res,:,:] += body_full_image.array

    return resultants
