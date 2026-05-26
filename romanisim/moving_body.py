"""This module contains routines to add moving object to simulated data products.
"""
from os import times
from time import perf_counter
import copy

import numpy as np
import galsim
from romanisim import psf
from romanisim.models import bandpass, parameters

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
        self.extra_flux_factor = abflux 
        
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

        self.galsim_profile = self._get_galsim_profile()

    def calculate_read_end_position(self, delta_t):
        read_movement = np.array([self.x_vel, self.y_vel]) * delta_t
        self.read_end_position = self.read_start_position + read_movement

    def _get_galsim_profile(self):
        if self.angular_speed <= 0:
            return galsim.DeltaFunction().withFlux(self.photon_flux) * self.extra_flux_factor * parameters.read_time
        width = self.angular_speed * parameters.read_time
        profile = galsim.Box(width, self.height).withFlux(self.photon_flux).rotate(galsim.Angle(self.direction, unit=galsim.degrees))
        return profile * self.extra_flux_factor * parameters.read_time

def simulate_body(
    resultants,
    times,
    moving_bodies_catalog,
    wcs=None,
    rng=None,
    seed=47,
    oversample=4,
    inv_linearity=None,
    filter_name=None,
    detector_number=None,
    enable_timing=True,
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
    enable_timing : bool
        if True, logs component timings and identifies the slowest section

    Returns
    -------
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    """

    ## TODO: Implement persistence

    timing = {
        'setup': 0.0,
        'position_update': 0.0,
        'profile_gen': 0.0,
        'psf_select': 0.0,
        'convolve': 0.0,
        'wcs': 0.0,
        'draw_stamp': 0.0,
        'poisson_noise': 0.0,
        'accumulate': 0.0,
        'linearity': 0.0,
        'resultant_finalize': 0.0,
    }
    overall_start = perf_counter()
    setup_start = perf_counter()

    pixel_scale = parameters.pixel_scale
    if wcs is None:
        wcs = galsim.PixelScale(pixel_scale)
    if rng is None:
        rng = galsim.BaseDeviate(seed)

    moving_body_list = []
    for row in moving_bodies_catalog:
        moving_body_list.append(MovingBody(row, filter_name, detector_number))
    
    if psf.saved_psf is None:
        moving_psf = psf.make_psf(detector_number, filter_name, wcs=wcs, variable=True, oversample=oversample)
    else:
        moving_psf = psf.saved_psf

    timing['setup'] += perf_counter() - setup_start

    body_accum_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)
    body_resultant_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)
    
    expected_num_reads = times[-1][-1] / parameters.read_time
    if expected_num_reads % 1 > 0:
        log.error('times not divisible by read time!!!!')
        raise ValueError('Last time in t_ij is not divisible by read time')
    num_reads = round(expected_num_reads)
    saved_reads = []
    for ts in times:
        saved_reads += [round(t / parameters.read_time) for t in ts]

    last_read_number_per_resultant = [round(ts[-1]/parameters.read_time) for ts in times]

    num_reads_in_resultant = 0
    resultant_i = 0
    for read_i in range(num_reads):
        read_num = read_i + 1
        # loop over the moving bodies and add their points to the accumulated image
        for j,mb in enumerate(moving_body_list):
            t0 = perf_counter()
            mb.calculate_read_end_position(parameters.read_time)
            psf_position = (mb.read_start_position + mb.read_end_position) / 2. #adjust to make the boxes not overlap
            timing['position_update'] += perf_counter() - t0

            # add new psf at the read position
            if hasattr(moving_psf, 'at_position'):
                psf0 = moving_psf.at_position(psf_position[1], psf_position[0])
            else:
                psf0 = moving_psf

            t0 = perf_counter() 
            body_conv = galsim.Convolve(mb.galsim_profile, psf0)
            timing['convolve'] += perf_counter() - t0
            t0 = perf_counter()
            image_pos = galsim.PositionD(psf_position[0], psf_position[1])
            pwcs = wcs.local(image_pos)
            timing['wcs'] += perf_counter() - t0
            t0 = perf_counter()
            stamp = body_conv.drawImage(center=image_pos, wcs=pwcs)
            timing['draw_stamp'] += perf_counter() - t0

            t0 = perf_counter()
            stamp.addNoise(galsim.PoissonNoise(rng))
            timing['poisson_noise'] += perf_counter() - t0

            t0 = perf_counter()
            overlapping_bounds = stamp.bounds & body_resultant_image.bounds
            if overlapping_bounds.area() > 0:
                body_accum_image[overlapping_bounds] += stamp[overlapping_bounds]
            timing['accumulate'] += perf_counter() - t0

            # Update the position for the next step
            moving_body_list[j].read_start_position = mb.read_end_position

        if read_num in saved_reads:
            if inv_linearity is not None:
                # Apply inverse linearity
                t0 = perf_counter()
                body_resultant_image += inv_linearity.apply(
                        body_accum_image.array, electrons=True)
                timing['linearity'] += perf_counter() - t0
            else:
                body_resultant_image += body_accum_image.copy()
            
            num_reads_in_resultant += 1

        t0 = perf_counter()
        if read_num in last_read_number_per_resultant:
            # add the new PSF to the resultant
            resultants[resultant_i,:,:] += body_resultant_image.array / num_reads_in_resultant

            resultant_i += 1
            # zero out the resultant array for the next resultant
            body_resultant_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)
            num_reads_in_resultant = 0    

        timing['resultant_finalize'] += perf_counter() - t0

    if enable_timing:
        total_elapsed = perf_counter() - overall_start
        slowest_component = max(timing, key=timing.get)
        log.info('simulate_body timing summary (s): total=%.6f', total_elapsed)
        for component, dt in sorted(timing.items(), key=lambda item: item[1], reverse=True):
            frac = (dt / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            log.info('  %s: %.6f (%.1f%%)', component, dt, frac)
        log.info('simulate_body slowest component: %s (%.6f s)',
                 slowest_component, timing[slowest_component])
        stop

    return resultants
