import numpy as np
import galsim
import webbpsf as wpsf
from romanisim import psf

from . import parameters
from . import log

def simulate_body(
    resultants,
    times,
    tstart=None,
    wcs=None,
    rng=None,
    seed=47,
    magnitude=None,
    start_position=None,
    angular_radius=None,
    angular_speed=None,
    direction=None,
    oversample=4,
):
    #TODO: update this doc string!
    """Adds a moving body to an existing image.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    times : list[list[float]]
        list of list of readout times for each read entering a resultant
    tstart : astropy.time.Time
        Time of exposure start.  Used only if persistence is not None.
    wcs : galsim.image.wcs
        WCS associated with the galsim image
    rng : np.random.Generator
        Random number generator to use
    seed : int
        seed to use for random number generator
    magnitude : float
        Moving body flux, units of mag. 
    start_position : tuple of 2 floats
        Pixel starting position of the moving body, with science frame (X,Y).
    angular_radius : float
        Radius of the body in arcsec. If <= 0, assumes point source.
    angular_speed : float
        Speed of the body moving across the sky in arcsec s^-1.
    direction : float
        Angle relative to the science frame in degrees.
    oversample : int
        oversampling with which to sample WebbPSF PSF

    Returns
    -------
    resultants : np.ndarray[n_resultant, nx, ny]
        array of n_resultant images giving each resultant
    """

    # TODO: add persistence once prototype is working
    # TODO: add WCS querying to allow RA and DEC inputs for more realistic obs
    #       xpos, ypos = wcs._xy(coord[:, 0], coord[:, 1])

    pixel_scale = parameters.pixel_scale
    if wcs is None:
        wcs = galsim.PixelScale(pixel_scale)
    if rng is None:
        rng = galsim.BaseDeviate(seed)

    if magnitude is None:
        magnitude = parameters.moving_body["magnitude"]
    if start_position is None:
        start_position = parameters.moving_body["start_position"]
    if angular_radius is None:
        angular_radius = parameters.moving_body["angular_radius"]
    if angular_speed is None:
        angular_speed = parameters.moving_body["angular_speed"]
    if direction is None:
        direction = parameters.moving_body["direction"]

    filter_name = parameters.default_parameters_dictionary['instrument']['optical_element']
    detector = parameters.default_parameters_dictionary['instrument']['detector']

    detector_ind = int(detector[-2:])
    
    moving_psf = psf.make_psf(detector_ind, filter_name, wcs=wcs, variable=False, oversample=oversample) ## TODO: should variable=True?

    photon_flux = 1000000#10.**(magnitude / -2.5)
    #profile = galsim.DeltaFunction().withFlux(photon_flux)

    if angular_radius < 0:
        height = 0.0001
    else:
        height = 2*angular_radius

    cos_dir = np.cos(direction * np.pi / 180.0)
    sin_dir = np.sin(direction * np.pi / 180.0)

    x_vel = angular_speed * cos_dir / pixel_scale
    y_vel = angular_speed * sin_dir / pixel_scale

    body_full_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)

    last_time = 0
    if not isinstance(start_position, np.ndarray):
        last_position = np.array(start_position)
    else:
        last_position = start_position

    for i_res in range(resultants.shape[0]):
        # make blank reads with PSF at that location
        body_resultant_image = galsim.Image(resultants.shape[1], resultants.shape[2], init_value=0)
        for read_time in times[i_res]:
            # calculate new position
            elapsed_time = read_time - last_time
            end_position = np.array([last_position[0] + elapsed_time * x_vel,
                                      last_position[1] + elapsed_time * y_vel])
            psf_position = (last_position + end_position) / 2. #adjust to make the boxes not overlap

            last_time = read_time
            last_position = end_position

            width = angular_speed * elapsed_time
            profile = galsim.Box(width, height).withFlux(photon_flux * elapsed_time).rotate(galsim.Angle(direction, unit=galsim.degrees))

            # add new psf at the read position
            if hasattr(moving_psf, 'at_position'):
                psf0 = moving_psf.at_position(psf_position[1], psf_position[0])
            else:
                psf0 = moving_psf

            body_conv = galsim.Convolve(profile, psf0)
            image_pos = galsim.PositionD(psf_position[0], psf_position[1])
            pwcs = wcs.local(image_pos)
            stamp = body_conv.drawImage(center=image_pos, wcs=pwcs)
            stamp.addNoise(galsim.PoissonNoise(rng))

            overlapping_bounds = stamp.bounds & body_resultant_image.bounds
            if overlapping_bounds.area() > 0:
                body_resultant_image[overlapping_bounds] += stamp[overlapping_bounds]
        
        # average the reads into a single resultant
        body_resultant_image /= len(times[i_res])
        body_full_image += body_resultant_image

        # add the new PSF to the resultant
        resultants[i_res,:,:] += body_full_image.array

    return resultants
