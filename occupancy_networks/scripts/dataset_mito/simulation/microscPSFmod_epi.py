#!/usr/bin/env python
"""
Generate a PSF using the Gibson and Lanni model.

Note: All distance units are microns.

This is slightly reworked version of the Python code provided by Kyle
Douglass, "Implementing a fast Gibson-Lanni PSF solver in Python".

http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python


References:

1. Li et al, "Fast and accurate three-dimensional point spread function computation
   for fluorescence microscopy", JOSA, 2017.

2. Gibson, S. & Lanni, F. "Experimental test of an analytical model of
   aberration in an oil-immersion objective lens used in three-dimensional
   light microscopy", J. Opt. Soc. Am. A 9, 154-166 (1992), [Originally
   published in J. Opt. Soc. Am. A 8, 1601-1613 (1991)].

3. Kirshner et al, "3-D PSF fitting for fluorescence microscopy: implementation
   and localization application", Journal of Microscopy, 2012.

Hazen 04/18
"""
import cmath
import math
import numpy
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.special

import time
import ipdb
from scipy.ndimage.interpolation import map_coordinates

import matplotlib.pyplot as plt

# Internal constants.
num_basis = 100     # Number of rescaled Bessels that approximate the phase function.
rho_samples = 1000  # Number of pupil sample along the radial direction.

# Microscope parameters.
# m_params = {"M" : 60.0,             # magnification
#             "NA" : 1.4,              # numerical aperture
#             "ng0" : 1.515,           # coverslip RI design value
#             "ng" : 1.515,            # coverslip RI experimental value
#             "ni0" : 1.515,           # immersion medium RI design value
#             "ni" : 1.515,            # immersion medium RI experimental value
#             "ns" : 1.33,             # specimen refractive index (RI)
#             "ti0" : 150,             # microns, working distance (immersion medium thickness) design value
#             "tg" : 170,              # microns, coverslip thickness experimental value
#             "tg0" : 170,             # microns, coverslip thickness design value
#             "zd0" : 200.0 * 1.0e+3}  # microscope tube length (in microns).


def calcRv(dxy, xy_size, sampling=4):
    """
    Calculate rv vector, this is 2x up-sampled.
    """
    # ipdb.set_trace()
    rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1
    return numpy.arange(0, rv_max * dxy, dxy / sampling)


def configure(mp, wvl):
    # Scaling factors for the Fourier-Bessel series expansion
    min_wavelength = 0.436 # 0.603 # microns
    scaling_factor = mp["NA"] * (3 * numpy.arange(1, num_basis + 1) - 2) * min_wavelength / wvl

    # Not sure this is completely correct for the case where the axial
    # location of the flourophore is 0.0.
    #
    # ipdb.set_trace()
    max_rho = min([mp["NA"], mp["ng0"], mp["ng"], mp["ni0"], mp["ni"], mp["ns"]]) / mp["NA"]

    return [scaling_factor, max_rho]


def deltaFocus(mp, zd):
    """
    Return focal offset needed to compensate for the camera being at zd.

    mp - The microscope parameters dictionary.
    zd - Actual camera position in microns.
    """
    a = mp["NA"] * mp["zd0"] / mp["M"]  # Aperture radius at the back focal plane.
    return a*a*(mp["zd0"] - zd)/(2.0*mp["zd0"]*zd)


def gLXYZCameraScan(mp, dxy, xy_size, zd, normalize = True, pz = 0.0, wvl = 0.6, zv = 0.0):
    """
    NOTE: Does not work!

    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the
    camera position (changing the microscope tube length).

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    zd - A numpy array containing the camera positions in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRCameraScan(mp, rv, zd, normalize = normalize, pz = pz, wvl = wvl, zv = zv)

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ(dxy, xy_size, zv, rv, PSF_rz)


def gLXYZFocalScan(mp, dxy, xy_size, zv, normalize = True, pz = 0.0, wvl = 0.6, zd = None):
    """
    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the microscopes
    focus.

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    # ipdb.set_trace()
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRFocalScan(mp, rv, zv, normalize = normalize, pz = pz, wvl = wvl, zd = zd)

    # Create XYZ PSF by interpolation.
    # return psfRZToPSFXYZ(dxy, xy_size, pz, zv, rv, PSF_rz)
    return psfRZToPSFXYZ(dxy, xy_size, pz, zv, rv, PSF_rz)
    # return psfRZToPSFXYZ(dxy, xy_size, pz, pz_, rv, PSF_rz, px, py).reshape((px.shape[0],-1))


def gLXYZParticleScan(mp, dxy, xy_size, pz, normalize = True, wvl = 0.688, zd = None, zv = 0.0, px=0, py=0):
    """
    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning a particle
    through the microscopes focus.

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)
    # ipdb.set_trace()

    # Sampling points in Z used for interpolation later
    pz_ = numpy.arange(-1.5, 1.5, step=0.001)
    # pz_ = numpy.arange(0, 1.5, step=0.001)
    # pz_ = numpy.arange(0, 5, step=0.001)


    # Calculate radial/Z PSF.
    PSF_rz = gLZRParticleScan(mp, rv, pz_, normalize = normalize, wvl = wvl, zd = zd, zv = zv)
    # ipdb.set_trace()
    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ(dxy, xy_size, pz, pz_, rv, PSF_rz, px, py).reshape((px.shape[0],-1))


def gLZRScan(mp, pz, rv, zd, zv, normalize = True, wvl = 0.6):
    """
    Calculate radial G-L at specified radius. This function is primarily designed
    for internal use. Note that only one pz, zd and zv should be a numpy array
    with more than one element. You can simulate scanning the focus, the particle
    or the camera but not 2 or 3 of these values at the same time.

    mp - The microscope parameters dictionary.
    pz - A numpy array containing the particle z position above the coverslip (positive values only).
    rv - A numpy array containing the radius values.
    zd - A numpy array containing the actual camera position in microns.
    zv - A numpy array containing the relative z offset value of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    """
    # ipdb.set_trace()
    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    a = mp["NA"] * mp["zd0"] / math.sqrt(mp["M"]*mp["M"] + mp["NA"]*mp["NA"])  # Aperture radius at the back focal plane.
    k = 2.0 * numpy.pi/wvl
    # ipdb.set_trace()
    ti = zv.reshape(-1,1) + mp["ti0"]
    pz = pz.reshape(-1,1)
    zd = zd.reshape(-1,1)

    opdt = OPD(mp, rho, ti, pz, wvl, zd)

    # Sample the phase
    #phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
    phase = numpy.exp(1j * opdt)

    # Define the basis of Bessel functions
    # Shape is (number of basis functions by number of rho samples)
    J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

    # Compute the approximation to the sampled pupil phase by finding the least squares
    # solution to the complex coefficients of the Fourier-Bessel expansion.
    # Shape of C is (number of basis functions by number of z samples).
    # Note the matrix transposes to get the dimensions correct.
    C, residuals, _, _ = numpy.linalg.lstsq(J.T, phase.T, rcond=-1)

    rv = rv*mp["M"]
    b = k * a * rv.reshape(-1, 1)/zd

    # Convenience functions for J0 and J1 Bessel functions
    J0 = lambda x: scipy.special.jv(0, x)
    J1 = lambda x: scipy.special.jv(1, x)

    # See equation 5 in Li, Xue, and Blu
    denom = scaling_factor * scaling_factor - b * b
    R = (scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho)
    R /= denom

    # The transpose places the axial direction along the first dimension of the array, i.e. rows
    # This is only for convenience.
    PSF_rz = (numpy.abs(R.dot(C))**2).T

    # Normalize to the maximum value
    if normalize:
        PSF_rz /= numpy.max(PSF_rz)

    return PSF_rz


def gLZRCameraScan(mp, rv, zd, normalize = True, pz = 0.0, wvl = 0.6, zv = 0.0):
    """
    NOTE: Does not work!

    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the camera position (changing the microscope
    tube length).

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zd - A numpy array containing the camera positions in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    pz = numpy.array([pz])
    zv = numpy.array([zv])

    return gLZRScan(mp, pz, rv, zd, zv, normalize = normalize, wvl = wvl)


def gLZRFocalScan(mp, rv, zv, normalize = True, pz = 0.0, wvl = 0.6, zd = None):
    """
    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the microscopes focus.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is
         closer to the objective) in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    if zd is None:
        zd = mp["zd0"]

    pz = numpy.array([pz])
    zd = numpy.array([zd])

    return gLZRScan(mp, pz, rv, zd, zv, normalize = normalize, wvl = wvl)


def gLZRParticleScan(mp, rv, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
    """
    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the particle relative to the microscopes focus.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    if zd is None:
        zd = mp["zd0"]

    zd = numpy.array([zd])
    zv = numpy.array([zv])

    return gLZRScan(mp, pz, rv, zd, zv, normalize = normalize, wvl = wvl)


def OPD(mp, rho, ti, pz, wvl, zd):
    """
    Calculate phase aberration term.

    mp - The microscope parameters dictionary.
    rho - Rho term.
    ti - Coverslip z offset in microns.
    pz - Particle z position above the coverslip in microns.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns.
    """
    NA = mp["NA"]
    ns = mp["ns"]
    ng0 = mp["ng0"]
    ng = mp["ng"]
    ni0 = mp["ni0"]
    ni = mp["ni"]
    ti0 = mp["ti0"]
    tg = mp["tg"]
    tg0 = mp["tg0"]
    zd0 = mp["zd0"]

    a = NA * zd0 / mp["M"]  # Aperture radius at the back focal plane.
    k = 2.0 * numpy.pi/wvl  # Wave number of emitted light.

    OPDs = pz * numpy.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample.
    OPDi = ti * numpy.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * numpy.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium.
    OPDg = tg * numpy.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * numpy.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip.
    OPDt = a * a * (zd0 - zd) * rho * rho / (2.0 * zd0 * zd) # OPD in camera position.

    return k * (OPDs + OPDi + OPDg + OPDt)


def psfRZToPSFXYZ(dxy, xy_size, zv, zv_, rv, PSF_rz, px=0, py=0):
    """
    # Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.
    # Modified
    # """
    # # Create XY grid of radius values.
    px = numpy.array(px).astype(numpy.float32)
    py = numpy.array(py).astype(numpy.float32)
    PSF_rz = PSF_rz.astype(numpy.float32)
    # t0 = time.perf_counter()
    # ipdb.set_trace()
    c_xy = float(xy_size) * 0.5
    c_xy = dxy * c_xy
    xy = numpy.mgrid[0:xy_size, 0:xy_size] + 0.5
    xy = dxy * xy
    x = xy[1].astype(numpy.float32)
    y = xy[0].astype(numpy.float32)
    X = numpy.repeat(x[None, :, :], zv.shape[0], axis=0)
    Y = numpy.repeat(y[None, :, :], zv.shape[0], axis=0)
    # r_pixel = dxy * numpy.sqrt((xy[1] - c_xy) * (xy[1] - c_xy) + (xy[0] - c_xy) * (xy[0] - c_xy))
    # r_pixel = numpy.sqrt((xy[1] - (c_xy - px) * (xy[1] - c_xy - px) + (xy[0] - c_xy - py) * (xy[0] - c_xy - py)))
    R_pixel = numpy.sqrt( \
        (X - c_xy - px[:,None,None]) * (X - c_xy - px[:,None,None]) + \
        (Y - c_xy - py[:,None,None]) * (Y - c_xy - py[:,None,None]) \
        ).ravel()
    R_pixel *= 1/(rv[1] - rv[0])
    # print("coordinates: ", time.perf_counter() - t0)
    """
    # Create XYZ PSF by interpolation.
    PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
    for i in range(PSF_rz.shape[0]):
        psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i,:], bounds_error=False, fill_value=0.0)
        PSF_xyz[i,:,:] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)
    """
    # #PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
    # #psf_rz_interp = scipy.interpolate.interp2d(rv, zv, PSF_rz, bounds_error=False, fill_value=0.0)
    # t0 = time.perf_counter()
    # #RV, ZV = numpy.meshgrid(rv, zv_)
    # #RV = RV.astype(numpy.float32).ravel()
    # #ZV = ZV.astype(numpy.float32).ravel()
    # #psf_interp =  scipy.interpolate.SmoothBivariateSpline(RV.ravel(), ZV.ravel(), PSF_rz.ravel())
    # #psf_interp =  scipy.interpolate.RectBivariateSpline(zv_, rv, PSF_rz)
    # # print("creating function: ", time.perf_counter() - t0)
    # t0 = time.perf_counter()
    # #for i in range(PSF_rz.shape[0]):
    # #    PSF_xyz[i,:,:] = psf_rz_interp(zv[i], r_pixel.ravel()).reshape(xy_size, xy_size)
    # #psf_rz_interp(zv, r_pixel.ravel()).reshape(xy_size, xy_size)
    # #r_pixel = r_pixel.ravel()
    # #r_pixel = numpy.repeat(r_pixel, PSF_rz.shape[0])
    # #Z = 700*numpy.ones(R_pixel.ravel().shape[0], numpy.float32)
    Z = numpy.repeat(zv/(zv_[1] - zv_[0]), x.shape[0] * x.shape[1])
    # #PSF_xyz = psf_interp(R_pixel.ravel(), Z, grid=False).reshape(zv.shape[0], xy_size, xy_size)
    # #PSF_xyz = scipy.interpolate.griddata((RV, ZV), PSF_rz.ravel(), (R_pixel.ravel(), Z), 'linear', fill_value=0).reshape(zv.shape[0], xy_size, xy_size)
    sampling_points = numpy.zeros((2,R_pixel.shape[0]), numpy.float32)
    sampling_points[0,:] = Z
    sampling_points[1,:] = R_pixel
    PSF_xyz = map_coordinates(PSF_rz, sampling_points)
    # # print("interpolation: ", time.perf_counter() - t0)

    return PSF_xyz


def slowGL(mp, max_rho, rv, zv, pz, wvl, zd):
    """
    Calculate a single point in the G-L PSF using integration. This
    is primarily provided for testing / reference purposes. As the
    function name implies, this is going to be slow.

    mp - The microscope parameters dictionary.
    max_rho - The maximum rho value.
    rv - A radius value in microns.
    zv - A z offset value (of the coverslip) in microns.
    pz - Particle z position above the coverslip in microns.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns.
    """
    a = mp["NA"] * mp["zd0"] / math.sqrt(mp["M"]*mp["M"] + mp["NA"]*mp["NA"])  # Aperture radius at the back focal plane.
    k = 2.0 * numpy.pi/wvl
    ti = zv + mp["ti0"]

    rv = rv*mp["M"]

    def integral_fn_imag(rho):
        t1 = k * a * rho * rv/zd
        t2 = scipy.special.jv(0, t1)
        t3 = t2*cmath.exp(1j*OPD(mp, rho, ti, pz, wvl, zd))*rho
        return t3.imag

    def integral_fn_real(rho):
        t1 = k * a * rho * rv/zd
        t2 = scipy.special.jv(0, t1)
        t3 = t2*cmath.exp(1j*OPD(mp, rho, ti, pz, wvl, zd))*rho
        return t3.real

    int_i = scipy.integrate.quad(lambda x: integral_fn_imag(x), 0.0, max_rho)[0]
    int_r = scipy.integrate.quad(lambda x: integral_fn_real(x), 0.0, max_rho)[0]

    t1 = k * a * a / (zd * zd)
    return t1 * (int_r * int_r + int_i * int_i)


def gLZRFocalScanSlow(mp, rv, zv, normalize = True, pz = 0.0, wvl = 0.6, zd = None):
    """
    This is the integration version of gLZRFocalScan.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    if zd is None:
        zd = mp["zd0"]

    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    psf_rz = numpy.zeros((zv.size, rv.size))
    for i in range(zv.size):
        for j in range(rv.size):
            psf_rz[i,j] = slowGL(mp, max_rho, rv[j], zv[i], pz, wvl, zd)

    if normalize:
        psf_rz = psf_rz/numpy.max(psf_rz)

    return psf_rz


def gLZRParticleScanSlow(mp, rv, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
    """
    This is the integration version of gLZRParticleScan.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    if zd is None:
        zd = mp["zd0"]

    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    psf_rz = numpy.zeros((pz.size, rv.size))
    for i in range(pz.size):
        for j in range(rv.size):
            psf_rz[i,j] = slowGL(mp, max_rho, rv[j], zv, pz[i], wvl, zd)

    if normalize:
        psf_rz = psf_rz/numpy.max(psf_rz)

    return psf_rz
