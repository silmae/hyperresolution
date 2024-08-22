""""
This file contains miscellaneous utility functions, mostly related to preprocessing of training data
"""
import copy
import math

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
import torch
import spectral
from scipy import ndimage, misc, interpolate
from scipy.integrate import trapezoid
import scipy
import cv2 as cv

from src import constants
import utils

# Epsilon value for outlier removal
_num_eps = 0.000001


def solar_irradiance(distance: float, wavelengths=constants.ASPECT_wavelengths, plot=False, resample=False):
    """
    Calculate solar spectral irradiance at a specified heliocentric distance, interpolated to match wl-vector.
    Solar spectral irradiance data at 1 AU outside the atmosphere was taken from NREL:
    https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html

    :param distance:
        Heliocentric distance in astronomical units
    :param wavelengths: vector of floats
        Wavelength vector (in µm), to which the insolation will be interpolated
    :param plot:
        Whether a plot will be shown of the calculated spectral irradiance

    :return: ndarray
        wavelength vector (in nanometers) and spectral irradiance in one ndarray
    """

    sol_path = constants.solar_path  # A collection of channels from 0.45 to 2.50 µm saved into a txt file

    solar = np.loadtxt(sol_path)

    # # Convert from µm to nm, and 1/µm to 1/nm. Comment these two lines away if working with micrometers
    # solar[:, 0] = solar[:, 0] * 1000
    # solar[:, 1] = solar[:, 1] / 1000

    # Scale with heliocentric distance, using the inverse square law
    solar[:, 1] = solar[:, 1] / distance**2

    if resample: # Resample to match the given wavelength vector
        resampled_solar = np.zeros((len(wavelengths), 2))
        resample = spectral.BandResampler(solar[:, 0], wavelengths)
        resampled_solar[:, 0] = wavelengths
        resampled_solar[:, 1] = resample(solar[:, 1])
        final = resampled_solar
    else:
        final = solar

    if plot == True:
        plt.plot(final[:, 0], final[:, 1])
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Irradiance [W / m² / µm]')
        plt.show()

    return final


def find_outliers(y: np.ndarray, x: np.ndarray or None = None,
                  z_thresh: float = 1., num_eps: float = _num_eps) -> np.ndarray:
    if x is None: x = np.arange(len(y))
    """Function by David Korda"""

    if len(np.unique(x)) != len(x):
        raise ValueError('"x" input must be unique.')

    inds = np.argsort(x)
    x_iterate, y_iterate = x[inds], y[inds]

    z_thresh = np.clip(z_thresh, a_min=num_eps, a_max=None)

    i = 0  # counts iterations (now needed only for edge outlier removal)

    def return_mean_std(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

    while True:
        deriv = np.diff(y_iterate) / np.diff(x_iterate)
        mu, sigma = return_mean_std(deriv)
        z_score = (deriv - mu) / sigma

        positive = np.where(z_score > z_thresh)[0]
        negative = np.where(-z_score > z_thresh)[0]

        # noise -> the points are next to each other (overlap if compensated for "diff" shift)
        # outliers = np.stack((np.intersect1d(positive, negative + 1), np.intersect1d(negative, positive + 1)))
        outliers = np.array([])
        outliers = np.append(outliers, np.intersect1d(positive, negative + 1))
        outliers = np.append(outliers, np.intersect1d(negative, positive + 1))

        if i == 0:  # check edges of the original data
            if 0 in positive or 0 in negative:  # first index is outlier
                outliers = np.append(outliers, [0])

            # last index is outlier
            if (len(z_score) - 1) in positive or (len(z_score) - 1) in negative:  # -1 to count "len" from 0
                outliers = np.append(outliers, [len(x_iterate) - 1])

        if np.size(outliers) == 0:
            break

        outliers = outliers.astype(int)
        x_iterate, y_iterate = np.delete(x_iterate, outliers), np.delete(y_iterate, outliers)
        i += 1

    # outliers are shifted due to deleting the two iterables
    return np.where([x_test not in x_iterate for x_test in x])[0]


def remove_outliers(y: np.ndarray, x: np.ndarray or None = None,
                    z_thresh: float = 1., num_eps: float = _num_eps) -> np.ndarray or tuple[np.ndarray, ...]:
    inds_to_remove = find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)
    """Function by David Korda"""

    if x is None:
        return np.delete(y, inds_to_remove)

    return np.delete(y, inds_to_remove), np.delete(x, inds_to_remove)


def interpolate_outliers(y: np.ndarray, x: np.ndarray or None = None,
                    z_thresh: float = 1., num_eps: float = _num_eps) -> np.ndarray:
    if x is None: x = np.arange(len(y))
    """Function by David Korda"""

    inds_to_remove = find_outliers(y=y, x=x, z_thresh=z_thresh, num_eps=num_eps)
    x_no_out, y_no_out = np.delete(x, inds_to_remove), np.delete(y, inds_to_remove)

    # interpolation first
    inds_in = np.logical_and(x >= np.min(x_no_out), x <= np.max(x_no_out))
    x_in = x[inds_in]  # x corrected for possible edges to avoid cubic extrapolation
    y_in = interpolate.interp1d(x_no_out, y_no_out, kind=gimme_kind(x_no_out))(x_in)

    # linearly extrapolate the interpolated values if needed
    return interpolate.interp1d(x_in, y_in, kind="linear", fill_value="extrapolate")(x)


def gimme_kind(x: np.ndarray) -> str:
    """Function by David Korda"""

    if len(x) > 3:
        return "cubic"
    if len(x) > 1:
        return "linear"
    return "nearest"


def denoise_array(array: np.ndarray, sigma: float, x: np.ndarray or None = None,
                  remove_mean: bool = False, sum_or_int: str or None = None) -> np.ndarray:
    """Function by David Korda.
    Modified to always use the method for equidistant data and never use the normalization"""

    if x is None:
        x = np.arange(0., np.shape(array)[-1])  # 0. to convert it to float

    # equidistant_measure = np.var(np.diff(x))

    # if equidistant_measure == 0.:  # equidistant step -> standard gaussian convolution
    step = x[1] - x[0]
    # correction = ndimage.gaussian_filter1d(np.ones(len(x)), sigma=sigma / step, mode="constant")
    array_denoised = ndimage.gaussian_filter1d(array, sigma=sigma / step, mode="nearest")

        # array_denoised = normalise_in_columns(array_denoised, norm_vector=correction)


    # else:  # transmission application
    #     if sum_or_int is None:  # 3 is randomly chosen. Better to do sum if there are too large gaps in wavelengths
    #         sum_or_int = "sum" if equidistant_measure > 3. else "int"
    #
    #     filter = scipy.norm.pdf(np.reshape(x, (len(x), 1)), loc=x, scale=sigma)  # Gaussian filter
    #
    #     # need num_filters x num_wavelengths
    #     if np.ndim(filter) == 1:
    #         filter = np.reshape(filter, (1, -1))
    #     if np.ndim(filter) > 2:
    #         raise ValueError("Filter must be 1-D or 2-D array.")
    #
    #     if sum_or_int == "sum":
    #         filter = normalise_in_rows(filter)
    #     else:
    #         filter = normalise_in_rows(filter, trapezoid(y=filter, x=x))
    #
    #     if sum_or_int == "sum":
    #         array_denoised = array @ np.transpose(filter)
    #     else:
    #         array_denoised = trapezoid(y=np.einsum('...j, kj -> ...kj', array, filter), x=x)

    if remove_mean:  # here I assume that the noise has a zero mean
        mn = np.mean(array_denoised - array, axis=-1, keepdims=True)
    else:
        mn = 0.

    return array_denoised - mn


def reflectance2SSA(reflectance, mu=1, mu0=1):
    """Convert spectral reflectance vector or cube to single-scattering albedo, for ndarray or torch tensor

    Equation from "HapkeCNN: Blind Nonlinear Unmixing for Intimate Mixtures Using Hapke Model and Convolutional
    Neural Network", Rasti et al. (2022).
    """
    if type(reflectance) == np.ndarray:
        ssa = 1 \
              - (
                (((mu + mu0)**2 * reflectance**2 + (1 + 4 * mu * mu0 * reflectance)*(1 - reflectance))**0.5 - (mu + mu0) * reflectance)
                / (1 + 4 * mu * mu0 * reflectance)
                 )**2

    elif type(reflectance) == Tensor:  # Here the same operations should probably work for arrays and tensors, right? Unless the powers don't
        ssa = 1 \
                - (
                    (((mu + mu0) ** 2 * reflectance ** 2 + (1 + 4 * mu * mu0 * reflectance) * (1 - reflectance)) ** 0.5 - (mu + mu0) * reflectance)
                    / (1 + 4 * mu * mu0 * reflectance)
                ) ** 2

    return ssa


def SSA2reflectance(ssa, mu=1, mu0=1):
    """Convert single scattering albedo vector or cube to spectral reflectance, for ndarray or torch tensor

    Equation from "HapkeCNN: Blind Nonlinear Unmixing for Intimate Mixtures Using Hapke Model and Convolutional
    Neural Network", Rasti et al. (2022).
    """
    if type(ssa) == np.ndarray:
        reflectance = ssa / ((1 + 2 * mu * np.sqrt((1 - ssa))) * (1 + 2 * mu0 * np.sqrt((1 - ssa))))
    elif type(ssa) == Tensor:
        reflectance = ssa / ((1 + 2*mu * torch.sqrt((1 - ssa))) * (1 + 2*mu0 * torch.sqrt((1 - ssa))))

    return reflectance


## The rest are currently not used
# def normalise_array(array: np.ndarray,
#                     axis: int or None = None,
#                     norm_vector: np.ndarray or None = None,
#                     norm_constant: float = 1.,
#                     num_eps: float = _num_eps) -> np.ndarray:
#     """Function by David Korda"""
#
#     if norm_vector is None:
#         norm_vector = np.nansum(array, axis=axis, keepdims=True)
#
#     # to force correct dimensions (e.g. when passing the output of interp1d)
#     if np.ndim(norm_vector) != np.ndim(array) and np.ndim(norm_vector) > 0:
#         norm_vector = np.expand_dims(norm_vector, axis=axis)
#
#     if np.any(np.abs(norm_vector) < num_eps):
#         print("You normalise with (almost) zero values. Check the normalisation vector.")
#
#     return array / norm_vector * norm_constant
#
#
# def normalise_in_columns(array: np.ndarray,
#                          norm_vector: np.ndarray or None = None,
#                          norm_constant: float = 1.) -> np.ndarray:
#     """Function by David Korda"""
#
#     return normalise_array(array, axis=0, norm_vector=norm_vector, norm_constant=norm_constant)
#
#
# def normalise_in_rows(array: np.ndarray,
#                       norm_vector: np.ndarray or None = None,
#                       norm_constant: float = 1.) -> np.ndarray:
#     """Function by David Korda"""
#
#     return normalise_array(array, axis=1, norm_vector=norm_vector, norm_constant=norm_constant)

