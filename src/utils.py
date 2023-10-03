""""
This file contains miscellaneous utility functions
"""
import math

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
import torch
import spectral
from scipy import ndimage, misc
import cv2 as cv

from src import constants


def apply_circular_mask(data: np.ndarray or torch.Tensor, h: int, w: int, center: tuple = None,
                        radius: int = None, masking_value=1) -> np.ndarray or torch.Tensor:
    """
    Create a circular mask and apply it to an image cube. Works for ndarrays and torch tensors.
    Mask creation from https://stackoverflow.com/a/44874588
    :param data:
        Data cube to be masked
    :param h:
        Height of image
    :param w:
        Width of image
    :param center:
        Center of the mask; if not given, will use center of the data
    :param radius:
        Radius of the mask, in pixels; if not given, will use the largest possible radius which will fit the whole circle
    :param masking_value:
        Value to be used to replace the masked values. Default is 1, other useful values are np.nan (or float('nan')) and 0.
    :return:
        Masked datacube
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    # # Plot to check mask shape
    # plt.imshow(mask)
    # plt.show()
    if type(data) == Tensor:
        device = data.device  # Check where the data is: GPU or CPU
        mask = Tensor(mask).to(device)  # Convert mask to tensor and move it to same device as data
        masked_data = data * mask
        masked_data = torch.where(masked_data == 0, masking_value, masked_data)
        # masked_data = masked_data + Tensor(
        #     abs(mask - 1))  # Convert masked values to ones instead of zeros to avoid problems with backprop
    else:
        # mask = abs((mask * 1))  # the above returns a mask of booleans, this converts it to int (somehow)
        masked_data = data * np.expand_dims(mask, axis=2)  # Need to have the same number of dimensions on mask and data
        masked_data = np.where(masked_data == 0, masking_value, masked_data)

    return masked_data


def crop2aspect_ratio(cube, aspect_ratio=1, keep_dim=None):
    """
    Crop a spectral image cube (numpy ndarray) according to aspect ratio given as parameter.
    :param cube:
        Spectral image cube to be cropped, dimension order (h, w, l)
    :param aspect_ratio:
        Desired aspect ratio of cropped data, default is 1/1
    :param keep_dim: int, 0 or 1
        Index of dimension IN THE INPUT CUBE that should be kept the same, default is None
    :return: cube:
        Cropped image cube as ndarray
    """

    # Data dimension order is (h, w, l)
    orig_h = cube.shape[0]
    orig_w = cube.shape[1]

    def cut_horizontally(cube, h):
        """Make two horizontal cuts removing data from top and bottom rows of image"""
        half_leftover = (orig_h - h) / 2
        start_i = math.floor(half_leftover)
        end_i = math.ceil(half_leftover)
        cube = cube[start_i:-end_i, :, :]

        # cube.h = h
        return cube

    def cut_vertically(cube, w):
        """Make two vertical cuts removing data from left and right of center"""
        half_leftover = (orig_w - w) / 2
        start_i = math.floor(half_leftover)
        end_i = math.ceil(half_leftover)

        cube = cube[:, start_i:-end_i, :]

        # cube.w = w
        return cube

    if orig_h > orig_w:  # if image is not horizontal or square, rotate 90 degrees
        cube = np.rot90(cube, axes=(0, 1))

        w = orig_h
        h = orig_w
        # I am so sorry, this is confusing
        orig_h = h
        orig_w = w

        if keep_dim is not None:
            keep_dim = abs(keep_dim - 1)  # if rotated, dimension to be kept the same changes from 0 to 1, or 1 to 0

    if orig_w > int(orig_h * aspect_ratio) and keep_dim != 1:
        h = orig_h
        w = int(orig_h * aspect_ratio)
        cube = cut_vertically(cube, w)
    else:
        h = int(orig_w * aspect_ratio)
        w = orig_w
        cube = cut_horizontally(cube, h)

    return cube


def join_VIR_VIS_and_IR(vis_cube, ir_cube, vis_wavelengths, ir_wavelengths, vis_fwhms, ir_fwhms):
    """
    Concatenate VIS and IR cubes from the VIR instrument of NASA Dawn. For the overlapping part of the cubes this
    function discards the VIS channels and uses only IR, since the VIS ones are much noisier. The function also does
    not attempt to adjust the levels of VIS and IR, but just assumes that the result is somewhat continuous.
    Args:
        vis_cube:
            VIR-VIS datacube as ndarray
        ir_cube:
            VIR-IR datacube as ndarray
        vis_wavelengths:
            Band center wavelengths for VIS as list
        ir_wavelengths:
            Band center wavelengths for IR as list
        vis_fwhms:
            Band full-width half-maximum values for VIS
        ir_fwhms:
            Band full-width half-maximum values for IR

    Returns:
            cube, wavelength vector, FWHM vector
    """
    # # Plot of VIS and IR spectrum in same fig: in the overlapping section the VIS is noisy, so discard it and use IR
    # plt.figure()
    # plt.plot(vis_wavelengths, vis_cube[60, 220, :])
    # plt.plot(ir_wavelengths, ir_cube[60, 220, :])
    # plt.show()

    min_wavelength_ir = ir_wavelengths[0]
    vis_indices = np.where(np.asarray(vis_wavelengths) < min_wavelength_ir)
    vis_cube = np.squeeze(vis_cube[:, :, vis_indices])  # remove extra dimension created from selecting with the indices

    vis_wavelengths = np.asarray(vis_wavelengths)[vis_indices]
    ir_wavelengths = np.asarray(ir_wavelengths)

    vis_fwhms = np.asarray(vis_fwhms)[vis_indices]
    ir_fwhms = np.asarray(ir_fwhms)

    # Combine the VIS and IR cubes, wavelengths and FWHMs
    cube = np.append(vis_cube, ir_cube, axis=2)
    wavelengths = np.append(vis_wavelengths, ir_wavelengths)
    fwhms = np.append(vis_fwhms, ir_fwhms)

    # plt.figure()
    # plt.plot(wavelengths, cube[60, 230, :])
    # plt.show()

    return cube, wavelengths, fwhms


def ASPECT_resampling(cube: np.ndarray, wavelengths, FWHMs):
    """
    Resampling a spectral image cube to match the wavelength channels of ASPECT NIR and SWIR. ASPECT channel center
    wavelengths and FWHMs are defined in constants.
    :param cube:
        Spectral image cube
    :param wavelengths:
        Wavelength vector of the spectral image cube
    :param FWHMs:
        Full-width-half-maximum values for all the wavelength channels of the cube
    :return:
        Resampled image cube
    """

    ASPECT_wavelengths = constants.ASPECT_wavelengths
    ASPECT_FWHMs = constants.ASPECT_FWHMs

    # if min(ASPECT_wavelengths) <= min(wavelengths):
    #     minimum = min(wavelengths)
    # else:
    #     minimum = min(ASPECT_wavelengths)
    #
    # if max(ASPECT_wavelengths) >= max(wavelengths):
    #     maximum = max(wavelengths)
    # else:
    #     maximum = max(ASPECT_wavelengths)

    resample = spectral.BandResampler(wavelengths, ASPECT_wavelengths, FWHMs, ASPECT_FWHMs)

    cube_resampled = np.zeros(shape=(cube.shape[0], cube.shape[1], len(ASPECT_wavelengths)))
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            cube_resampled[i, j, :] = resample(cube[i, j, :])

    return cube_resampled, ASPECT_wavelengths, ASPECT_FWHMs


def rot_and_crop_Dawn_VIR_ISIS(data, rot_deg, crop_indices_x, crop_indices_y, edge_detection=False):
    """
    Rotate and crop a DAWN VIR IR or VIS ISIS image cube, and detect edges with opencv Canny if so specified.
    :param data:
        Image cube to be rotated and cropped
    :param rot_deg:
        Degrees of rotation
    :param crop_indices_x:
        Start and stop indices in x-direction, as tuple
    :param crop_indices_y:
        Start and stop indices in y-direction, as tuple
    :param edge_detection:
        Whether edge detection is applied. If True returns edges, if False returns a None -type edge item.
    :return: data, edges:
        Rotated and cropped image cube, and detected edges
    """
    data = np.clip(data, 0, 1000)  # clip to get rid of the absurd masking values
    data = ndimage.rotate(data, rot_deg, mode='constant', axes=(1, 2))  # rotate to get the interesting area horizontal
    # plt.figure()
    # plt.imshow(data[100, :, :])
    # plt.show()
    data = data[:, crop_indices_y[0]:crop_indices_y[1], crop_indices_x[0]:crop_indices_x[1]]  # crop the masking values away
    # plt.figure()
    # plt.imshow(data[100, :, :])
    # plt.show()

    if edge_detection:
        image = data[constants.edge_detection_channel, :, :]  # pick one channel for edge detection
        image = image / np.max(image)  # normalization
        image = image * 256  # convert to 8-bit integer to make compatible for edge detection
        image = image.astype(np.uint8)
        edges = cv.Canny(image, constants.edge_detector_params[0], constants.edge_detector_params[1])  # Edge detection
    else:
        edges = None

    return data, edges


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


def ASPECT_NIR_SWIR_from_Dawn_VIR(cube, wavelengths, FWHMs, convert_rad2refl=True):
    """Take a spectral image from Dawn VIR and make it look like data from Milani's ASPECT's NIR and SWIR"""

    # Resample spectra to resemble ASPECT data
    cube, wavelengths, FWHMs = ASPECT_resampling(cube, wavelengths, FWHMs)

    # Convert radiances to I/F
    if convert_rad2refl:
        insolation = solar_irradiance(distance=constants.ceres_hc_dist, wavelengths=constants.ASPECT_wavelengths,
                                            plot=False, resample=True)
        cube = cube / insolation[:, 1]

    # # Sanity check plot
    # plt.imshow(cube[:, :, 20])
    # plt.show()

    # Crop the image to aspect ratio where one side is the larger of NIR FOV and one side is SWIR FOV
    aspect_ratio = max(constants.ASPECT_NIR_FOV) / constants.ASPECT_SWIR_FOV
    cube = crop2aspect_ratio(cube, aspect_ratio)

    # Interpolate each channel to have more pixels: width from NIR width, height from what the height would be
    height = constants.ASPECT_SWIR_equivalent_radius * 2  # height should be diameter of FOV, so 2 * radius
    width = constants.ASPECT_NIR_channel_shape[1]
    resized = np.zeros((height, width, cube.shape[2]))
    for channel in range(cube.shape[2]):
        resized[:, :, channel] = cv.resize(cube[:, :, channel], (width, height), interpolation=cv.INTER_AREA)
    cube = resized

    cube = apply_circular_mask(cube, height, width, masking_value=float('nan'))

    test_cube = crop2aspect_ratio(cube, aspect_ratio=constants.ASPECT_NIR_channel_shape[0] /
                                                            constants.ASPECT_NIR_channel_shape[1], keep_dim=1)

    cube_short = test_cube[:, :, :int(len(wavelengths) / 2)]
    cube_long = cube[:, :, int(len(wavelengths) / 2):]

    SWIR_data = np.nanmean(cube_long, axis=(0, 1))
    NIR_data = cube_short
    test_data = test_cube

    # # Sanity check plot
    # plt.figure()
    # plt.imshow(cube_short[:, :, 20])
    # plt.figure()
    # plt.imshow(cube_long[:, :, 20])
    # plt.show()

    return NIR_data, SWIR_data, test_data

# TODO Make this work someday
# def ASPECTify(cube, wavelengths, FWHMs, VIS=False, NIR1=True, NIR2=True, SWIR=True):
#     """Take a spectral image and make it look like data from Milani's ASPECT"""
#     if VIS:
#         print('Sorry, the function can not currently work with the VIS portion of ASPECT')
#         print('Stopping execution')
#         exit(1)
#
#     # # ASPECT wavelength vectors: change these values later, if the wavelengths change!
#     # ASPECT_VIS_wavelengths = np.linspace(start=0.650, stop=0.950, num=14)
#     # ASPECT_NIR1_wavelengths = np.linspace(start=0.850, stop=0.1250, num=14)
#     # ASPECT_NIR2_wavelengths = np.linspace(start=1.200, stop=0.1600, num=14)
#     # ASPECT_SWIR_wavelengths = np.linspace(start=1.650, stop=2.500, num=30)
#     # # ASPECT FOVs in degrees
#     # ASPECT_VIS_FOV = 10  # 10x10 deg square
#     # ASPECT_NIR_FOV_w = 6.7  # width
#     # ASPECT_NIR_FOV_h = 5.4  # height
#     # ASPECT_SWIR_FOV = 5.85  # circular
#
#     # Resample spectra to resemble ASPECT data
#     cube, wavelengths, FWHMs = ASPECT_resampling(cube, wavelengths, FWHMs)
#
#     if (NIR1 or NIR2) and SWIR:
#         # The largest is the SWIR circular FOV, and so it is the limiting factor
#         # Cut the largest possible rectangle where one side is circular FOV, other is width of NIR. Then divide
#         # along wavelength into NIR and SWIR, mask SWIR into circle and take mean spectrum, and cut NIR to size.
#         return None





