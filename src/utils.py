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
                        radius: int = None) -> np.ndarray or torch.Tensor:
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
    if type(data) == Tensor:
        device = data.device  # Check where the data is: GPU or CPU
        mask = Tensor(mask).to(device)  # Convert mask to tensor and move it to same device as data
        masked_data = data * mask
        masked_data = masked_data + Tensor(
            abs(mask - 1))  # Convert masked values to ones instead of zeros to avoid problems with backprop
    else:
        mask = abs((mask * 1))  # the above returns a mask of booleans, this converts it to int (somehow)
        masked_data = data * mask

    return masked_data


def crop_and_mask(dataset, aspect_ratio=1, radius=None):
    """
    Crop X, Y, and cube of a torch dataset according to aspect ratio given as parameter. Then create circular mask
    according to radius parameter, and apply it to the cropped data.
    :param dataset:
        torch dataset object with X, Y, and cube
    :param aspect_ratio:
        Desired aspect ratio of cropped data, default is 1/1
    :param radius:
        Radius of applied circular mask, in pixels
    :return:
        torch dataset object with X, Y, and cube cropped and masked
    """

    orig_w = dataset.w
    orig_h = dataset.h

    # Data dimension order is (l, w, h)

    def cut_horizontally(dataset, h):
        '''Make two horizontal cuts removing data from top and bottom rows of image'''
        half_leftover = (orig_h - h) / 2
        start_i = math.floor(half_leftover)
        end_i = math.ceil(half_leftover)

        dataset.X = dataset.X[:, :, start_i:-end_i]
        dataset.Y = dataset.Y[:, :, start_i:-end_i]
        dataset.cube = dataset.cube[:, :, start_i:-end_i]

        dataset.h = h
        return dataset

    def cut_vertically(dataset, w):
        '''Make two vertical cuts removing data from left and right of center'''
        half_leftover = (orig_w - w) / 2
        start_i = math.floor(half_leftover)
        end_i = math.ceil(half_leftover)

        dataset.X = dataset.X[:, start_i:-end_i, :]
        dataset.Y = dataset.Y[:, start_i:-end_i, :]
        dataset.cube = dataset.cube[:, start_i:-end_i, :]

        dataset.w = w
        return dataset

    if orig_h > orig_w:  # if image is not horizontal or square, rotate 90 degrees
        dataset.X = torch.rot90(dataset.X, dims=(1, 2))
        dataset.Y = torch.rot90(dataset.Y, dims=(1, 2))
        dataset.cube = np.rot90(dataset.cube, axes=(1, 2))

        dataset.w = orig_h
        dataset.h = orig_w

        orig_h = dataset.h
        orig_w = dataset.w

    if orig_w > int(orig_h * aspect_ratio):
        h = orig_h
        w = int(orig_h * aspect_ratio)
        dataset = cut_vertically(dataset, w)
    else:
        h = int(orig_w * (1 / aspect_ratio))
        w = orig_w
        dataset = cut_horizontally(dataset, h)

    if radius is None:  # if no radius is given, use the minimum of h and w after cropping
        radius = int(min([dataset.h, dataset.w]) / 2)

    # # Plot before and after mask is applied
    # plt.imshow(np.nanmean(dataset.X,
    #                       0))  # The plot will appear in wrong orientation due to matplotlib expecting the indices in a certain order
    # plt.show()

    dataset.X = apply_circular_mask(dataset.X, dataset.w, dataset.h, radius=radius)
    dataset.Y = apply_circular_mask(dataset.Y, dataset.w, dataset.h, radius=radius)
    dataset.cube = apply_circular_mask(dataset.cube, dataset.w, dataset.h, radius=radius)

    # # Sanity check plot
    # plt.imshow(np.nanmean(dataset.X,
    #                       0) + 1)  # matplotlib wants its dimensions in a different order, which makes the plot look like h and w are mixed
    # plt.show()

    return dataset


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


# TODO Make this function work, but later
    # def ASPECTify(cube, wavelengths, VIS=False, NIR1=True, NIR2=True, SWIR=True):
    #     """Take a spectral image and make it look like data from Milani's ASPECT"""
    #     if VIS:
    #         print('Sorry, the function can not currently work with the VIS portion of ASPECT')
    #         print('Stopping execution')
    #         exit(1)
    #
    #     # ASPECT wavelength vectors: change these values later, if the wavelengths change!
    #     ASPECT_VIS_wavelengths = np.linspace(start=0.650, stop=0.950, num=14)
    #     ASPECT_NIR1_wavelengths = np.linspace(start=0.850, stop=0.1250, num=14)
    #     ASPECT_NIR2_wavelengths = np.linspace(start=1.200, stop=0.1600, num=14)
    #     ASPECT_SWIR_wavelengths = np.linspace(start=1.650, stop=2.500, num=30)
    #     # ASPECT FOVs in degrees
    #     ASPECT_VIS_FOV = 10  # 10x10 deg square
    #     ASPECT_NIR_FOV_w = 6.7  # width
    #     ASPECT_NIR_FOV_h = 5.4  # height
    #     ASPECT_SWIR_FOV = 5.85  # circular
    #
    #     if (NIR1 or NIR2) and SWIR:
    #         # The largest is the SWIR circular FOV, and so it is the limiting factor
    #         # Cut the largest possible rectangle where one side is circular FOV, other is width of NIR. Then divide
    #         # along wavelength into NIR and SWIR, mask SWIR into circle and take mean spectrum, and cut NIR to size.
    #         return None
    #
    #
    # training_data_ASPECT = ASPECTify(training_data.cube, training_data.wavelengths)


