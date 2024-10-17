"""Functions related to simulated data: constructing simulated Didymos images from simulated data by Penttilä et al.,
and simulating ASPECT-like images from spectral image cubes"""

import copy
import math

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
import torch
import spectral
from scipy import ndimage, misc, interpolate
import scipy
import cv2 as cv

from src import utils
from src import constants


def ASPECT_resampling(cube: np.ndarray, wavelengths, FWHMs=None):
    """
    Resampling a spectral image cube to match the wavelength channels of ASPECT NIR and SWIR. ASPECT channel center
    wavelengths and FWHMs are defined in constants.
    :param cube:
        Spectral image cube, or point spectrum
    :param wavelengths:
        Wavelength vector of the spectral image cube
    :param FWHMs:
        Full-width-half-maximum values for all the wavelength channels of the cube
    :return:
        Resampled image cube, ASPECT wavelengths, ASPECT FWHMs
    """

    ASPECT_wavelengths = constants.ASPECT_wavelengths
    ASPECT_FWHMs = constants.ASPECT_FWHMs

    resample = spectral.BandResampler(wavelengths, ASPECT_wavelengths, FWHMs, ASPECT_FWHMs)

    if len(cube.shape) == 3: # spectral image cube has two spatial dimensions and one spectral
        cube_resampled = np.zeros(shape=(cube.shape[0], cube.shape[1], len(ASPECT_wavelengths)))
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                cube_resampled[i, j, :] = resample(cube[i, j, :])
    elif len(cube.shape) == 1:
        cube_resampled = resample(cube)

    return cube_resampled, ASPECT_wavelengths, ASPECT_FWHMs


def ASPECT_NIR_SWIR_from_cube(cube: np.ndarray, wavelengths, FWHMs, convert_rad2refl=True, smoothing=True, vignetting=True, data_shape='actual'):
    """Take a spectral image and make it look like data from Milani's ASPECT's NIR and SWIR. Resamples
    the spectra to match ASPECT wavelengths given in constants.py, converts radiances of the original into I/F if
    specified in parameters. Calculates a mean spectrum from an area corresponding to SWIR FOV, cuts the shorter
    wavelength image into an image matching ASPECT NIR FOV and interpolates to match the pixel count of NIR.

    :param cube:
        Spectral image cube to be converted to look like ASPECT data
    :param wavelengths:
        Wavelength vector of the input spectral image cube
    :param FWHMs:
        Full-width-half-maximum vector of the wavelength channels, same length as the wavelength vector
    :param convert_rad2refl:
        Whether radiances of the input cube are converted to reflectances (to I/F)
    :param smoothing:
        Whether the spectra of the input cube image should go through outlier removal and Gaussian smoothing
    :param vignetting:
        Whether a vignette is applied on the SWIR spectral image before it is averaged to get a single spectrum
    :param datashape: 'actual' or 'full_cube' or 'VNIR_cube'
        Shape of returned data. 'actual' refers to the real ASPECT configuration with a spectral image cube in VIS and
        NIR and a point spectrum for SWIR. 'full_cube' means a full length (VIS+NIR+SWIR) image cube, and 'VNIR_cube' means
        an image cube covering VIS and NIR areas with nothing at SWIR.
    :return: VIS_and_NIR_data, SWIR_data, test_data
        Short wavelength spectral image cube, long wavelength point spectrum, complete spectral image cube with
        wavelength channels covering the whole ASPECT wavelength range
    """

    if smoothing:
        # Outlier removal and denoising: asteroid spectra should be very smooth with features tens or even hundreds nm wide
        unsmoothed_spectrum = copy.deepcopy(cube[50, 50, :])  # save an unsmoothed spectrum for comparison later
        orig_wls = copy.deepcopy(wavelengths)
        # Remove outliers separately for each spectrum
        smoothed_cube = np.zeros(shape=np.shape(cube))
        for i in range(np.shape(cube)[0]):
            for j in range(np.shape(cube)[1]):
                test_spectrum = cube[i, j, :]
                smoothed_cube[i, j, :] = utils.interpolate_outliers(test_spectrum, wavelengths, num_eps=1e-20)  # smaller epsilon increases sensitivity
                # print(f'Removed outliers from spectrum ({i}, {j}) out of {np.shape(cube)[:2]}')
        # plt.figure()
        # plt.plot(wavelengths, cube[50, 50, :])
        # plt.plot(wavelengths, smoothed_cube[50, 50, :])
        # plt.show()
        cube = smoothed_cube

        # Smoothing with a Gaussian kernel: once before resampling and converting to I/F, again after
        smoothed_cube = utils.denoise_array(cube, 0.01, wavelengths)
        # plt.figure()
        # plt.plot(wavelengths, cube[50, 50, :])
        # plt.plot(wavelengths, smoothed_cube[50, 50, :])
        # plt.show()
        cube = smoothed_cube

    # Resample spectra to resemble ASPECT data
    # if not wavelengths == constants.ASPECT_wavelengths:
    cube, wavelengths, FWHMs = ASPECT_resampling(cube, wavelengths, FWHMs)

    # Convert radiances to I/F
    if convert_rad2refl:
        insolation = utils.solar_irradiance(distance=constants.didymos_hc_dist, wavelengths=constants.ASPECT_wavelengths,
                                            plot=False, resample=True)
        cube = cube / insolation[:, 1]

    if smoothing:
        # Convert an unsmoothed test spectrum
        unsmoothed_spectrum = unsmoothed_spectrum / utils.solar_irradiance(distance=constants.vesta_hc_dist, wavelengths=orig_wls,
                                                plot=False, resample=True)[:, 1]

        # Smoothing with a Gaussian kernel: second time, now for the resampled and I/F converted spectra
        smoothed_cube = utils.denoise_array(cube, 0.02, wavelengths)
        # plt.figure()
        # # plt.plot(wavelengths, cube[50, 50, :])
        # plt.plot(orig_wls, unsmoothed_spectrum, label='Original')
        # plt.plot(wavelengths, smoothed_cube[50, 50, :], label='Smoothed')
        # plt.xlabel('Wavelength [µm]')
        # plt.ylabel('I/F')
        # plt.legend()
        # plt.show()
        cube = smoothed_cube

    add_noise = True
    # Apply gaussian noise to the image
    if add_noise:
        rng = np.random.default_rng(seed=42)
        mu, sigma = 0, 0.01  # mean and standard deviation
        s = rng.normal(mu, sigma, np.shape(cube))
        s = np.clip(s, a_min=-2*sigma, a_max=2*sigma)
        cube = cube + s + 2*sigma
        # plt.figure()
        # plt.imshow(np.sum(cube, axis=-1))
        # plt.plot(cube[100, 100, :])
        # cube = np.clip(cube, a_max=1, a_min=0)
        # plt.plot(cube[100, 100, :])
        # plt.show()
        # print('')

    VIS_and_NIR_data, SWIR_data, test_data = cube2ASPECT_data(cube, vignetting)

    # Sanity check plots
    # plt.figure()
    # plt.imshow(cube_short[:, :, 20])
    # plt.figure()
    # plt.imshow(cube_long[:, :, 20])
    # plt.figure()
    # plt.plot(test_cube[300, 300, :])
    # plt.plot(test_cube[200, 200, :])
    # plt.plot(test_cube[400, 400, :])
    # plt.show()
    if data_shape == 'actual':
        return VIS_and_NIR_data, SWIR_data, test_data
    elif data_shape == 'full_cube':
        return test_data, None, test_data
    elif data_shape == 'VNIR_cube':
        return VIS_and_NIR_data, None, VIS_and_NIR_data


def cube2ASPECT_data(cube: np.ndarray, vignetting=True):
    """
    Manipulate a spectral image cube to match the spatial dimensions of ASPECT data. Returns the VIS and NIR as one
    cube that has the spatial dimensions of the NIR data. Returns also the SWIR point spectrum, and a cube that extends
    the whole wavelength range and has the NIR spatial dimensions.
    :param cube:
        Spectral image cube as ndarray
    :param vignetting:
        Whether to apply vignetting to the SWIR part
    :return VIS_and_NIR_data, SWIR_data, test_data:
        Image cube with combined wavelengths range of the VIS and NIR modules and spatial dimensions of NIR, SWIR point
        spectrum, and full length image cube with spatial dimensions of NIR
    """

    # Crop the image to aspect ratio where one side is the larger of NIR FOV and one side is SWIR FOV
    aspect_ratio = max(constants.ASPECT_NIR_FOV) / constants.ASPECT_SWIR_FOV
    cube = crop2aspect_ratio(cube, aspect_ratio)
    # Interpolate each channel to have more pixels: width from NIR width, height from what the height would be
    height = constants.ASPECT_SWIR_equivalent_radius * 2  # height should be diameter of FOV, so 2 * radius
    width = constants.ASPECT_NIR_channel_shape[1]
    minimi = np.min(np.nan_to_num(cube))

    cube = resize_image_cube(cube, height, width)

    cube = apply_circular_mask(cube + 1e-6, height, width, masking_value=float('nan'))

    test_cube = crop2aspect_ratio(cube, aspect_ratio=constants.ASPECT_NIR_channel_shape[0] /
                                                     constants.ASPECT_NIR_channel_shape[1], keep_dim=1)

    if len(test_cube[0, 0, :]) < constants.ASPECT_SWIR_start_channel_index:
        # If less channels than VNIR, the calculated cube is abundance maps ground truth and is returned now
        return test_cube

    cube_short = test_cube[:, :, :constants.ASPECT_SWIR_start_channel_index]
    cube_long = cube[:, :, constants.ASPECT_SWIR_start_channel_index:]

    # Apply vignetting on the SWIR part, the shorter channels are considered ideally flat-fielded
    if vignetting:
        cube_long = apply_vignetting(cube_long)

    SWIR_data = np.nanmean(cube_long, axis=(0, 1))
    VIS_and_NIR_data = cube_short
    test_data = test_cube

    return VIS_and_NIR_data, SWIR_data, test_data


def apply_vignetting(cube: np.ndarray, sigma=None, keep_mean_intensity=True):
    """
    Applies Gaussian vignetting to each channel of spectral image.
    From https://www.geeksforgeeks.org/create-a-vignette-filter-using-python-opencv/
    :param cube:
        Spectral image cube
    :param sigma:
        Width of Gaussian kernel. If None, uses half of image height
    :param keep_mean_intensity:
        Whether to keep the mean intensity of the input cube: if true, adjust vignetted cube by multiplying with constant
    :return:
        Vignetted spectral image cube
    """
    cols, rows = cube.shape[0], cube.shape[1]

    if sigma == None:
        sigma = cols * 0.5

    # generating vignette mask using Gaussian
    # resultant_kernels
    Y_resultant_kernel = cv.getGaussianKernel(rows, sigma)
    X_resultant_kernel = cv.getGaussianKernel(cols, sigma)

    # generating resultant_kernel matrix
    resultant_kernel = X_resultant_kernel * Y_resultant_kernel.T

    # creating mask and normalising by using np.linalg
    # function
    # mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    mask = resultant_kernel / np.max(resultant_kernel)
    vignetted_cube = np.copy(cube)

    # applying the mask to each channel in the input image
    for i in range(cube.shape[2]):  # Looping is slow, but there are few channels in ASPECT data
        vignetted_cube[:, :, i] = vignetted_cube[:, :, i] * mask

    if keep_mean_intensity:
        orig_mean_intensity = np.nanmean(cube)
        vignetted_mean_intensity = np.nanmean(vignetted_cube)
        vignetted_cube = vignetted_cube / (vignetted_mean_intensity / orig_mean_intensity)
        vignetted_mean_intensity = np.nanmean(vignetted_cube)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].imshow(cube[:, :, 10])
    # ax[1].imshow(vignetted_cube[:, :, 10])
    # plt.show()

    return vignetted_cube



def apply_circular_mask(data: np.ndarray or torch.Tensor, h: int, w: int, center: tuple = None,
                        radius: int = None, masking_value=1, return_mask=False) -> np.ndarray or torch.Tensor:
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
    if return_mask:
        return masked_data, mask
    else:
        return masked_data


def resize_image_cube(cube, height, width):
    """Interpolate spectra image cube to have spatial dimensions given as inputs (no spectral interpolation)"""
    resized = np.zeros((height, width, cube.shape[2]))

    for channel in range(cube.shape[2]):
        resized[:, :, channel] = cv.resize(cube[:, :, channel], (width, height), interpolation=cv.INTER_LINEAR)

    return resized


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


def resample_spectrum(spectrum, old_wls, new_wls, old_FWHMs=None, new_FWHMs=None):
    """Resample a spectrum to a new wavelength vector. Uses FWHM values in the resampling, if those are given."""
    if old_FWHMs is None and new_FWHMs is None:
        resample = spectral.BandResampler(old_wls, new_wls)
    else:
        resample = spectral.BandResampler(old_wls, new_wls, old_FWHMs, new_FWHMs)
    spectrum = resample(spectrum)
    return spectrum


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
