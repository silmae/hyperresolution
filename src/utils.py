""""
This file contains miscellaneous utility functions
"""
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import spectral.io.envi as envi
from torch import Tensor
import torch
from planetaryimage import CubeFile
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


def open_Dawn_VIR_ISIS(cub_path='./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub'):
    isisimage = CubeFile.open(str(cub_path))
    cube = isisimage.data
    return cube, isisimage


def open_DAWN_VIR_PDS3_as_ENVI(label_path='./datasets/DAWN/VIR_IR_1B_1_488154033_1.LBL'):
    """
    Open a PDS3 qube file from the DAWN spacecraft VIR instrument as an ENVI file. The function generates an ENVI
    header file from the PDS3 label file given as parameter, saves the header to disc, and uses it to open
    the qube file associated with the label (same filename apart from extension). Can open both VIR IR and VIR VIS data.
    :param label_path:
        Path to the .LBL file, which must be in the same folder as the .QUB file
    :return:
        Spectral image cube as numpy ndarray
    """

    # Replace file extension to get paths for ENVI header and the QUBE file associated with the label
    if type(label_path) is not str:
        label_path = str(label_path)
    hdr_path = label_path[:-3] + 'hdr'
    qube_path = label_path[:-3] + 'QUB'

    lbl_dict = {}
    with open(label_path, 'r') as PDS3label:
        lines = PDS3label.readlines()
        for line in lines:
            if '=' in line:  # only read lines where values are assigned, not titles and such
                key, value = line.split('=', 1)
                lbl_dict[key.strip()] = value.strip()  # strip away whitespaces, append to dictionary

        def list_cleaner(string_list):
            for i in range(len(string_list)):
                string_list[i] = string_list[i].replace("(", "")
                string_list[i] = string_list[i].replace(")", "")
                string_list[i] = string_list[i].strip()
            return string_list

        order = lbl_dict['AXIS_NAME'].split(',')  # this includes the order of the dimensions, e.g. (LINE, SAMPLE, BAND)
        order = list_cleaner(order)
        for i in range(3):
            order[i] = order[i] + 'S'  # ENVI wants the labels as "lineS", "bandS"

        number = lbl_dict['CORE_ITEMS'].split(',')
        number = list_cleaner(number)

    # Write and ENVI header file using the dimensions of bands, lines, samples extracted from the label
    # The wavelengths and FWHMs of the channels are hardcoded which is pretty lazy
    with open(hdr_path, 'w') as header:
        if '_IR_' in str(label_path):
            header.write('ENVI \n'
                         'description = {DAWN VIR IR data} \n'
                         f'{order[0]} = {number[0]} \n'
                         f'{order[1]} = {number[1]} \n'
                         f'{order[2]} = {number[2]} \n'
                         'header offset = 0 \n'
                         'data type = 4 \n'  # 4 means float32 (IEEE)
                         'interleave = BIP \n'
                         'sensor type = DAWN VIR IR \n'
                         'byte order = 1 \n'
                         'wavelength = { \n'
                         '1.021,1.030,1.040,1.049,1.059,1.068,1.078,1.087,1.096,1.106,1.115,1.125, \n'
                         '1.134,1.144,1.153,1.163,1.172,1.182,1.191,1.200,1.210,1.219,1.229,1.238, \n'
                         '1.248,1.257,1.267,1.276,1.286,1.295,1.305,1.314,1.323,1.333,1.342,1.352, \n'
                         '1.361,1.371,1.380,1.390,1.399,1.409,1.418,1.428,1.437,1.446,1.456,1.465, \n'
                         '1.475,1.484,1.494,1.503,1.513,1.522,1.532,1.541,1.550,1.560,1.569,1.579, \n'
                         '1.588,1.598,1.607,1.617,1.626,1.636,1.645,1.655,1.664,1.673,1.683,1.692, \n'
                         '1.702,1.711,1.721,1.730,1.740,1.749,1.759,1.768,1.777,1.787,1.796,1.806, \n'
                         '1.815,1.825,1.834,1.844,1.853,1.863,1.872,1.882,1.891,1.900,1.910,1.919, \n'
                         '1.929,1.938,1.948,1.957,1.967,1.976,1.986,1.995,2.005,2.014,2.023,2.033, \n'
                         '2.042,2.052,2.061,2.071,2.080,2.090,2.099,2.109,2.118,2.127,2.137,2.146, \n'
                         '2.156,2.165,2.175,2.184,2.194,2.203,2.213,2.222,2.232,2.241,2.250,2.260, \n'
                         '2.269,2.279,2.288,2.298,2.307,2.317,2.326,2.336,2.345,2.355,2.364,2.373, \n'
                         '2.383,2.392,2.402,2.411,2.421,2.430,2.440,2.449,2.459,2.468,2.477,2.487, \n'
                         '2.496,2.506,2.515,2.525,2.534,2.544,2.553,2.563,2.572,2.582,2.591,2.600, \n'
                         '2.610,2.619,2.629,2.638,2.648,2.657,2.667,2.676,2.686,2.695,2.705,2.714, \n'
                         '2.723,2.733,2.742,2.752,2.761,2.771,2.780,2.790,2.799,2.809,2.818,2.827, \n'
                         '2.837,2.846,2.856,2.865,2.875,2.884,2.894,2.903,2.913,2.922,2.932,2.941, \n'
                         '2.950,2.960,2.969,2.979,2.988,2.998,3.007,3.017,3.026,3.036,3.045,3.055, \n'
                         '3.064,3.073,3.083,3.092,3.102,3.111,3.121,3.130,3.140,3.149,3.159,3.168, \n'
                         '3.177,3.187,3.196,3.206,3.215,3.225,3.234,3.244,3.253,3.263,3.272,3.282, \n'
                         '3.291,3.300,3.310,3.319,3.329,3.338,3.348,3.357,3.367,3.376,3.386,3.395, \n'
                         '3.405,3.414,3.423,3.433,3.442,3.452,3.461,3.471,3.480,3.490,3.499,3.509, \n'
                         '3.518,3.527,3.537,3.546,3.556,3.565,3.575,3.584,3.594,3.603,3.613,3.622, \n'
                         '3.632,3.641,3.650,3.660,3.669,3.679,3.688,3.698,3.707,3.717,3.726,3.736, \n'
                         '3.745,3.754,3.764,3.773,3.783,3.792,3.802,3.811,3.821,3.830,3.840,3.849, \n'
                         '3.859,3.868,3.877,3.887,3.896,3.906,3.915,3.925,3.934,3.944,3.953,3.963, \n'
                         '3.972,3.982,3.991,4.000,4.010,4.019,4.029,4.038,4.048,4.057,4.067,4.076, \n'
                         '4.086,4.095,4.104,4.114,4.123,4.133,4.142,4.152,4.161,4.171,4.180,4.190, \n'
                         '4.199,4.209,4.218,4.227,4.237,4.246,4.256,4.265,4.275,4.284,4.294,4.303, \n'
                         '4.313,4.322,4.332,4.341,4.350,4.360,4.369,4.379,4.388,4.398,4.407,4.417, \n'
                         '4.426,4.436,4.445,4.454,4.464,4.473,4.483,4.492,4.502,4.511,4.521,4.530, \n'
                         '4.540,4.549,4.559,4.568,4.577,4.587,4.596,4.606,4.615,4.625,4.634,4.644, \n'
                         '4.653,4.663,4.672,4.682,4.691,4.700,4.710,4.719,4.729,4.738,4.748,4.757, \n'
                         '4.767,4.776,4.786,4.795,4.804,4.814,4.823,4.833,4.842,4.852,4.861,4.871, \n'
                         '4.880,4.890,4.899,4.909,4.918,4.927,4.937,4.946,4.956,4.965,4.975,4.984, \n'
                         '4.994,5.003,5.013,5.022,5.032,5.041,5.050,5.060,5.069,5.079,5.088,5.098} \n'
                         'fwhm = {\n'
                         '0.0140,0.0140,0.0140,0.0140,0.0140,0.0140,0.0140,0.0140,0.0140,0.0139,       \n'
                         '0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,\n'
                         '0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0139,0.0138,0.0138,0.0138,\n'
                         '0.0138,0.0138,0.0138,0.0138,0.0138,0.0138,0.0138,0.0137,0.0137,0.0137,0.0137,\n'
                         '0.0137,0.0137,0.0137,0.0137,0.0137,0.0136,0.0136,0.0136,0.0136,0.0136,0.0136,\n'
                         '0.0136,0.0135,0.0135,0.0135,0.0135,0.0135,0.0135,0.0135,0.0134,0.0134,0.0134,\n'
                         '0.0134,0.0134,0.0134,0.0134,0.0133,0.0133,0.0133,0.0133,0.0133,0.0133,0.0132,\n'
                         '0.0132,0.0132,0.0132,0.0132,0.0132,0.0131,0.0131,0.0131,0.0131,0.0131,0.0131,\n'
                         '0.0130,0.0130,0.0130,0.0130,0.0130,0.0129,0.0129,0.0129,0.0129,0.0129,0.0129,\n'
                         '0.0128,0.0128,0.0128,0.0128,0.0128,0.0128,0.0127,0.0127,0.0127,0.0127,0.0127,\n'
                         '0.0126,0.0126,0.0126,0.0126,0.0126,0.0126,0.0125,0.0125,0.0125,0.0125,0.0125,\n'
                         '0.0125,0.0124,0.0124,0.0124,0.0124,0.0124,0.0124,0.0123,0.0123,0.0123,0.0123,\n'
                         '0.0123,0.0123,0.0122,0.0122,0.0122,0.0122,0.0122,0.0122,0.0121,0.0121,0.0121,\n'
                         '0.0121,0.0121,0.0121,0.0121,0.0120,0.0120,0.0120,0.0120,0.0120,0.0120,0.0120,\n'
                         '0.0119,0.0119,0.0119,0.0119,0.0119,0.0119,0.0119,0.0118,0.0118,0.0118,0.0118,\n'
                         '0.0118,0.0118,0.0118,0.0118,0.0118,0.0117,0.0117,0.0117,0.0117,0.0117,0.0117,\n'
                         '0.0117,0.0117,0.0117,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,\n'
                         '0.0116,0.0116,0.0116,0.0116,0.0116,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,\n'
                         '0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,\n'
                         '0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,\n'
                         '0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,\n'
                         '0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,0.0115,\n'
                         '0.0115,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,0.0116,\n'
                         '0.0116,0.0117,0.0117,0.0117,0.0117,0.0117,0.0117,0.0117,0.0117,0.0117,0.0118,\n'
                         '0.0118,0.0118,0.0118,0.0118,0.0118,0.0118,0.0119,0.0119,0.0119,0.0119,0.0119,\n'
                         '0.0119,0.0120,0.0120,0.0120,0.0120,0.0120,0.0121,0.0121,0.0121,0.0121,0.0121,\n'
                         '0.0122,0.0122,0.0122,0.0122,0.0122,0.0123,0.0123,0.0123,0.0123,0.0124,0.0124,\n'
                         '0.0124,0.0124,0.0125,0.0125,0.0125,0.0125,0.0126,0.0126,0.0126,0.0126,0.0127,\n'
                         '0.0127,0.0127,0.0128,0.0128,0.0128,0.0128,0.0129,0.0129,0.0129,0.0130,0.0130,\n'
                         '0.0130,0.0131,0.0131,0.0131,0.0132,0.0132,0.0132,0.0133,0.0133,0.0133,0.0134,\n'
                         '0.0134,0.0134,0.0135,0.0135,0.0135,0.0136,0.0136,0.0137,0.0137,0.0137,0.0138,\n'
                         '0.0138,0.0139,0.0139,0.0139,0.0140,0.0140,0.0141,0.0141,0.0141,0.0142,0.0142,\n'
                         '0.0143,0.0143,0.0144,0.0144,0.0144,0.0145,0.0145,0.0146,0.0146,0.0147,0.0147,\n'
                         '0.0148,0.0148,0.0148,0.0149,0.0149,0.0150,0.0150,0.0151,0.0151,0.0152,0.0152,\n'
                         '0.0153,0.0153,0.0154,0.0154,0.0155,0.0155,0.0156,0.0156,0.0157,0.0157,0.0158,\n'
                         '0.0158,0.0159,0.0159,0.0160,0.0160,0.0161,0.0162,0.0162,0.0163,0.0163,0.0164,\n'
                         '0.0164,0.0165,0.0165,0.0166,0.0167,0.0167,0.0168,0.0168,0.0169,0.0169,0.0170,\n'
                         '0.0171,0.0171,0.0172,0.0172,0.0173,0.0173,0.0174,0.0175,0.0175,0.0176,0.0176,\n'
                         '0.0177,0.0178,0.0178,0.0179,0.0180,0.0180,0.0181,0.0181,0.0182,0.0183,0.0183,\n'
                         '0.0184,0.0185,0.0185,0.0186}')
        elif '_VIS_' in str(label_path):
            header.write('ENVI \n'
                         'description = {DAWN VIR VIS data} \n'
                         f'{order[0]} = {number[0]} \n'
                         f'{order[1]} = {number[1]} \n'
                         f'{order[2]} = {number[2]} \n'
                         'header offset = 0 \n'
                         'data type = 4 \n'  # 4 means float32 (IEEE)
                         'interleave = BIP \n'
                         'sensor type = DAWN VIR VIS \n'
                         'byte order = 1 \n'
                         'wavelength = { \n'
                         '0.255,0.257,0.259,0.261,0.263,0.265,0.266,0.268,0.270,0.272,0.274,0.276,\n'
                         '0.278,0.280,0.282,0.284,0.285,0.287,0.289,0.291,0.293,0.295,0.297,0.299,\n'
                         '0.301,0.302,0.304,0.306,0.308,0.310,0.312,0.314,0.316,0.318,0.319,0.321,\n'
                         '0.323,0.325,0.327,0.329,0.331,0.333,0.335,0.336,0.338,0.340,0.342,0.344,\n'
                         '0.346,0.348,0.350,0.352,0.354,0.355,0.357,0.359,0.361,0.363,0.365,0.367,\n'
                         '0.369,0.371,0.372,0.374,0.376,0.378,0.380,0.382,0.384,0.386,0.388,0.389,\n'
                         '0.391,0.393,0.395,0.397,0.399,0.401,0.403,0.405,0.407,0.408,0.410,0.412,\n'
                         '0.414,0.416,0.418,0.420,0.422,0.424,0.425,0.427,0.429,0.431,0.433,0.435,\n'
                         '0.437,0.439,0.441,0.442,0.444,0.446,0.448,0.450,0.452,0.454,0.456,0.458,\n'
                         '0.459,0.461,0.463,0.465,0.467,0.469,0.471,0.473,0.475,0.477,0.478,0.480,\n'
                         '0.482,0.484,0.486,0.488,0.490,0.492,0.494,0.495,0.497,0.499,0.501,0.503,\n'
                         '0.505,0.507,0.509,0.511,0.512,0.514,0.516,0.518,0.520,0.522,0.524,0.526,\n'
                         '0.528,0.529,0.531,0.533,0.535,0.537,0.539,0.541,0.543,0.545,0.547,0.548,\n'
                         '0.550,0.552,0.554,0.556,0.558,0.560,0.562,0.564,0.565,0.567,0.569,0.571,\n'
                         '0.573,0.575,0.577,0.579,0.581,0.582,0.584,0.586,0.588,0.590,0.592,0.594,\n'
                         '0.596,0.598,0.600,0.601,0.603,0.605,0.607,0.609,0.611,0.613,0.615,0.617,\n'
                         '0.618,0.620,0.622,0.624,0.626,0.628,0.630,0.632,0.634,0.635,0.637,0.639,\n'
                         '0.641,0.643,0.645,0.647,0.649,0.651,0.652,0.654,0.656,0.658,0.660,0.662,\n'
                         '0.664,0.666,0.668,0.670,0.671,0.673,0.675,0.677,0.679,0.681,0.683,0.685,\n'
                         '0.687,0.688,0.690,0.692,0.694,0.696,0.698,0.700,0.702,0.704,0.705,0.707,\n'
                         '0.709,0.711,0.713,0.715,0.717,0.719,0.721,0.723,0.724,0.726,0.728,0.730,\n'
                         '0.732,0.734,0.736,0.738,0.740,0.741,0.743,0.745,0.747,0.749,0.751,0.753,\n'
                         '0.755,0.757,0.758,0.760,0.762,0.764,0.766,0.768,0.770,0.772,0.774,0.775,\n'
                         '0.777,0.779,0.781,0.783,0.785,0.787,0.789,0.791,0.793,0.794,0.796,0.798,\n'
                         '0.800,0.802,0.804,0.806,0.808,0.810,0.811,0.813,0.815,0.817,0.819,0.821,\n'
                         '0.823,0.825,0.827,0.828,0.830,0.832,0.834,0.836,0.838,0.840,0.842,0.844,\n'
                         '0.846,0.847,0.849,0.851,0.853,0.855,0.857,0.859,0.861,0.863,0.864,0.866,\n'
                         '0.868,0.870,0.872,0.874,0.876,0.878,0.880,0.881,0.883,0.885,0.887,0.889,\n'
                         '0.891,0.893,0.895,0.897,0.898,0.900,0.902,0.904,0.906,0.908,0.910,0.912,\n'
                         '0.914,0.916,0.917,0.919,0.921,0.923,0.925,0.927,0.929,0.931,0.933,0.934,\n'
                         '0.936,0.938,0.940,0.942,0.944,0.946,0.948,0.950,0.951,0.953,0.955,0.957,\n'
                         '0.959,0.961,0.963,0.965,0.967,0.968,0.970,0.972,0.974,0.976,0.978,0.980,\n'
                         '0.982,0.984,0.986,0.987,0.989,0.991,0.993,0.995,0.997,0.999,1.001,1.003,\n'
                         '1.004,1.006,1.008,1.010,1.012,1.014,1.016,1.018,1.020,1.021,1.023,1.025,\n'
                         '1.027,1.029,1.031,1.033,1.035,1.037,1.039,1.040,1.042,1.044,1.046,1.048,\n'
                         '1.050,1.052,1.054,1.056,1.057,1.059,1.061,1.063,1.065,1.067,1.069,1.071}\n'
                         'fwhm = {\n'
                         '0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021,\n'
                         '0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018, 0.0018,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019,\n'
                         '0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020,\n'
                         '0.0020, 0.0020, 0.0021, 0.0021}')

    # Open the qube as ENVI using the created header
    img = envi.open(hdr_path, qube_path)

    numpyimage = np.asarray(img.asarray())
    numpyimage = np.nan_to_num(numpyimage, nan=0)
    numpyimage = np.clip(numpyimage, a_min=0,
                         a_max=100)  # Without clipping the minimum is -32 767: this values is used in original processing to mark bad pixels
    # plt.imshow(np.mean(numpyimage, 2), vmin=0)
    # plt.imshow(np.mean(numpyimage, axis=2), vmin=0)
    # plt.show()

    return numpyimage, img  # Return both the numpy array and the whole ENVI thing


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


def ASPECT_resampling(cube, wavelengths, FWHMs):
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
