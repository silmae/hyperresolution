""""
This file contains miscellaneous utility functions
"""
import math

import numpy as np
from matplotlib import pyplot as plt
import spectral.io.envi as envi
from torch import Tensor
import torch


def apply_circular_mask(data: np.ndarray or torch.Tensor, h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray or torch.Tensor:
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
        masked_data = masked_data + Tensor(abs(mask - 1))  # Convert masked values to ones instead of zeros to avoid problems with backprop
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


def open_DAWN_VIR_IR_PDS3_as_ENVI(label_path='./datasets/DAWN/VIR_IR_1B_1_488154033_1.LBL'):
    """
    Open a PDS3 qube file from the DAWN spacecraft VIR IR instrument as an ENVI file. The function generates an ENVI
    header file from the PDS3 label file given as parameter, saves the header to disc, and uses it to open
    the qube file associated with the label (same filename apart from extension).
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

        print('test')

    # Write and ENVI header file using the dimensions of bands, lines, samples extracted from the label
    with open(hdr_path, 'w') as header:
        header.write('ENVI \n'
                    'description = {DAWN VIR IR data} \n' 
                    f'{order[0]} = {number[0]} \n' 
                    f'{order[1]} = {number[1]} \n'
                    f'{order[2]} = {number[2]} \n' 
                    'header offset = 0 \n' 
                    'data type = 4 \n'   # 4 means float32 (IEEE)
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
                    'fwhm = {'
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

    # Open the qube as ENVI using the created header
    img = envi.open(hdr_path, qube_path)

    numpyimage = np.asarray(img.asarray())
    numpyimage = np.nan_to_num(numpyimage, nan=0)
    numpyimage = np.clip(numpyimage, a_min=0, a_max=100)  # Without clipping the minimum is -32 767: this values is used in original processing to mark bad pixels
    plt.imshow(np.mean(numpyimage, 2), vmin=0)
    plt.imshow(np.mean(numpyimage, axis=2), vmin=0)
    plt.show()

    return numpyimage, img  # Return both the numpy array and the whole ENVI thing
