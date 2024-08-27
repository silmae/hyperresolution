import os
from pathlib import Path
import logging

import numpy as np
import scipy.io  # for loading Matlab matrices
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import cv2 as cv
from planetaryimage import CubeFile

from src import constants
from src import utils
from src import simulation


def load_spectral_csv(filepath, convert2micron=True):
    """Reads csv file containing one spectrum, where the first column is wavelengths and second is intensity."""
    data = np.loadtxt(filepath)
    wavelengths = data[:, 0]
    spectrum = data[:, 1]
    if convert2micron and wavelengths[-1] > 100:  # if last wavelength value is over 100, assume they are nanometers
        wavelengths = wavelengths / 1000

    return spectrum, wavelengths


def file_loader_Dawn_PDS3(filepath):
    """
    Loads PDS3 qube files of Dawn VIR-VIS and VIR-IR data when given path to lbl file associated with either VIS or IR.
    Concatenates the cubes into one, using the IR channels for the overlapping part since the VIS ones are noisier.
    :param filepath
        Path to .lbl file
    :return: h, w, l, cube, wavelengths, FWHMs:
        Height, width, length, image cube as ndarray, wavelength vector, full-width-half-maximum -vector of wl channels
    """
    filepath = str(filepath)
    if '_VIS_' in filepath:
        vis_path = Path(filepath)
        ir_path = Path(filepath.replace('_VIS_', '_IR_'))
    elif '_IR_' in filepath:
        ir_path = Path(filepath)
        vis_path = Path(filepath.replace('_IR_', '_VIS_'))

    try:
        vis_cube, vis_data = open_DAWN_VIR_PDS3_as_ENVI(vis_path)
    except FileNotFoundError:
        logging.info('VIR-VIS file not found')
        vis_cube, vis_data = None, None

    try:
        ir_cube, ir_data = open_DAWN_VIR_PDS3_as_ENVI(ir_path)
    except FileNotFoundError:
        logging.info('VIR-IR file not found')
        ir_cube, ir_data = None, None

    # # Sanity check plot
    # plt.imshow(cube[:, :, 80])
    # plt.show()

    vis_wavelengths = vis_data.metadata['wavelength']
    vis_wavelengths = [float(x) for x in vis_wavelengths]
    vis_fwhms = vis_data.metadata['fwhm']
    vis_fwhms = [float(x) for x in vis_fwhms]

    ir_wavelengths = ir_data.metadata['wavelength']
    ir_wavelengths = [float(x) for x in ir_wavelengths]
    ir_fwhms = vis_data.metadata['fwhm']
    ir_fwhms = [float(x) for x in ir_fwhms]

    # Join the VIS and IR cubes and wavelength vectors
    cube, wavelengths, fwhms = simulation.join_VIR_VIS_and_IR(vis_cube, ir_cube, vis_wavelengths, ir_wavelengths, vis_fwhms,
                                                         ir_fwhms)

    # # Crop the cube a bit in horizontal direction
    # cube = cube[:, :100, :]

    shape = cube.shape
    h = shape[0]
    w = shape[1]
    l = shape[2]

    return h, w, l, cube, wavelengths, fwhms


def file_loader_Dawn_ISIS(filepath):
    """
    File loader to supply neural network with data from Dawn VIR instrument. Loads the .cub ISIS file of the path given
    as parameter. If the path was to a VIR VIS file, also loads the corresponding VIR IR file, and vice versa.
    Rotates and crops the two files according to the corresponding dictionaries in constants.py, then concatenates the
    two image cubes into one. Then resamples the spectra to correspond to ASPECT wavelength channels given in
    constants.py, converts the radiance values to I/F, and interpolates the spatial pixel counts to also match
    ASPECT NIR specifications in constants.py.
    :param filepath:
        Path to either VIR IR or VIR VIS .cub ISIS file
    :return: h, w, l, cube, wavelengths, FWHMs:
        Height, width, length, image cube as ndarray, wavelength vector, full-width-half-maximum -vector of wl channels
    """
    filepath = str(filepath)
    if '_VIS_' in filepath:
        vis_path = Path(filepath)
        ir_path = Path(filepath.replace('_VIS_', '_IR_'))
    elif '_IR_' in filepath:
        ir_path = Path(filepath)
        vis_path = Path(filepath.replace('_IR_', '_VIS_'))

    def _load_and_crop_(cub_path):
        cube, isis = open_Dawn_VIR_ISIS(cub_path)
        try:
            rot_crop_dict = constants.Dawn_ISIS_rot_deg_and_crop_indices[f'{cub_path.name}']
        except:
            logging.info(f'No dictionary of rotation and crop indices corresponding to {cub_path.name} was found, check constants.py')
            exit(1)

        cube, edges = simulation.rot_and_crop_Dawn_VIR_ISIS(data=cube,
                                                       rot_deg=rot_crop_dict['rot_deg'],
                                                       crop_indices_x=rot_crop_dict['crop_indices_x'],
                                                       crop_indices_y=rot_crop_dict['crop_indices_y'],
                                                       edge_detection=True)
        bands = isis.label['IsisCube']['BandBin']
        wavelengths = bands['Center']
        FWHMs = bands['Width']
        return cube, wavelengths, FWHMs, edges

    vis_cube, vis_wavelengths, vis_FWHMs, vis_edges = _load_and_crop_(vis_path)
    ir_cube, ir_wavelengths, ir_FWHMs, ir_edges = _load_and_crop_(ir_path)

    # # Plots to check if the offset between IR and VIS is good: edges detected from one frame of both
    # edges = np.zeros(shape=(vis_edges.shape[0], vis_edges.shape[1], 3))  # Plot IR edges in red channel, VIS in green
    # edges[:, :, 0] = ir_edges
    # edges[:, :, 1] = vis_edges
    #
    # showable_VIS = vis_cube[constants.edge_detection_channel, :, :]
    # showable_IR = ir_cube[constants.edge_detection_channel, :, :]
    # showable = np.zeros(shape=(showable_VIS.shape[0], showable_VIS.shape[1], 3))  # Plot one channel from IR in red channel, VIS in green
    # showable[:, :, 0] = showable_IR / np.max(showable_IR)
    # showable[:, :, 1] = showable_VIS / np.max(showable_VIS)
    #
    # fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained')
    # ax = axs[0]
    # ax.imshow(edges)
    # ax.set_title('Edges')
    # ax = axs[1]
    # ax.imshow(showable)
    # ax.set_title('One channel from each')
    # plt.show()

    # ISIS image cubes have their dimensions in a different order, wavelengths first: transpose to wl last
    vis_cube = np.transpose(vis_cube, (2, 1, 0))
    ir_cube = np.transpose(ir_cube, (2, 1, 0))

    # Join the VIS and IR cubes into one, using IR channels for overlapping part
    cube, wavelengths, FWHMs = simulation.join_VIR_VIS_and_IR(vis_cube=vis_cube, ir_cube=ir_cube,
                                                         vis_wavelengths=vis_wavelengths, ir_wavelengths=ir_wavelengths,
                                                         vis_fwhms=vis_FWHMs, ir_fwhms=ir_FWHMs)

    shape = cube.shape
    h = shape[0]
    w = shape[1]
    l = shape[2]

    # # Plot of one spectrum
    # plt.plot(wavelengths, cube[155, 285, :])
    # plt.show()

    return h, w, l, cube, wavelengths, FWHMs


def load_Didymos_reflectance_spectrum(denoise=True):
    # Load Didymos spectrum used in the simulation
    spectrum_path = constants.didymos_path  # A collection of channels from 0.45 to 2.50 µm saved into a txt file
    didymos_data = np.loadtxt(spectrum_path)
    didymos_reflectance = didymos_data[:, 1]
    didymos_wavelengths = didymos_data[:, 0] / 1000  # Convert nm to µm

    if denoise:
        # Denoise the Didymos reflectance spectrum
        original_reflectance = np.copy(didymos_reflectance)
        didymos_reflectance = utils.interpolate_outliers(didymos_reflectance, z_thresh=1.2)
        didymos_reflectance = utils.denoise_array(didymos_reflectance, x=didymos_wavelengths, sigma=0.045)

        # # Plot of denoised and original spectrum
        # plt.figure()
        # plt.plot(didymos_wavelengths, original_reflectance)
        # plt.plot(didymos_wavelengths, didymos_reflectance)
        # plt.show()

    return didymos_wavelengths, didymos_reflectance


def file_loader_simulated_Didymos(filepath, spectrum='Didymos', crater='px10'):
    """Loads a simulated Didymos image by loading a frame from a simulated DN image, using that as a brightness map
    for a spectrum. Can create a circular area of different material to simulate a crater. """

    def load_file(filepath):
        data_dict = scipy.io.loadmat(filepath)
        datacube = data_dict['cube']
        wavelengths = np.squeeze(data_dict['wavelengths'] / 1000)
        return datacube, wavelengths

    cube, nir_wavelengths = load_file(filepath)

    # Dark correction
    cube = cube - np.mean(cube[:10, :10, :], axis=(0, 1))

    # Extract one frame from the image cube and normalize it so that maximum value is 1
    frame = cube[:, :, 0] / np.max(cube[:, :, 0])

    if spectrum == 'Didymos': # reflectance spectrum of Didymos
        data_wavelengths, data_reflectance = load_Didymos_reflectance_spectrum(denoise=True)
    else:  # laboratory mixture of pyroxene and olivine
        data_reflectance, data_wavelengths = load_spectral_csv(Path(constants.lab_mixtures_path, f'{spectrum}.csv'))

        # Ground truth abundance maps using the normalized frame and the pyroxene percentage
        perkele = (np.copy(frame) > 1e-20) * 1
        px_abundance = (np.copy(frame) > 1e-20) * float(spectrum[2:]) / 100
        ol_abundance = (np.copy(frame) > 1e-20) * (100 - float(spectrum[2:])) / 100
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(px_abundance)
        axs[1].imshow(ol_abundance)
        plt.show()
        gt_abundances = [px_abundance, ol_abundance]

    # Instead of treating the cubes properly and trying to join them:
    # just take one frame from the cube for brightness variation, and
    # create the spectral features from the theoretical spectrum
    wavelengths = constants.ASPECT_wavelengths
    didymos_reflectance, _, FWHMs = simulation.ASPECT_resampling(data_reflectance, data_wavelengths)
    cube = np.ones(shape=(frame.shape[0], frame.shape[1], len(wavelengths)))
    cube = cube * didymos_reflectance
    cube = cube * np.expand_dims(frame, axis=2)

    if crater is not None:
        crater_reflectance, crater_wavelengths = load_spectral_csv(Path(constants.lab_mixtures_path, f'{crater}.csv'))
        crater_reflectance, _, _ = simulation.ASPECT_resampling(crater_reflectance, crater_wavelengths)

        crater_masked, mask = simulation.apply_circular_mask(data=np.copy(cube),
                                                       h=cube.shape[0],
                                                       w=cube.shape[1],
                                                       radius=constants.ASPECT_SWIR_equivalent_radius / 5,
                                                       masking_value=0,
                                                       return_mask=True)

        # Set the crater area to 0 in the spectral image cube
        cube = cube - crater_masked

        # Calculate a cube that has proper data only in the crater and rest is zeroes
        crater_frame = crater_masked[:, :, 0] / np.max(crater_masked[:, :, 0])
        crater_masked = np.expand_dims(crater_frame, axis=2) * crater_reflectance

        # Plug the crater cube into the actual cube
        cube = cube + crater_masked

        # Similar treatment for the abundance maps: calculate abundaces of the crater, set crater area to zero in
        # original abundance maps, then plug in the crater abundance maps

        # Normalized abundance map of crater
        norm_abundance_masked = np.copy(gt_abundances[0])
        norm_abundance_masked = (norm_abundance_masked * mask) / np.max(norm_abundance_masked * mask)
        # Calculate crater abundance maps
        px_abundance_masked = (np.copy(norm_abundance_masked) > 1e-20) * float(crater[2:]) / 100
        ol_abundance_masked = (np.copy(norm_abundance_masked) > 1e-20) * (100 - float(crater[2:])) / 100

        # Set the crater area to zero
        for i in range(2):
            gt_abundances[i] = gt_abundances[i] - (gt_abundances[i] * mask)

        # Plug the crater abundance maps in the original maps
        gt_abundances[0] = gt_abundances[0] + px_abundance_masked
        gt_abundances[1] = gt_abundances[1] + ol_abundance_masked

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(gt_abundances[0])
        axs[1].imshow(gt_abundances[1])
        plt.show()

    # Add a small epsilon value to not get 0 at the areas not occupied by the asteroid - this is done avoid division by
    # zero later
    cube = cube + 1e-4

    h = np.shape(cube)[0]
    w = np.shape(cube)[1]
    l = np.shape(cube)[2]

    # fig, axs = plt.subplots(1, 2)
    # # axs[0].imshow(cube[:, :, 10])
    # axs[0].imshow(frame)
    # axs[1].plot(wavelengths, cube[200, 300, :])
    # axs[1].plot(wavelengths, cube[200, 350, :])
    # axs[1].plot(wavelengths, cube[250, 350, :])
    # axs[1].plot(wavelengths, cube[300, 300, :])
    # axs[1].plot(wavelengths, cube[300, 350, :])
    # plt.show()

    return h, w, l, cube, wavelengths, FWHMs, gt_abundances


def open_Dawn_VIR_ISIS(cub_path='./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub'):
    """
    Open a Dawn VIR IR or VIS ISIS image from disc using the planetaryimage package.
    :param cub_path:
        Path to the .cub file on disc
    :return: cube, isisimage:
        Cube as ndarray and the whole ISIS image which also contains metadata
    """
    isisimage = CubeFile.open(str(cub_path))
    cube = isisimage.data
    return cube, isisimage


def open_DAWN_VIR_PDS3_as_ENVI(label_path='./datasets/DAWN/VIR_IR_1B_1_488154033_1.LBL'):
    """
    Open a PDS3 qube file from the Dawn spacecraft VIR instrument as an ENVI file. The function generates an ENVI
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
    # Without clipping the minimum is -32 767: this values is used in original processing to mark bad pixels
    numpyimage = np.clip(numpyimage, a_min=0, a_max=100)
    # plt.imshow(np.mean(numpyimage, 2), vmin=0)
    # plt.imshow(np.mean(numpyimage, axis=2), vmin=0)
    # plt.show()

    return numpyimage, img  # Return both the numpy array and the whole ENVI thing
