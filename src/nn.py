from pathlib import Path
import logging
import sys
import math
import os
import numpy as np
import h5py
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import save
from torch import load
from torch import from_numpy
import torch.optim as optim
import torchmetrics
import scipy.io  # for loading Matlab matrices
import matplotlib.pyplot as plt
import cv2 as cv

from src import plotter
from src import utils
from src import constants

# Set manual seed for comparable results between training runs
torch.manual_seed(42)

bands = 80  # TODO what?

torch.autograd.set_detect_anomaly(True)  # this will provide traceback if stuff turns into NaN


def SAM(s1, s2):
    """
    From https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html#SAM
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle


class Encoder(nn.Module):

    def __init__(self, enc_layer_count=3, e_filter_count=48, e_kernel_size=7, kernel_reduction=2, band_count=100,
                 endmember_count=3):
        super(Encoder, self).__init__()

        self.band_count = band_count
        self.endmember_count = endmember_count
        self.enc_layer_count = enc_layer_count
        self.layers = nn.ModuleList()

        self.filter_counts = []
        for i in range(enc_layer_count - 1):
            self.filter_counts.append(e_filter_count)
            if e_filter_count >= 2:
                e_filter_count = int(e_filter_count / 2)

        # self.layers.append(nn.Conv3d(in_channels=band_count, out_channels=band_count, kernel_size=(3,3,3), padding='same'))
        # self.layers.append(nn.Flatten())
        self.layers.append(nn.Conv2d(in_channels=band_count,
                                     out_channels=self.filter_counts[0],
                                     kernel_size=e_kernel_size, padding='same',
                                     stride=1,
                                     bias=False))
        if enc_layer_count > 2:
            for i in range(1, enc_layer_count - 1):
                if e_kernel_size - kernel_reduction >= 1:
                    e_kernel_size = e_kernel_size - kernel_reduction
                self.layers.append(nn.Conv2d(in_channels=self.filter_counts[i - 1],
                                             out_channels=self.filter_counts[i],
                                             kernel_size=e_kernel_size, padding='same',
                                             stride=1,
                                             bias=False))

        self.layers.append(nn.Conv2d(in_channels=int(self.filter_counts[-1]),
                                     out_channels=endmember_count, kernel_size=1,
                                     padding='same',
                                     stride=1,
                                     bias=False))

        self.soft_max = nn.Softmax(dim=1)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout2d(0.2)

        self.norms = nn.ModuleList()
        for count in self.filter_counts:
            self.norms.append(nn.BatchNorm2d(count))
        self.norms.append(nn.BatchNorm2d(endmember_count))

        # self.activation = F.leaky_relu_
        self.activation = F.selu_

    def forward(self, x):

        for i in range(self.enc_layer_count):
            x = self.activation(self.layers[i](x))
            x = self.norms[i](x)
            # x = self.dropout(x)  # dropout does not do good apparently

        out = self.soft_max(x * 3.5)  # the magical constant is from Palsson's paper, there called alpha

        return out


class Decoder(nn.Module):

    def __init__(self, band_count=200, endmember_count=3, d_kernel_size=13):
        super(Decoder, self).__init__()

        self.band_count = band_count
        self.endmember_count = endmember_count
        self.kernel_size = d_kernel_size
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Conv2d(in_channels=self.endmember_count,
                      out_channels=self.band_count,
                      kernel_size=self.kernel_size,
                      padding='same',
                      stride=1,
                      bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # Force the weights to be positive, these will be the endmember spectra:
            layer.weight.data = layer.weight.data.clamp(min=0)
            # # The endmembers are very noisy: calculate derivative and subtract it, then replace the weights with result
            # variation = layer.weight.data[1:, :, :, :] - layer.weight.data[:-1, :, :, :]
            # layer.weight.data[1:, :, :, :] = layer.weight.data[1:, :, :, :] - variation * 0.001  # Not a good idea to subtract all of the variation, adjust the percentage

            # Set weights of the first decoder kernel to match the value used for masks
            orig_kernel = layer.weight.data[:, 0, :, :]
            mask_endmember = np.zeros(orig_kernel.shape)
            mid_index = int((orig_kernel.shape[1] - 1) / 2)
            mask_endmember[:, mid_index, mid_index] = 0.5  # masking value
            layer.weight.data[:, 0, :, :] = torch.tensor(mask_endmember)

        return x


def file_loader_rem_sens(filepath="./datasets/TinyAPEX.mat"):
    mat = scipy.io.loadmat(filepath)
    # mat = scipy.io.loadmat("./datasets/Samson.mat")
    w = mat['W'][0][0]  # W is 2 dim matrix with 1 element
    h = mat['H'][0][0]
    l = mat['L'][0][0]
    abundance_count = mat['p'][0][0]
    cube_flat = np.array(mat['Y'])
    cube_flat = cube_flat.transpose()
    # plt.imshow(cube_flat)
    # plt.show()
    cube = cube_flat.reshape(h, w, l)

    return h, w, l, abundance_count, cube


def file_loader_rock(filepath):
    d = h5py.File(filepath, 'r')
    contents = list(d.keys())
    cube = np.transpose(d['/hdr'], (1, 2, 0))
    cube = np.nan_to_num(cube, nan=1)  # Original rock images are masked with NaN, replace those with a number

    wavelengths = d['/wavelengths'][:]

    # cube = cube[50:, 50:, :]

    # # Sanity check plot
    # plt.imshow(np.nanmean(cube, 2))
    # plt.show()

    dimensions = cube.shape
    h = dimensions[0]
    w = dimensions[1]
    l = dimensions[2]

    return h, w, l, cube, wavelengths


def file_loader_luigi(filepath):
    data = xr.open_dataset(filepath)['reflectance']
    cube = data.values

    # The image is very large, crop to get manageable training time
    cube = cube[75:300, 100:300, 0:120]

    # # Sanity check plot
    # plt.imshow(cube[:, :, 80])
    # plt.show()

    shape = cube.shape
    w = shape[1]
    h = shape[0]
    l = shape[2]

    wavelengths = data.wavelength.values

    return h, w, l, cube, wavelengths


def file_loader_Dawn_PDS3(filepath):
    """
    Loads PDS3 qube files of Dawn VIR-VIS and VIR-IR data when given path to lbl file associated with either VIS or IR.
    Concatenates the cubes into one, using the IR channels for the overlapping part since the VIS ones are noisier.
    Args:
        filepath:

    Returns:

    """
    filepath = str(filepath)
    if '_VIS_' in filepath:
        vis_path = Path(filepath)
        ir_path = Path(filepath.replace('_VIS_', '_IR_'))
    elif '_IR_' in filepath:
        ir_path = Path(filepath)
        vis_path = Path(filepath.replace('_IR_', '_VIS_'))

    try:
        vis_cube, vis_data = utils.open_DAWN_VIR_PDS3_as_ENVI(vis_path)
    except FileNotFoundError:
        logging.info('VIR-VIS file not found')
        vis_cube, vis_data = None, None

    try:
        ir_cube, ir_data = utils.open_DAWN_VIR_PDS3_as_ENVI(ir_path)
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
    cube, wavelengths, fwhms = utils.join_VIR_VIS_and_IR(vis_cube, ir_cube, vis_wavelengths, ir_wavelengths, vis_fwhms,
                                                         ir_fwhms)

    # # Crop the cube a bit in horizontal direction
    # cube = cube[:, :100, :]

    # Resample spectra to resemble ASPECT data
    cube, wavelengths, fwhms = utils.ASPECT_resampling(cube, wavelengths, fwhms)
    # TODO Interpolate data spatially to have same pixel count as ASPECT NIR?

    shape = cube.shape
    w = shape[1]
    h = shape[0]
    l = shape[2]

    return h, w, l, cube, wavelengths, fwhms


def file_loader_Dawn_ISIS(filepath="./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub"):
    filepath = str(filepath)
    if '_VIS_' in filepath:
        vis_path = Path(filepath)
        ir_path = Path(filepath.replace('_VIS_', '_IR_'))
    elif '_IR_' in filepath:
        ir_path = Path(filepath)
        vis_path = Path(filepath.replace('_IR_', '_VIS_'))

    def _load_and_crop_(cub_path):
        cube, isis = utils.open_Dawn_VIR_ISIS(cub_path)
        try:
            rot_crop_dict = constants.Dawn_ISIS_rot_deg_and_crop_indices[f'{cub_path.name}']
        except:
            logging.info(f'No dictionary of rotation and crop indices corresponding to {cub_path.name} was found, check constants.py')
            exit(1)

        cube, edges = utils.rot_and_crop_Dawn_VIR_ISIS(data=cube,
                                                       rot_deg=rot_crop_dict['rot_deg'],
                                                       crop_indices_x=rot_crop_dict['crop_indices_x'],
                                                       crop_indices_y=rot_crop_dict['crop_indices_y'],
                                                       edge_detection=False)
        bands = isis.label['IsisCube']['BandBin']
        wavelengths = bands['Center']
        FWHMs = bands['Width']
        return cube, wavelengths, FWHMs, edges

    vis_cube, vis_wavelengths, vis_FWHMs, vis_edges = _load_and_crop_(vis_path)
    ir_cube, ir_wavelengths, ir_FWHMs, ir_edges = _load_and_crop_(ir_path)

    # # Plots to check if the offset between IR and VIS is good: edges detected from one frame of both
    # edges = np.zeros(shape=(vis_edges.shape[0], vis_edges.shape[1], 3))  # Plot VIS edges in red channel, IR in green
    # edges[:, :, 0] = vis_edges
    # edges[:, :, 1] = ir_edges
    #
    # showable_VIS = vis_cube[100, :, :]
    # showable_IR = ir_cube[100, :, :]
    # showable = np.zeros(shape=(showable_VIS.shape[0], showable_VIS.shape[1], 3))  # Plot one channel from VIS in red channel, IR in green
    # showable[:, :, 0] = showable_VIS / np.max(showable_VIS)
    # showable[:, :, 1] = showable_IR / np.max(showable_IR)
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

    cube, wavelengths, FWHMs = utils.join_VIR_VIS_and_IR(vis_cube=vis_cube, ir_cube=ir_cube,
                                                         vis_wavelengths=vis_wavelengths, ir_wavelengths=ir_wavelengths,
                                                         vis_fwhms=vis_FWHMs, ir_fwhms=ir_FWHMs)

    # Resampling cube to ASPECT wavelengths
    cube, wavelengths, FWHMs = utils.ASPECT_resampling(cube, wavelengths, FWHMs)

    # Convert radiances to I/F
    insolation = utils.solar_irradiance(distance=constants.ceres_hc_dist, wavelengths=constants.ASPECT_wavelengths,
                                        plot=False, resample=True)
    cube = cube / insolation[:, 1]

    # # Sanity check plot
    # plt.imshow(cube[:, :, 20])
    # plt.show()

    # Interpolate each wavelength channel to ASPECT NIR spatial pixel count
    width = constants.ASPECT_NIR_channel_shape[1]
    height = constants.ASPECT_NIR_channel_shape[0]
    resized = np.zeros((height, width, cube.shape[2]))
    for channel in range(cube.shape[2]):
        resized[:, :, channel] = cv.resize(cube[:, :, channel], (width, height), interpolation=cv.INTER_AREA)
    cube = resized

    # # Sanity check plot
    # plt.imshow(cube[:, :, 20])
    # plt.show()

    shape = cube.shape
    h = shape[0]
    w = shape[1]
    l = shape[2]

    # # Plot of one spectrum
    # plt.plot(wavelengths, cube[100, 100, :])
    # plt.show()

    return h, w, l, cube, wavelengths, FWHMs


class TrainingData(Dataset):
    """Handles catering the training data from disk to NN."""

    def __init__(self, type, filepath):

        if type == 'remote_sensing':
            h, w, l, abundance_count, cube = file_loader_rem_sens(filepath)
        elif type == 'rock':
            h, w, l, cube, wavelengths = file_loader_rock(filepath)
        elif type == 'luigi':
            h, w, l, cube, wavelengths = file_loader_luigi(filepath)
        elif type == 'DAWN_PDS3':
            h, w, l, cube, wavelengths, FWHMs = file_loader_Dawn_PDS3(filepath)
            self.FWHMs = FWHMs
        elif type == 'DAWN_ISIS':
            h, w, l, cube, wavelengths, FWHMs = file_loader_Dawn_ISIS(filepath)
        else:
            logging.info('Invalid training data type, ending execution')
            exit(1)

        self.w = w
        self.h = h
        self.l = l
        if type != 'remote_sensing':  # no wavelength data for the remote sensing images used here
            self.wavelengths = wavelengths
            # self.abundance_count = abundance_count

        l_half = int(self.l / 2)

        # Dimensions of the image must be [batches, bands, width, height] for convolution
        # Transform from original [h, w, bands] with transpose
        self.cube = np.transpose(cube, (2, 1, 0))

        first_half = self.cube[:l_half, :, :]

        self.X = torch.from_numpy(first_half).float()
        self.Y = torch.from_numpy(self.cube).float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X, self.Y


def init_network(enc_params, dec_params, common_params):
    enc = Encoder(enc_layer_count=2,
                  band_count=int(common_params['bands'] / 2),
                  endmember_count=common_params['endmember_count'],
                  e_filter_count=enc_params['e_filter_count'],
                  e_kernel_size=enc_params['e_kernel_size'],
                  kernel_reduction=enc_params['kernel_reduction'], )
    dec = Decoder(band_count=common_params['bands'],
                  endmember_count=common_params['endmember_count'],
                  d_kernel_size=dec_params['d_kernel_size'])
    enc = enc.float()
    dec = dec.float()

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.3)

    enc.apply(init_weights)
    dec.apply(init_weights)

    # # Print network structure into log
    # logging.info(enc)
    # logging.info(dec)
    # logging.info(enc.parameters())
    # logging.info(dec.parameters())

    return enc, dec


def cubeSAM(predcube, groundcube):
    '''Calculate mean SAM of masked image cube. Omits the arccos, replacing it with subtracting result from 1'''
    prednorm = torch.linalg.norm(predcube, dim=1)
    groundnorm = torch.linalg.norm(groundcube, dim=1)
    upper = torch.linalg.vecdot(predcube, groundcube, dim=1)
    # lower = torch.linalg.vecdot(prednorm, groundnorm, dim=1)
    lower = prednorm * groundnorm
    cossimilarity = upper / lower
    coserror = torch.mean(torch.abs(1 - cossimilarity))
    return coserror


def train(training_data, enc_params, dec_params, common_params, epochs=1, plots=True, prints=True):
    bands = training_data.l
    half_point = int(bands / 2)

    # cube_original = training_data.Ys[1].numpy()  #training_data.cube
    cube_original = training_data.cube
    training_data.X = training_data.X.float()
    training_data.Y = training_data.Y.float()

    w = training_data.w
    h = training_data.h

    data_loader = DataLoader(training_data, shuffle=False, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Build and initialize the encoder and decoder
    enc, dec = init_network(enc_params, dec_params, common_params)

    # Move network to GPU memory
    enc = enc.to(device)
    dec = dec.to(device)

    def loss_fn(y_true, y_pred):
        # # Chop both true and predicted cubes in half with respect of wavelength

        # https://discuss.pytorch.org/t/custom-loss-functions/29387 -Kimmo

        # def MAPE(Y_actual, Y_Predicted):
        #     abs_diff = torch.abs((Y_actual - Y_Predicted))
        #     # zero mask
        #     mask = (Y_actual != 0)
        #     # initialize output tensor with desired value
        #     norm = torch.full_like(Y_actual, fill_value=float('nan'))
        #     norm[mask] = torch.div(abs_diff[mask], Y_actual[mask]) # divide by non-zero elements
        #     norm = torch.nan_to_num(norm, posinf=1e10) # replace infinities with finite big number
        #     mean_diff = torch.mean(norm)
        #     mape = mean_diff * 100
        #     return mape
        #
        # def mean_spectral_gradient(cube):
        #     abs_grad = torch.abs(cube[:, 1:, :, :] - cube[:, :-1, :, :])
        #     sum_grad = torch.sum(abs_grad, dim=(1))
        #     mean_grad = torch.mean(sum_grad)
        #     # sum_grad = torch.sum(mean_grad)
        #     return mean_grad

        short_y_true = y_true[:, :half_point, :, :]
        long_y_true = y_true[:, half_point:, :, :]

        y_pred = utils.apply_circular_mask(y_pred, w, h)
        short_y_pred = y_pred[:, :half_point, :, :]
        long_y_pred = y_pred[:, half_point:, :, :]

        # Calculate short wavelength loss by comparing cubes. For loss metric MAPE
        # loss_short = MAPE(short_y_true, short_y_pred)
        metric_mape = torchmetrics.MeanAbsolutePercentageError().to(device)

        # This thing can kill backprop even when called for a single spectrum
        # metric_SAM = torchmetrics.SpectralAngleMapper(reduction='none').to(device)

        loss_short = metric_mape(short_y_pred, short_y_true)
        # loss_short_SAM = metric_SAM(short_y_pred, short_y_true)  # Using this function for masked cube breaks backprop: "RuntimeError: Function 'CatBackward0' returned nan values in its 0th output."
        loss_short_SAM = cubeSAM(short_y_pred, short_y_true)

        # Calculate long wavelength loss by comparing mean spectra of long wavelength cubes
        long_y_true = torch.mean(long_y_true, dim=(
            2, 3))  # TODO Move this calculation to preprocessing and only feed the mean spectrum into here
        long_y_true = torch.unsqueeze(long_y_true, 2)
        long_y_true = torch.unsqueeze(long_y_true, 3)

        long_y_pred = torch.mean(long_y_pred, dim=(2, 3))
        long_y_pred = torch.unsqueeze(long_y_pred, 2)
        long_y_pred = torch.unsqueeze(long_y_pred, 3)

        loss_long = metric_mape(long_y_pred, long_y_true)
        loss_long_SAM = cubeSAM(long_y_pred, long_y_true)
        # loss_long_SAM = metric_SAM(long_y_pred, long_y_true)

        # # TV over endmember spectra
        # total_variation = torch.norm(dec.layers[-1].weight.data[half_point+1:, :, :, :] - dec.layers[-1].weight.data[half_point:-1, :, :, :], p=2)

        # # TV over output spectra
        # total_variation = torch.norm(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :], p=2)

        # print(f'loss_short: {loss_short}, loss_long: {loss_long}, loss_long_SAM: {loss_long_SAM}, loss_short_SAM: {loss_short_SAM}, total variation: {total_variation}')

        # Loss as sum of the calculated components
        loss_sum = loss_short + loss_short_SAM + 10 * loss_long + 10 * loss_long_SAM #+ 0.01 * total_variation

        return loss_sum

    def test_fn(predcube, groundcube):
        """Calculate a test score for a prediction by comparing the predicted cube to a ground truth one.
        Very similar to loss_fn, but the long wavelengths are not averaged into single spectrum.
        N.B. This sort of testing is not possible if the approach is ever applied in practice!"""

        # Only the errors of the latter half are really interesting, so discard the shorter channels
        predcube = predcube[:, half_point:, :, :]
        groundcube = groundcube[:, half_point:, :, :]

        score_SAM = cubeSAM(predcube, groundcube)
        metric_MAPE = torchmetrics.MeanAbsolutePercentageError().to(device)
        score_MAPE = metric_MAPE(predcube, groundcube)
        # TODO calculate penalty for noise in output spectra
        return score_SAM + score_MAPE

    # FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
    params_to_optimize = [
        {'params': enc.parameters()},
        {'params': dec.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=common_params['learning_rate'])

    # For storing performance results
    train_losses = []
    test_scores = []

    n_epochs = epochs
    best_loss = 1e10
    best_index = 0

    best_test_loss = 1e10
    best_test_index = 0

    final_pred = None

    # Training loop
    for epoch in range(n_epochs):

        enc.train(True)
        dec.train(True)

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)  # Move data to GPU memory
            optimizer.zero_grad()  # Reset gradients
            enc_pred = enc(x)
            enc_pred = torch.nan_to_num(enc_pred)  # check nans again
            dec_pred = dec(enc_pred)
            final_pred = dec_pred
            loss = loss_fn(y, dec_pred)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()

            test_score = test_fn(y, dec_pred)
            test_item = test_score.item()

        if prints:
            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch {epoch}/{n_epochs} loss: {loss_item}   test: {test_item}")

            # logging.info(f"Epoch {epoch}/{n_epochs} loss: {loss_item}")
            # logging.info(f"Epoch {epoch}/{n_epochs} test: {test_item}")
        train_losses.append(loss_item)
        test_scores.append(test_item)

        if loss_item < best_loss:
            best_loss = loss_item
            best_index = epoch

        if test_item < best_test_loss:
            best_test_loss = test_item
            best_test_index = epoch

            # # This will save the whole shebang, which is a bit stupid
            # enc_save_name = "encoder.pt"
            # dec_save_name = "decoder.pt"
            # torch.save(enc, f"./{enc_save_name}")
            # torch.save(dec, f"./{dec_save_name}")

        # every n:th epoch plot endmember spectra and false color images from longer end
        if plots is True and (epoch % 1000 == 0 or epoch == n_epochs - 1):
            # Get weights of last layer, the endmember spectra, bring them to CPU and convert to numpy
            endmembers = dec.layers[-1].weight.data.detach().cpu().numpy()
            # Retrieve endmember spectra from middle of decoder kernels and plot them
            dec_kernel_mid = int((dec_params['d_kernel_size'] - 1) / 2)
            endmembers_mid = endmembers[:, :, dec_kernel_mid, dec_kernel_mid]
            plotter.plot_endmembers(endmembers_mid, epoch)

            final_pred = torch.squeeze(final_pred)
            final_pred = final_pred.detach().cpu().numpy()
            final_pred = utils.apply_circular_mask(final_pred, w, h)  # Use same circular mask on the output

            # Construct 3 channels to plot as false color images
            # Average the data for plotting over a few channels from the original cube and the reconstruction
            false_col_org = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            false_col_rec = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            for i in range(10):
                false_col_org = false_col_org + cube_original[(half_point + 5 + i, half_point + 15 + i, bands - 5 - i),
                                                :, :]  # TODO replace hardcoded indices
                false_col_rec = false_col_rec + final_pred[(half_point + 5 + i, half_point + 15 + i, bands - 5 - i), :,
                                                :]
            false_col_org = false_col_org / np.max(false_col_org)
            false_col_rec = false_col_rec / np.max(false_col_org)

            # juggle dimensions for plotting
            false_col_org = np.transpose(false_col_org, (2, 1, 0))
            false_col_rec = np.transpose(false_col_rec, (2, 1, 0))

            shape = np.shape(final_pred)
            spectral_angles = np.zeros((shape[1], shape[2]))
            best_SAM = 5
            best_indices = (0, 0)
            worst_SAM = 1e-5  # np.zeros((shape[0], 1))
            worst_indices = (0, 0)

            for i in range(shape[1]):
                for j in range(shape[2]):
                    orig = cube_original[:, i, j]
                    pred = final_pred[:, i, j]

                    spectral_angle = SAM(orig, pred)
                    spectral_angles[i, j] = spectral_angle

                    if spectral_angle < best_SAM:
                        best_SAM = spectral_angle
                        best_indices = (i, j)
                    if spectral_angle > worst_SAM and np.mean(orig) != 1:
                        worst_SAM = spectral_angle
                        worst_indices = (i, j)

            plotter.plot_SAM(spectral_angles, epoch)

            fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
            worst_ax = plotter.plot_spectra(cube_original[:, worst_indices[0], worst_indices[1]],
                                 final_pred[:, worst_indices[0], worst_indices[1]], tag='worst', ax=axs[0,0])
            best_ax = plotter.plot_spectra(cube_original[:, best_indices[0], best_indices[1]],
                                 final_pred[:, best_indices[0], best_indices[1]], tag='best', ax=axs[0,1])
            mid_ax = plotter.plot_spectra(cube_original[:, int(training_data.w / 2), int(training_data.h / 2)],
                                 final_pred[:, int(training_data.w / 2), int(training_data.h / 2)], tag='middle', ax=axs[1,0])
            folder = './figures/'
            image_name = f"spectra_worst_best_mid_e{epoch}.png"
            path = Path(folder, image_name)
            logging.info(f"Saving the image to '{path}'.")
            plt.savefig(path, dpi=300)
            plt.close(fig)

            plotter.plot_false_color(false_org=false_col_org, false_reconstructed=false_col_rec, dont_show=True,
                                     epoch=epoch)

    plotter.plot_nn_train_history(train_loss=train_losses,
                                  best_epoch_idx=best_index,
                                  test_scores=test_scores,
                                  best_test_epoch_idx=best_test_index,
                                  file_name='nn_history',
                                  log_y=True)

    return best_loss, best_test_loss
