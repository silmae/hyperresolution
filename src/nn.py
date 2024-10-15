import copy
from pathlib import Path
import logging
import sys
import math
import os
import gc

import numpy as np
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
import matplotlib.pyplot as plt

from src import plotter
from src import utils
from src import file_handling
from src import constants
from src import simulation

torch.autograd.set_detect_anomaly(True)  # this will provide traceback if stuff turns into NaN

epsilon = 1e-10  # An epsilon value is added in some places, mostly so that backprop will not break


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


def R2_distance(s1, s2):
    return np.sqrt(np.sum((s1 - s2)**2))


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
                                     padding_mode='reflect',
                                     stride=1,
                                     bias=False))
        if enc_layer_count > 2:
            for i in range(1, enc_layer_count - 1):
                if e_kernel_size - kernel_reduction >= 1:
                    e_kernel_size = e_kernel_size - kernel_reduction
                self.layers.append(nn.Conv2d(in_channels=self.filter_counts[i - 1],
                                             out_channels=self.filter_counts[i],
                                             kernel_size=e_kernel_size, padding='same',
                                             padding_mode='reflect',
                                             stride=1,
                                             bias=False))

        self.layers.append(nn.Conv2d(in_channels=int(self.filter_counts[-1]),
                                     out_channels=endmember_count, kernel_size=1,
                                     padding='same',
                                     padding_mode='reflect',
                                     stride=1,
                                     bias=False))

        self.soft_max = nn.Softmax(dim=1)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout2d(0.2)

        self.norms = nn.ModuleList()
        for count in self.filter_counts:
            self.norms.append(nn.BatchNorm2d(count))
        self.norms.append(nn.BatchNorm2d(endmember_count))

        self.activation = F.leaky_relu_
        # self.activation = F.selu_

    def forward(self, x):

        for i in range(self.enc_layer_count):
            x = self.activation(self.layers[i](x))
            x = self.norms[i](x)
            # x = self.dropout(x)  # dropout does not do good apparently

        out = self.soft_max(x * 10)  # the magical constant is from Palsson's paper, there called alpha

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
                      padding_mode='reflect',
                      stride=1,
                      bias=False)
        )

    def forward(self, x):

        for layer in self.layers:
            # Run input data through layer
            x = layer(x)

            # # Force the weights to be positive, these will be the endmember spectra:
            # layer.weight.data = layer.weight.data.clamp(min=0)
            # # The endmembers are very noisy: calculate derivative and subtract it, then replace the weights with result
            # variation = layer.weight.data[1:, :, :, :] - layer.weight.data[:-1, :, :, :]
            # layer.weight.data[1:, :, :, :] = layer.weight.data[1:, :, :,
            #                                  :] - variation * 0.001  # Not a good idea to subtract all of the variation, adjust the percentage

        # The mixing uses single-scattering albedos for endmember signals, so convert output cube back to reflectance
        x = utils.SSA2reflectance(x)

        return x


class TrainingData(Dataset):
    """Handles catering the training data from disk to NN."""

    def __init__(self, type, filepath, data_shape='actual'):

        if type == 'DAWN_PDS3':
            h, w, l, cube, wavelengths, FWHMs = file_handling.file_loader_Dawn_PDS3(filepath)
            self.FWHMs = FWHMs
        elif type == 'DAWN_ISIS':
            h, w, l, cube, wavelengths, FWHMs = file_handling.file_loader_Dawn_ISIS(filepath)
            # Crop the VIR cube to contain a bit more than the useful wavelengths: no useless processing, and less edge artifacts
            cube = cube[:, :, constants.VIR_channels_start_index:constants.VIR_channels_stop_index]
            wavelengths = wavelengths[constants.VIR_channels_start_index:constants.VIR_channels_stop_index]
            FWHMs = FWHMs[constants.VIR_channels_start_index:constants.VIR_channels_stop_index]
        elif type == 'simulated_Didymos':
            h, w, l, cube, wavelengths, FWHMs, gt_abundances = file_handling.file_loader_simulated_Didymos(filepath, spectrum='px75', crater='px50')
            # Transpose the abundance maps to get matching dimensions with network predictions
            for index, abundance in enumerate(gt_abundances):
                gt_abundances[index] = np.transpose(abundance)
            self.gt_abundances = gt_abundances
        elif type == 'simulated_Didymos_pyroxenes':
            h, w, l, cube, wavelengths, FWHMs, gt_abundances = file_handling.file_loader_simulated_Didymos_pyroxenes(frame_filepath=filepath,
                                                                                                                     spectrum1_filepath='datasets/RELAB_pyroxenes/c1dl10.tab',
                                                                                                                    spectrum2_filepath='datasets/RELAB_pyroxenes/c1dl13.tab')
            # Transpose the abundance maps to get matching dimensions with network predictions
            for index, abundance in enumerate(gt_abundances):
                gt_abundances[index] = np.transpose(abundance)
            self.gt_abundances = gt_abundances
        else:
            logging.info('Invalid training data type, ending execution')
            exit(1)

        # Make the data look like it came from ASPECT
        input_cube, SWIR_data, test_cube = simulation.ASPECT_NIR_SWIR_from_cube(cube, wavelengths, FWHMs, vignetting=True, smoothing=False, convert_rad2refl=False, data_shape=data_shape)
        input_cube = np.nan_to_num(input_cube, nan=1)  # Convert nans of short cube to ones
        if data_shape != 'actual':
            test_cube = np.nan_to_num(test_cube, nan=1)  # Convert nans of test cube to ones if no need to calculate nanmean

        # Dimension order is [h, w, l]
        self.h = test_cube.shape[0]
        self.w = test_cube.shape[1]
        self.l = test_cube.shape[2]

        # Dimensions of the image must be [batches, bands, width, height] for convolution
        # Transform from original [h, w, bands] with transpose
        test_cube = np.transpose(test_cube, (2, 1, 0))
        input_cube = np.transpose(input_cube, (2, 1, 0))

        if data_shape == 'actual':
            Y = np.zeros((2, input_cube.shape[0], input_cube.shape[1],
                          input_cube.shape[2]))  # add a dimension where the SWIR spectrum can be placed
            # Y[0, :, 0, 0] = SWIR_data
            SWIR_length = len(constants.ASPECT_wavelengths) - constants.ASPECT_SWIR_start_channel_index
            Y[0, :SWIR_length, 0, 0] = SWIR_data
            Y[1, :, :, :] = input_cube
        else:
            Y = test_cube

        X = input_cube

        # l_half = int(self.l / 2)

        # Convert the numpy arrays to torch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.cube = test_cube

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X, self.Y


def init_network(enc_params, dec_params, common_params, endmembers=None):
    enc = Encoder(enc_layer_count=enc_params['enc_layer_count'],
                  band_count=enc_params['band_count'],
                  endmember_count=common_params['endmember_count'] + 1,  # dark endmember not given, brightness evaluated as abundance and then used to scale the prediction
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

    # If endmember spectra are given as parameters, set them as decoder kernel weights
    if endmembers is not None:
        for i, endmember in enumerate(endmembers):
            dec.layers[-1].weight.data[:, i, 0, 0] = torch.tensor(endmember)
    # plt.figure()
    # plt.plot(dec.layers[-1].weight.data[:, 0, 0, 0].detach().cpu().numpy())
    # plt.plot(dec.layers[-1].weight.data[:, 1, 0, 0].detach().cpu().numpy())
    # plt.plot(dec.layers[-1].weight.data[:, 2, 0, 0].detach().cpu().numpy())
    # plt.show()

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


def cube_SID(x, y, return_map=False):
    """
    Computes the spectral information divergence (SID) between two spectral image cubes input as torch tensors.

    Based on the SID function found here: https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html#SID
    Modified to work on torch tensors instead of numpy vectors.

    Parameters:
        x: `torch tensor`
            Predicted image cube

        y: `torch tensor`
            Ground truth image cube

        return_map: 'Boolean'
            Whether to return a map of the SID values, or sum all of them into one value

    Returns: `float`
            Spectral information divergence between s1 and s2.

    Reference
        C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
        Similarity, and Discrimination for Hyperspectral Image"
        IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.

    """
    p = (x / torch.sum(x, dim=1)) + epsilon
    q = (y / torch.sum(y, dim=1)) + epsilon

    Dxy = p * torch.log(p / q)
    Dyx = q * torch.log(q / p)

    if return_map:
        SID = torch.sum(Dxy + Dyx, dim=1)
    else:
        SID = torch.sum(Dxy + Dyx)

    return SID


def tensor_image_corrcoeff(y_true, y_pred):
    """Calculate mean image and flatten to make compatible with torch Pearson correlation coefficient"""
    y_true_mean = torch.mean(y_true, dim=1)
    y_true_mean = torch.flatten(y_true_mean)
    y_pred_mean = torch.mean(y_pred, dim=1)
    y_pred_mean = torch.flatten(y_pred_mean)

    # The masking values being identical would push the correlation too high, so find their indices and discard them
    correlation_indices = torch.nonzero(torch.abs(y_true_mean - 1))
    y_true_mean = y_true_mean[correlation_indices]
    y_pred_mean = y_pred_mean[correlation_indices]

    flattened_means = torch.zeros(size=(2, len(y_true_mean)))
    flattened_means[0, :] = torch.flatten(y_true_mean)
    flattened_means[1, :] = torch.flatten(y_pred_mean)
    spatial_correlation = torch.corrcoef(flattened_means)[0, 1]  # Returns 2-by-2 matrix, take non-diagonal element
    if torch.isnan(spatial_correlation):
        spatial_correlation = 1e-10
    return spatial_correlation


def train(training_data, enc_params, dec_params, common_params, epochs=1, plots=True, prints=True, initial_endmembers=None, data_shape='actual'):
    bands = training_data.l
    SWIR_cutoff_index = constants.ASPECT_SWIR_start_channel_index

    # cube_original = training_data.Ys[1].numpy()  #training_data.cube
    cube_original = training_data.cube
    training_data.X = training_data.X.float()
    training_data.Y = training_data.Y.float()

    w = training_data.w
    h = training_data.h

    # A data loader object needed to feed the data to the network
    data_loader = DataLoader(training_data, shuffle=False, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Build and initialize the encoder and decoder
    enc, dec = init_network(enc_params, dec_params, common_params, endmembers=np.copy(initial_endmembers))

    # Move network to GPU memory
    enc = enc.to(device)
    dec = dec.to(device)

    test_cube = torch.from_numpy(cube_original).float()
    test_cube = torch.nan_to_num(torch.unsqueeze(test_cube, dim=0), nan=1)
    test_cube = test_cube.to(device)

    def loss_fn(y_true, y_pred):
        """Calculating loss by comparing predicted spectral image cube to ground truth"""

        short_y_true = y_true[:, 1, :, :, :]
        long_y_true = y_true[:, 0, :len(constants.ASPECT_wavelengths) - constants.ASPECT_SWIR_start_channel_index, 0, 0]

        short_y_pred = y_pred[:, :SWIR_cutoff_index, :, :]
        long_y_pred = y_pred[:, SWIR_cutoff_index:, :, :]

        short_y_pred = simulation.apply_circular_mask(short_y_pred, w, h, radius=constants.ASPECT_SWIR_equivalent_radius,
                                                 masking_value=1)
        long_y_pred = simulation.apply_circular_mask(long_y_pred, w, h, radius=constants.ASPECT_SWIR_equivalent_radius,
                                                masking_value=torch.nan)

        # Calculate short wavelength loss by comparing cubes. For loss metrics MAPE and SAM
        metric_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
        loss_short = metric_mape(short_y_pred, short_y_true)

        # loss_short_SAM = cubeSAM(short_y_pred, short_y_true)

        loss_short_SID = cube_SID(short_y_pred, short_y_true)

        # Calculate long wavelength loss by comparing mean spectrum of the masked prediction to GT spectrum
        long_y_pred = torch.nanmean(long_y_pred, dim=(2, 3))
        loss_long = metric_mape(long_y_pred, long_y_true)

        # loss_long_SAM = cubeSAM(long_y_pred, long_y_true)

        loss_long_SID = cube_SID(long_y_pred, long_y_true)


        # Correlation of GT cube spatial features and full reconstruction cube spatial features
        spatial_correlation = tensor_image_corrcoeff(short_y_true, y_pred)

        # # # Printing the losses separately for debugging purposes
        # print(f'loss_short: {loss_short}, loss_long: {loss_long}, loss_long_SAM: {loss_long_SAM}, loss_short_SAM: {loss_short_SAM}')

        # Loss as sum of the calculated components
        loss_sum = loss_short + (loss_short_SID * 100) + 10 * loss_long + 10 * (loss_long_SID * 100) + 100 * (1 - spatial_correlation)
        # loss_sum = loss_short_SAM + 10 * loss_long_SAM + 10 * (1 - spatial_correlation)

        return loss_sum

    def test_fn(groundcube, predcube, only_SWIR=True):
        """Calculate a test score for a prediction by comparing the predicted cube to a ground truth one.
        Very similar to loss_fn, but the long wavelengths are not averaged into single spectrum.
        N.B. This sort of testing is not possible if the approach is ever applied in practice!"""

        if only_SWIR:
            # Only the errors of the latter half are really interesting, so discard the shorter channels
            predcube = predcube[:, SWIR_cutoff_index:, :, :]
            groundcube = groundcube[:, SWIR_cutoff_index:, :, :]

        predcube = simulation.apply_circular_mask(predcube, w, h,
                                                      radius=constants.ASPECT_SWIR_equivalent_radius,
                                                      masking_value=1)

        SID = cube_SID(predcube, groundcube)  # spectral information divergence

        # score_SAM = cubeSAM(predcube, groundcube)
        # metric_MAPE = torchmetrics.MeanAbsolutePercentageError().to(device)
        # score_MAPE = metric_MAPE(predcube, groundcube)
        #
        # score_spatial_corr = 1 - tensor_image_corrcoeff(groundcube, predcube)
        # return 100 * score_SAM + score_MAPE + 100 * score_spatial_corr
        return SID

    def test_fn_unmixing(ground_abundances, pred_abundances, return_maps=False):
        """Calculate a score to quantify the unmixing performance by comparing produced abundance maps
        to ideal ones. """
        RMSE_scores = np.zeros((2, pred_abundances[0].shape[0], pred_abundances[0].shape[1]))
        for index in range(len(ground_abundances)):
            error = np.abs(pred_abundances[index] - ground_abundances[index]) ** 2
            RMSE_scores[index, :, :] = np.sqrt((1 / len(ground_abundances)) * error)

        if return_maps:
            return RMSE_scores
        else:
            return np.nanmean(RMSE_scores)

    # FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
    params_to_optimize = [
        {'params': enc.parameters()},
        {'params': dec.parameters()}
    ]

    # Defining the optimizer and setting its learning rate
    optimizer = torch.optim.AdamW(params_to_optimize, lr=common_params['learning_rate'], weight_decay=1e-3, amsgrad=True)

    # For storing performance results
    train_losses = []
    test_scores = []
    test_scores_unmixing = []

    n_epochs = epochs
    best_loss = 1e10
    best_index = 0

    best_test_loss = 1e10
    best_test_index = 0

    best_unmixing_test_loss = 1e10
    best_unmixing_test_index = 0

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
            abundances, brightness_map = torch.split(enc_pred, split_size_or_sections=[2, 1], dim=1)
            dec_pred = dec(abundances + epsilon)
            dec_pred = torch.nan_to_num(dec_pred)  # ... and again
            brightness_map = torch.abs(1 - brightness_map) + epsilon  # sum-to-one constraint means that brightness map is actually a darkness map: has high values where there are no spectral signals
            final_pred = torch.multiply(dec_pred, brightness_map)
            if data_shape == 'actual':
                loss = loss_fn(y, final_pred)  # If the data is shaped like ASPECT data, do a different loss calculation
            else:
                loss = test_fn(y, final_pred, only_SWIR=False)
            loss.backward()
            optimizer.step()

            # Implement freezing the endmembers by just plugging the original back in after backprop
            if initial_endmembers is not None:
                for i, endmember in enumerate(initial_endmembers):
                    endmember_tensor = torch.tensor(np.expand_dims(endmember, axis=(1, 2)))
                    dec.layers[-1].weight.data[:, i, :, :] = endmember_tensor

            loss_item = loss.item()
            test_score = test_fn(test_cube, final_pred, only_SWIR=False)
            test_item = test_score.item()

            # Unmixing performance by comparing predicted abundance maps to ground truth ones
            abundances = simulation.apply_circular_mask(enc_pred, h=w, w=h,
                                                        radius=constants.ASPECT_SWIR_equivalent_radius,
                                                        masking_value=torch.nan)
            abundances = np.squeeze(abundances.cpu().detach().numpy())
            pred_abundances = [abundances[0, :, :], abundances[1, :, :]]
            test_item_unmixing = test_fn_unmixing(training_data.gt_abundances, pred_abundances)
            test_scores_unmixing.append(test_item_unmixing)

            # Apply the mask, but a bit smaller than the SWIR FOV: the edges are discarded, because they have errors
            # from the decoder kernel operating on the masked values
            final_pred = simulation.apply_circular_mask(final_pred, w, h,
                                                   radius=constants.ASPECT_SWIR_equivalent_radius - dec_params[
                                                       'd_kernel_size'], masking_value=1)
            # test_cube = simulation.apply_circular_mask(test_cube, w, h,
            #                                       radius=constants.ASPECT_SWIR_equivalent_radius - dec_params[
            #                                           'd_kernel_size'], masking_value=1)

        if prints:
            memory_usage = torch.cuda.memory_allocated() * (1024 ** -3)  # Fetch used memory in bytes and convert to GB
            sys.stdout.write('\r')
            sys.stdout.write(f"Epoch {epoch}/{n_epochs} loss: {loss_item:.4f}   test: {test_item_unmixing:.4f}   (GPU memory usage: {memory_usage:.2f} GB)")

        train_losses.append(loss_item)
        test_scores.append(test_item)

        if loss_item < best_loss:
            best_loss = loss_item
            best_index = epoch
        if test_item < best_test_loss:
            best_loss = test_item
            best_test_index = epoch
        if test_item_unmixing < best_unmixing_test_loss:
            best_unmixing_test_loss = test_item_unmixing
            best_unmixing_test_index = epoch

        # early_stop_thresh = 50
        # if test_item < best_test_loss:
        #     best_test_loss = test_item
        #     best_test_index = epoch
        # elif (epoch > 500) and (epoch - best_test_index > early_stop_thresh):
        #     print("Early stopped training at epoch %d" % epoch)
        #     break  # terminate the training loop

        # Every n:th epoch make plots of the results
        if plots is True and (epoch % 50 == 0 or epoch == n_epochs - 1):

            plot_endmembers = False
            if plot_endmembers:  # Plot endmember spectra if they are not given as parameters
                # Get weights of last layer, the endmember spectra, bring them to CPU and convert to numpy
                endmembers = dec.layers[-1].weight.data.detach().cpu().numpy()
                # Retrieve endmember spectra by summing the weights of each kernel
                endmembers = np.sum(np.sum(endmembers, axis=-1), axis=-1)  # sum over both spatial axes
                plotter.plot_endmembers(endmembers, epoch)

            # Plot brightness map
            br_map = simulation.apply_circular_mask(brightness_map, h=w, w=h, radius=constants.ASPECT_SWIR_equivalent_radius,
                                                   masking_value=torch.nan)
            br_map = np.squeeze(br_map.cpu().detach().numpy())
            fig = plt.figure()
            plt.imshow(br_map)
            folder = './figures/'
            image_name = f"brightness_map_e{epoch}.png"
            path = Path(folder, image_name)
            logging.info(f"Saving the image to '{path}'.")
            plt.savefig(path, dpi=300)
            plt.close(fig)

            # Get abundance maps from encoder predictions
            abundances = simulation.apply_circular_mask(enc_pred, h=w, w=h, radius=constants.ASPECT_SWIR_equivalent_radius,
                                                   masking_value=torch.nan)
            abundances = np.squeeze(abundances.cpu().detach().numpy())
            # plotter.plot_abundance_maps(abundances, epoch)

            # Calculate RMSE error maps of abundance predictions
            pred_abundances = [abundances[0, :, :], abundances[1, :, :]]
            abundance_error_maps = test_fn_unmixing(training_data.gt_abundances, pred_abundances, return_maps=True)
            # plotter.plot_abundance_maps(abundance_error_maps, epoch=f'{epoch}_RMSE')

            gt = copy.deepcopy(training_data.gt_abundances)
            for i in range(2):
                gt[i] = simulation.apply_circular_mask(np.expand_dims(gt[i] + 1e-6, axis=-1), h=w, w=h,
                                                       radius=constants.ASPECT_SWIR_equivalent_radius,
                                                       masking_value=np.nan)
            plotter.plot_abundance_maps_with_gt(pred_abundances, gt, abundance_error_maps, epoch)
            del gt

            final_pred = torch.squeeze(final_pred)
            # Use same circular mask on the output, note that the order of width and height is opposite here
            final_pred = simulation.apply_circular_mask(final_pred, w, h, radius=constants.ASPECT_SWIR_equivalent_radius,
                                                   masking_value=0)
            final_pred = final_pred.detach().cpu().numpy()

            # # Construct 3 channels to plot as false color images
            # # Average the data for plotting over a few channels from the original cube and the reconstruction
            # false_col_org = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            # false_col_rec = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            # for i in range(10):
            #     false_col_org = false_col_org + cube_original[
            #                                     (SWIR_cutoff_index + 5 + i, SWIR_cutoff_index + 15 + i, bands - 5 - i),
            #                                     :, :]  # TODO replace hardcoded indices
            #
            #     false_col_rec = false_col_rec + final_pred[
            #                                     (SWIR_cutoff_index + 5 + i, SWIR_cutoff_index + 15 + i, bands - 5 - i),
            #                                     :, :]
            # # juggle dimensions for plotting
            # false_col_org = np.transpose(false_col_org, (2, 1, 0))
            # false_col_rec = np.transpose(false_col_rec, (2, 1, 0))
            # # Convert nans to zeros and normalize with maximum value in the image
            # false_col_org = np.nan_to_num(false_col_org, nan=0)
            # false_col_rec = np.nan_to_num(false_col_rec, nan=0)
            # false_col_org = (false_col_org / np.max(false_col_org))
            # false_col_rec = (false_col_rec / np.max(false_col_rec))
            # plotter.plot_false_color(false_org=false_col_org, false_reconstructed=false_col_rec, dont_show=True,
            #                          epoch=epoch)

            # Calculate spectral angle between ground and pred in each pixel, then plot the values as a colormap
            shape = np.shape(final_pred)
            spectral_angles = np.zeros((shape[1], shape[2]))
            R2_distances = np.zeros((shape[1], shape[2]))
            best_score = 5
            best_indices = (0, 0)
            worst_score = 1e-5  # np.zeros((shape[0], 1))
            worst_indices = (0, 0)

            for i in range(shape[1]):
                for j in range(shape[2]):
                    orig = cube_original[:, i, j]
                    pred = final_pred[:, i, j]

                    spectral_angle = SAM(orig, pred)
                    spectral_angles[i, j] = spectral_angle

                    R2_dist = R2_distance(orig, pred)
                    R2_distances[i, j] = R2_dist

                    if ((spectral_angle + R2_dist) < best_score) and (not np.isnan(orig[0])) and (np.mean(abs(orig[:-1] - orig[1:])) > 0.001):
                        best_score = spectral_angle + R2_dist
                        best_indices = (i, j)
                    if (spectral_angle + R2_dist) > worst_score and np.mean(orig) != 1 and np.mean(orig) > 0.01:
                        worst_score = spectral_angle + R2_dist
                        worst_indices = (i, j)

            # Plots to illustrate reconstruction error
            plotter.plot_SAM(spectral_angles, epoch)
            # plotter.plot_R2(R2_distances, epoch)

            # Find the worst, best and middle pixels from the pred and original images
            worst_pred = final_pred[:, worst_indices[0], worst_indices[1]]
            worst_orig = cube_original[:, worst_indices[0], worst_indices[1]]
            best_pred = final_pred[:, best_indices[0], best_indices[1]]
            best_orig = cube_original[:, best_indices[0], best_indices[1]]
            mid_pred = final_pred[:, int(training_data.w / 2), int(training_data.h / 2)]
            mid_orig = cube_original[:, int(training_data.w / 2), int(training_data.h / 2)]

            normalize_example_spectra = True
            if normalize_example_spectra:
                worst_pred = worst_pred / np.max(worst_pred)
                worst_orig = worst_orig / np.max(worst_orig)
                best_pred = best_pred / np.max(best_pred)
                best_orig = best_orig / np.max(best_orig)
                mid_pred = mid_pred / np.max(mid_pred)
                mid_orig = mid_orig / np.max(mid_orig)

            fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
            worst_ax = plotter.plot_spectra(worst_orig, worst_pred, tag='worst', ax=axs[0, 0])
            best_ax = plotter.plot_spectra(best_orig, best_pred, tag='best', ax=axs[0, 1])
            mid_ax = plotter.plot_spectra(mid_orig, mid_pred, tag='middle', ax=axs[1, 0])
            axs[1, 1].imshow(cube_original[20, :, :])
            axs[1, 1].scatter(worst_indices[1], worst_indices[0], color='r', marker='o')
            axs[1, 1].scatter(best_indices[1], best_indices[0], color='g', marker='o')
            folder = './figures/'
            image_name = f"spectra_worst_best_mid_e{epoch}.png"
            path = Path(folder, image_name)
            logging.info(f"Saving the image to '{path}'.")
            plt.savefig(path, dpi=300)
            plt.close(fig)

            plotter.plot_nn_train_history(train_loss=train_losses,
                                          best_epoch_idx=best_index,
                                          test_scores=test_scores_unmixing,
                                          best_test_epoch_idx=best_unmixing_test_index,
                                          file_name='figures/nn_history',
                                          log_y=True)

        # Delete some stuff to free up GPU memory
        del loss, loss_item, test_score, test_item, test_item_unmixing, final_pred, enc_pred, abundances, pred_abundances
        torch.cuda.empty_cache()
        # Run garbage collection
        gc.collect()

    last_loss = train_losses[-1]
    last_test_loss = test_scores[-1]
    last_unmixing_test_loss = test_scores_unmixing[-1]

    return best_loss, best_test_loss, best_unmixing_test_loss, last_loss, last_test_loss, last_unmixing_test_loss
