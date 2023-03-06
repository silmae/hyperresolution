"""
Leaf model implementation as a neural network.

Greatly accelerates prediction time compared to the original
optimization method with some loss to accuracy.

"""

import logging
import sys
import math
import os
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
import scipy.io # for loading Matlab matrices
import matplotlib.pyplot as plt

import plotter

# Set manual seed when doing hyperparameter search for comparable results
# between training runs.
torch.manual_seed(42)

bands = 80

torch.autograd.set_detect_anomaly(True) # this will provide traceback if stuff turns into NaN

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

    def __init__(self, enc_layer_count = 3, e_filter_count=48, kernel_size=7, kernel_reduction=2, band_count=100, endmember_count=3):
        super(Encoder, self).__init__()

        self.band_count = band_count
        self.endmember_count = endmember_count
        self.enc_layer_count = enc_layer_count
        self.layers = nn.ModuleList()

        self.filter_counts = []
        for i in range(enc_layer_count-1):
            self.filter_counts.append(e_filter_count)
            e_filter_count = int(e_filter_count / 2)

        # self.layers.append(nn.Conv3d(in_channels=band_count, out_channels=band_count, kernel_size=(3,3,3), padding='same'))
        # self.layers.append(nn.Flatten())
        self.layers.append(nn.Conv2d(in_channels=band_count, out_channels=self.filter_counts[0], kernel_size=kernel_size, padding='same', stride=1, bias=False))
        if enc_layer_count > 2:
            for i in range(1, enc_layer_count-1):
                kernel_size = kernel_size - kernel_reduction
                self.layers.append(nn.Conv2d(in_channels=self.filter_counts[i-1], out_channels=self.filter_counts[i], kernel_size=kernel_size, padding='same', stride=1, bias=False))

        self.layers.append(nn.Conv2d(in_channels=int(self.filter_counts[-1]), out_channels=endmember_count, kernel_size=1, padding='same', stride=1, bias=False))

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
        # for layer in self.layers:
        #     x = self.activation(layer(x))
        #     x = self.norm(x)
        #     x = self.dropout(x) # dropout does not do good apparently
        # out = self.soft_max(x)

        for i in range(self.enc_layer_count):
            x = self.activation(self.layers[i](x))
            x = self.norms[i](x)

        out = self.soft_max(x*3.5)  # this really seems to be important XD

        return out


class Decoder(nn.Module):

    def __init__(self, band_count=200, endmember_count=3):
        super(Decoder, self).__init__()

        self.band_count = band_count
        self.endmember_count = endmember_count
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Conv2d(in_channels=self.endmember_count,
                      out_channels=self.band_count,
                      kernel_size=13,
                      padding='same',
                      stride=1,
                      bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # Force the weights to be positive, these will be the endmember spectra:
            layer.weight.data = layer.weight.data.clamp(min=0)
            # The endmembers are very noisy: calculate derivative and subtract it, then replace the weights with result
            variation = layer.weight.data[1:, :, :, :] - layer.weight.data[:-1, :, :, :]
            layer.weight.data[1:, :, :, :] = layer.weight.data[1:, :, :, :] - variation*0.01  # Not a good idea to subtract all of the variation, adjust the percentage

        return x


def file_loader():
    mat = scipy.io.loadmat("./datasets/TinyAPEX.mat")
    # mat = scipy.io.loadmat("./datasets/Samson.mat")
    w = mat['W'][0][0] # W is 2 dim matrix with 1 element
    h = mat['H'][0][0]
    l = mat['L'][0][0]
    abundance_count = mat['p'][0][0]
    cube_flat = np.array(mat['Y'])
    cube_flat = cube_flat.transpose()
    # plt.imshow(cube_flat)
    # plt.show()
    cube = cube_flat.reshape(h, w, l)

    return h, w, l, abundance_count, cube


class TrainingData(Dataset):
    """Handles catering the training data from disk to NN."""

    def __init__(self):
        h, w, l, abundance_count, cube = file_loader()

        self.w = w
        self.h = h
        self.l = l
        self.abundace_count = abundance_count
        l_half = int(self.l / 2)

        self.cube = np.transpose(cube, (2, 1, 0))

        first_half = self.cube[:l_half, :, :]

        # Dimensions of the image must be [batches, bands, width, height] for convolution
        # We make transformation from [h, w, bands] in here once at the loading time.
        self.X = torch.from_numpy(first_half).float()
        self.Y = torch.from_numpy(self.cube).float()
        # half_w = 50
        # half_h = 50
        # self.Xs = [
        #     self.X[:, :half_h, :half_w],
        #     self.X[:, half_h:half_h+half_h, half_w:half_w+half_w],
        #     self.X[:, half_h:half_h+half_h, :half_w],
        #     self.X[:, :half_h, half_w:half_w+half_w]
        # ]
        # self.Ys = [
        #     self.Y[:, :half_h, :half_w],
        #     self.Y[:, half_h:half_h + half_h, half_w:half_w + half_w],
        #     self.Y[:, half_h:half_h + half_h, :half_w],
        #     self.Y[:, :half_h, half_w:half_w + half_w]
        # ]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # add one empty dimension https://stackoverflow.com/questions/57237381/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-32-3-3-but
        # convert to float (the same must be done for the NNs later https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9)
        # return torch.unsqueeze(self.Xs[1], 0), torch.unsqueeze(self.Ys[1], 0)
        # return self.Xs[1], self.Ys[1]
        return self.X, self.Y


def train(epochs=1):

    training_data = TrainingData()
    bands = training_data.l
    half_point = int(bands/2)
    endmember_count = training_data.abundace_count
    # endmember_count = 15 # hard-coded value for testing if changing this has any effect
    # cube_original = training_data.Ys[1].numpy()  #training_data.cube
    cube_original = training_data.Y.numpy()  #training_data.cube
    data_loader = DataLoader(training_data, shuffle=False, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    enc = Encoder(enc_layer_count=2, band_count=int(bands/2), endmember_count=endmember_count, e_filter_count=128, kernel_size=7, kernel_reduction=2)
    dec = Decoder(band_count=bands, endmember_count=endmember_count)
    enc = enc.float()
    dec = dec.float()

    # Move network to GPU memory
    enc = enc.to(device)
    dec = dec.to(device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.3)

    enc.apply(init_weights)
    dec.apply(init_weights)

    logging.info(enc)
    logging.info(dec)
    logging.info(enc.parameters())
    logging.info(dec.parameters())

    def loss_fn(y_true, y_pred):
        # # Chop both true and predicted cubes in half with respect of wavelength

        # https://discuss.pytorch.org/t/custom-loss-functions/29387 -Kimmo

        def MAPE(Y_actual, Y_Predicted):
            abs_diff = torch.abs((Y_actual - Y_Predicted))
            # zero mask
            mask = (Y_actual != 0)
            # initialize output tensor with desired value
            norm = torch.full_like(Y_actual, fill_value=float('nan'))
            norm[mask] = torch.div(abs_diff[mask], Y_actual[mask]) # divide by non-zero elements
            norm = torch.nan_to_num(norm, posinf=1e10) # replace infinities with finite big number
            mean_diff = torch.mean(norm)
            mape = mean_diff * 100
            return mape

        def mean_spectral_gradient(cube):
            abs_grad = torch.abs(cube[:, 1:, :, :] - cube[:, :-1, :, :])
            sum_grad = torch.sum(abs_grad, dim=(1))
            mean_grad = torch.mean(sum_grad)
            # sum_grad = torch.sum(mean_grad)
            return mean_grad

        short_y_true = y_true[:, :half_point, :, :]
        long_y_true = y_true[:, half_point:, :, :]
        short_y_pred = y_pred[:, :half_point, :, :]
        long_y_pred = y_pred[:, half_point:, :, :]

        # Calculate short wavelength loss by comparing cubes. For loss metric MAPE
        # loss_short = MAPE(short_y_true, short_y_pred)
        metric_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
        metric_SAM = torchmetrics.SpectralAngleMapper().to(device)

        loss_short = metric_mape(short_y_pred, short_y_true)
        loss_short_SAM = metric_SAM(short_y_pred, short_y_true)

        # Calculate long wavelength loss by comparing mean spectra of long wavelength cubes
        long_y_true = torch.mean(long_y_true, dim=(2, 3))
        long_y_true = torch.unsqueeze(long_y_true, 2)
        long_y_true = torch.unsqueeze(long_y_true, 3)

        long_y_pred = torch.mean(long_y_pred, dim=(2, 3))
        long_y_pred = torch.unsqueeze(long_y_pred, 2)
        long_y_pred = torch.unsqueeze(long_y_pred, 3)

        # loss_long = MAPE(long_y_true, long_y_pred)
        loss_long = metric_mape(long_y_pred, long_y_true)
        loss_long_SAM = metric_SAM(long_y_pred, long_y_true)

        # Calculate the gradients of predicted spectra to quantify the noise
        # loss_grad = mean_spectral_gradient(y_pred) - mean_spectral_gradient(y_true)

        loss_sum = loss_short + loss_long + loss_long_SAM + loss_short_SAM  #+ loss_grad*2

        return loss_sum

    # FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
    params_to_optimize = [
        {'params': enc.parameters()},
        {'params': dec.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize)

    train_losses = []

    n_epochs = epochs
    best_loss = 1e10
    best_index = 0

    final_pred = None

    # Training loop
    for epoch in range(n_epochs):

        enc.train(True)
        dec.train(True)

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)  # Move data to GPU memory
            optimizer.zero_grad()  # Reset gradients
            enc_pred = enc(x)
            enc_pred = torch.nan_to_num(enc_pred) # check nans again
            dec_pred = dec(enc_pred)
            final_pred = dec_pred
            loss = loss_fn(y, dec_pred)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()

        logging.info(f"Epoch {epoch} loss: {loss_item}")
        train_losses.append(loss_item)
        if loss_item < best_loss:
            best_loss = loss_item
            best_index = epoch

            # # This will save the whole shebang, which is a bit stupid
            # enc_save_name = "encoder.pt"
            # dec_save_name = "decoder.pt"
            # torch.save(enc, f"./{enc_save_name}")
            # torch.save(dec, f"./{dec_save_name}")

        # plot every n:th epoch false color images
        if epoch % 1000 == 0 or epoch == n_epochs-1:
            # Get weights of last layer, the endmember spectra, bring them to CPU and convert to numpy
            endmembers = dec.layers[-1].weight.data.detach().cpu().numpy()
            endmembers_mid = endmembers[:, :, 6, 6]
            plotter.plot_endmembers(endmembers_mid, epoch)

            # pick 3 channels to plot as false color images
            # Average the data for plotting over a few channels from the original cube and the reconstruction
            final_pred = torch.squeeze(final_pred)
            final_pred = final_pred.detach().cpu().numpy()
            false_col_org = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            false_col_rec = np.zeros((3, np.shape(cube_original)[1], np.shape(cube_original)[2]))
            for i in range(10):
                false_col_org = false_col_org + cube_original[(half_point+5+i, half_point+50+i, bands-10-i), :, :]
                false_col_rec = false_col_rec + final_pred[(half_point+5+i, half_point+50+i, bands-10-i), :, :]
            false_col_org = false_col_org / 10
            false_col_rec = false_col_rec / 10

            # juggle dimensions for plotting
            false_col_org = np.transpose(false_col_org, (2,1,0))
            false_col_rec = np.transpose(false_col_rec, (2,1,0))

            shape = np.shape(final_pred)
            spectral_angles = np.zeros((shape[1], shape[2]))
            best_SAM = 5
            best_indices = (0, 0)
            worst_SAM = 5 # np.zeros((shape[0], 1))
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
                    if spectral_angle > worst_SAM:
                        worst_SAM = spectral_angle
                        worst_indices = (i, j)

                    # if i == 20 and j == 20:
                    #     plotter.plot_spectra(orig, pred, epoch)

            plotter.plot_SAM(spectral_angles, epoch)
            plotter.plot_spectra(cube_original[:, worst_indices[0], worst_indices[1]], final_pred[:, worst_indices[0], worst_indices[1]], epoch, tag='worst')
            plotter.plot_spectra(cube_original[:, best_indices[0], best_indices[1]], final_pred[:, best_indices[0], best_indices[1]], epoch, tag='best')
            plotter.plot_false_color(false_org=false_col_org, false_reconstructed=false_col_rec,dont_show=True, epoch=epoch)

    plotter.plot_nn_train_history(train_losses, best_index, file_name='nn_history', log_y=True)

    return best_loss


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    train(epochs=15000)



# The rest is abandoned code saved for snippets if needed

# def predict(r_m, t_m, nn_name='nn_default'):
#     """Use neural network to predict HyperBlend leaf model parameters from measured reflectance and transmittance.
#
#     :param r_m:
#         Measured reflectance.
#     :param t_m:
#         Measured transmittance.
#     :param nn_name:
#         Neural network name. Default name 'nn_default' is used if not given.
#         Provide only if you want to use your trained custom NN.
#     :return:
#         Lists ad, sd, ai, mf (absorption density, scattering desnity, scattering anisotropy, and mixing factor).
#         Use ``leaf_commons._convert_raw_params_to_renderable()`` before passing them to rendering method.
#     """
#
#     net = _load_model(nn_name=nn_name)
#     r_m = np.array(r_m)
#     t_m = np.array(t_m)
#     res = net(from_numpy(np.column_stack([r_m, t_m])))
#     res_item = res.detach().numpy()
#     ad = np.clip(res_item[:,0], 0., 1.)
#     sd = np.clip(res_item[:,1], 0., 1.)
#     ai = np.clip(res_item[:,2], 0., 1.)
#     mf = np.clip(res_item[:,3], 0., 1.)
#     return ad, sd, ai, mf


# def _load_model(nn_name):
#     """Loads the NN from disk.
#
#     :param nn_name:
#         Name of the model file.
#     :return:
#         Returns loaded NN.
#     :exception:
#         ModuleNotFoundError can happen if the network was trained when the name
#         of this script was something else than what it is now. Your only help
#         is to train again.
#     """
#     try:
#         p = _get_model_path(nn_name)
#         net = load(p)
#         net.eval()
#         logging.info(f"NN model loaded from '{p}'")
#     except ModuleNotFoundError as e:
#         logging.error(f"Pytorch could not load requested neural network. "
#                       f"This happens if class or file names associated with "
#                       f"NN are changed. You must train a new model to fix this.")
#         raise
#     return net


# def _get_model_path(nn_name='nn_default'):
#     """Returns path to the NN model.
#
#     :param nn_name:
#         Name of the NN.
#     :return:
#         Returns path to the NN model.
#     :exception:
#         FileNotFoundError if the model cannot be found.
#     """
#
#     if not nn_name.endswith('.pt'):
#         nn_name = nn_name + '.pt'
#     model_path = PH.join(PH.path_directory_surface_model(), nn_name)
#     if os.path.exists(model_path):
#         return model_path
#     else:
#         raise FileNotFoundError(f"Model '{model_path}' was not found. Check spelling.")


# def exists(nn_name='nn_default.pt'):
#     """Checks whether NN with given name exists.
#
#     :return:
#         True if found, False otherwise.
#     """
#
#     if not nn_name.endswith('.pt'):
#         nn_name = nn_name + '.pt'
#     model_path = PH.join(PH.path_directory_surface_model(), nn_name)
#     return os.path.exists(model_path)
