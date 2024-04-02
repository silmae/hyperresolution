"""
This file contains plotting-related code.

Tips for plotting:
https://towardsdatascience.com/5-powerful-tricks-to-visualize-your-data-with-matplotlib-16bc33747e05

"""

import os
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib import cm
# from scipy.optimize import curve_fit
import cv2 as cv

from src import utils
from src import constants

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 300})
# LaTeX font for all text in all figures
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

figsize = (12,6)
"""Figure size for two plot figures."""
figsize_single = (6,6)
"""Figure size for single plot figures."""
fig_title_font_size = 18
"""Title font size."""
axis_label_font_size = 16
"""Axis label font size"""

variable_space_ylim = [0.0, 1.0]
"""Y-axis limit for leaf material parameter plot."""

# Colors
color_reflectance = 'royalblue'
color_transmittance = 'deeppink'
color_reflectance_measured = 'black'
color_transmittance_measured = 'black'
color_ad = 'olivedrab'
color_sd = 'darkorange'
color_ai = 'brown'
color_mf = 'darkorchid'
color_history_target = 'black'

alpha_error = 0.2
"""Alpha for std shadow."""

max_ticks = 8
"""Max tick count for wavelength."""

image_type = 'png'


def plot_false_color(false_org, false_reconstructed, epoch, dont_show=True, save_thumbnail=True) -> None:
    """

    :param save_thumbnail:
        If True, a PNG image is saved to result/plot folder. Default is True.
    :param dont_show:
        If True, the plot is not plotted on the monitor. Use together with save_thumbnail. Default is True.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle(f"False color cubes", fontsize=fig_title_font_size)
    ax[0].set_title('Original')
    ax[1].set_title('Reconstruct')
    ax[0].imshow(false_org)
    ax[1].imshow(false_reconstructed)

    if save_thumbnail is not None:
        folder = './figures/'
        image_name = f"false_e{epoch}.{image_type}"
        path = Path(folder, image_name)
        logging.info(f"Saving the image to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)


def plot_SAM(map, epoch):
    """
    Plots a spectral angle map and saves the figure on disc.

    :param map:
    The spectral angle map to be plotted, a 2D array where each element is the spectral angle in radians
    :param epoch:
    Training epoch where the map was calculated, will be included in filename of saved figure
    """

    fig = plt.figure()
    fig.suptitle('SAM')
    plt.imshow(map, vmin=0, vmax=3.1415 / 4)
    plt.colorbar()

    folder = './figures/'
    image_name = f"SAM_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


def plot_R2(map, epoch):
    """
    Plots a map of R2 distance and saves the figure on disc.

    :param map:
    Map to be plotted, a 2D array where each element is the R2 distance between two spectra
    :param epoch:
    Training epoch where the map was calculated, will be included in filename of saved figure
    """

    fig = plt.figure()
    fig.suptitle('Euclidean distance')
    plt.imshow(map, vmin=0, vmax=0.5)
    plt.colorbar()

    folder = './figures/'
    image_name = f"R2_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


def plot_spectra(orig, pred, tag, ax):
    """
    Plots two spectra, original and predicted, into same figure. Calculates for both curves a gradient to quantify the
    amount of variation, and includes the results in a legend. Plots into axis object given as parameter, returns the
    object.

    :param orig:
        Original spectrum
    :param pred:
        Predicted spectrum, same length as original
    :param tag:
        A tag to be included in filename, for example 'best' or 'worst' prediction
    :param ax:
        Matplotlib axis object
    :return:
        Matplotlib axis object
    """

    # To quantify noise, calculate gradients from both
    orig_grad = sum(abs(orig[1:] - orig[:-1]))
    pred_grad = sum(abs(pred[1:] - pred[:-1]))

    ax.plot(constants.ASPECT_wavelengths, orig, label=f'Original, grad: {orig_grad:.2f}')
    ax.plot(constants.ASPECT_wavelengths, pred, label=f'Prediction, grad: {pred_grad:.2f}')
    ax.legend()
    ax.set_title(f'{tag}')

    return ax

def plot_endmembers(endmembers, epoch):
    """
    Plots endmember spectra into one figure, and saves it on disc.

    :param endmembers:
    All endmembers included into one ndarray
    :param epoch:
    Training epoch where the endmembers were saved, will be included in filename of saved figure
    """

    fig = plt.figure()
    fig.suptitle('Endmember spectra')
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('I/F')
    for i in range(len(endmembers[0, :])):
        plt.plot(constants.ASPECT_wavelengths, endmembers[:, i])

    folder = './figures/'
    image_name = f"endmembers_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


def plot_nn_train_history(train_loss, best_epoch_idx, best_test_epoch_idx=None, dont_show=True, save_thumbnail=True,
                          test_scores=None, file_name="nn_train_history.png", log_y = False) -> None:
    """Plot training history of neural network.

    :param train_loss:
        List of training losses (per epoch).
    :param best_epoch_idx:
        Index of best epoch for highlighting, according to training data.
    :param best_test_epoch_idx:
        Index of best epoch for highlighting, according to test data.
    :param dont_show:
        If true, does not show the interactive plot (that halts excecution). Always use True when
        running multiple times in a loop (hyperparameter tuning). Default True.
    :param save_thumbnail:
        If True, save plot to disk. Default True.
    :param test_scores:
        Scores from testing the network, not needed for the train history plot. Default is None.
    :param file_name:
        Filename for saving the plot. Postfix '.png' is added if missing. Default name is "nn_train_history.png".
    :param log_y:
        Whether to use logarithmic y-axis
    :return:
    """

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Training history", fontsize=fig_title_font_size)
    # ax1.plot(train_loss, label="Training loss")
    color = 'tab:orange'
    ax1.set_xlabel('Epoch', fontsize=axis_label_font_size)
    ax1.set_ylabel('Training loss', color=color)
    ax1.plot(train_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    if log_y == True:
        ax1.set_yscale('log')
    ax1.scatter(best_epoch_idx, train_loss[best_epoch_idx], facecolors='none', edgecolors='g')
    # ax1.legend()

    if test_scores is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Test loss', color=color)  # we already handled the x-label with ax1
        ax2.plot(test_scores, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if log_y == True:
            ax2.set_yscale('log')
        ax2.scatter(best_test_epoch_idx, test_scores[best_test_epoch_idx], facecolors='none', edgecolors='r')

    fig.tight_layout()

    if save_thumbnail:
        if not file_name.endswith(".png"):
            file_name = file_name + '.png'
        path = Path('./', file_name)
        logging.info(f"Saving NN training history to '{path}'.")
        plt.savefig(path, dpi=300)
    if not dont_show:
        plt.show()

    # close the figure to avoid memory consumption warning when over 20 figs
    plt.close(fig)


def plot_abundance_maps(abundances, epoch, log_scale=False):

    # Sorry about the next lines, can't be bothered to think about this
    count = abundances.shape[0]
    if count <= 4:
        n_row = 2
        n_col = 2
    elif count <= 6:
        n_row = 2
        n_col = 3
    elif count <= 9:
        n_row = 3
        n_col = 3
    else:
        n_row = 3
        n_col = 4

    fig, axs = plt.subplots(n_row, n_col, layout='constrained')  # , figsize=(12, 12))
    axs = axs.flatten()
    for i in range(count):
        if log_scale:
            im = axs[i].imshow(abundances[i, :, :], norm=colors.LogNorm(vmin=1e-3, vmax=10))
        else:
            im = axs[i].imshow(abundances[i, :, :], vmin=0, vmax=1)
            mid_point_abundance = abundances[i, int(constants.ASPECT_NIR_channel_shape[0] / 2), int(constants.ASPECT_NIR_channel_shape[1] / 2)]
            axs[i].title.set_text(f'Middle point: {mid_point_abundance:.2f}')
        im.axes.xaxis.set_ticks([])
        im.axes.yaxis.set_ticks([])
    fig.colorbar(im, ax=axs.ravel().tolist())
    folder = './figures/'
    image_name = f"abundance_maps_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)


def illustrate_ASPECT_FOV(background_image=False):
    image_path = './datasets/Vesta_FC21B0014724_11354131448F1H.png' #Vesta_FC21B0003982_11223231340F7E.png'):
    frame = cv.imread(image_path)
    if background_image is False:
        frame = frame / frame
    height = constants.ASPECT_VIS_channel_shape[0] + 100
    width = constants.ASPECT_VIS_channel_shape[1] + 100
    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    fig, ax = plt.subplots()
    ax.imshow(frame)#, cmap='gray')

    rect_VIS = patches.Rectangle(xy=(50, 50),
                                 width=constants.ASPECT_VIS_channel_shape[1],
                                 height=constants.ASPECT_VIS_channel_shape[0],
                                 linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect_VIS)
    ax.text(65, 65, 'VIS: FOV 10x10 deg, 1024x1024 pixels',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=10, color='red')

    height_NIR =  int(constants.ASPECT_NIR_FOV[0] / constants.ASPECT_VIS_FOV[0] * constants.ASPECT_VIS_channel_shape[0])
    width_NIR = int(constants.ASPECT_NIR_FOV[1] / constants.ASPECT_VIS_FOV[1] * constants.ASPECT_VIS_channel_shape[1])
    rect_NIR = patches.Rectangle(xy=(int(width / 2 - width_NIR / 2),
                                     int(height / 2 - height_NIR / 2)),
                                 width=width_NIR,
                                 height=height_NIR,
                                 linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect_NIR)
    ax.text(width/2 - width_NIR/2, height/2 - height_NIR/2 - 30, f'NIR: FOV {constants.ASPECT_NIR_FOV[1]}x{constants.ASPECT_NIR_FOV[0]} deg, {constants.ASPECT_NIR_channel_shape[1]}x{constants.ASPECT_NIR_channel_shape[0]} pixels',
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=10, color='red')

    radius_SWIR = int(constants.ASPECT_SWIR_FOV / constants.ASPECT_VIS_FOV[0] * constants.ASPECT_VIS_channel_shape[0] / 2)
    circ_SWIR = patches.Circle(xy=(int(width/2), int(height/2)), radius=radius_SWIR,
                               linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(circ_SWIR)
    ax.text(width/2, height/2, 'SWIR: \nFOV 5.85 deg, 1 pixel',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='red')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.show()


