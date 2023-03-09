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
from matplotlib import cm
from scipy.optimize import curve_fit


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


def join(*args) -> str:
    """Custom join function to avoid problems using os.path.join. """

    n = len(args)
    s = ''
    for i,arg in enumerate(args):
        if i == n-1:
            s = s + arg
        else:
            s = s + arg + '/'
    p = os.path.normpath(s)
    return p


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
    plt.imshow(map, vmin=0, vmax=3.1415 / 2)
    plt.colorbar()

    folder = './figures/'
    image_name = f"SAM_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


def plot_spectra(orig, pred, epoch, tag):
    """
    Plots two spectra, original and predicted, into same figure. Calculates for both curves a gradient to quantify the
    amount of variation, and includes the results in a legend. Constructs a filename and saves the figure on disc.

    :param orig:
        Original spectrum
    :param pred:
        Predicted spectrum, same length as original
    :param epoch:
        Training epoch where the prediction was made, will be included in filename
    :param tag:
        A tag to be included in filename, for example 'best' or 'worst' prediction
    """

    # To quantify noise, calculate gradients from both
    orig_grad = sum(abs(orig[1:] - orig[:-1]))
    pred_grad = sum(abs(pred[1:] - pred[:-1]))

    fig = plt.figure()
    plt.plot(orig, label=f'Original, grad: {orig_grad}')
    plt.plot(pred, label=f'Prediction, grad: {pred_grad}')
    plt.legend()

    folder = './figures/'
    image_name = f"spectra_{tag}_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


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
    for i in range(len(endmembers[0, :])):
        plt.plot(endmembers[:, i])

    folder = './figures/'
    image_name = f"endmembers_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)

    plt.close(fig)


def plot_nn_train_history(train_loss, best_epoch_idx, dont_show=True, save_thumbnail=True,
                          file_name="nn_train_history.png", log_y = False) -> None:
    """Plot training history of neural network.

    :param train_loss:
        List of training losses (per epoch).
    :param best_epoch_idx:
        Index of best epoch for highlighting.
    :param dont_show:
        If true, does not show the interactive plot (that halts excecution). Always use True when
        running multiple times in a loop (hyperparameter tuning). Default True.
    :param save_thumbnail:
        If True, save plot to disk. Default True.
    :param file_name:
        Filename for saving the plot. Postfix '.png' is added if missing. Default name is "nn_train_history.png".
    :param log_y:
        Whether to use logarithmic y-axis
    :return:
    """


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_single)
    fig.suptitle(f"Training history", fontsize=fig_title_font_size)
    ax.plot(train_loss, label="Training loss")
    if log_y == True:
        ax.set_yscale('log')
    ax.scatter(best_epoch_idx, train_loss[best_epoch_idx], facecolors='none', edgecolors='r')
    ax.set_xlabel('Epoch', fontsize=axis_label_font_size)
    ax.legend()

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
