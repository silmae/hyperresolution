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
from src import file_handling
from src import simulation

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 300})
# LaTeX font for all text in all figures
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

figsize = (12, 6)
"""Figure size for two plot figures."""
figsize_single = (6, 6)
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

    # # To quantify noise, calculate gradients from both
    # orig_grad = sum(abs(orig[1:] - orig[:-1]))
    # pred_grad = sum(abs(pred[1:] - pred[:-1]))

    if len(orig) == 70:
        wls = constants.ASPECT_wavelengths
    elif len(orig) == 64:
        wls = constants.Itokawa_wavelengths

    ax.plot(wls, orig, label=f'Original')
    ax.plot(wls, pred, label=f'Prediction')
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
                          test_scores=None, file_name="nn_train_history.png", log_y=False) -> None:
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


def plot_abundance_maps(abundances, epoch, log_scale=False, titles=None):
    count = len(abundances)
    n_col = count
    n_row = 1

    fig, axs = plt.subplots(n_row, n_col, layout='constrained')  # , figsize=(12, 12))
    axs = axs.flatten()
    for i in range(count):
        if log_scale:
            im = axs[i].imshow(abundances[i], norm=colors.LogNorm(vmin=1e-3, vmax=10))
        else:
            im = axs[i].imshow(abundances[i], vmin=0, vmax=1)
            # mid_point_abundance = abundances[i, int(constants.ASPECT_NIR_channel_shape[0] / 2), int(constants.ASPECT_NIR_channel_shape[1] / 2)]
            if titles is not None:
                axs[i].title.set_text(f'{titles[i]}')
        im.axes.xaxis.set_ticks([])
        im.axes.yaxis.set_ticks([])
    fig.colorbar(im, ax=axs.ravel().tolist())
    folder = './figures/'
    image_name = f"abundance_maps_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_abundance_maps_with_gt(abundances, gt, RMSE_maps, epoch):
    """Plot predicted abundance maps together with ground truth maps and RMSE of abundance estimations"""

    n_row, n_col = len(abundances), 3
    fig, axs = plt.subplots(n_row, n_col, layout='constrained')  # , figsize=(12, 12))
    # axs = axs.flatten()
    # data_name = ['Ground truth', 'Predicted', 'RMSE']
    vmax = 0.75
    for i in range(len(abundances)):
        im = axs[i, 0].imshow(gt[i], vmin=0, vmax=vmax)
        axs[i, 0].title.set_text(f'Ground truth, EM {i + 1}')
        im.axes.xaxis.set_ticks([])
        im.axes.yaxis.set_ticks([])
        im = axs[i, 1].imshow(abundances[i], vmin=0, vmax=vmax)
        axs[i, 1].title.set_text(f'Predicted, EM {i + 1}')
        im.axes.xaxis.set_ticks([])
        im.axes.yaxis.set_ticks([])
        im = axs[i, 2].imshow(RMSE_maps[i, :, :], vmin=0, vmax=vmax)
        axs[i, 2].title.set_text(f'RMSE, EM {i + 1}')
        im.axes.xaxis.set_ticks([])
        im.axes.yaxis.set_ticks([])

    fig.colorbar(im, ax=axs.ravel().tolist())
    folder = './figures/'
    image_name = f"abundance_maps_with_gt_e{epoch}.{image_type}"
    path = Path(folder, image_name)
    logging.info(f"Saving the image to '{path}'.")
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_Itokawa_abundances(abundance_S, abundance_Q, epoch):
    """Plot abundance maps of Itokawa surface by overlaying them on an image to provide context. The coordinates
    for the abundance maps are fetched from the constants-file"""

    def safe_arange(start: float, stop: float or None = None, step: float = 1.0, dtype: type = float,
                    endpoint: bool = False, linspace_like: bool = True) -> np.ndarray:
        if stop is None:
            start, stop = 0.0, start

        if linspace_like:
            n = int(np.round((stop - start) / step)) + int(endpoint == True)
            return np.linspace(start, stop, n, endpoint=endpoint, dtype=dtype)

        return np.array(step * np.arange(start / step, stop / step), dtype=dtype)

    def plot_surface_abundance(abundance_map: np.ndarray, title='') -> None:

        font_size_axis = 36

        cmap = "jet"
        vmin = 0
        vmax = 1
        alpha = 0.4
        s = 10.

        xticks, yticks = safe_arange(0., 360., 10., endpoint=True), safe_arange(-90., 90., 10., endpoint=True)
        left, right = 0.0, 360.
        bottom, top = -90., 90.

        cticks, ctickslabel = safe_arange(vmin, vmax, .1, endpoint=True), safe_arange(vmin, vmax, .1, endpoint=True)

        background_image = "./datasets/Korda/new_itokawa_mosaic.jpg"

        indices_file = np.load('./datasets/Korda/Itokawa-denoised-norm.npz', allow_pickle=True)
        indices = np.array(indices_file["metadata"][:, :2], dtype=float)

        # mean_of_predictions = np.mean(y_pred, axis=0) * 100.

        # if what_type == "taxonomy":
        #     _, most_probable_classes_1 = get_most_probable_classes()
        #     _, most_probable_classes_2 = get_most_winning_classes()
        #     most_probable_classes = stack((most_probable_classes_1,
        #                                    np.setdiff1d(most_probable_classes_2, most_probable_classes_1)))
        #     n_probable_classes = len(most_probable_classes)
        #
        #     titles = ["".join((name, " ", classes2[most_probable_classes[i]],
        #                        "-type predictions")) for i in range(n_probable_classes)]
        #
        #     labels = [classes2[most_probable_classes[i]] for i in range(n_probable_classes)]
        #
        # elif what_type == "composition":
        #     # set titles (this should work well)
        #
        #     titles_all = [mineral_names] + endmember_names
        #     titles_all = flatten_list(titles_all)[used_indices(minerals_used, endmembers_used)]
        #     # titles_all = flatten_list(titles_all)[unique_indices(minerals_used, endmembers_used, all_minerals=True)]
        #
        #     most_probable_classes = unique_indices(minerals_used, endmembers_used, return_digits=True)
        #     labels = titles_all[most_probable_classes]
        #
        #     n_probable_classes = len(most_probable_classes)
        #
        #     print("\nSelected mineralogy:")
        #     for i, cls in enumerate(most_probable_classes):
        #         print("{:14s} {:5.2f}%".format(labels[i], round(mean_of_predictions[cls], 2)))
        #
        #     titles = ["".join((name, " ", labels[i], " predictions"))
        #               for i in range(n_probable_classes)]
        #
        # else:
        #     raise ValueError('"what_type" must be either "taxonomy" or "composition"')

        # Color code dominant classes / labels
        # probability_values = np.transpose(np.array([y_pred[:, most_probable_classes[i]]
        #                                             for i in range(n_probable_classes)]))

        # Plot the coverage map using latitude and longitude from HB
        img = plt.imread(background_image)  # Background image
        fig, ax = plt.subplots(figsize=(30, 25))
        ax.imshow(img, cmap="gray", extent=[0, 360, -90, 90], alpha=1)

        # # Draw the predictions map
        # values = y_pred[:, 20]
        # im = ax.scatter(indices[:, 0], indices[:, 1], s=s, c=values,
        #                 marker=",", cmap='jet', vmin=vmin, vmax=vmax, alpha=alpha)

        im = ax.imshow(abundance_map, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, extent=[constants.Itokawa_lon_min,
                                                                                            constants.Itokawa_lon_max,
                                                                                            constants.Itokawa_lat_min,
                                                                                            constants.Itokawa_lat_max])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        plt.xticks(rotation=90., fontsize=font_size_axis - 4)
        plt.yticks(fontsize=font_size_axis - 4)

        ax.grid()

        ax.set_xlabel("Longitude (deg)", fontsize=font_size_axis)  # \N{DEGREE SIGN}
        ax.set_ylabel("Latitude (deg)", fontsize=font_size_axis)
        ax.set_title(title, fontsize=font_size_axis + 4)

        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=bottom, top=top)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes(**cbar_kwargs)
        cbar = plt.colorbar(im, orientation='horizontal')  # , cax=cax)
        # cax = divider.append_axes("bottom", size="10%", pad=1.35)
        # cbar = plt.colorbar(abundance_map, orientation="horizontal")  # , cax=cax)

        # cbar.ax.set_yticklabels([f'{x:.1f}' for x in cticks])

        cbar.set_ticks(cticks)
        cbar.set_ticklabels([f'{x:.1f}' for x in cticks])
        cbar.ax.tick_params(labelsize=font_size_axis - 4)

        plt.draw()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figures/{title}_e{epoch}.png', dpi=100)
        plt.close(fig)

    plot_surface_abundance(abundance_Q, title='Q-type abundance')
    plot_surface_abundance(abundance_S, title='S-type abundance')


def illustrate_ASPECT_FOV(background_image=False):
    image_path = './datasets/Vesta_FC21B0014724_11354131448F1H.png'  # Vesta_FC21B0003982_11223231340F7E.png'):
    frame = cv.imread(image_path)
    if background_image is False:
        frame = frame / frame
    height = constants.ASPECT_VIS_channel_shape[0] + 100
    width = constants.ASPECT_VIS_channel_shape[1] + 100
    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    fig, ax = plt.subplots()
    ax.imshow(frame)  # , cmap='gray')

    rect_VIS = patches.Rectangle(xy=(50, 50),
                                 width=constants.ASPECT_VIS_channel_shape[1],
                                 height=constants.ASPECT_VIS_channel_shape[0],
                                 linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect_VIS)
    ax.text(65, 65, 'VIS: FOV 10x10 deg, 1024x1024 pixels',
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=10, color='red')

    height_NIR = int(constants.ASPECT_NIR_FOV[0] / constants.ASPECT_VIS_FOV[0] * constants.ASPECT_VIS_channel_shape[0])
    width_NIR = int(constants.ASPECT_NIR_FOV[1] / constants.ASPECT_VIS_FOV[1] * constants.ASPECT_VIS_channel_shape[1])
    rect_NIR = patches.Rectangle(xy=(int(width / 2 - width_NIR / 2),
                                     int(height / 2 - height_NIR / 2)),
                                 width=width_NIR,
                                 height=height_NIR,
                                 linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect_NIR)
    ax.text(width / 2 - width_NIR / 2, height / 2 - height_NIR / 2 - 30,
            f'NIR: FOV {constants.ASPECT_NIR_FOV[1]}x{constants.ASPECT_NIR_FOV[0]} deg, {constants.ASPECT_NIR_channel_shape[1]}x{constants.ASPECT_NIR_channel_shape[0]} pixels',
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=10, color='red')

    radius_SWIR = int(
        constants.ASPECT_SWIR_FOV / constants.ASPECT_VIS_FOV[0] * constants.ASPECT_VIS_channel_shape[0] / 2)
    circ_SWIR = patches.Circle(xy=(int(width / 2), int(height / 2)), radius=radius_SWIR,
                               linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(circ_SWIR)
    ax.text(width / 2, height / 2, 'SWIR: \nFOV 5.85 deg, 1 pixel',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10, color='red')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.show()


def illustrate_mixing_nonlinearity():
    pyroxene, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px100.csv'))
    olivine, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))
    px10, _ = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px10.csv'))
    px25, _ = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px25.csv'))
    px50, _ = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px50.csv'))
    px75, _ = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px75.csv'))
    px90, _ = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px90.csv'))

    px_factor = 0.10
    plt.figure()
    plt.plot(wls, pyroxene, label='Pyroxene', linestyle='--')
    plt.plot(wls, olivine, label='Olivine', linestyle='--')

    if px_factor == 0.10:
        plt.plot(wls, px10, label='Laboratory mixture', linestyle='-.')
    elif px_factor == 0.25:
        plt.plot(wls, px25, label='Laboratory mixture', linestyle='-.')
    elif px_factor == 0.50:
        plt.plot(wls, px50, label='Laboratory mixture', linestyle='-.')
    elif px_factor == 0.75:
        plt.plot(wls, px75, label='Laboratory mixture', linestyle='-.')
    elif px_factor == 0.90:
        plt.plot(wls, px90, label='Laboratory mixture', linestyle='-.')

    plt.plot(wls, np.exp(np.log(pyroxene) * px_factor + np.log(olivine) * (1 - px_factor)),
             label='Logarithmic mixture (simulated)')
    plt.plot(wls, pyroxene * px_factor + olivine * (1 - px_factor), label='Linear mixture (simulated)')
    plt.plot(wls, (utils.SSA2reflectance(
        utils.reflectance2SSA(pyroxene) * px_factor + utils.reflectance2SSA(olivine) * (1 - px_factor))),
             label='Hapke mixture (simulated)')
    plt.legend()
    plt.xlabel('Wavelengths [µm]')
    plt.ylabel('Reflectance')
    plt.show()
