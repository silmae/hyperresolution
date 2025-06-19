from pathlib import Path
import logging
import os
import math
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from planetaryimage import CubeFile

import optuna
import cv2 as cv

from src import nn
from src import utils
from src import plotter
from src import constants
from src import file_handling
from src import simulation

if __name__ == '__main__':
    # Set manual seed for comparable results between training runs
    torch.manual_seed(42)

    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    # # Save logs into file
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)  # Setup the root logger.
    #
    # now = datetime.now()
    # filename = now.strftime("%Y-%M-%d_%H:%M:%S")
    #
    # logger.addHandler(logging.FileHandler(f"{filename}.log", mode="w"))

    ############# SANDBOX ###############

    # # Plot to illustrate how the FOVs of the ASPECT modules overlap each other
    # plotter.illustrate_ASPECT_FOV()
    # # Plot to illustrate nonlinearity of spectral mixing
    # plotter.illustrate_mixing_nonlinearity()

    # # Loading RELAB spectra
    # file_handling.load_RELAB_spectrum(filepath=Path('datasets/RELAB_pyroxenes/c1dl51a.tab'))
    # pyroxene_spectra = []
    # pyroxene_filenames = []
    # filelist = os.listdir(Path('datasets/RELAB_pyroxenes'))
    # # plt.figure()
    # for i, filename in enumerate(filelist):
    #     if '.tab' in filename:# and not 'a.tab' in filename:# and i>=120:
    #         wls, refl = file_handling.load_RELAB_spectrum(Path('datasets/RELAB_pyroxenes', filename))
    #         # plt.plot(wls, refl, label=filename)
    #         refl, new_wls, _ = simulation.ASPECT_resampling(refl, wls)
    #         pyroxene_spectra.append(refl)
    #         pyroxene_filenames.append(filename)
    # best_SAM_score = 1000
    # best_indices = [0, 0]
    #
    # for i in range(len(pyroxene_spectra)):
    #     for j in range(i + 1, len(pyroxene_spectra)):
    #         spectrum = pyroxene_spectra[i]
    #         comparison_spectrum = pyroxene_spectra[j]
    #         SAM_short = nn.SAM(spectrum[:constants.ASPECT_SWIR_start_channel_index],
    #                            comparison_spectrum[:constants.ASPECT_SWIR_start_channel_index])
    #         SAM_long = nn.SAM(spectrum[constants.ASPECT_SWIR_start_channel_index:],
    #                           comparison_spectrum[constants.ASPECT_SWIR_start_channel_index:])
    #         SAM_score = 2*SAM_short - SAM_long
    #         if SAM_score < best_SAM_score:
    #             best_SAM_score = SAM_score
    #             best_indices = [i, j]
    #
    # best_pair_filenames = [pyroxene_filenames[best_indices[0]], pyroxene_filenames[best_indices[1]]]
    # best_pair_spectra = [pyroxene_spectra[best_indices[0]], pyroxene_spectra[best_indices[1]]]
    # plt.figure()
    # plt.plot(new_wls, best_pair_spectra[0] / np.max(best_pair_spectra[0]), label=best_pair_filenames[0])
    # plt.plot(new_wls, best_pair_spectra[1] / np.max(best_pair_spectra[1]), label=best_pair_filenames[1])
    # plt.legend()
    # plt.show()

    # # Load data received from David
    file_handling.file_loader_Itokawa_NIRS(path='./datasets/Korda/Itokawa-denoised-norm.npz')

    #
    # def safe_arange(start: float, stop: float or None = None, step: float = 1.0, dtype: type = float,
    #                 endpoint: bool = False, linspace_like: bool = True) -> np.ndarray:
    #     if stop is None:
    #         start, stop = 0.0, start
    #
    #     if linspace_like:
    #         n = int(np.round((stop - start) / step)) + int(endpoint == True)
    #         return np.linspace(start, stop, n, endpoint=endpoint, dtype=dtype)
    #
    #     return np.array(step * np.arange(start / step, stop / step), dtype=dtype)
    #
    # def plot_surface_spectra(y_pred: np.ndarray) -> None:
    #
    #     font_size_axis = 36
    #
    #     cmap = "viridis_r"
    #     vmin, vmax = 0.8, 0.9
    #     alpha = 0.4
    #     s = 10.
    #
    #     xticks, yticks = safe_arange(0., 360., 10., endpoint=True), safe_arange(-90., 90., 10., endpoint=True)
    #     left, right = 0.0, 360.
    #     bottom, top = -90., 90.
    #
    #     cticks, ctickslabel = safe_arange(vmin, vmax, .1, endpoint=True), safe_arange(vmin, vmax, .1, endpoint=True)
    #
    #     # if "Itokawa" in filename:
    #     background_image = "./datasets/Korda/new_itokawa_mosaic.jpg"
    #     name = "Itokawa"
    #
    #
    #     # indices_file = np.load("".join((_path_data, filename)), allow_pickle=True)
    #     indices_file = np.load('./datasets/Korda/Itokawa-denoised-norm.npz', allow_pickle=True)
    #     indices = np.array(indices_file["metadata"][:, :2], dtype=float)
    #
    #     # mean_of_predictions = np.mean(y_pred, axis=0) * 100.
    #
    #     # if what_type == "taxonomy":
    #     #     _, most_probable_classes_1 = get_most_probable_classes()
    #     #     _, most_probable_classes_2 = get_most_winning_classes()
    #     #     most_probable_classes = stack((most_probable_classes_1,
    #     #                                    np.setdiff1d(most_probable_classes_2, most_probable_classes_1)))
    #     #     n_probable_classes = len(most_probable_classes)
    #     #
    #     #     titles = ["".join((name, " ", classes2[most_probable_classes[i]],
    #     #                        "-type predictions")) for i in range(n_probable_classes)]
    #     #
    #     #     labels = [classes2[most_probable_classes[i]] for i in range(n_probable_classes)]
    #     #
    #     # elif what_type == "composition":
    #     #     # set titles (this should work well)
    #     #
    #     #     titles_all = [mineral_names] + endmember_names
    #     #     titles_all = flatten_list(titles_all)[used_indices(minerals_used, endmembers_used)]
    #     #     # titles_all = flatten_list(titles_all)[unique_indices(minerals_used, endmembers_used, all_minerals=True)]
    #     #
    #     #     most_probable_classes = unique_indices(minerals_used, endmembers_used, return_digits=True)
    #     #     labels = titles_all[most_probable_classes]
    #     #
    #     #     n_probable_classes = len(most_probable_classes)
    #     #
    #     #     print("\nSelected mineralogy:")
    #     #     for i, cls in enumerate(most_probable_classes):
    #     #         print("{:14s} {:5.2f}%".format(labels[i], round(mean_of_predictions[cls], 2)))
    #     #
    #     #     titles = ["".join((name, " ", labels[i], " predictions"))
    #     #               for i in range(n_probable_classes)]
    #     #
    #     # else:
    #     #     raise ValueError('"what_type" must be either "taxonomy" or "composition"')
    #
    #     # Color code dominant classes / labels
    #     # probability_values = np.transpose(np.array([y_pred[:, most_probable_classes[i]]
    #     #                                             for i in range(n_probable_classes)]))
    #
    #
    #     # Plot the coverage map using latitude and longitude from HB
    #     img = plt.imread(background_image)  # Background image
    #     fig, ax = plt.subplots(figsize=(30, 25))
    #     ax.imshow(img, cmap="gray", extent=[0, 360, -90, 90], alpha=1)
    #
    #     # Draw the predictions map
    #     values = y_pred[:, 20]
    #     im = ax.scatter(indices[:, 0], indices[:, 1], s=s, c=values,
    #                     marker=",", cmap='jet', vmin=vmin, vmax=vmax, alpha=alpha)
    #
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(yticks)
    #     plt.xticks(rotation=90., fontsize=font_size_axis - 4)
    #     plt.yticks(fontsize=font_size_axis - 4)
    #
    #     ax.grid()
    #
    #     ax.set_xlabel("Longitude (deg)", fontsize=font_size_axis)  # \N{DEGREE SIGN}
    #     ax.set_ylabel("Latitude (deg)", fontsize=font_size_axis)
    #     # ax.set_title(titles[i], fontsize=font_size_axis + 4)
    #
    #     ax.set_xlim(left=left, right=right)
    #     ax.set_ylim(bottom=bottom, top=top)
    #
    #     # divider = make_axes_locatable(ax)
    #     # cax = divider.append_axes(**cbar_kwargs)
    #     # cbar = plt.colorbar(im, cax=cax)
    #     # cax = divider.append_axes("bottom", size="10%", pad=1.35)
    #     cbar = plt.colorbar(im, orientation="horizontal")#, cax=cax)
    #
    #     cbar.set_ticks(cticks)
    #     cbar.set_ticklabels(ctickslabel)
    #     cbar.ax.tick_params(labelsize=font_size_axis - 4)
    #
    #     plt.draw()
    #     plt.tight_layout()
    #     plt.show()
    #
    # plot_surface_spectra(spectra)

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")

        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    ############################

    # # Vesta data from NASA Dawn VIR
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_VIS_1B_1_366641356_1.cub'))  # Vesta, HAMO, Marcia-Calpurnia-Minucia
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_366636556_1.cub'))  # Vesta, survey
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_367917915_1.cub'))  # Vesta, survey

    # # Same as above, from Ceres
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub'))  # Ceres
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_486828195_1.cub')) # another Ceres image, survey
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_486875439_1.cub')) # Ceres, survey, Kumitoga

    data_shape = 'actual'
    # data_shape = 'full_cube'
    if data_shape == 'VNIR_cube':
        constants.ASPECT_wavelengths = constants.ASPECT_wavelengths[:constants.ASPECT_SWIR_start_channel_index]

    # # Load endmember spectra, resample to ASPECT wavelengths, arrange into a list
    # didymos_wavelengths, didymos_reflectance = file_handling.load_Didymos_reflectance_spectrum(denoise=True)
    # didymos_reflectance, _, _ = simulation.ASPECT_resampling(didymos_reflectance, didymos_wavelengths)
    # # TODO S and Q type asteroid mean spectra as endmembers, for the measured mixtures?

    # # Load pyroxene and olivine spectra
    # pyroxene, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px100.csv'))
    # olivine, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))

    # Load pyroxene spectra
    # wls, endmember1 = file_handling.load_RELAB_spectrum('datasets/RELAB_pyroxenes/c1dl10.tab')  # "Clinopyroxene- Wo 10 En 63 Fs 27 (EFW13-4: 100% cpx, trCrist) 0 - 100 μm"
    # wls, endmember2 = file_handling.load_RELAB_spectrum('datasets/RELAB_pyroxenes/c1dl13.tab')  # "Clinopyroxene- Wo 8 En 46 Fs 46 (E40-1: 99.5% cpx, 0.5% glass, Crist) 0 - 100 μm"
    # wls1, endmember1 = file_handling.load_RELAB_spectrum(
    #     'datasets/RELAB_pyroxenes/c1dl28a.tab')  # "Orthopyroxene- En 25 Fs 75 (C)"
    # wls2, endmember2 = file_handling.load_RELAB_spectrum(
    #     'datasets/RELAB_pyroxenes/c1dl50a.tab')  # "Clinopyroxene- Wo 15 En 21 Fs 64 (B)"
    # wls3, endmember3 = file_handling.load_RELAB_spectrum(
    #     'datasets/RELAB_pyroxenes/c1dl10.tab')  # "Clinopyroxene- Wo 10 En 63 Fs 27"
    # endmember3, wls3 = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))

    # endmembers = [endmember1, endmember2, endmember3]
    # wl_vectors = [wls1, wls2, wls3]

    # Load endmembers: mean spectra of S and Q type asteroids from Bus-DeMeo taxonomy (http://smass.mit.edu/busdemeoclass.html)
    em_s, wls_s = file_handling.load_spectral_csv(Path('./datasets/S_type_mean_spectrum.csv'))
    em_q, wls_q = file_handling.load_spectral_csv(Path('./datasets/Q_type_mean_spectrum.csv'))
    # Compensate for phase reddening: remove linear continuum from both spectra my multiplying with a line
    x = np.linspace(0, 1, len(em_s))
    s_slope, s_offset = -0.19, 0.5
    q_slope, q_offset = -0.06, 0.5
    em_s = em_s * (x * s_slope + s_offset)
    em_q = em_q * (x * q_slope + q_offset)
    endmembers = [em_s / 5, em_q / 5]
    wl_vectors = [wls_s, wls_q]
    # plt.figure()
    # plt.plot(em_s)
    # plt.plot(em_q)
    # plt.show()

    def prepare_endmember(em, wls):
        # Interpolate the endmember spectra to ASPECT wavelengths
        em, new_wls, _ = simulation.ASPECT_resampling(em, wls)

        # Convert endmembers from reflectances to single-scattering albedos: mixing should be more linear in this space
        em = utils.reflectance2SSA(em)

        return em

    for i in range(len(endmembers)):
        endmembers[i] = prepare_endmember(endmembers[i], wl_vectors[i])

    # plt.figure()
    # for i in range(len(endmembers)):
    #     plt.plot(endmembers[i])
    # plt.show()

    # Simulated images of the Didymos system, by Penttilä et al.
    # training_data = nn.TrainingData(type='simulated_Didymos',
    #                                 filepath=Path('./datasets/Didymos_simulated/AIS simulated data v5/D1v5-10km-noiseless-40ms.mat'),
    #                                 data_shape=data_shape)
    # training_data = nn.TrainingData(type='simulated_Didymos_pyroxenes',
    #                                 filepath=Path(
    #                                     './datasets/Didymos_simulated/AIS simulated data v5/D1v5-10km-noiseless-40ms.mat'),
    #                                 data_shape=data_shape,
    #                                 endmembers=endmembers)
    training_data = nn.TrainingData(type='simulated_Didymos_pyroxenes',
                                    filepath=Path(
                                        './datasets/Didymos_simulated/AIS simulated data v5/D1v5-10km-noiseless-40ms.mat'),
                                    data_shape=data_shape,
                                    endmembers=endmembers,
                                    no_abundance_gt=True)
    # D1v5-3km-noiseless-40ms.mat asteroid fills the frame
    # D1D2v5-10km-noiseless-40ms.mat moon shadow on main

    # mineral_spectra = endmembers
    # # Load endmembers: mean spectra of S and Q type asteroids from Bus-DeMeo taxonomy (http://smass.mit.edu/busdemeoclass.html)
    # em_s, wls_s = file_handling.load_spectral_csv(Path('./datasets/S_type_mean_spectrum.csv'))
    # em_q, wls_q = file_handling.load_spectral_csv(Path('./datasets/Q_type_mean_spectrum.csv'))
    # endmembers = [em_s, em_q]
    # wl_vectors = [wls_s, wls_q]

    # # Load pyroxene and olivine spectra
    # pyroxene, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px100.csv'))
    # olivine, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))
    # endmembers = [pyroxene, olivine]
    # wl_vectors = [wls, wls]
    # wls1, endmember1 = file_handling.load_RELAB_spectrum(
    #     'datasets/RELAB_pyroxenes/c1dl28a.tab')  # "Orthopyroxene- En 25 Fs 75 (C)"
    # wls2, endmember2 = file_handling.load_RELAB_spectrum(
    #     'datasets/RELAB_pyroxenes/c1dl50a.tab')  # "Clinopyroxene- Wo 15 En 21 Fs 64 (B)"

    # Following Wo-En-Fs ratios approximately according to Korda et al.: https://doi.org/10.1051/0004-6361/202346290
    wls1, endmember1 = file_handling.load_RELAB_spectrum(
        'datasets/RELAB_pyroxenes/c1dl27a.tab')  # "Orthopyroxene- En 80 Fs 20 (C)"
    wls2, endmember2 = file_handling.load_RELAB_spectrum(
        'datasets/RELAB_pyroxenes/c1dl43a.tab')  # "Clinopyroxene- Wo 50 En 40 Fs 10"
    endmember3, wls3 = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))

    endmembers = [endmember1, endmember2, endmember3]
    wl_vectors = [wls1, wls2, wls3]

    # plt.figure()
    # # plt.plot(wls_s, em_s)
    # # plt.plot(wls_q, em_q)
    # plt.plot(pyroxene)
    # plt.plot(olivine)
    # plt.show()
    for i in range(len(endmembers)):
        # endmember = endmembers[i] / 10
        # endmember = endmember / np.max(endmember)
        # endmember = endmembers[i] / np.max(endmembers[i])
        endmembers[i] = prepare_endmember(endmembers[i], wl_vectors[i])

    # mixture = mineral_spectra[0]*0.5 + mineral_spectra[1]*0.5# + mineral_spectra[2]*0.33
    # # mixture = utils.SSA2reflectance(mixture)
    #
    # plt.figure()
    # plt.plot(mixture / np.max(mixture))
    # mixing_factors = np.linspace(0, 1, 5)
    # # endmembers[0] = utils.SSA2reflectance(endmembers[0])
    # # endmembers[1] = utils.SSA2reflectance(endmembers[1])
    # for factor in mixing_factors:
    #     unmixed = factor*endmembers[0] + (1-factor)*endmembers[1]
    #     # unmixed = utils.SSA2reflectance(unmixed)
    #     plt.plot(unmixed / np.max(unmixed))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(endmembers[0])
    # plt.plot(endmembers[1])
    # plt.show()


    bands = training_data.l

    endmember_count = len(endmembers)

    common_params = {'bands': bands,
                     'endmember_count': endmember_count,
                     'learning_rate': 0.000416}

    if data_shape == 'full_cube':
        band_count = bands
    else:
        band_count = constants.ASPECT_SWIR_start_channel_index

    enc_params = {'enc_layer_count': 7,
                  'band_count': band_count,
                  'endmember_count': common_params['endmember_count'],
                  'e_filter_count': 1024,
                  'e_kernel_size': 3,
                  'kernel_reduction': 0}

    dec_params = {'band_count': common_params['bands'],
                  'endmember_count': common_params['endmember_count'],
                  'd_kernel_size': 1}

    # Build and train a neural network
    nn.train(training_data,
             enc_params=enc_params,
             dec_params=dec_params,
             common_params=common_params,
             initial_endmembers=endmembers,
             epochs=8000,
             data_shape=data_shape,
             prints=True,
             plots=True,
             save_weights=False)

    # Build and train a neural network, loading saved weights at the start
    # common_params['learning_rate'] = common_params['learning_rate'] / 10  # Reduce learning rate
    # nn.train(training_data,
    #          enc_params=enc_params,
    #          dec_params=dec_params,
    #          common_params=common_params,
    #          initial_endmembers=endmembers,
    #          initial_enc_weights_path=Path('./enc_weights'),
    #          initial_dec_weights_path=Path('./dec_weights'),
    #          epochs=8000,
    #          data_shape=data_shape,
    #          prints=True,
    #          plots=True,
    #          save_weights=False)

    # ############### Hyperparameter optimization ##################
    # epochs = 3000
    #
    # # Optuna without ray
    # def objective(trial):
    #
    #     common_params = {'bands': bands,
    #                      'endmember_count': 3,  # For number of endmembers use an educated guess by a geologist
    #                      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)}
    #
    #     enc_params = {'enc_layer_count': trial.suggest_int('enc_layer_count', 1, 7),
    #                   'band_count': band_count,
    #                   'endmember_count': common_params['endmember_count'],
    #                   'e_filter_count': trial.suggest_int('e_filter_count', 8, 640),
    #                   'e_kernel_size': trial.suggest_int('e_kernel_size', 3, 9),
    #                   'kernel_reduction': trial.suggest_int('kernel_reduction', 0, 4)}
    #
    #     dec_params = {'band_count': common_params['bands'],
    #                   'd_endmember_count': common_params['endmember_count'],
    #                   'd_kernel_size': 1}
    #
    #     try:
    #         best_loss, best_test_loss, best_unmixing_test_loss, last_loss, last_test_loss, last_unmixing_test_loss = \
    #             nn.train(training_data,
    #             enc_params=enc_params,
    #             dec_params=dec_params,
    #             common_params=common_params,
    #             epochs=epochs,
    #             initial_endmembers=endmembers,
    #             data_shape=data_shape,
    #             plots=False,
    #             prints=True)
    #     except:
    #         logging.info('Something went wrong, terminating and trying next configuration')
    #         last_unmixing_test_loss = 1e5
    #     return last_unmixing_test_loss
    #
    #
    # optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    #
    # # Create a study object and optimize the objective function.
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=300)
    #
    # # Print summary of optimization run into log
    # logging.info(study.trials_dataframe(attrs=('value', 'params')).to_string())
