from pathlib import Path
import logging
import os
import math
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
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

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    data_shape = 'full_cube'
    if data_shape == 'VNIR_cube':
        constants.ASPECT_wavelengths = constants.ASPECT_wavelengths[:constants.ASPECT_SWIR_start_channel_index]

    # Simulated images of the Didymos system, by Penttilä et al.
    training_data = nn.TrainingData(type='simulated_Didymos',
                                    filepath=Path('./datasets/Didymos_simulated/AIS simulated data v5/D1v5-10km-noiseless-40ms.mat'),
                                    data_shape=data_shape)

    bands = training_data.l

    endmember_count = 2  # endmember_count = training_data.abundance_count

    common_params = {'bands': bands,
                     'endmember_count': endmember_count,
                     'learning_rate': 0.00025}

    if data_shape == 'full_cube':
        band_count = bands
    else:
        band_count = constants.ASPECT_SWIR_start_channel_index

    enc_params = {'enc_layer_count': 3,
                  'band_count': band_count,
                  'endmember_count': common_params['endmember_count'],
                  'e_filter_count': 128,
                  'e_kernel_size': 3,
                  'kernel_reduction': 0}

    dec_params = {'band_count': common_params['bands'],
                  'endmember_count': common_params['endmember_count'],
                  'd_kernel_size': 1}

    # Load endmember spectra, resample to ASPECT wavelengths, arrange into a list
    didymos_wavelengths, didymos_reflectance = file_handling.load_Didymos_reflectance_spectrum(denoise=True)
    didymos_reflectance, _, _ = simulation.ASPECT_resampling(didymos_reflectance, didymos_wavelengths)

    # Load pyroxene and olivine spectra
    pyroxene, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px100.csv'))
    olivine, wls = file_handling.load_spectral_csv(Path(constants.lab_mixtures_path, 'px0.csv'))

    # Interpolate the endmember spectra to ASPECT wavelengths
    pyroxene, new_wls, _ = simulation.ASPECT_resampling(pyroxene, wls)
    olivine, new_wls, _ = simulation.ASPECT_resampling(olivine, wls)

    # Flat spectra to adjust lightness and darkness of pixels
    # dark_em = np.ones(shape=didymos_reflectance.shape) * 0.001
    # light_em = np.ones(shape=didymos_reflectance.shape) * 0.50

    # Convert endmembers from reflectances to single-scattering albedos: mixing should be more linear in this space
    pyroxene = utils.reflectance2SSA(pyroxene)
    olivine = utils.reflectance2SSA(olivine)
    # dark_em = utils.reflectance2SSA(dark_em)

    endmembers = [pyroxene, olivine]#, dark_em]  # , light_em]

    # Build and train a neural network
    nn.train(training_data,
             enc_params=enc_params,
             dec_params=dec_params,
             common_params=common_params,
             initial_endmembers=endmembers,
             epochs=8000,
             data_shape=data_shape,
             prints=True,
             plots=True)

    # # ############### Hyperparameter optimization ##################
    # epochs = 3000
    #
    # # Optuna without ray
    # def objective(trial):
    #
    #     common_params = {'bands': bands,
    #                      'endmember_count': 3,  # For number of endmembers use an educated guess by a geologist
    #                      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)}
    #
    #     enc_params = {'enc_layer_count': trial.suggest_int('enc_layer_count', 1, 10),
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
