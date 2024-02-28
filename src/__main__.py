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

    # # Vesta data, the primary data
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_VIS_1B_1_366641356_1.cub'))  # Vesta, HAMO, Marcia-Calpurnia-Minucia
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_366636556_1.cub'))  # Vesta, survey
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_367917915_1.cub'))  # Vesta, survey

    # # ISIS images of Ceres: test generalizability with some of these after optimizing network with Vesta data
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub'))  # Ceres
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_486828195_1.cub')) # another Ceres image, survey
    # training_data = nn.TrainingData(type='DAWN_ISIS', filepath=Path('./datasets/DAWN/ISIS/m-VIR_IR_1B_1_486875439_1.cub')) # Ceres, survey, Kumitoga

    # # Simulated images of the Didymos system, by Penttilä et al.
    training_data = nn.TrainingData(type='simulated_Didymos',
                                    filepath=Path('./datasets/Didymos_simulated/AIS simulated data v5/D1v5_noiseless_10km'))

    bands = training_data.l

    # Optimization result for Vesta data including VIS, with 12 endmembers: this architecture works great for
    # reconstruction of Vesta images, but the endmembers are not realistic
    # parameters: {'learning_rate': 0.00017412188150195052, 'enc_layer_count': 1, 'e_filter_count': 501,
    #              'e_kernel_size': 6, 'kernel_reduction': 3, 'd_kernel_size': 9}

    # Optimization result for 5 endmembers
    # learning_rate
    # ': 0.0004503804275361948, '
    # enc_layer_count
    # ': 4, '
    # e_filter_count
    # ': 161, '
    # e_kernel_size
    # ': 6, '
    # kernel_reduction
    # ': 0, '
    # d_kernel_size
    # ': 5

    endmember_count = 3   # endmember_count = training_data.abundance_count

    common_params = {'bands': bands,
                     'endmember_count': endmember_count,
                     'learning_rate': 0.000450}

    enc_params = {'enc_layer_count': 5,
                  'band_count': constants.ASPECT_SWIR_start_channel_index,
                  'endmember_count': common_params['endmember_count'],
                  'e_filter_count': 256,
                  'e_kernel_size': 6,
                  'kernel_reduction': 0}

    dec_params = {'band_count': common_params['bands'],
                  'endmember_count': common_params['endmember_count'],
                  'd_kernel_size': 1}

    # Load endmember spectra, resample to ASPECT wavelengths, arrange into a list
    didymos_wavelengths, didymos_reflectance = file_handling.load_Didymos_reflectance_spectrum(denoise=True)
    didymos_reflectance, _, _ = utils.ASPECT_resampling(didymos_reflectance, didymos_wavelengths)

    # Flat spectra to adjust lightness and darkness of pixels
    dark_em = np.ones(shape=didymos_reflectance.shape) * 0.01
    # light_em = np.ones(shape=didymos_reflectance.shape) * 0.50

    endmembers = [didymos_reflectance, dark_em] #, light_em]

    # Build and train a neural network
    nn.train(training_data,
             enc_params=enc_params,
             dec_params=dec_params,
             common_params=common_params,
             initial_endmembers=endmembers,
             epochs=5000,
             prints=True,
             plots=True)

    # ############### Hyperparameter optimization ##################
    # epochs = 6000
    #
    # # Optuna without ray
    # def objective(trial):
    #
    #     common_params = {'bands': bands,
    #                      'endmember_count': 5,  # For number of endmembers use an educated guess by a geologist and then add one or two to that
    #                      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)}
    #
    #     enc_params = {'enc_layer_count': trial.suggest_int('enc_layer_count', 1, 8),
    #                   'band_count': constants.ASPECT_SWIR_start_channel_index,
    #                   'endmember_count': common_params['endmember_count'],
    #                   'e_filter_count': trial.suggest_int('e_filter_count', 8, 640),
    #                   'e_kernel_size': trial.suggest_int('e_kernel_size', 3, 9),
    #                   'kernel_reduction': trial.suggest_int('kernel_reduction', 0, 4)}
    #
    #     dec_params = {'band_count': common_params['bands'],
    #                   'd_endmember_count': common_params['endmember_count'],
    #                   'd_kernel_size': trial.suggest_int('d_kernel_size', 1, 9)}
    #
    #     try:
    #         best_loss, best_test_loss, last_loss, last_test_loss = nn.train(training_data,
    #                                                                         enc_params=enc_params,
    #                                                                         dec_params=dec_params,
    #                                                                         common_params=common_params,
    #                                                                         epochs=epochs,
    #                                                                         plots=False,
    #                                                                         prints=True)
    #     except:
    #         logging.info('Something went wrong, terminating and trying next configuration')
    #         last_test_loss = 100
    #     return last_test_loss
    #
    #
    # optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    #
    # # Create a study object and optimize the objective function.
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=200)
    #
    # # Print summary of optimization run into log
    # logging.info(study.trials_dataframe(attrs=('value', 'params')).to_string())
