from pathlib import Path
import logging
import os
import math
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import torch
from planetaryimage import CubeFile

import optuna

import spectral

import cv2 as cv

from src import nn
from src import utils


if __name__ == '__main__':
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

    # TODO Vinkkejä mistä jatkaa tätä projektia:
    #  ISIS kuution avaaminen funktiossa ja tuon datan syöttö verkolle toisella funktiolla
    #  Treenaa verkkoa tuolla datalla ja katso kuinka käy
    #  Kanavien ja aallonpituusalueen muokkaus vastaamaan ASPECTia

    # Cross test for two images
    ############# SANDBOX ###############

    # testin = utils.open_DAWN_VIR_IR_PDS3_as_ENVI('./datasets/DAWN/VIR_IR_1B_1_368033917_3.LBL')

    # TODO Make a function for opening these ISIS cube files. Will most likely need to tailor
    #  the angle for rotation and the crop indices for each image.
    #  Also write a function for feeding that data to the network.

    # isisimage = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_589173531_1.cub")  # This is the largest image of these, Ceres
    # isisimage = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_487367451_1.cub")  # A bit smaller image, Ceres
    # isisimage = CubeFile.open("./datasets/DAWN/ISIS/f-VIR_IR_1B_1_483703316_1.cub")  # This is taken from afar and has half of Ceres in view
    isisimage_IR = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_IR_1B_1_494387713_1.cub")
    isisimage_VIS = CubeFile.open("./datasets/DAWN/ISIS/m-VIR_VIS_1B_1_494387713_1.cub")

    # bands = isisimage.label['IsisCube']['BandBin']
    # wavelengths = bands['Center']
    # FWHMs = bands['Width']

    def rot_and_crop_Dawn_VIR_ISIS(isisimage, rot_deg, crop_indices_x=(130, 580), crop_indices_y=(190, 460)):

        data = isisimage.data
        data = np.clip(data, 0, 1000)  # clip to get rid of the absurd masking values
        data = ndimage.rotate(data, rot_deg, mode='constant')  # rotate to get the interesting area horizontal
        data = data[:, crop_indices_y[0]:crop_indices_y[1], crop_indices_x[0]:crop_indices_x[1]]  # crop the masking values away

        showable = data[200, :, :]  # pick one channel for plotting
        showable = showable / np.max(showable)  # normalization
        showable = showable * 256  # convert to 8-bit integer to make compatible for edge detection
        showable = showable.astype(np.uint8)
        edges = cv.Canny(showable, 60, 40)  # Edge detection

        return data, showable, edges

    data_IR, showable_IR, edges_IR = rot_and_crop_Dawn_VIR_ISIS(isisimage_IR, -19, (130, 580), (190, 460))
    data_VIS, showable_VIS, edges_VIS = rot_and_crop_Dawn_VIR_ISIS(isisimage_VIS, -19, (132, 582), (193, 463))

    # showable_VIS = isisimage_VIS.data[200, :, :]
    # showable_VIS = np.clip(showable_VIS, 0, 1000)
    # showable_VIS = ndimage.rotate(showable_VIS, -19, mode='constant')
    # showable_VIS = showable_VIS[193:463, 132:582]  # offset in y direction
    # showable_VIS = showable_VIS / np.max(showable_VIS)
    # showable_VIS = showable_VIS * 256
    # showable_VIS = showable_VIS.astype(np.uint8)
    # edges_VIS = cv.Canny(showable_VIS, 60, 40)

    showable = np.zeros(shape=(edges_VIS.shape[0], edges_VIS.shape[1], 3))  # Plot VIS edges in red channel, IR in green
    showable[:, :, 0] = edges_VIS
    showable[:, :, 1] = edges_IR

    plt.imshow(showable, vmin=0)
    plt.show()

    print('stop')

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    ############################

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")

        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    # training_data = nn.TrainingData(type='remote_sensing', filepath=Path('./datasets/TinyAPEX.mat'))
    # training_data = nn.TrainingData(type='rock', filepath=Path('./datasets/0065/A.mhdr.h5'))
    # training_data = nn.TrainingData(type='luigi', filepath=Path('./datasets/Luigi_stone/30klx_G2.nc'))
    # training_data = nn.TrainingData(type='DAWN', filepath=Path('./datasets/DAWN/PDS3/VIR_IR_1B_1_520299107_1.LBL'))

    training_data = nn.TrainingData(type='DAWN', filepath=Path('./datasets/DAWN/PDS3/VIR_VIS_1B_1_487349955_1.LBL'))
    # training_data_IR = nn.TrainingData(type='DAWN', filepath=Path('./datasets/DAWN/PDS3/VIR_IR_1B_1_487349955_1.LBL'))

    print('test')
    def ASPECT_resampling(cube, wavelengths, FWHMs):
        ASPECT_wavelengths = np.asarray(np.linspace(start=0.850, stop=2.500, num=60))
        ASPECT_FWHMs = np.zeros(shape=ASPECT_wavelengths.shape) + 0.040

        if min(ASPECT_wavelengths) <= min(wavelengths):
            minimum = min(wavelengths)
        else:
            minimum = min(ASPECT_wavelengths)

        if max(ASPECT_wavelengths) >= max(wavelengths):
            maximum = max(wavelengths)
        else:
            maximum = max(ASPECT_wavelengths)

        resample = spectral.BandResampler(wavelengths, ASPECT_wavelengths, FWHMs, ASPECT_FWHMs)

        cube_resampled = np.zeros(shape=(len(ASPECT_wavelengths), cube.shape[1], cube.shape[2]))
        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                cube_resampled[:, i, j] = resample(cube[:, i, j])

        return cube_resampled, ASPECT_wavelengths, ASPECT_FWHMs


    training_data = ASPECT_resampling(training_data.cube, training_data.wavelengths, training_data.FWHMs)

    # TODO Make this function work, but later
    # def ASPECTify(cube, wavelengths, VIS=False, NIR1=True, NIR2=True, SWIR=True):
    #     """Take a spectral image and make it look like data from Milani's ASPECT"""
    #     if VIS:
    #         print('Sorry, the function can not currently work with the VIS portion of ASPECT')
    #         print('Stopping execution')
    #         exit(1)
    #
    #     # ASPECT wavelength vectors: change these values later, if the wavelengths change!
    #     ASPECT_VIS_wavelengths = np.linspace(start=0.650, stop=0.950, num=14)
    #     ASPECT_NIR1_wavelengths = np.linspace(start=0.850, stop=0.1250, num=14)
    #     ASPECT_NIR2_wavelengths = np.linspace(start=1.200, stop=0.1600, num=14)
    #     ASPECT_SWIR_wavelengths = np.linspace(start=1.650, stop=2.500, num=30)
    #     # ASPECT FOVs in degrees
    #     ASPECT_VIS_FOV = 10  # 10x10 deg square
    #     ASPECT_NIR_FOV_w = 6.7  # width
    #     ASPECT_NIR_FOV_h = 5.4  # height
    #     ASPECT_SWIR_FOV = 5.85  # circular
    #
    #     if (NIR1 or NIR2) and SWIR:
    #         # The largest is the SWIR circular FOV, and so it is the limiting factor
    #         # Cut the largest possible rectangle where one side is circular FOV, other is width of NIR. Then divide
    #         # along wavelength into NIR and SWIR, mask SWIR into circle and take mean spectrum, and cut NIR to size.
    #         return None
    #
    #
    # training_data_ASPECT = ASPECTify(training_data.cube, training_data.wavelengths)

    # Crop data and apply a circular mask: aspect ratio from ASPECT NIR module  # TODO make radius comparable with aspect ratio
    training_data = utils.crop_and_mask(training_data, aspect_ratio=6.7/5.4)#, radius=100)
    bands = training_data.l

    # endmember_count = 12    # endmember_count = training_data.abundance_count
    #
    # common_params = {'bands': bands,
    #                  'endmember_count': endmember_count,
    #                  'learning_rate': 0.009362}
    #
    # enc_params = {'enc_layer_count': 5,
    #               'band_count': int(common_params['bands'] / 2),
    #               'endmember_count': common_params['endmember_count'],
    #               'e_filter_count': 392,
    #               'e_kernel_size': 10,
    #               'kernel_reduction': 0}
    #
    # dec_params = {'band_count': common_params['bands'],
    #               'endmember_count': common_params['endmember_count'],
    #               'd_kernel_size': 1}
    #
    # # # Build and train a neural network
    # nn.train(training_data, enc_params=enc_params, dec_params=dec_params, common_params=common_params, epochs=10000, prints=True, plots=True)

    epochs = 5000

    # Optuna without ray
    def objective(trial):

        common_params = {'bands': bands,
                         'endmember_count': trial.suggest_int('endmember_count', 3, 12),  # TODO optimize this separately?
                         'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e1, log=True)}

        enc_params = {'enc_layer_count': trial.suggest_int('enc_layer_count', 1, 7),
                      'band_count': int(common_params['bands'] / 2),
                      'endmember_count': common_params['endmember_count'],
                      'e_filter_count': trial.suggest_int('e_filter_count', 8, 512),
                      'e_kernel_size': trial.suggest_int('e_kernel_size', 3, 15),
                      'kernel_reduction': trial.suggest_int('kernel_reduction', 0, 4)}

        dec_params = {'band_count': common_params['bands'],
                      'd_endmember_count': common_params['endmember_count'],
                      'd_kernel_size': trial.suggest_int('d_kernel_size', 1, 15)}

        try:
            best_loss, best_test_loss = nn.train(training_data,
                                                 enc_params=enc_params,
                                                 dec_params=dec_params,
                                                 common_params=common_params,
                                                 epochs=epochs,
                                                 plots=False,
                                                 prints=True)
        except:
            logging.info('Something went wrong, terminating and trying next configuration')
            best_test_loss = 100
        return best_test_loss

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=400)

    # Print summary of optimization run into log
    logging.info(study.trials_dataframe(attrs=('value', 'params')).to_string())




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
