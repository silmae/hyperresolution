from pathlib import Path
import logging
import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from src import nn
from src import utils


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    logging.basicConfig(stream=sys.stdout, level='INFO')

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ############################

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")

        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    # TODO Docstrings everywhere

    training_data = nn.TrainingData(type='remote_sensing', filepath=Path('./datasets/TinyAPEX.mat'))
    # training_data = nn.TrainingData(type='rock', filepath=Path('./datasets/0065/A.mhdr.h5'))
    # training_data = nn.TrainingData(type='luigi', filepath=Path('./datasets/Luigi_stone/30klx_G2.nc'))

    def crop_and_mask(dataset, aspect_ratio=1):

        orig_w = dataset.w
        orig_h = dataset.h
        # Data dimension order is (l, w, h)

        def cut_horizontally(dataset, h):
            half_leftover = (orig_h - h) / 2
            start_i = math.floor(half_leftover)
            end_i = math.ceil(half_leftover)

            dataset.X = dataset.X[:, :, start_i:-end_i]
            dataset.Y = dataset.Y[:, :, start_i:-end_i]
            dataset.cube = dataset.cube[:, :, start_i:-end_i]

            dataset.h = h
            return dataset

        def cut_vertically(dataset, w):
            '''Make two vertical cuts removing data from left and right of center'''
            half_leftover = (orig_w - w) / 2
            start_i = math.floor(half_leftover)
            end_i = math.ceil(half_leftover)

            dataset.X = dataset.X[:, start_i:-end_i, :]
            dataset.Y = dataset.Y[:, start_i:-end_i, :]
            dataset.cube = dataset.cube[:, start_i:-end_i, :]

            dataset.w = w
            return dataset

        if orig_h > orig_w:  # if image is not horizontal or square, rotate 90 degrees
            plt.imshow(np.nanmean(dataset.X, 0) + 1)
            plt.show()
            dataset.X = np.rot90(dataset.X, axes=(1, 2))
            dataset.Y = np.rot90(dataset.Y, axes=(1, 2))
            dataset.cube = np.rot90(dataset.cube, axes=(1, 2))

            plt.imshow(np.nanmean(dataset.X, 0))  # The plot will appear in wrong orientation due to matplotlib expecting the indices in a certain order
            plt.show()

            dataset.w = orig_h
            dataset.h = orig_w

            orig_h = dataset.h
            orig_w = dataset.w

        if orig_w > int(orig_h * aspect_ratio):
            h = orig_h
            w = int(orig_h * aspect_ratio)
            dataset = cut_vertically(dataset, w)
        else:
            h = int(orig_w * (1/aspect_ratio))
            w = orig_w
            dataset = cut_horizontally(dataset, h)

            # half_leftover = (orig_w - w) / 2
            # start_i = math.floor(half_leftover)
            # end_i = math.ceil(half_leftover)
            #
            # dataset.X = dataset.X[:, start_i:-end_i, :]
            # dataset.Y = dataset.Y[:, start_i:-end_i, :]
            # dataset.cube = dataset.cube[:, start_i:-end_i, :]
            #
            # dataset.w = w

        # # REPLACED THIS WITH ROTATING THE DATA 90DEG IF THE IMAGE IS VERTICAL
        # elif orig_h >= orig_w:
        #     if orig_h > int(orig_w * aspect_ratio):
        #         h = int(orig_w * aspect_ratio)
        #         w = orig_w
        #         dataset = cut_horizontally(dataset, h, w)
        #     else:
        #         h = orig_h
        #         w = int(orig_h * (1/aspect_ratio))
        #         dataset = cut_vertically(dataset, w)

            # half_leftover = (orig_h - h) / 2
            # start_i = math.floor(half_leftover)
            # end_i = math.ceil(half_leftover)

            # dataset.X = dataset.X[:, :, start_i:-end_i]
            # dataset.Y = dataset.Y[:, :, start_i:-end_i]
            # dataset.cube = dataset.cube[:, :, start_i:-end_i]
            # dataset.h = h

        radius = int(min([dataset.h, dataset.w]) / 2)

        dataset.X = utils.apply_circular_mask(dataset.X, dataset.w, dataset.h, radius=radius)  # dataset.X * mask
        # dataset.Y = utils.apply_circular_mask(dataset.Y, dataset.w, dataset.h, radius=radius)
        dataset.cube = utils.apply_circular_mask(dataset.cube, dataset.w, dataset.h, radius=radius)

        half_point = int(dataset.l / 2)
        Y_short = dataset.cube[:half_point, :, :]
        Y_long = dataset.cube[half_point:, :, :]
        Y_long = np.mean(Y_long, axis=(1, 2))
        dataset.Y = [Y_short, Y_long]

        plt.imshow(np.nanmean(dataset.X, 0) + 1)  # matplotlib wants its dimensions in a different order, which makes the plot look like h and w are mixed
        plt.show()

        return dataset

    training_data = crop_and_mask(training_data, aspect_ratio=6.7/5.4)
    bands = training_data.l

    endmember_count = 5

    # endmember_count = training_data.abundance_count

    common_params = {'bands': bands,
                     'endmember_count': endmember_count}

    enc_params = {'enc_layer_count': 3,
                  'band_count': int(common_params['bands'] / 2),
                  'endmember_count': common_params['endmember_count'],
                  'e_filter_count': 128,
                  'kernel_size': 5,
                  'kernel_reduction': 2}

    dec_params = {'band_count': common_params['bands'],
                  'endmember_count': common_params['endmember_count']}

    # Build and train a neural network
    nn.train(training_data, enc_params=enc_params, dec_params=dec_params, common_params=common_params, epochs=10000)


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
