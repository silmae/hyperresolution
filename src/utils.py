""""
This file contains miscellaneous utility functions
"""
import numpy as np
from torch import Tensor
import torch


def apply_circular_mask(data: np.ndarray or torch.Tensor, h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray or torch.Tensor:
    """
    Create a circular mask and apply it to an image cube. Works for ndarrays and torch tensors.
    Mask creation from https://stackoverflow.com/a/44874588
    :param data:
        Data cube to be masked
    :param h:
        Height of image
    :param w:
        Width of image
    :param center:
        Center of the mask; if not given, will use center of the data
    :param radius:
        Radius of the mask, in pixels; if not given, will use the largest possible radius which will fit the whole circle
    :return:
        Masked datacube
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    if type(data) == Tensor:
        device = data.device  # Check where the data is: GPU or CPU
        mask = Tensor(mask).to(device)  # Convert mask to tensor and move it to same device as data
        masked_data = data * mask
        masked_data = masked_data + Tensor(abs(mask - 1))  # Convert masked values to ones instead of zeros to avoid problems with backprop
    else:
        mask = abs((mask * 1))  # the above returns a mask of booleans, this converts it to int (somehow)
        masked_data = data * mask

    return masked_data
