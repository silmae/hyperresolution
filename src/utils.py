""""
This file contains miscellaneous utility functions
"""
import numpy as np
from torch import Tensor
import torch
from torch.masked import masked_tensor


def apply_circular_mask(data, h, w, center=None, radius=None):
    """From https://stackoverflow.com/a/44874588"""

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    if type(data) == Tensor:
        # length = data.shape[0]
        # mask = np.broadcast_to(mask, (length,) + mask.shape)
        # mask = Tensor(mask).bool()
        # masked_data = masked_tensor(data, mask)
        masked_data = data * Tensor(mask)
        masked_data = masked_data + Tensor(abs(mask - 1))
    else:
        mask = abs((mask * 1))  # the above returns a mask of booleans, this converts it to int (somehow)
        masked_data = data * mask
    return masked_data
