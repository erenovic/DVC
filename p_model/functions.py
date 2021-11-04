# Functions used in RAE, RPM and BitEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda")


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )
    

def init_xavier(module):
    """Xavier initialize the Conv2d weights"""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)