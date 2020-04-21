import util

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class SpairEncoder(nn.Module):
    """Encoder inspired by SPAIR.

    Args:
        embedding_dim: Dimensionality of latent state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        num_objects: Number of object slots.
    """

    def __init__(self, input_dims, feature_dim_space, num_objects,
                 obj_extractor_hidden_dim=32, obj_encoder_hidden_dim=512):
        super(SpairEncoder, self).__init__()

        self.H, self.W = feature_dim_space


        self.feature_extractor = FeatureExtractor()

    def forward(self, inp):

        return

