import torch.nn as nn
from .layer2d import Layer2D
import logging
from util import config
import numpy as np

logger = logging.getLogger(__name__)


class Downsample(Layer2D):

    MODES = ["nearest", "bilinear"]

    def __init__(self, scale=2, size=None):
        super().__init__(activation_type=None, normalize=False, use_dropout=False)
        self.scale = scale
        self.in_channels = None
        self.out_channels = None
        self.size = size
        self.min_shape = [3, 3]

    def changed(self):
        logger.debug(f"changed {self.input_shape[-1] * self.scale != self.output_shape[-1]} {self.input_shape[-1]} {self.scale} {self.output_shape[-1]}")
        return self.input_shape[-1] * self.scale != self.output_shape[-1]

    def _create_phenotype(self, input_shape):
        logger.debug(f"downsample {input_shape} {self.input_shape} {self.final_output_shape}")
        self.size = self.size or [self.input_shape[-2] * self.scale, self.input_shape[-1] * self.scale]
        self.size = [max(self.size[0], self.min_shape[0]), max(self.size[1], self.min_shape[1])]
        return nn.AvgPool2d(kernel_size=2)

    def _create_normalization(self):
        return None

    def spectral_norm(self, module):
        return module
