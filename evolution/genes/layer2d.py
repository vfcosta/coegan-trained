from evolution.layers.pixelwise_norm import PixelwiseNorm
import numpy as np
from torch import nn

from evolution.layers.skip_module import SkipModule
from util import config
from .layer import Layer


class Layer2D(Layer):
    """Represents an generic 2d layer in the evolving model."""

    def __init__(self, activation_params={}, activation_type="random", normalize=config.gan.normalization,
                 use_dropout=False, skip_conn=False):
        super().__init__(activation_params=activation_params, activation_type=activation_type, normalize=normalize,
                         use_dropout=use_dropout)
        self.skip_module = None
        self.skip_conn = skip_conn
        if skip_conn == "random":
            self.skip_conn = bool(np.random.choice([False, True]))

    def is_linear(self):
        return False

    def _create_normalization(self):
        if self.normalize == "pixelwise":
            return PixelwiseNorm()
        elif self.normalize == "batch":
            return nn.BatchNorm2d(self.out_channels)
        return None

    def create_skip_module(self, module):
        if self.skip_conn:
            skip_module = SkipModule(module)
            if self.skip_module is not None:
                skip_module.conv = self.skip_module.conv
            self.skip_module = skip_module

    def create_module(self):
        if self.skip_module is not None:
            return self.skip_module
        return super().create_module()
