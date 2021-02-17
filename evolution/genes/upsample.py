import torch.nn as nn
from .layer2d import Layer2D
import logging
from util import config
import numpy as np

logger = logging.getLogger(__name__)


class Upsample(Layer2D):

    MODES = ["nearest", "bilinear"]

    def __init__(self, scale=2, size=None, mode=None):
        super().__init__(activation_type=None, normalize=False, use_dropout=False)
        self.scale = scale
        self.in_channels = None
        self.out_channels = None
        self.size = size
        self.mode = mode

    def changed(self):
        logger.debug(f"changed {self.input_shape[-1] * self.scale != self.output_shape[-1]} {self.input_shape[-1]} {self.scale} {self.output_shape[-1]}")
        return self.input_shape[-1] * self.scale != self.output_shape[-1]

    def _create_phenotype(self, input_shape):
        logger.debug(f"UPSAMPLE {input_shape} {self.input_shape} {self.final_output_shape}")
        if self.mode is None:
            self.mode = np.random.choice(self.MODES)
        self.size = self.size or [self.input_shape[-2] * self.scale, self.input_shape[-1] * self.scale]
        if config.layer.upsample.limit_output_size:
            limit = self.final_output_shape[-2:] if not isinstance(self.final_output_shape, int) else self.initial_input_shape[-2:]
            self.size = [min(self.size[0], limit[0]), min(self.size[1], limit[1])]
        return nn.Upsample(size=self.size, mode=self.mode)

    def _create_normalization(self):
        return None

    def spectral_norm(self, module):
        return module
