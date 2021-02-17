import torch.nn as nn
from .conv2d import Conv2d
from util import config
import numpy as np
from util import tools
import logging

logger = logging.getLogger(__name__)


class Deconv2dUpsample(Conv2d):
    """Represents a convolution layer."""
    MODES = ["nearest"]

    def __init__(self, out_channels=None, kernel_size=None, stride=1, activation_type="random", activation_params={},
                 normalize=config.gan.normalization, size=None, skip_conn=config.layer.deconv2d.enable_skip_conn,
                 mode=None):
        super().__init__(out_channels, kernel_size, stride, activation_type=activation_type,
                         activation_params=activation_params, normalize=normalize, skip_conn=skip_conn)
        self._current_input_shape = None
        self.size = size
        self.padding = 1
        self.scale_factor = 2
        self.mode = mode

    def _create_pad(self):
        return None

    def _create_upsample_pad(self):
        out_shape = self.size
        pad_w, pad_h = tools.conv2d_get_padding(tuple(out_shape), tuple(out_shape), self.kernel_size, self.stride)
        self.padding = [*pad_w, *pad_h]
        return nn.ZeroPad2d(self.padding)

    def changed(self):
        module_kernel_size = self.module[-1].kernel_size
        if not isinstance(self.module[-1].kernel_size, int):
            module_kernel_size = module_kernel_size[0]
        return self.module[-1].out_channels != self.out_channels or self.module[-1].in_channels != self.in_channels or \
               self.kernel_size != module_kernel_size

    def _create_phenotype(self, input_shape):
        if self.mode is None:
            self.mode = np.random.choice(self.MODES)
        self.size = self.size or [input_shape[-2] * self.scale_factor, input_shape[-1] * self.scale_factor]
        if config.layer.upsample.limit_output_size:
            limit = self.final_output_shape[-2:] if not isinstance(self.final_output_shape, int) else self.initial_input_shape[-2:]
            self.size = [min(self.size[0], limit[0]), min(self.size[1], limit[1])]
        upsample = nn.Upsample(size=self.size, mode=self.mode)
        conv2d = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=self.bias)
        if self.module is not None and config.layer.resize_weights:
            conv2d, self.adjusted = tools.resize_conv(self.module[-1], conv2d)
        module = nn.Sequential(upsample, self._create_upsample_pad(), conv2d)
        self.create_skip_module(module)
        return module

    def spectral_norm(self, module):
        nn.utils.spectral_norm(module[-1])
        return module

    def remove_spectral_norm(self):
        self._remove_spectral_norm(self.module[-1])

    def setup(self):
        calc_out_channels = self.out_channels is None
        super().setup()
        if calc_out_channels:
            if config.layer.conv2d.random_out_channels:
                self.out_channels = 2 ** np.random.randint(config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power+1)
            else:
                self.out_channels = max(1, self.in_channels//2)
        if config.layer.conv2d.force_double:
            self.out_channels = min(self.out_channels, self.in_channels//2)

    def first_deconv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Deconv2dUpsample)
