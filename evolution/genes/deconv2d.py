import torch.nn as nn
from .conv2d import Conv2d
from ..layers.convupsample import ConvUpsample
import numpy as np
from util import config
import math
from util import tools
import logging

logger = logging.getLogger(__name__)


class Deconv2d(Conv2d):
    """Represents a convolution layer."""

    def __init__(self, out_channels=None, kernel_size=None, stride=2, activation_type="random", activation_params={},
                 normalize=config.gan.normalization, skip_conn=config.layer.deconv2d.enable_skip_conn, size=None):
        super().__init__(out_channels, kernel_size, stride, activation_type=activation_type,
                         activation_params=activation_params, normalize=normalize, skip_conn=skip_conn)
        self._current_input_shape = None
        self.output_padding = 1

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

        if self.is_last_layer() and self.final_output_shape is not None:
            self.out_channels = self.final_output_shape[0]
        # if self.out_channels > self.in_channels:
        #     self.stride = 1
        # else:
        self.stride = 2

    def changed(self):
        return super().changed() or self._current_input_shape != self.input_shape

    def _create_phenotype(self, input_shape):
        # adjust output size
        output_size = None
        padding = self.kernel_size // 2
        logger.debug(f"CREATE DECONV2d {self.stride} {self.final_output_shape} {input_shape}")
        if not isinstance(self.final_output_shape, int):
            logger.debug("adjust deconv")
            in_dimension = np.array(input_shape[2:])
            if self.input_shape[-1] >= self.final_output_shape[-1]:
                self.stride = 1
                output_size = np.array(self.final_output_shape[1:])
                logger.debug(f"reset stride to 1 {self.kernel_size} {self.padding} {output_size}")
            else:
                self.stride = 2
                out_dimension = np.array(self.final_output_shape[1:])
                div = np.round(out_dimension/(in_dimension*2)).astype(np.int32)
                div[div == 0] = 1
                output_size = out_dimension//div
                logger.debug(f"output_size {output_size} {div} {out_dimension} {in_dimension} {self.kernel_size}")
                logger.debug(f"stride {self.stride} {padding}")
            padding = max(0, int(math.ceil(((in_dimension[0] - 1) * self.stride - output_size[0] + self.kernel_size) / 2)))
            if output_size is not None:
                output_size = [x.item() for x in output_size]
        self._current_input_shape = self.input_shape
        self.output_padding = self.kernel_size // 2
        layer = ConvUpsample(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, output_size,
            padding=padding, bias=self.bias)

        if self.module is None or not config.layer.resize_weights:
            if self.has_wscale():
                nn.init.normal_(layer.module.weight)
                if layer.module.bias is not None:
                    nn.init.zeros_(layer.module.bias)
            else:
                nn.init.xavier_uniform_(layer.module.weight)
        else:  # resize and copy weights
            layer.module, self.adjusted = tools.resize_conv(self.module.module, layer.module)
        self.create_skip_module(layer)
        return layer

    def spectral_norm(self, module):
        nn.utils.spectral_norm(module.module)
        return module

    def remove_spectral_norm(self):
        self._remove_spectral_norm(self.module.module)

    def first_deconv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Deconv2d)

    def _create_pad(self):
        # return None
        if self.stride > 1:
            return None
        out_shape = np.ceil(np.array(self.input_shape[-2:]) * self.stride)
        logger.debug(f"{self.input_shape[-2:]} {tuple(out_shape)} {self.stride} {self.kernel_size}")
        pad_w, pad_h = tools.convtransp2d_get_padding(self.input_shape[-2:], tuple(out_shape), self.kernel_size, self.stride, out_pad=self.output_padding)
        logger.debug(f"DECONV {pad_w} {pad_h} {out_shape} {self.padding}")
        self.padding = [*pad_w, *pad_h]
        logger.debug(tools.convtransp2d_output_shape(self.input_shape[-2:], self.kernel_size, self.stride, (pad_h, pad_w), out_pad=self.output_padding))
        return nn.ZeroPad2d(self.padding)
