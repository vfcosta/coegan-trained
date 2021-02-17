import logging
from collections.abc import Iterable

import numpy as np
import torch.nn as nn

from util import config
from util import tools
from .layer2d import Layer2D

logger = logging.getLogger(__name__)


class Conv2d(Layer2D):
    """Represents a convolution layer."""

    def __init__(self, out_channels=None, kernel_size=None, stride=config.layer.conv2d.stride,
                 activation_type="random", padding=1, activation_params={}, normalize=config.gan.normalization,
                 use_dropout=None, skip_conn=config.layer.conv2d.enable_skip_conn):
        super().__init__(activation_type=activation_type, activation_params=activation_params, normalize=normalize,
                         use_dropout=use_dropout or config.gan.dropout, skip_conn=skip_conn)
        kernel_size = kernel_size or config.layer.conv2d.kernel_size
        if isinstance(kernel_size, Iterable):
            kernel_size = int(np.random.choice(kernel_size))
        if stride == "random":
            self.stride = np.random.randint(1, 3)
        else:
            self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = None
        self.out_channels = out_channels
        self.padding = padding

    def setup(self):
        super().setup()
        self.in_channels = self.input_shape[1] if self.input_shape and len(self.input_shape) > 2 else 1
        # limit the kernel size to the input shape
        if self.input_shape is not None:
            self.kernel_size = min(self.kernel_size, self.input_shape[2])
        if self.out_channels is None:
            if self.first_conv() or config.layer.conv2d.random_out_channels:
                self.out_channels = 2 ** np.random.randint(config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power+1)
            else:
                self.out_channels = min(2 * self.in_channels, 2**(config.layer.conv2d.max_channels_power+1))
        if config.layer.conv2d.force_double:
            self.out_channels = max(self.out_channels, self.in_channels*2)

    def changed(self):
        module_kernel_size = self.module.kernel_size
        if not isinstance(self.module.kernel_size, int):
            module_kernel_size = module_kernel_size[0]
        return self.module.out_channels != self.out_channels or self.module.in_channels != self.in_channels or \
               self.kernel_size != module_kernel_size

    def _create_phenotype(self, input_shape):
        super()._create_phenotype(input_shape)
        module = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=self.bias)

        if self.module is None or not config.layer.resize_weights:
            if self.has_wscale():
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
        elif self.kernel_size == self.module.kernel_size[0]:  # resize only when the kernel size was not changed
            module, self.adjusted = tools.resize_conv(self.module, module)
        self.create_skip_module(module)
        return module

    def _create_pad(self):
        out_shape = np.ceil(np.array(self.input_shape[-2:]) / self.stride)
        pad_w, pad_h = tools.conv2d_get_padding(self.input_shape[-2:], tuple(out_shape), self.kernel_size, self.stride)
        self.padding = [*pad_w, *pad_h]
        return nn.ZeroPad2d(self.padding)

    def apply_mutation(self):
        self.out_channels = None
        super().apply_mutation()

    def first_conv(self):
        return not self.previous_layer or not isinstance(self.previous_layer, Conv2d)

    def __repr__(self):
        return self.__class__.__name__ + f"(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
                                         f"kernel_size={self.kernel_size}, stride={self.stride}, " \
                                         f"padding={self.padding}, " \
                                         f"normalize={self.normalize}, " \
                                         f"activation_type={self.activation_type})"
