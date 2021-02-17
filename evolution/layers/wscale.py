import torch
from torch import nn
from util import tools
import numpy as np
import math


# https://github.com/nashory/pggan-pytorch/blob/master/custom_layers.py#L100
# https://github.com/facebookresearch/pytorch_GAN_zoo/blob/b75dee40918caabb4fe7ec561522717bf096a8cb/models/networks/custom_layers.py#L29

def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = np.prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self, module, equalized=True, lr_mul=1.0, init_bias_to_zero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """
        super().__init__()
        self.module = module
        self.equalized = equalized
        if init_bias_to_zero and self.module.bias is not None:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_mul
            self.scale = getLayerNormalizationFactor(self.module) * lr_mul

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.scale
        return x

    @property
    def bias(self):
        return self.module.bias

    @property
    def out_channels(self):
        return self.module.out_channels

    @property
    def in_channels(self):
        return self.module.in_channels

    @property
    def out_features(self):
        return self.module.out_features

    @property
    def in_features(self):
        return self.module.in_features

    @property
    def kernel_size(self):
        return self.module.kernel_size

    @property
    def weight(self):
        return self.module.weight


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming

        # self.scale = torch.sqrt(torch.mean(self.incoming.weight.data ** 2))
        # self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)

        fan = nn.init._calculate_correct_fan(self.incoming.weight, "fan_in")
        gain = nn.init.calculate_gain("relu")  # TODO: make calculate gain dependent of the activation
        self.scale = torch.tensor(gain / math.sqrt(fan))

        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        self.scale = tools.cuda(self.scale) if x.is_cuda else self.scale.cpu()
        x = x.mul(self.scale)
        if self.bias is not None:
            dims = [1, 1] if len(x.size()) == 4 else []
            x += self.bias.view(1, -1, *dims).expand_as(x)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % self.incoming.__class__.__name__
        return self.__class__.__name__ + param_str
