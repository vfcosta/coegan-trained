from util import config
from util import tools
import torch.nn as nn
from torch.nn import functional as F


class SkipModule(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.skip = nn.Identity()
        self.conv = None

    def create_conv(self, input_shape, out_shape):
        if self.conv is None and input_shape[1] == out_shape[1]:
            return
        if self.conv is not None and self.conv.weight.shape == out_shape:
            return
        conv = nn.Conv2d(input_shape[1], out_shape[1], kernel_size=1, bias=False)
        if self.conv is not None and config.layer.resize_weights:
            conv, _ = tools.resize_conv(self.conv, conv)
        self.conv = conv

    def forward(self, input_data):
        x = self.skip(input_data)
        out = self.module(x)
        if x.shape[2:] != out.shape[2:]:
            x = F.interpolate(x, out.shape[2:])
        self.create_conv(x.shape, out.shape)
        if self.conv is not None:
            x = self.conv(x)
        return x + out
