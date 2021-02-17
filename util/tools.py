import torch.nn as nn
import torch
import os
import numpy as np
from torch.nn import functional as F
from functools import lru_cache
import math
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(discriminator, generator, epoch, data_folder):
    out_dir = '%s/models' % data_folder
    os.makedirs(out_dir)
    torch.save(discriminator.state_dict(), '%s/D_epoch_%d' % (out_dir, epoch))
    torch.save(generator.state_dict(), '%s/G_epoch_%d' % (out_dir, epoch))


def round_array(array, max_sum, invert=False):
    if invert and len(array) > 1:
        # invert sizes as low fitness values is better than high values
        array = (max_sum - np.array(array)) / (len(array) - 1)

    array = np.array(array)
    array = np.clip(array, 0, max_sum)
    rounded = np.floor(array)
    diff = int(max_sum) - int(np.sum(rounded))
    if diff > 0:
        for i in range(diff):
            max_index = (array - rounded).argmax()
            if len(array) == 2:
                max_index = array.argmin()
            rounded[max_index] += 1
    return rounded


def coord_1d_2d(x, rows):
    return x // rows, x % rows


def coord_2d_1d(r, c, rows):
    return r * rows + c


def get_neighbors(center, rows, cols):
    r, c = coord_1d_2d(center, rows)
    top = coord_2d_1d((r - 1) % rows, c, rows)
    bottom = coord_2d_1d((r + 1) % rows, c, rows)
    right = coord_2d_1d(r, (c + 1) % cols, rows)
    left = coord_2d_1d(r, (c - 1) % cols, rows)
    return [center, top, bottom, right, left]


@lru_cache(maxsize=10)
def _permutations(len_a1, len_a2):
    pairs = []
    for start_j in range(len_a2):
        j = 0
        for i in range(len_a1):
            pairs.append((i, (j + start_j) % len_a2))
            j = (j + 1) % len_a2
    return pairs


def permutations(a1, a2, random=False):
    len_a1, len_a2 = len(a1), len(a2)
    if random:
        pairs = np.array(np.meshgrid(range(len_a1), range(len_a2))).T.reshape(-1, 2)
        np.random.shuffle(pairs)
        return pairs
    return _permutations(len_a1, len_a2)


def is_cuda_available(condition=True):
    return condition and torch.cuda.is_available()


def device_name(condition=True):
    return "cuda" if is_cuda_available(condition) else "cpu"


def cuda(variable, condition=True):
    return variable.cuda() if is_cuda_available(condition) else variable


def resize_channels(x, size):
    x = x.permute(2, 3, 0, 1)
    out = F.interpolate(x, size, mode='bilinear')
    return out.permute(2, 3, 0, 1)


# based on https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
def resize_activations_avg(v, so):
    """
    Resize activation tensor 'v' of shape 'si' to match shape 'so'.
    :param v:
    :param so:
    :return:
    """
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so)# and si[0] == so[0]

    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]
    if si[0] > so[0]:
        v = v[:so[0], :]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = (si[2] // so[2], si[3] // so[3])
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

    # Extend spatial axes. Below is a wrong implementation
    # shape = [1, 1]
    # for i in range(2, len(si)):
    #     if si[i] < so[i]:
    #         assert so[i] % si[i] == 0
    #         shape += [so[i] // si[i]]
    #     else:
    #         shape += [1]
    # v = v.repeat(*shape)
    if si[2] != so[2]:
        assert so[2] / si[2] == so[3] / si[3]  # currently only support this case
        v = F.interpolate(v, size=so[2], mode='nearest')#, align_corners=True)

    # Increase feature maps.
    if si[1] < so[1]:
        z = torch.zeros([v.shape[0], so[1] - si[1]] + so[2:])
        v = torch.cat([v, z], 1)
    if si[0] < so[0]:
        z = torch.zeros([so[0] - si[0], v.shape[1]] + so[2:])
        v = torch.cat([v, z], 0)
    return v


def resize_1d(x, size):
    return _resize(x, size, "nearest")


def resize_2d(x, size):
    return _resize(x, size, "bilinear", align_corners=True)


def _resize(x, size, mode, align_corners=None):
    x = x.clone().detach()
    x = x.expand(1, 1, *x.size())
    ret = F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    return ret[0, 0]


def resize_activations(w, size):
    out = w
    if w.size()[2] != size[3] or w.size()[2] != size[3]:
        out = F.interpolate(w, size=size[-2:])
    if w.size()[0] != size[0] or w.size()[1] != size[1]:
        out = resize_channels(w, size[:2])
    return out


def resize_conv(source_module, target_module):
    success = False
    if target_module.bias is not None and source_module.bias is not None and source_module.bias.size() != target_module.bias.size():
        target_module.bias = nn.Parameter(resize_1d(source_module.bias, target_module.bias.size()[0]))
        logger.debug(f"bias {target_module.bias.size()} {source_module.bias.size()}")
    try:
        w = resize_activations(source_module.weight, target_module.weight.size())
        logger.debug(f"{source_module.weight.size()} {target_module.weight.size()} {w.size()}")
        target_module.weight = nn.Parameter(w)
        success = True
    except Exception as e:
        logger.debug("error resizing weights")
        logger.exception(e)
    return target_module, success


def calc_div_factor(w, num_layers, max_layers, min_w=3):
    n = min(max_layers, int(math.log(w // min_w, 2)))
    return 2 ** max(0, n - num_layers)


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
                                              num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(
        dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return h, w


def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
                                                       num2tuple(kernel_size), num2tuple(stride), num2tuple(
        pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (h_w[0] - 1) * stride[0] - sum(pad[0]) + dilation[0] * (kernel_size[0] - 1) + out_pad[0] + 1
    w = (h_w[1] - 1) * stride[1] - sum(pad[1]) + dilation[1] * (kernel_size[1] - 1) + out_pad[1] + 1

    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
                                                     num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)

    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))


def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(h_w_in), num2tuple(h_w_out), \
                                                              num2tuple(kernel_size), num2tuple(stride), num2tuple(
        dilation), num2tuple(out_pad)

    p_h = -(h_w_out[0] - 1 + 2*out_pad[0] - dilation[0] * (kernel_size[0] - 1) - (h_w_in[0] - 1) * stride[0]) / 2
    p_w = -(h_w_out[1] - 1 + 2*out_pad[1] - dilation[1] * (kernel_size[1] - 1) - (h_w_in[1] - 1) * stride[1]) / 2

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))
