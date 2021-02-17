import torch
import torch.nn as nn
from .layer2d import Layer2D


class SelfAttention(Layer2D):

    def __init__(self):
        super().__init__(activation_type=None, normalize=False, use_dropout=False)
        self.in_channels = None
        self.out_channels = None

    def _create_phenotype(self, input_shape):
        self.in_channels = input_shape[1]
        self.out_channels = self.in_channels
        return SelfAttentionModule(self.in_channels)

    def changed(self):
        return self.in_channels != self.module.in_channels or self.out_channels != self.module.in_channels


class SelfAttentionModule(nn.Module):
    """ Self attention Layer"""
    # based on https://github.com/heykeetae/Self-Attention-GAN

    def __init__(self, in_dim):
        super().__init__()
        self.kernel_size = 1
        self.in_channels = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=self.kernel_size)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=self.kernel_size)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=self.kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
