import torch.nn as nn
from .layer import Layer
from util import config
import numpy as np
from util import tools
import logging

logger = logging.getLogger(__name__)


class Linear(Layer):
    """Represents a linear layer (pytorch module) in the evolving model."""

    def __init__(self, out_features=None, activation_type="random", activation_params={},
                 normalize=config.gan.normalization, bias=True):
        super().__init__(activation_type=activation_type, activation_params=activation_params, normalize=normalize)
        self.out_features = out_features
        self.in_features = None
        self.original_out_features = out_features
        self.bias = bias

    def setup(self):
        super().setup()
        if self.input_shape:
            self.in_features = int(np.prod(self.input_shape[1:]))
            if self.has_minibatch_stddev():
                self.in_features += 1
        if self.original_out_features is None:
            self.original_out_features = 2 ** np.random.randint(config.layer.linear.min_features_power,
                                                                config.layer.linear.max_features_power+1)
        if self.out_features is None:
            self.out_features = self.original_out_features

    def changed(self):
        return self.module.out_features != self.out_features or self.module.in_features != self.in_features or \
               (not self.is_last_linear() and self.original_out_features != self.out_features)

    def _create_normalization(self):
        if self.normalize == "batch" or self.normalize is True:
            return nn.BatchNorm1d(self.out_features)

    def _create_phenotype(self, input_shape):
        self.out_features = self.out_features if self.is_last_linear() else self.original_out_features
        module = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        if self.has_wscale():
            nn.init.normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
        if self.module is not None and config.layer.resize_linear_weights:
            logger.debug(f"LINEAR RESIZE {self.module.weight.size()} {module.weight.size()}")
            if module.bias is not None and self.module.bias is not None and self.module.bias.size() != module.bias.size():
                module.bias = nn.Parameter(tools.resize_1d(self.module.bias, module.bias.size()[-1]))
            if self.module.weight.size() != module.weight.size():
                logger.debug("resize linear weights")
                w = tools.resize_2d(self.module.weight, module.weight.size())
                logger.debug(f"linear {self.module.weight.size()} {module.weight.size()} {w.size()}")
                module.weight = nn.Parameter(w)
        return module

    def apply_mutation(self):
        self.out_features = None
        super().apply_mutation()

    def is_last_linear(self):
        return not self.next_layer or not isinstance(self.next_layer, Linear)

    def __repr__(self):
        return self.__class__.__name__ + f"(in_features={self.in_features}, out_features={self.out_features}," \
                                         f"normalize={self.normalize}, " \
                                         f"activation_type={self.activation_type})"
