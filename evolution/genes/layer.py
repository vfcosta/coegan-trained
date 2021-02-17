from collections.abc import Iterable
import torch.nn as nn
import numpy as np
import uuid
from .gene import Gene
from util import config
from ..layers.wscale import WScaleLayer, ConstrainedLayer
from ..layers.minibatch_stddev import MinibatchStdDev
import logging

logger = logging.getLogger(__name__)


class Layer(Gene):
    """Represents a Layer (e.g., FC, conv, deconv) and its
    hyperparameters (e.g., activation function) in the evolving model."""

    def __init__(self, activation_type="random", activation_params={}, normalize=config.gan.normalization,
                 use_dropout=False, input_layer=False, output_layer=False):
        super().__init__()
        self.activation_type = None
        self._original_activation_type = activation_type
        self._original_normalization = normalize
        self.activation_params = activation_params
        self.module = None
        self.module_name = None
        self.input_shape = None
        self.final_output_shape = None
        self.initial_input_shape = None
        self.output_shape = None
        self.next_layer = None
        self.previous_layer = None
        self.normalization = None
        self.pad = None
        self.normalize = None
        self.use_dropout = use_dropout
        self.wscale = None
        self.freezed = False
        self.adjusted = False
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.bias = True

    def setup(self):
        if self.normalize is None:
            if isinstance(self._original_normalization, Iterable) and not isinstance(self._original_normalization, str):
                self.normalize = np.random.choice(self._original_normalization)
            else:
                self.normalize = self._original_normalization
        if self._original_activation_type != "random":
            self.activation_type = self._original_activation_type
        elif self.activation_type is None:
            self.activation_type = np.random.choice(config.layer.activation_functions)
            self.activation_params = {"negative_slope": 0.2} if self.activation_type == "LeakyReLU" else {}

    def create_phenotype(self, input_shape, final_output_shape):
        self.input_shape = input_shape
        self.final_output_shape = final_output_shape
        self.setup()
        if self.module is not None:
            self.remove_spectral_norm()
        modules = []

        if self.has_minibatch_stddev():
            modules.append(MinibatchStdDev())

        if self.module is None or self.changed() or not config.layer.keep_weights:
            self.module = self._create_phenotype(self.input_shape)
            self.normalization = self.create_normalization()
            self.pad = self._create_pad()
            if self.has_wscale():
                self.module = ConstrainedLayer(self.module)
                # self.wscale = WScaleLayer(self.module)
            if not self.adjusted:
                self.used = 0
            self.adjusted = False

        if self.normalize == "spectral":
            self.module = self.spectral_norm(self.module)
        if self.pad is not None:
            modules.append(self.pad)
        modules.append(self.create_module())
        if self.wscale:
            modules.append(self.wscale)
        if self.activation_type:
            modules.append(getattr(nn, self.activation_type)(**self.activation_params))
        if self.normalization:
            modules.append(self.normalization)
        if self.use_dropout:
            modules.append(nn.Dropout2d(p=0.2))
        return nn.Sequential(*modules)

    def spectral_norm(self, module):
        return nn.utils.spectral_norm(module)

    def remove_spectral_norm(self):
        self._remove_spectral_norm(self.module)

    def _remove_spectral_norm(self, module):
        try:
            nn.utils.remove_spectral_norm(module)
        except Exception as e:
            logger.debug(e)

    def create_module(self):
        return self.module

    def create_normalization(self):
        if self.normalize is not False and self.normalize != "none":
            return self._create_normalization()
        return None

    def apply_mutation(self):
        if self._original_activation_type == "random":
            self.activation_type = None
        self.setup()

    def _create_normalization(self):
        return None

    def _create_pad(self):
        return None

    def _create_phenotype(self, input_size):
        pass

    def changed(self):
        return False

    def has_wscale(self):
        return config.gan.use_wscale and not self.is_last_layer()

    def has_minibatch_stddev(self):
        return config.gan.use_minibatch_stddev and self.is_last_layer() and self.is_linear()

    def is_last_layer(self):
        return self.next_layer is None

    def is_linear(self):
        return True

    def reset(self):
        self.module = None  # reset weights
        self.uuid = str(uuid.uuid4())  # assign a new uuid

    def named_parameters(self):
        if self.module is None:
            return None
        return self.module.named_parameters()

    def freeze(self):
        # XXX: If the optimizer has momentum, the parameters will still change. The gradients will be zeroed though.
        if not config.evolution.freeze_when_change:
            return
        logger.debug(f"freeze layer {self.freezed}")
        self.freezed = True
        self.module.zero_grad()
        for param in self.module.parameters():
            param.requires_grad = False

    def unfreeze(self):
        if not config.evolution.freeze_when_change:
            return
        logger.debug(f"unfreeze layer {self.freezed}")
        self.freezed = False
        for param in self.module.parameters():
            param.requires_grad = True

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'activation_type=' + str(self.activation_type) + ')'
