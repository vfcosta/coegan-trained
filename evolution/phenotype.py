import torch
import torch.nn as nn
from .genes import Linear
from .layers.reshape import Reshape
import numpy as np
import copy
import traceback
from util import config
import logging
import json
from metrics.glicko2 import glicko2
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class Phenotype(nn.Module):

    def forward(self, x):
        try:
            out = self.model(x)
            return out
        except Exception as err:
            logger.exception(err)
            traceback.print_exc()
            self.optimizer = None
            self.invalid = True

    def __init__(self, output_size, genome=None, input_shape=None, optimizer_conf={}):
        super().__init__()
        self.genome = genome
        self.optimizer = None
        self.optimizer_conf = optimizer_conf
        self.scheduler = None
        self.current_adversarial = None
        self.model = None
        self.fitness_values = []
        self.errors = []
        self.win_rate = 0
        self.invalid = False
        self.output_size = output_size
        self.input_shape = input_shape
        self.original_input_shape = input_shape
        self.original_output_size = output_size
        self.trained_samples = 0
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=config.evolution.fitness.skill_rating.sigma, tau=config.evolution.fitness.skill_rating.tau)
        self.skill_rating = self.glicko.create_rating()
        self.skill_rating_games = []
        self.rank = None
        self.crowding_distance = None
        self.objectives = []

    def breed(self, mate=None, skip_mutation=False, freeze=False):
        mate_genome = mate.genome if mate else None
        genome = self.genome.breed(skip_mutation=skip_mutation, mate=mate_genome, freeze=freeze)
        p = self.__class__(output_size=self.original_output_size or self.output_size, genome=genome,
                           input_shape=self.original_input_shape or self.input_shape,
                           optimizer_conf=self.optimizer_conf)
        try:
            p.setup()
            self.copy_to(p)
            if mate:
                mate.copy_to(p)
            p.model.zero_grad()
        except Exception as err:
            logger.exception(err)
            traceback.print_exc()
            logger.debug(genome)
            p.optimizer = None
            p.invalid = True
            # p.error = 100
            if not skip_mutation or mate is not None:
                logger.debug("fallback to parent copy")
                return self.breed(mate=None, skip_mutation=True)
        return p

    def setup(self):
        # create some input data
        with torch.no_grad():
            x = torch.randn([1] + list(self.input_shape[1:]))
        self.create_model(x)

    def clone(self):
        self.genome.optimizer_gene.optimizer = None
        genome = self.genome.clone()
        p = self.__class__(output_size=self.output_size, genome=genome, input_shape=self.input_shape)
        try:
            p.setup()
            self.copy_to(p)
            p.model.zero_grad()
        except Exception as err:
            logger.exception(err)
        return p

    def copy_to(self, target):
        """
        Copy the phenotype parameters to the target.
        This copy will keep the parameters that match in size from the optimizer.
        """
        target.trained_samples = self.trained_samples
        # if not target.genome.mutated:
        target.skill_rating = glicko2.Rating(self.skill_rating.mu, self.skill_rating.phi, self.skill_rating.sigma)
        # else:
            # target.skill_rating = self.glicko.create_rating()
            # target.skill_rating.mu = min(self.skill_rating.mu, target.skill_rating.mu)
        if not self.optimizer_conf.get("copy_optimizer_state"):
            return

        old_state_dict = self.optimizer.state_dict()
        if len(old_state_dict['state']) == 0:
            logger.info("no state to copy")
            return  # there is no state to copy
        try:
            target.optimizer.load_state_dict(old_state_dict)
        except Exception as e:
            logger.info("failed to copy optimizer state")
            logger.exception(e)

    def do_train(self, phenotype, images):
        # if self.genome.freeze:
        #     return self.last_error
        if phenotype.invalid:
            return
        if self.invalid:
            return
        try:
            if self.scheduler is not None and self.current_adversarial != phenotype:
                self.current_adversarial = phenotype
                self.scheduler.step()
            error = self.train_step(phenotype, images)
            self.errors.append(error)
            self.trained_samples += len(images)
            self.genome.increase_usage_counter()
        except Exception as err:
            traceback.print_exc()
            logger.exception(err)
            logger.error(self.model)
        # phenotype.genome.gan_type = original_gan_type
        self.current_adversarial = None

    def do_eval(self, phenotype, images):
        self.eval_step(phenotype, images)
        # if self.invalid:
        #     self.error = 100
        #     return
        # self.error = self.error or 0
        # self.error += self.eval_step(phenotype, images)

    def create_model(self, input_data):
        self.input_shape = input_data.size()
        if "model" not in self._modules:
            self.model = self.transform_genotype(input_data)
        if self.optimizer is None:
            self.optimizer = self.genome.optimizer_gene.create_phenotype(self)
            if config.evolution.adjust_learning_rate:
                self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.update_learning_rate)
        self.genome.num_params = self.num_params()

    def update_learning_rate(self, epoch):
        logger.debug(f"UPDATE LEARNING RATE: {epoch} {self.error} {self.fitness()}")
        return 1

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def transform_genotype(self, input_data):
        """Generates a generic model using pytorch."""
        layers = []
        genes = self.genome.all_genes()
        min_div_scale = config.evolution.min_div_scale
        has_new_gene = len([g for g in genes if g.used == 0]) > 0
        next_input_shape = input_data.shape
        next_input_size = int(np.prod(next_input_shape[1:]))

        # link with previous and next genes
        for i, gene in enumerate(genes):
            gene.initial_input_shape = next_input_shape
            if i + 1 < len(genes):
                gene.next_layer = genes[i + 1]
            if i > 0:
                gene.previous_layer = genes[i - 1]

        # iterate over genes to create a pytorch sequential model
        for i, gene in enumerate(genes):
            # adjust shape for linear layer
            if gene.is_linear() and len(next_input_shape) > 2:
                layers.append(Reshape((-1, next_input_size)))
            # adjust shape for 2d layer
            if not gene.is_linear() and len(next_input_shape) == 2:
                # adjust shape based on the next layers
                w, h = int(np.ceil(self.output_size[1]/min_div_scale)), int(np.ceil(self.output_size[2]/min_div_scale))
                next_input_shape = (-1, next_input_size//w//h, w, h)
                layers.append(Reshape(next_input_shape))

            # adjust out_features of the last linear layer
            if isinstance(gene, Linear) and gene.is_last_linear() and not isinstance(self.output_size, int) and \
                    not gene.next_layer.is_linear():
                if gene.next_layer is not None and gene.next_layer.in_channels is None:
                    if gene.next_layer.out_channels is None or gene.next_layer.is_last_layer():
                        gene.next_layer.in_channels = 2 ** np.random.randint(
                            config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power + 1)
                    else:
                        gene.next_layer.in_channels = 2 * gene.next_layer.out_channels
                gene.out_features = gene.next_layer.in_channels * int(np.ceil(self.output_size[1] / min_div_scale)) * int(np.ceil(self.output_size[2] / min_div_scale))
                logger.debug(f"OUT {gene.out_features} {gene.next_layer.in_channels} {gene.next_layer.out_channels} {gene.next_layer.in_channels} {self.output_size[1] / min_div_scale, min_div_scale}")

            new_layer = gene.create_phenotype(next_input_shape, self.output_size)
            if gene.used > 0 and has_new_gene:
                gene.freeze()
            else:
                gene.unfreeze()
            gene.module_name = "model.%d" % len(layers)
            layers.append(new_layer)
            next_input_size, next_input_shape = self.calc_output_size(layers, input_data)
            gene.output_shape = next_input_shape[1:]

        return nn.Sequential(*layers)

    def calc_output_size(self, layers, input_data):
        current_model = nn.Sequential(*layers)
        current_model.eval()
        # print("CALC", current_model, input_data.shape, self.__class__)
        # print(self.genome, self.__class__)
        forward_pass = current_model(input_data)
        # return the product of the vector array (ignoring the batch size)
        return int(np.prod(forward_pass.size()[1:])), forward_pass.size()

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def fitness(self):
        return np.mean(self.fitness_values)

    def save(self, path):
        torch.save(self.cpu(), path)

    def valid(self):
        return not self.invalid and len(self.errors) > 0

    def skill_rating_enabled(self):
        return config.stats.calc_skill_rating or config.evolution.fitness.discriminator == "skill_rating" or\
               config.evolution.fitness.generator == "skill_rating"

    def calc_skill_rating(self, adversarial):
        if not self.skill_rating_enabled():
            return
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu, phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        # self.skill_rating_games.append((1 if self.win_rate > 0.5 else 0, rating))
        self.skill_rating_games.append((self.win_rate, rating))

    def finish_calc_skill_rating(self):
        if not self.skill_rating_enabled():
            return
        if len(self.skill_rating_games) == 0:
            logger.warning("no games to update the skill rating")
            return
        logger.debug("finish_calc_skill_rating {self.skill_rating_games}")
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []

    def reset_optimizer_state(self):
        self.optimizer = self.genome.optimizer_gene.create_phenotype(self)

    @property
    def error(self):
        return np.mean(self.errors)

    @classmethod
    def load(cls, path):
        return torch.load(path, map_location="cpu")

    def __repr__(self):
        return self.__class__.__name__ + f"(genome={self.genome})"

    def to_json(self):
        """Create a json representing the model"""
        ret = []
        for gene in self.genome.all_genes():
            d = dict(gene.__dict__)
            del d["uuid"], d["module"], d["next_layer"], d["previous_layer"], d["normalization"], d["wscale"], d["pad"]
            d.pop("skip_module", None)
            d.pop("_original_normalization", None)
            ret.append({
                "type": gene.__class__.__name__,
                "wscale": gene.has_wscale(),
                "minibatch_stddev": gene.has_minibatch_stddev(),
                "normalization": gene.normalization.__class__.__name__ if gene.normalization is not None else None,
                **d,
            })
        return json.dumps({"genes": ret, "gan_type": self.genome.gan_type})
