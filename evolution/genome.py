from torch import nn
from .genes import Linear, Conv2d, Optimizer, Deconv2d, Deconv2dUpsample
import numpy as np
from util import config
import logging
import copy
import uuid

logger = logging.getLogger(__name__)


class Genome:
    """Hold a tree with connected genes that compose a pytorch model."""

    def __init__(self, linear_at_end=True, random=False, **kwargs):
        self.max_layers = kwargs.get("max_layers", config.evolution.max_layers)
        self.add_layer_prob = kwargs.get("add_layer_prob", config.evolution.add_layer_prob)
        self.rm_layer_prob = kwargs.get("rm_layer_prob", config.evolution.rm_layer_prob)
        self.gene_mutation_prob = kwargs.get("gene_mutation_prob", config.evolution.gene_mutation_prob)
        self.mutate_gan_type_prob = kwargs.get("mutate_gan_type_prob", config.evolution.mutate_gan_type_prob)
        self.crossover_rate = kwargs.get("crossover_rate", config.evolution.crossover_rate)
        # TODO: now it is a simple sequential list of genes, but it should be a graph to represent non-sequential models
        self._genes = []
        self.optimizer_gene = Optimizer()
        self._output_genes = []
        self._input_genes = []
        self.linear_at_end = linear_at_end
        self.random = random
        self.generation = 0
        self.age = 0
        self.possible_genes = [(Linear, {}), (Conv2d, {}), (Deconv2d, {})]
        self.num_params = 0
        self.mutated = False
        self.uuid = str(uuid.uuid4())
        self.parents_uuid = []
        self.freeze = False
        self.gan_type = None
        self.original_genome = None
        if self.mutate_gan_type_prob > 0:
            self.mutate_gan_type()
        else:
            self.gan_type = config.gan.type

    def clone(self):
        # remove spectral norm to avoid errors in deepcopy
        for gene in self.all_genes():
            if gene.module is not None:
                gene.remove_spectral_norm()
        return copy.deepcopy(self)

    def breed(self, mate=None, skip_mutation=False, freeze=False):
        # reset the optimizer to prevent copy
        self.optimizer_gene.optimizer = None
        genome = self.clone() if self.original_genome is None else self.original_genome.clone()
        genome.uuid = str(uuid.uuid4())
        genome.parents_uuid = [self.uuid]
        genome.age = self.age + 1
        if mate is not None:
            genome.crossover(mate)
        if not skip_mutation:
            genome.mutate()
        self.freeze = freeze
        self.original_genome = copy.deepcopy(genome) if freeze else None
        return genome

    def crossover(self, mate):
        """Apply crossover operator using as cut_point the change from 1d to 2d (or 2d to 1d)"""
        if np.random.rand() <= self.crossover_rate:
            self.age = 0
            cut_point, cut_point_mate = self.cut_point(target_linear=True), mate.cut_point(target_linear=False)
            mate = mate.clone()
            if cut_point and cut_point_mate:
                self._genes = self._genes[cut_point]
                for gene in mate.genes[cut_point_mate]:
                    self.add(gene, force_sequence=True)

    def cut_point(self, target_linear=True):
        linear = None
        for i, gene in enumerate(self.genes):
            if linear is not None and linear != gene.is_linear():
                if gene.is_linear() == target_linear:
                    return slice(i, len(self.genes))
                else:
                    return slice(0, i)
            linear = gene.is_linear()
        if linear == target_linear:
            return slice(len(self.genes))
        return None

    def add_random_gene(self):
        new_gene_index = np.random.choice(len(self.possible_genes))
        new_gene_type = self.possible_genes[new_gene_index]
        new_gene = new_gene_type[0](**new_gene_type[1])
        logger.info("new_gene: %s", new_gene)
        return self.add(new_gene)

    def add(self, gene, force_sequence=None):
        """Add a gene keeping linear layers at the end"""
        first_half = (gene.is_linear() and not self.linear_at_end) or (not gene.is_linear() and self.linear_at_end)
        div_index = len(self._genes)
        for i, g in enumerate(self._genes):
            if g.is_linear() == self.linear_at_end:
                div_index = i
                break

        insert_at = len(self._genes)
        if first_half:
            insert_at = div_index

        # for D, add convolutional layers in the beginning
        if self.linear_at_end and not gene.is_linear():
            insert_at = 0

        if self.random and (force_sequence is None or force_sequence is False):
            if first_half:
                insert_at = np.random.randint(0, insert_at + 1)
            else:
                insert_at = np.random.randint(div_index, insert_at+1) if insert_at > div_index else insert_at

        self._genes.insert(insert_at, gene)
        return insert_at + 1

    def mutate(self):
        # FIXME: improve this method
        self.optimizer_gene.mutate(probability=self.gene_mutation_prob)
        self.mutated = False
        if np.random.rand() < self.gene_mutation_prob and len(self.genes) > 0:
            gene = np.random.choice(self.genes)
            gene.mutate(probability=1)
            self.mutated = True
        if np.random.rand() < self.add_layer_prob and len(self.all_genes()) < self.max_layers:
            logger.info("MUTATE add")
            self.add_random_gene()
            self.mutated = True
        if np.random.rand() < self.rm_layer_prob and len(self.genes) > 1:
            self.remove_random_gene()
            self.mutated = True
        if np.random.rand() < self.mutate_gan_type_prob:
            self.mutate_gan_type()
            self.mutated = True
        if self.mutated:
            self.age = 0

    def remove_random_gene(self):
        removed = np.random.choice(self.genes)
        logger.info("MUTATE rm")
        self._genes.remove(removed)

    def increase_usage_counter(self):
        """Increase gene usage counter"""
        for gene in self.all_genes():
            if not gene.freezed:
                gene.used += 1

    def mutate_gan_type(self):
        possibilities = config.gan.possible_gan_types
        new_gan_type = np.random.choice(possibilities)
        logger.info(f"MUTATE gan type: {self.gan_type} -> {new_gan_type}")
        self.gan_type = new_gan_type

    def get_gene_by_module_name(self, name):
        return next((gene for gene in self.genes if name.startswith(gene.module_name)), None)

    def get_gene_by_uuid(self, uuid):
        return next((gene for gene in self.genes if gene.uuid == uuid), None)

    def new_genes(self, genome):
        current_uuids = [g.uuid for g in self.genes]
        uuids = [g.uuid for g in genome.genes]
        new_uuids = set(uuids) - set(current_uuids)
        return [uuids.index(u) for u in new_uuids]

    def clean_copy(self):
        """Return the genome without heavy parameters (weights, etc)"""
        genome = Genome(random=self.random)
        for g in self.input_genes:
            g_copy = copy.copy(g)
            g_copy.module = None
            genome.input_genes.append(g_copy)
        for g in self.genes:
            g_copy = copy.copy(g)
            g_copy.module = None
            genome.genes.append(g_copy)
        for g in self.output_genes:
            g_copy = copy.copy(g)
            g_copy.module = None
            genome.output_genes.append(g_copy)
        return genome

    @classmethod
    def from_json(cls, data):
        genome = Genome(random=False)
        for i, row in enumerate(data):
            if row["type"] == "Deconv2d":
                layer = Deconv2d(activation_type=row["activation_type"], out_channels=row["out_channels"])
                layer.in_channels = row["in_channels"]
            elif row["type"] == "Deconv2dUpsample":
                layer = Deconv2dUpsample(activation_type=row["activation_type"], out_channels=row["out_channels"])
                layer.in_channels = row["in_channels"]
            elif row["type"] == "Conv2d":
                layer = Conv2d(activation_type=row["activation_type"], out_channels=row["out_channels"])
                layer.in_channels = row["in_channels"]
            else:
                layer = Linear(activation_type=row["activation_type"], out_features=row["out_features"])
            if layer.input_layer:
                genome.input_genes = [layer]
            elif layer.output_layer:
                genome.output_genes = [layer]
            else:
                genome.add(layer)
        return genome

    def distance(self, genome, c=1, mode=None):
        """Calculates the distance between the current genome and the genome passed as parameter."""
        mode = mode or config.evolution.speciation.distance
        n = 1  # max(len(genome.genes), len(self.genes))  # get the number of genes in the larger genome
        d = 0

        if mode == "uuid":
            # get uuids from both genomes
            current_uuids = set([g.uuid for g in self.genes])
            uuids = set([g.uuid for g in genome.genes])
            # calculates the number of different genes
            d = len(current_uuids.symmetric_difference(uuids))

        # use num_genes difference and internal setup to compute the difference
        # TODO: consider the age
        if mode == "num_genes":
            d = abs(len(self.genes) - len(genome.genes))
            if d == 0:
                for g1, g2 in zip(self.genes, genome.genes):
                    if g1.__class__ != g2.__class__:
                        d += 0.3
                    if g1.activation_type != g2.activation_type:
                        d += 0.1
                    if isinstance(g1, Linear) and isinstance(g2, Linear):
                        if g1.in_features != g2.in_features:
                            d += 0.1
                        if g1.out_features != g2.out_features:
                            d += 0.1
                    if isinstance(g1, Conv2d) and isinstance(g2, Conv2d):
                        if g1.in_channels != g2.in_channels:
                            d += 0.1
                        if g1.out_channels != g2.out_channels:
                            d += 0.1

        if mode == "num_genes_per_type":
            linear_genes = [g for g in self.genes if g.is_linear()]
            non_linear_genes = [g for g in self.genes if not g.is_linear()]
            linear_genes2 = [g for g in genome.genes if g.is_linear()]
            non_linear_genes2 = [g for g in genome.genes if not g.is_linear()]
            d = abs(len(linear_genes) - len(linear_genes2)) + abs(len(non_linear_genes) - len(non_linear_genes2))

        if mode == "parameter":
            d = abs(self.num_params - genome.num_params)

        # compare the gan types
        if self.gan_type != genome.gan_type:
            d += 1

        return c*d/n

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes):
        self._genes = genes

    @property
    def output_genes(self):
        return self._output_genes

    @output_genes.setter
    def output_genes(self, output_genes):
        self._output_genes = output_genes
        for g in output_genes or []:
            g.output_layer = True

    @property
    def input_genes(self):
        return self._input_genes

    @input_genes.setter
    def input_genes(self, input_genes):
        self._input_genes = input_genes
        for g in input_genes or []:
            g.input_layer = True

    def all_genes(self):
        return self.input_genes + self.genes + self.output_genes

    def __repr__(self):
        genes_str = " -> ".join([f"{g.input_shape}:{g}:{g.output_shape}" for g in self.all_genes()])
        return self.__class__.__name__ + f"(genes={genes_str})"
