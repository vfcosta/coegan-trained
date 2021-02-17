from util import config
import numpy as np
from .population import Population
from .base_evolution import BaseEvolution
from .discriminator import Discriminator
from scipy.stats import median_absolute_deviation
import logging

logger = logging.getLogger(__name__)


class Lexicase(BaseEvolution):

    def lexicase_selection(self, individuals, test_individuals):
        np.random.shuffle(test_individuals)
        for i in range(len(test_individuals)):
            logger.info(f"iteration {i}")
            test_p = test_individuals.pop(0)
            for p in individuals:
                p.fitness_values = []
                self.evaluate(p, test_p)
            all_fitness = {p: p.fitness() for p in individuals}
            best_fitness = min(all_fitness.values())
            mad = median_absolute_deviation(list(all_fitness.values()))
            logger.debug(f"best fitness: {best_fitness}, mad: {mad}")
            individuals = [p for p, fitness in all_fitness.items() if fitness <= best_fitness + mad]
            logger.debug(f"remaining individuals: {len(individuals)}")
            if not test_individuals or len(individuals) == 1:
                return np.random.choice(individuals)

    def evaluate(self, p, test_p):
        if isinstance(p, Discriminator):
            self.evaluator.evaluate_validation(test_p, p)
        else:
            self.evaluator.evaluate_validation(p, test_p)

    def generate_children(self, population, test_population):
        children = []
        for _ in range(len(population.phenotypes())):
            selected = self.lexicase_selection(population.phenotypes(), test_population.phenotypes())
            logger.info(f"individual selected: {selected}")
            children.append(self.generate_child([selected]))
        return children

    def next_population(self, generators_population, discriminators_population):
        logger.info(f"next population {self.generation}")
        discriminator_children = self.generate_children(discriminators_population, generators_population)
        generator_children = self.generate_children(generators_population, discriminators_population)
        self.evaluate_population(generator_children, discriminator_children)
        logger.debug(f"generator_children: {len(generator_children)}, discriminator_children: {len(discriminator_children)}")
        return Population(generator_children), Population(discriminator_children)
