from .base_evolution import BaseEvolution
import logging

logger = logging.getLogger(__name__)


class NonEvolutionary(BaseEvolution):

    def next_population(self, generators_population, discriminators_population):
        logger.info("next population")
        self.evaluator.evaluate_population(generators_population.phenotypes(), discriminators_population.phenotypes())
        return generators_population, discriminators_population
