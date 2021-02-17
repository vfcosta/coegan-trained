import util.tools as tools
from util import config
import torch
import logging
import numpy as np
from evolution.population import Population

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, train_loader, validation_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.best_discriminators = []
        self.best_generators = []
        self.initial = True
        self.batches = []
        self.eval_batches = []

    def init_generation(self, generation):
        self.batches = []
        self.eval_batches = []

    def train_evaluate(self, G, D, batches_limit):
        logger.debug(f"train: G({G.genome.gan_type}) x D({D.genome.gan_type}), batches: {batches_limit}")
        if config.evolution.evaluation.reset_optimizer:
            D.reset_optimizer_state()
            G.reset_optimizer_state()

        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return
        torch.cuda.empty_cache()
        n = 0
        G, D = tools.cuda(G), tools.cuda(D)  # load everything on gpu (cuda)
        G.train()
        D.train()
        G.win_rate, D.win_rate = 0, 0
        while n < batches_limit:
            image_loader = self.batches if config.evolution.evaluation.same_batches and self.batches else self.train_loader
            for images, _ in image_loader:
                if config.evolution.evaluation.same_batches and image_loader != self.batches:
                    self.batches.append((images, _))
                n += 1
                images = tools.cuda(images)
                if n % config.gan.generator_iterations == 0:
                    D.do_train(G, images)
                if n % config.gan.critic_iterations == 0:
                    G.do_train(D, images)
                if n >= config.gan.batches_limit:
                    break
        D.win_rate /= n
        G.win_rate = 1 - D.win_rate
        D.calc_skill_rating(G)
        G.calc_skill_rating(D)
        # print("train GLICKO G:", G.skill_rating, G.win_rate, ", D:", D.skill_rating, D.win_rate)

        G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()

    def evaluate_population(self, generators, discriminators, batches_limit=None, evaluation_type=None, calc_fid=True):
        """Evaluate the population using all-vs-all pairing strategy"""
        batches_limit = batches_limit or config.gan.batches_limit
        evaluation_type = evaluation_type or config.evolution.evaluation.type
        for i in range(config.evolution.evaluation.iterations):
            if evaluation_type == "random":
                for D in discriminators:
                    for g in np.random.choice(generators, 2, replace=False):
                        self.train_evaluate(g, D, batches_limit)
                for G in generators:
                    for d in np.random.choice(discriminators, 2, replace=False):
                        self.train_evaluate(G, d, batches_limit)
            elif evaluation_type == "spatial":
                rows = 3
                cols = len(discriminators)//rows
                pairs = []
                for center in range(len(discriminators)):
                    pairs.append([(center, n) for n in tools.get_neighbors(center, rows, cols)])
                # reorder pairs to avoid sequential training
                pairs = np.transpose(np.array(pairs), (1, 0, 2)).reshape(-1, 2)
                for g, d in pairs:
                    self.train_evaluate(generators[g], discriminators[d], batches_limit)
            elif evaluation_type == "spatial2":
                rows = 3
                cols = len(discriminators)//rows
                for center in range(len(discriminators)):
                    neighbors = tools.get_neighbors(center, rows, cols)
                    norm = len(neighbors)
                    for n in neighbors:
                        self.train_evaluate(generators[center], discriminators[n].clone(), batches_limit)
                        self.train_evaluate(generators[n].clone(), discriminators[center], batches_limit)

            elif evaluation_type == "all-vs-all" and config.evolution.evaluation.clone_adversarial:
                # train all-vs-all in a non-sequential order
                pairs = tools.permutations(generators, discriminators)
                original_generators = [g.clone() for g in generators]
                original_discriminators = [d.clone() for d in discriminators]
                for g, d in pairs:
                    self.train_evaluate(generators[g], original_discriminators[d].clone(), batches_limit)
                    self.train_evaluate(original_generators[g].clone(), discriminators[d], batches_limit)
            elif evaluation_type == "all-vs-all":
                # train all-vs-all in a non-sequential order
                pairs = tools.permutations(generators, discriminators)
                for g, d in pairs:
                    self.train_evaluate(generators[g], discriminators[d], batches_limit)
            elif evaluation_type in ["all-vs-best", "all-vs-species-best", "all-vs-kbest", "all-vs-kbest-previous"]:
                if config.evolution.evaluation.initialize_all and self.initial:
                    self.initial = False
                    # as there are no way to determine the best G and D, we rely on all-vs-all for the first evaluation
                    return self.evaluate_population(generators, discriminators, batches_limit,
                                                    evaluation_type="all-vs-all")

                pairs = tools.permutations(discriminators, self.best_generators)
                for d, g in pairs:
                    adversarial = self.best_generators[g]
                    if config.evolution.evaluation.clone_adversarial:
                        adversarial = adversarial.clone()
                    self.train_evaluate(adversarial, discriminators[d], batches_limit)
                pairs = tools.permutations(generators, self.best_discriminators)
                for g, d in pairs:
                    adversarial = self.best_discriminators[d]
                    if config.evolution.evaluation.clone_adversarial:
                        adversarial = adversarial.clone()
                    self.train_evaluate(generators[g], adversarial, batches_limit)

        # reset FID
        for G in generators:
            G.fid_score = None

        images, n = None, 0
        for batch, _ in self.validation_loader:
            if images is None:
                images = batch
            else:
                images = torch.cat((images, batch))
            n += 1
            if n >= config.evolution.fitness.evaluation_batches:
                break
        images = tools.cuda(images)
        if len(generators) > 0:
            for p in discriminators:
                p = tools.cuda(p)
                p.calc_global_metrics(self.best_generators or [Population(generators).best()], images)
                p.cpu()
        if len(discriminators) > 0:
            for p in generators:
                p = tools.cuda(p)
                p.calc_global_metrics(self.best_discriminators or [Population(discriminators).best()], images)
                p.cpu()

        # # update the skill rating for the next generation
        for p in discriminators + generators + self.best_discriminators + self.best_generators:
            p.finish_calc_skill_rating()
        for p in discriminators + generators:
            p.finish_generation(calc_fid=calc_fid)

    def evaluate_all_validation(self, generators, discriminators):
        # evaluate in validation
        logger.info(f"best G: {len(self.best_generators)}, best D: {len(self.best_discriminators)}")
        for D in discriminators:
            for G in self.best_generators + generators:
                with torch.no_grad():
                    self.evaluate_validation(G, D)
        for G in generators:
            for D in self.best_discriminators:
                with torch.no_grad():
                    self.evaluate_validation(G, D)

        # # update the skill rating for the next generation
        for p in discriminators + generators + self.best_discriminators + self.best_generators:
            p.finish_calc_skill_rating()

    def update_bests(self, generators_population, discriminators_population):
        # store best of generation in coevolution memory
        self.best_discriminators = self.get_bests(discriminators_population, self.best_discriminators)
        self.best_generators = self.get_bests(generators_population, self.best_generators)

    def evaluate_validation(self, G, D, eval_generator=True, eval_discriminator=True):
        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return
        torch.cuda.empty_cache()
        G, D = tools.cuda(G), tools.cuda(D)
        G.eval(), D.eval()
        G.win_rate, D.win_rate = 0, 0
        n = 0
        while n < config.evolution.fitness.evaluation_batches:
            image_loader = self.eval_batches if config.evolution.evaluation.same_batches and self.eval_batches else self.validation_loader
            for images, _ in image_loader:
                if config.evolution.evaluation.same_batches and image_loader != self.eval_batches:
                    self.eval_batches.append((images, _))
                n += 1
                images = tools.cuda(images)
                if eval_discriminator:
                    D.do_eval(G, images)  # FIXME always eval D when skill rating is enabled
                if eval_generator:
                    G.do_eval(D, images)
                    G.win_rate = 1 - D.win_rate
                if n >= config.evolution.fitness.evaluation_batches:
                    break

        D.win_rate /= n
        G.win_rate = 1 - D.win_rate
        if eval_discriminator:
            D.calc_skill_rating(G)
        if eval_generator:
            G.calc_skill_rating(D)

        logger.debug(f"eval GLICKO G: {G.skill_rating} {G.win_rate}, D: {D.skill_rating} {D.win_rate}")
        G, D = G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()

    def get_bests(self, population, previous_best=[]):
        if config.evolution.evaluation.type == "all-vs-species-best":
            return [species.best() for species in population.species_list]
        elif config.evolution.evaluation.type == "all-vs-best":
            return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
        elif config.evolution.evaluation.type == "all-vs-kbest":
            return population.bests(config.evolution.evaluation.best_size)
        elif config.evolution.evaluation.type == "all-vs-kbest-previous":
            return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
        return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
