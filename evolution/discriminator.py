from evolution import Phenotype
from evolution.generator import Generator
import torch
import evolution
from evolution import Genome, Linear, Conv2d
import logging
import torch.autograd as autograd
from torch import Tensor
from util import config
from util import tools
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits

logger = logging.getLogger(__name__)


class Discriminator(Phenotype):

    labels = None
    fake_labels = None
    selected_loss = None

    def __init__(self, output_size=1, genome=None, input_shape=None, optimizer_conf=None):
        super().__init__(output_size=output_size, genome=genome, input_shape=input_shape)
        self.output_size = output_size
        self.optimizer_conf = optimizer_conf or config.gan.discriminator.optimizer

        if genome is None:
            if config.gan.discriminator.fixed:
                self.genome = Genome(random=False, add_layer_prob=0, rm_layer_prob=0, gene_mutation_prob=0,
                                     mutate_gan_type_prob=0)
                self.genome.add(Conv2d(256, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                self.genome.add(Conv2d(128, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                self.genome.add(Conv2d(64, stride=2, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                # self.genome.add(SelfAttention())
            else:
                self.genome = Genome(random=not config.evolution.sequential_layers)
                self.genome.possible_genes = [(getattr(evolution, l), {}) for l in config.gan.discriminator.possible_layers]

            if not config.gan.discriminator.fixed:
                self.genome.input_genes = [Conv2d(stride=1, normalize="spectral")]
            else:
                self.genome.input_genes = []
            if not config.gan.discriminator.fixed:
                self.genome.output_genes = [Linear(1, activation_type=None, normalize="spectral", bias=False)]
            else:
                self.genome.output_genes = [Linear(1, activation_type=None, normalize="none", bias=False)]

    def forward(self, x):
        out = super().forward(x)
        out = out.view(out.size(0), -1)
        return out

    def train_step(self, G, images):
        """Train the discriminator on real+fake"""
        self.zero_grad()

        if self.labels is None:
            self.labels = tools.cuda(Tensor(torch.ones(images.size(0))))
            self.labels = self.labels * 0.9 if config.gan.label_smoothing else self.labels

        if self.fake_labels is None:
            self.fake_labels = tools.cuda(Tensor(torch.zeros(images.size(0))))
            self.fake_labels = self.fake_labels + 0.1 if config.gan.label_smoothing else self.fake_labels

        error, real_decision, fake_decision, fake_data = self.loss(G, images)

        if self.use_gradient_penalty():
            gradient_penalty = self.gradient_penalty(images.data, fake_data.data)
            error += gradient_penalty

        error.backward()
        self.optimizer.step()

        # clip weights for WGAN
        if self.genome.gan_type == "wgan" and not self.use_gradient_penalty():
            clip_value = 0.01
            for p in self.parameters():
                p.data.clamp_(-clip_value, clip_value)

        self.calc_metrics(G, error.item(), real_decision, fake_decision)
        return error.item()

    def loss(self, G, images):
        real_decision = self(images)
        fake_data = G(G.generate_noise(images.size()[0])).detach()  # detach to avoid training G on these labels
        fake_decision = self(fake_data)
        loss_function = getattr(self, f"loss_{self.genome.gan_type}")
        error = loss_function(real_decision, fake_decision)
        return error, real_decision, fake_decision, fake_data

    def loss_wgan(self, real_decision, fake_decision):
        return -real_decision.mean() + fake_decision.mean()

    def loss_wgan_gp(self, real_decision, fake_decision):
        return self.loss_wgan(real_decision, fake_decision)

    def loss_rsgan(self, real_decision, fake_decision):
        return binary_cross_entropy_with_logits(real_decision.view(-1) - fake_decision.view(-1), self.labels)

    def loss_rasgan(self, real_decision, fake_decision):
        error = (binary_cross_entropy_with_logits(real_decision.view(-1) - torch.mean(fake_decision.view(-1)),
                                                  self.labels) +
                 binary_cross_entropy_with_logits(fake_decision.view(-1) - torch.mean(real_decision.view(-1)),
                                                  self.fake_labels)) / 2
        return error

    def loss_lsgan(self, real_decision, fake_decision):
        return (torch.mean((real_decision - 1) ** 2) + torch.mean(fake_decision ** 2)) / 2

    def loss_gan(self, real_decision, fake_decision):
        real_error = binary_cross_entropy_with_logits(real_decision.view(-1), self.labels)
        fake_error = binary_cross_entropy_with_logits(fake_decision.view(-1), self.fake_labels)
        return (real_error + fake_error) / 2

    def loss_hinge(self, real_decision, fake_decision):
        real_error = torch.mean(torch.nn.ReLU(inplace=True)(1 - real_decision))
        fake_error = torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_decision))
        return real_error + fake_error

    def eval_step(self, G, images):
        error, real_decision, fake_decision, _ = self.loss(G, images)
        self.calc_metrics(G, error.item(), real_decision, fake_decision)
        return error.item()

    def use_gradient_penalty(self):
        return config.gan.discriminator.use_gradient_penalty or self.genome.gan_type == "wgan_gp"

    def calc_global_metrics(self, best_generators, images):
        if Generator.noise_images is None:
            Generator.noise_images = tools.cuda(torch.FloatTensor(len(images), *images[0].shape).uniform_(-1, 1))
        G = tools.cuda(best_generators[0])
        labels = tools.cuda(Tensor(torch.ones(images.size(0))))
        fake_labels = tools.cuda(Tensor(torch.zeros(images.size(0))))
        if config.evolution.fitness.discriminator.startswith("validation_loss_"):
            fake_data = G(G.generate_noise(images.size(0)))
            loss_function = getattr(self, config.evolution.fitness.discriminator[11:])
            error = loss_function(self(images), self(fake_data)).item()
            self.fitness_values = [error]
        elif config.evolution.fitness.discriminator == "rel_avg":
            with torch.no_grad():
                real_decision = self(images)
                fake_data = G(G.generate_noise(images.size(0)))
                fake_decision = self(fake_data)

                noise_score = torch.mean(torch.sigmoid(self(Generator.noise_images)))
                train_score = torch.mean(torch.sigmoid(real_decision))
                gen_score = torch.mean(torch.sigmoid(fake_decision))

                d_conf = (1 + train_score - noise_score) / 2
                value = -d_conf * (1 - gen_score)
                self.fitness_values = [value.item()]
        elif config.evolution.fitness.discriminator == "rel_avg2":
            with torch.no_grad():
                real_decision = self(images)
                fake_data = G(G.generate_noise(images.size(0)))
                fake_decision = self(fake_data)
                noise_decision = self(Generator.noise_images)
                error = (binary_cross_entropy_with_logits(real_decision.view(-1) - torch.mean(fake_decision.view(-1)), labels) +
                         binary_cross_entropy_with_logits(fake_decision.view(-1) - torch.mean(real_decision.view(-1)), fake_labels) +
                         binary_cross_entropy_with_logits(noise_decision.view(-1) - torch.mean(real_decision.view(-1)), fake_labels)) / 3
                self.fitness_values = [error.item()]
        if config.evolution.fitness.discriminator == "rel_avg3":
            with torch.no_grad():
                real_decision = self(images)
                fake_data = G(G.generate_noise(images.size(0)))
                fake_decision = self(fake_data)
                noise_decision = self(Generator.noise_images)
                mean_noise = torch.mean(noise_decision)
                error = (binary_cross_entropy_with_logits(real_decision.view(-1) - (torch.mean(fake_decision.view(-1)) + mean_noise)/2, labels) +
                         binary_cross_entropy_with_logits(fake_decision.view(-1) - (torch.mean(real_decision.view(-1)) + mean_noise)/2, fake_labels)) / 2
                self.fitness_values = [error.item()]
        G.cpu()

    def calc_metrics(self, G, error, real_decision, fake_decision):
        self.calc_win_rate(torch.sigmoid(real_decision), torch.sigmoid(fake_decision), G)
        if config.evolution.fitness.discriminator == "rel_avg":
            pass
            # with torch.no_grad():
            #     c, w, h = self.input_shape[-3], self.input_shape[-2], self.input_shape[-1]
            #     if Generator.noise_images is None:
            #         Generator.noise_images = tools.cuda(torch.FloatTensor(real_decision.shape[0], c, w, h).uniform_(-1, 1))
            #     noise_score = torch.mean(torch.sigmoid(self(Generator.noise_images)))
            #     train_score = torch.mean(torch.sigmoid(real_decision))
            #     gen_score = torch.mean(torch.sigmoid(fake_decision))
            #     value = -train_score*(1-noise_score)*(1-gen_score)
            #     self.fitness_values.append(value.item())
        elif config.evolution.fitness.discriminator == "loss":
            self.fitness_values.append(error)
        elif config.evolution.fitness.discriminator.startswith("loss_"):
            loss_function = getattr(self, config.evolution.fitness.discriminator)
            error = loss_function(real_decision, fake_decision).item()
            self.fitness_values.append(error)
        elif config.evolution.fitness.discriminator == "AUC":
            full_decision = np.concatenate((torch.sigmoid(real_decision).detach().cpu().numpy().flatten(),
                                            torch.sigmoid(fake_decision).detach().cpu().numpy().flatten()))
            full_labels = np.concatenate((np.ones(real_decision.size()[0]), np.zeros(fake_decision.size()[0])))
            self.fitness_values.append(1 - roc_auc_score(full_labels, full_decision))
            # self.fitness_value -= average_precision_score(full_labels, full_decision)
            # self.fitness_value += 1 - np.mean(
            #     [accuracy_score(np.zeros(fake_decision.size()[0]), fake_decision.cpu().data.numpy().flatten() > 0.5),
            #      accuracy_score(np.ones(real_decision.size()[0]), real_decision.cpu().data.numpy().flatten() > 0.5)])
        elif config.evolution.fitness.discriminator == "BCE":
            self.fitness_values.append(self.loss_gan(real_decision, fake_decision).item())
        elif config.evolution.fitness.discriminator == "random_loss":
            if Discriminator.selected_loss is None:
                Discriminator.selected_loss = np.random.choice(config.gan.possible_gan_types)
                logger.info(f"using random loss function as fitness: {Discriminator.selected_loss}")
            loss_function = getattr(self, f"loss_{Discriminator.selected_loss}")
            self.fitness_values.append(loss_function(real_decision, fake_decision).item())

    def calc_win_rate(self, real_decision, fake_decision, G):
        if self.skill_rating_enabled():
            self.win_rate += (sum((real_decision > 0.5).float()) + sum((fake_decision < 0.5).float())).item()/(len(real_decision) + len(fake_decision))
            # self.win_rate += (torch.mean(real_decision).item() + (1-torch.mean(fake_decision)).item())/2
            # self.win_rate += (1-torch.mean(fake_decision)).item()

            # for match in [torch.mean((real_decision > 0.5).float()).item(), torch.mean((real_decision < 0.5).float()).item()]:
            # for match in [torch.mean(real_decision) > 0.5, torch.mean(fake_decision).item() < 0.5]:
            #     match = int(match)
            #     self.skill_rating_games.append((match, G.skill_rating))
            #     G.skill_rating_games.append((1 - match, self.skill_rating))

            # matches = [(p > 0.5) for p in real_decision] + [(p < 0.5) for p in fake_decision]
            # for match in matches:
            #     match = match.float().item()
            #     self.skill_rating_games.append((match, G.skill_rating))
            #     G.skill_rating_games.append((1 - match, self.skill_rating))

    def gradient_penalty(self, real_data, fake_data, epsilon=1e-12):
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = tools.cuda(alpha.expand_as(real_data))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.clone().detach().requires_grad_(True)

        disc_interpolates = tools.cuda(self(interpolates))
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=tools.cuda(torch.ones(disc_interpolates.size())),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + epsilon)
        return ((gradients_norm - 1) ** 2).mean() * config.gan.discriminator.gradient_penalty_lambda

    def finish_generation(self, **kwargs):
        if config.evolution.fitness.discriminator == "skill_rating":
            self.fitness_values.append(-self.skill_rating.mu) #+ 2*self.skill_rating.phi
        elif config.evolution.fitness.discriminator == "random":
            self.fitness_values = [np.random.rand()]
