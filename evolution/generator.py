from evolution import Phenotype
import torch
from torch.utils.data import Dataset, DataLoader
import evolution
from torch import Tensor
from evolution import Genome, Linear, Deconv2d, Conv2d, Deconv2dUpsample
import logging
from util import config
from util.inception_score import inception_score
from util import tools
import numpy as np
from sklearn.metrics import accuracy_score
from metrics import generative_score
import os
from metrics import rmse_score
from torch.nn.functional import binary_cross_entropy_with_logits


logger = logging.getLogger(__name__)


class Generator(Phenotype):

    fid_noise = None
    real_labels = None
    fake_labels = None
    selected_loss = None
    noise_images = None

    def __init__(self, output_size=(1, 28, 28), genome=None, input_shape=(1, config.gan.latent_dim), optimizer_conf=None):
        super().__init__(output_size=output_size, genome=genome, input_shape=input_shape)
        self.noise_size = int(np.prod(self.input_shape[1:]))
        self.inception_score_mean = 0
        self.fid_score = None
        self.rmse_score = None
        self.optimizer_conf = optimizer_conf or config.gan.generator.optimizer
        deconv2d_class = Deconv2dUpsample if config.layer.deconv2d.use_upsample else Deconv2d

        if genome is None:
            if config.gan.generator.fixed:
                self.genome = Genome(random=False, add_layer_prob=0, rm_layer_prob=0, gene_mutation_prob=0,
                                     mutate_gan_type_prob=0, linear_at_end=False)
                self.genome.add(deconv2d_class(128, stride=1, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
                self.genome.add(deconv2d_class(64, stride=1, activation_type="LeakyReLU", activation_params={"negative_slope": 0.2}))
            else:
                self.genome = Genome(random=not config.evolution.sequential_layers, linear_at_end=False)
                self.genome.possible_genes = [(getattr(evolution, l), {}) for l in config.gan.generator.possible_layers]
            self.genome.input_genes = [Linear(8 * int(np.prod(output_size)), activation_type=None, normalize=False)]
            deconv_out = Deconv2d if config.gan.generator.fixed else Deconv2dUpsample
            self.genome.output_genes = [deconv_out(output_size[0], size=output_size[-2:], activation_type="Tanh", normalize=False)]

    def forward(self, x):
        out = super().forward(x)
        if out is not None and len(out.size()) == 2:
            out = out.view(out.size(0), *self.output_size)
        return out

    def train_step(self, D, images):
        self.inception_score_mean = 0
        batch_size = images.size(0)
        # 2. Train G on D's response (but DO NOT train D on these labels)
        self.zero_grad()

        if self.real_labels is None:
            self.real_labels = tools.cuda(Tensor(torch.ones(batch_size)))
            self.real_labels = self.real_labels * 0.9 if config.gan.label_smoothing else self.real_labels

        if self.fake_labels is None:
            self.fake_labels = tools.cuda(Tensor(torch.zeros(images.size(0))))
            self.fake_labels = self.fake_labels + 0.1 if config.gan.label_smoothing else self.fake_labels

        error, decision = self.loss(D, images)
        error.backward()
        self.optimizer.step()  # Only optimizes G's parameters
        self.calc_metrics(D, error.item(), decision, images)
        return error.item()

    def loss(self, D, images, gen_input=None):
        if gen_input is None:
            gen_input = self.generate_noise(images.size(0))
        fake_data = self(gen_input)
        fake_decision = D(fake_data)
        loss_function = getattr(self, f"loss_{self.genome.gan_type}")
        error = loss_function(D, fake_decision, images)
        return error, fake_decision

    def loss_wgan(self, D, fake_decision, images):
        return -fake_decision.mean()

    def loss_wgan_gp(self, D, fake_decision, images):
        return self.loss_wgan(D, fake_decision, images)

    def loss_rsgan(self, D, fake_decision, images):
        real_decision = D(images)
        return binary_cross_entropy_with_logits(fake_decision.view(-1) - real_decision.view(-1), self.real_labels)

    def loss_rasgan(self, D, fake_decision, images):
        real_decision = D(images)
        error = (binary_cross_entropy_with_logits(real_decision.view(-1) - torch.mean(fake_decision.view(-1)),
                                                  self.fake_labels) +
                 binary_cross_entropy_with_logits(fake_decision.view(-1) - torch.mean(real_decision.view(-1)),
                                                  self.real_labels)) / 2
        return error

    def loss_lsgan(self, D, fake_decision, images):
        return torch.mean((fake_decision - 1) ** 2)

    def loss_gan(self, D, fake_decision, images):
        return binary_cross_entropy_with_logits(fake_decision.view(-1), self.real_labels)

    def loss_hinge(self, D, fake_decision, images):
        return -fake_decision.mean()

    def eval_step(self, D, images):
        error, decision = self.loss(D, images)
        self.calc_metrics(D, error.item(), decision, images)
        return error.item()

    def calc_global_metrics(self, best_discriminators, images):
        if Generator.noise_images is None:
            Generator.noise_images = tools.cuda(torch.FloatTensor(len(images), *images[0].shape).uniform_(-1, 1))
        D = tools.cuda(best_discriminators[0])
        labels = tools.cuda(Tensor(torch.ones(images.size(0))))
        fake_labels = tools.cuda(Tensor(torch.zeros(images.size(0))))
        if config.evolution.fitness.generator.startswith("validation_loss_"):
            loss_function = getattr(self, config.evolution.fitness.generator[11:])
            fake_data = self(self.generate_noise(images.size(0)))
            fake_decision = D(fake_data)
            error = loss_function(D, fake_decision, images).item()
            self.fitness_values = [error]
        elif config.evolution.fitness.generator == "rel_avg":
            with torch.no_grad():
                real_decision = D(images)
                fake_data = self(self.generate_noise(images.size(0)))
                fake_decision = D(fake_data)

                train_score = torch.mean(torch.sigmoid(real_decision))
                gen_score = torch.mean(torch.sigmoid(fake_decision))
                noise_score = torch.mean(torch.sigmoid(D(Generator.noise_images)))
                d_conf = (1 + train_score - noise_score) / 2
                value = -d_conf * gen_score
                self.fitness_values = [value.item()]
        elif config.evolution.fitness.generator == "rel_avg2":
            with torch.no_grad():
                real_decision = D(images)
                fake_data = self(self.generate_noise(images.size(0)))
                fake_decision = D(fake_data)
                noise_decision = D(Generator.noise_images)
                error = (binary_cross_entropy_with_logits(real_decision.view(-1) - torch.mean(fake_decision.view(-1)), fake_labels) +
                         binary_cross_entropy_with_logits(fake_decision.view(-1) - torch.mean(real_decision.view(-1)), labels) +
                         binary_cross_entropy_with_logits(noise_decision.view(-1) - torch.mean(real_decision.view(-1)), labels)) / 3
                self.fitness_values = [error.item()]
        elif config.evolution.fitness.generator == "rel_avg3":
            with torch.no_grad():
                real_decision = D(images)
                fake_data = self(self.generate_noise(images.size(0)))
                fake_decision = D(fake_data)
                noise_decision = D(Generator.noise_images)
                mean_noise = torch.mean(noise_decision)
                error = (binary_cross_entropy_with_logits(real_decision.view(-1) - (torch.mean(fake_decision.view(-1)) + mean_noise)/2, fake_labels) +
                         binary_cross_entropy_with_logits(fake_decision.view(-1) - (torch.mean(real_decision.view(-1)) + mean_noise)/2, labels)) / 2
                self.fitness_values = [error.item()]
        D.cpu()

    def calc_metrics(self, D, error, fake_decision, images):
        if config.evolution.fitness.discriminator == "rel_avg":
            pass
            # real_decision = D(images)
            # with torch.no_grad():
            #     c, w, h = D.input_shape[-3], D.input_shape[-2], D.input_shape[-1]
            #     if Generator.noise_images is None:
            #         Generator.noise_images = tools.cuda(torch.FloatTensor(real_decision.shape[0], c, w, h).uniform_(-1, 1))
            #     noise_score = torch.mean(torch.sigmoid(D(Generator.noise_images)))
            #     train_score = torch.mean(torch.sigmoid(real_decision))
            #     gen_score = torch.mean(torch.sigmoid(fake_decision))
            #     value = -train_score * gen_score * (1-noise_score)
            #     self.fitness_values.append(value.item())
        elif config.evolution.fitness.generator == "loss":
            self.fitness_values.append(error)
        elif config.evolution.fitness.generator.startswith("loss_"):
            loss_function = getattr(self, config.evolution.fitness.generator)
            error = loss_function(D, fake_decision, images).item()
            self.fitness_values.append(error)
        elif config.evolution.fitness.generator == "AUC":
            self.fitness_values.append(1 - accuracy_score(np.ones(fake_decision.size(0)), torch.sigmoid(fake_decision).detach().cpu() > 0.5))
        elif config.evolution.fitness.generator == "BCE":
            self.fitness_values.append(binary_cross_entropy_with_logits(fake_decision.squeeze(), self.real_labels).item())
        elif config.evolution.fitness.generator == "random_loss":
            if Generator.selected_loss is None:
                Generator.selected_loss = np.random.choice(config.gan.possible_gan_types)
                logger.info(f"using random loss function as fitness: {Generator.selected_loss}")
            loss_function = getattr(self, f"loss_{Generator.selected_loss}")
            self.fitness_values.append(loss_function(D, fake_decision, images).item())

    def calc_win_rate(self, fake_decision):
        if self.skill_rating_enabled():
            self.win_rate += sum((fake_decision >= 0.5).float()).item()/len(fake_decision)

    def generate_noise(self, batch_size, volatile=False, cuda=True):
        with torch.set_grad_enabled(not volatile):
            gen_input = tools.cuda(torch.randn([batch_size] + list(self.input_shape[1:]), requires_grad=True), condition=cuda)
        return gen_input

    def inception_score(self, batch_size=10, splits=10):
        """Computes the inception score of the generated images
        n -- amount of generated images
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        generated_images = self(self.generate_common_noise()).detach()
        self.inception_score_mean, _ = inception_score(generated_images,
                                                       batch_size=batch_size, resize=True, splits=splits)
        return self.inception_score_mean

    def calc_rmse_score(self):
        generated_images = self(self.generate_common_noise()).detach()
        self.rmse_score = rmse_score.rmse(generated_images)

    def generate_common_noise(self, noise_path='generator_noise.pt', size=None):
        """Generate a noise to be used as base for comparisons"""
        size = size or config.evolution.fitness.fid_sample_size
        if os.path.isfile(noise_path) and Generator.fid_noise is None:
            Generator.fid_noise = torch.load(noise_path)
            logger.info(f"generator noise loaded from file with shape {Generator.fid_noise.shape}")
            if Generator.fid_noise.shape[0] != size:
                logger.info(f"discard loaded generator noise because the sample size is different: {size}")
                Generator.fid_noise = None
        if Generator.fid_noise is None:
            Generator.fid_noise = self.generate_noise(size).cpu()
            torch.save(Generator.fid_noise, noise_path)
            logger.info(f"generator noise saved to file with shape {Generator.fid_noise.shape}")
        return Generator.fid_noise

    def finish_generation(self, **kwargs):
        if kwargs.get("calc_fid") and (config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score):
            self.calc_fid()
        if config.evolution.fitness.generator == "FID":
            self.fitness_values.append(self.fid_score)
        elif config.evolution.fitness.generator == "skill_rating":
            self.fitness_values.append(-self.skill_rating.mu) #+ 2*self.skill_rating.phi
        elif config.evolution.fitness.generator == "random":
            self.fitness_values = [np.random.rand()]

    def calc_fid(self):
        noise = self.generate_common_noise()
        self.fid_score = generative_score.fid_images(GeneratorDataset(self, noise=noise), size=noise.shape[0])


class GeneratorDataset(Dataset):
    def __init__(self, generator, size=None, batch_size=None, noise=None):
        self.batch_size = batch_size or config.evolution.fitness.fid_batch_size
        self.default_noise = noise
        self._noise = self.default_noise
        self.generator = generator
        self.size = size if self._noise is None else len(self._noise)
        self.images = []
        self.index = 0

    def __len__(self):
        return self.size

    @property
    def noise(self):
        if self.default_noise is not None:
            self._noise = self.default_noise
        if self._noise is None:
            with torch.no_grad():
                self._noise = self.generator.generate_noise(self.batch_size)
        return self._noise

    def __getitem__(self, idx):
        if len(self.images) <= self.index:
            self.index = 0
            self._noise = None
            with torch.no_grad():
                self.images = self.generator(self.noise.cpu()).detach()
        image = self.images[self.index]
        self.index += 1
        return image, np.array([])
