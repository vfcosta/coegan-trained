from metrics.fid import fid_score
from metrics.fid.inception import InceptionV3
from util import tools
import logging
import torch
from util import config
import time
import os


base_fid_statistics = None
inception_model = None
logger = logging.getLogger(__name__)
use_cuda = True


def build_inception_model():
    """Build the inception model without normalization (the range is already [-1, 1]"""
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[config.evolution.fitness.fid_dimension]], normalize_input=False)


def initialize_fid(train_loader, size=1000):
    global base_fid_statistics, inception_model
    if inception_model is None:
        inception_model = build_inception_model()
    inception_model = tools.cuda(inception_model, use_cuda)

    if base_fid_statistics is None:
        logger.info("calculate base fid statistics: %d", size)
        base_fid_statistics = fid_score.calculate_activation_statistics(
            train_loader.dataset, inception_model, dims=config.evolution.fitness.fid_dimension, size=size,
            batch_size=config.evolution.fitness.fid_batch_size)
        inception_model.cpu()


def fid(generator, size=1000, noise=None):
    generator.eval()
    with torch.no_grad():
        if noise is None:
            noise = generator.generate_noise(size)
        generated_images = generator(noise.cpu()).detach()
        score = fid_images(generated_images)
    generator.zero_grad()
    return score


def fid_images(dataloader, size=1000):
    global base_fid_statistics, inception_model
    inception_model = tools.cuda(inception_model, use_cuda)
    start_time = time.time()
    m1, s1 = fid_score.calculate_activation_statistics(
        dataloader, inception_model, dims=config.evolution.fitness.fid_dimension, size=size,
        batch_size=config.evolution.fitness.fid_batch_size)
    print("FID: calc activation --- %s seconds ---" % (time.time() - start_time))
    inception_model.cpu()
    m2, s2 = base_fid_statistics
    ret = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    return ret


if __name__ == "__main__":
    # Test FID score using images from the input dataset
    from hydra.experimental import compose, initialize
    initialize(config_path="../conf", strict=True)
    import util
    util.config._config = compose("config.yaml")
    import numpy as np
    from evolution.gan_train import GanTrain
    from evolution.generator import Generator, GeneratorDataset
    base_path = os.path.join(os.path.dirname(__file__), "..")

    fid_stat = ""#os.path.join(base_path, "fid_stats_cifar10_train.npz")
    util.config.gan.dataset = "CIFAR10"
    logger.info(f"start gan train")
    train = GanTrain(log_dir="/tmp")
    fid_size = 10000

    if fid_stat and os.path.exists(fid_stat):
        f = np.load(fid_stat)
        m, s = f['mu'][:], f['sigma'][:]
        print("fid_stats_cifar10_train", m, s)
        base_fid_statistics = m, s
        f.close()
        inception_model = build_inception_model()
    else:
        logger.info(f"init fid")
        initialize_fid(train.train_loader, size=fid_size)
        m, s = base_fid_statistics
        print("calc fid stats", m, s)
        logger.info(f"finish fid")

    images = []
    generator_path = os.path.join(base_path, "./generator.pkl")
    if generator_path:
        generator = Generator.load(generator_path)
        dataset = GeneratorDataset(generator, size=fid_size)
    else:
        dataset = train.validation_loader.dataset
    logger.info("start fid %d", fid_size)
    ret = fid_images(dataset, size=fid_size)
    logger.info("FID %s", ret)
