import os
from hydra.experimental import compose, initialize
import torch
from torchvision.utils import save_image

initialize(config_path="../../conf", strict=True)
import util

util.config._config = compose("config.yaml")
from evolution.gan_train import GanTrain
from evolution.generator import Generator, GeneratorDataset

base_path = os.path.join(os.path.dirname(__file__), "..", "..")


def store_from_dataset(dataset, path, size, batch_size=50):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    i = 0
    for batch, _ in dataloader:
        for image in batch:
            save_image((image + 1) / 2, os.path.join(path, f"image_{i}.png"))
            i += 1
            if i >= size:
                return


def store(path, size, dataset_name="CIFAR10", generator_path=None):
    path = os.path.join(base_path, path)
    os.makedirs(path, exist_ok=True)
    if generator_path:
        generator = Generator.load(generator_path)
        generator_dataset = GeneratorDataset(generator, size=size)
        store_from_dataset(generator_dataset, path, size)
    else:
        util.config.gan.dataset = dataset_name
        train = GanTrain(log_dir="/tmp")
        dataset = train.train_loader.dataset
        store_from_dataset(dataset, path, size)


if __name__ == "__main__":
    size = 10000
    store("runs/cifar10_images", size)
    store("runs/cifar10_gen", size, generator_path=os.path.join(base_path, "./generator.pkl"))
