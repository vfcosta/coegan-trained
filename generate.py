import os
from hydra.experimental import compose, initialize
initialize(config_path="./conf")
import util
util.config._config = compose("config.yaml")

import torch
from torchvision.utils import save_image
from evolution.generator import Generator, GeneratorDataset
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def store_from_dataset(dataset, path, batch_size=50):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    i = 0
    for batch, _ in tqdm(dataloader):
        for image in batch:
            save_image((image + 1) / 2, os.path.join(path, f"image_{i}.png"))
            i += 1


def main(samples=100, output="output"):
    path = os.path.join(os.path.dirname(__file__), "generator.pkl")
    generator = Generator.load(path)
    generator_dataset = GeneratorDataset(generator, size=samples)
    os.makedirs(output, exist_ok=True)
    store_from_dataset(generator_dataset, output)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", type=str, default="output", help='Output Path')
    parser.add_argument("-s", "--samples", default=100, type=int, help='Number of samples to generate')
    args = parser.parse_args()
    print(args)
    main(samples=args.samples, output=args.output)
