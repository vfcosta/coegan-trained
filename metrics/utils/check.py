import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from metrics.torch_fid.fid_score import ImagePathDataset


if __name__ == "__main__":
    transform_arr = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    # transform_arr = [transforms.ToTensor()]
    transform = transforms.Compose(transform_arr)
    dataset = dsets.CIFAR10("../data", download=True, transform=transform)

    img, _ = next(iter(dataset))
    img = (img + 1) / 2
    print(img)
    print(_)
    save_image(img, "test.png")
    dataset = ImagePathDataset(["test.png"], transforms=transforms.ToTensor())
    img2 = next(iter(dataset))
    print(img2)
    print(img.equal(img2))
    print(img.mean(), img2.mean())
