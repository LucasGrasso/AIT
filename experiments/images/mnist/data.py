import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_loaders(batch_size, subset=None, seed=0, root=".data/mnist"):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),  # (x - μ) / σ
        ]
    )
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    if subset:
        train = Subset(train, range(subset))
        test = Subset(test, range(min(subset, len(test))))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train,
            batch_size,
            shuffle=True,
            drop_last=True,
            generator=g,
        ),
        DataLoader(
            test,
            batch_size,
            shuffle=False,
            drop_last=False,
        ),
    )
