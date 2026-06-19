import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# per channel
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(
    batch_size, subset=None, seed=0, root=".data/cifar10", num_workers=4
):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    if subset:
        train = Subset(train, range(subset))
        test = Subset(test, range(min(subset, len(test))))
    g = torch.Generator().manual_seed(seed)
    persistent = num_workers > 0
    return (
        DataLoader(
            train,
            batch_size,
            shuffle=True,
            drop_last=True,
            generator=g,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent,
        ),
        DataLoader(
            test,
            batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent,
        ),
    )
