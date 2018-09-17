import os
from torchvision import datasets
from torchvision.transforms import (
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomResizedCrop,
    CenterCrop,
    RandomCrop,
    transforms
)
from base import BaseDataLoader, AutoAugmentDataset


def get_dataloader_instance(dataloader_name, config):
    try:
        dataloader = eval(dataloader_name)
    except NameError:
        raise NameError("Dataloader '{dataloader}' not found.".format(dataloader=dataloader_name))

    dataloader_instance = dataloader(config)

    return dataloader_instance


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loader
    """
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    # TODO add more augmentations. Move to separate file
    all_augmentations = [
        RandomVerticalFlip(p=1),
        RandomHorizontalFlip(p=1),
        RandomResizedCrop(28),
        CenterCrop(28)
    ]

    def __init__(self, config):
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'mnist')
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.MNIST(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                all_augmentations=self.all_augmentations
            ),
            'test': datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.base_transforms
            )
        }
        super(MnistDataLoader, self).__init__(self.dataset, config)


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    # TODO add more augmentations. Move to separate file
    all_augmentations = [
        RandomVerticalFlip(p=1),
        RandomHorizontalFlip(p=1),
        RandomResizedCrop(32),
        RandomCrop(32, padding=4),
        CenterCrop(32)
    ]

    def __init__(self, config):
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                all_augmentations=self.all_augmentations
            ),
            'test': datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=self.base_transforms
            )
        }
        super(CIFAR10DataLoader, self).__init__(self.dataset, config)
