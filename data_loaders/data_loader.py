import os
from torchvision import datasets
from albumentations import Normalize
from base import (
    BaseDataLoader,
    AutoAugmentDataset
)
from augmentations.augmentation import (
    get_strong_augmentations,
)


def get_dataloader_instance(dataloader_name, config):

    if dataloader_name == 'CIFAR10DataLoader':
        dataloader = CIFAR10DataLoader
    elif dataloader_name == 'SVHNDataLoader':
        dataloader = SVHNDataLoader
    else:
        raise NameError("Dataloader '{dataloader}' not found.".format(dataloader=dataloader_name))

    dataloader_instance = dataloader(config)

    return dataloader_instance


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        self.augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                max_size=self.max_size
            ),
            'test': datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=self.base_transforms
            )
        }
        super(CIFAR10DataLoader, self).__init__(self.dataset, config)


class SVHNDataLoader(BaseDataLoader):
    """
    SVHN data loader
    """
    base_transforms = [
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        self.augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'SVHN')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.SVHN(self.data_dir, split='train', download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                max_size=self.max_size
            ),
            'test': datasets.SVHN(
                self.data_dir,
                split='test',
                download=True,
                transform=self.base_transforms
            )
        }
        super(SVHNDataLoader, self).__init__(self.dataset, config)
