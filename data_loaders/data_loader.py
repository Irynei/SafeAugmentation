import os
from torchvision import datasets
from albumentations import Normalize, Resize
from base import (
    BaseDataLoader,
    AutoAugmentDataset,
    AlbumentationsDataset
)
from augmentations.augmentation import (
    get_strong_augmentations,
)
from utils.util import download_and_unzip


def get_dataloader_instance(dataloader_name, config):

    if dataloader_name == 'CIFAR10DataLoader':
        dataloader = CIFAR10DataLoader
    elif dataloader_name == 'SVHNDataLoader':
        dataloader = SVHNDataLoader
    elif dataloader_name == 'CIFAR10DataLoaderImageClassification':
        dataloader = CIFAR10DataLoaderImageClassification
    elif dataloader_name == 'TinyImageNetDataLoader':
        dataloader = TinyImageNetDataLoader
    elif dataloader_name == 'TinyImageNetDataLoaderImageClassification':
        dataloader = TinyImageNetDataLoaderImageClassification
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
                max_size=self.max_size,
                train=True
            ),
            'test': AutoAugmentDataset(
                dataset=datasets.CIFAR10(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                train=False
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
            'test': AutoAugmentDataset(
                dataset=datasets.SVHN(self.data_dir, split='test', download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                train=False
            )
        }
        super(SVHNDataLoader, self).__init__(self.dataset, config)


class TinyImageNetDataLoader(BaseDataLoader):

    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_transforms = [
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    def __init__(self, config):
        self.image_size = (64, 64)
        self.augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.base_transforms.append(Resize(width=self.image_size[0], height=self.image_size[1]))
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'tiny_imagenet')
        self.max_size = int(config['augmentation']['max_size'])
        download_and_unzip(self.url, self.filename, self.data_dir)
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/train')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            ),
            'val': AutoAugmentDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/val')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            ),
            'test': AutoAugmentDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/test')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                train=False
            )
        }
        super(TinyImageNetDataLoader, self).__init__(self.dataset, config)


class TinyImageNetDataLoaderImageClassification(BaseDataLoader):

    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_transforms = [
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    def __init__(self, config):
        self.image_size = (64, 64)
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'tiny_imagenet')
        download_and_unzip(self.url, self.filename, self.data_dir)
        augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.augm_index = int(config['augmentation']['index'])
        self.base_transforms.append(Resize(width=self.image_size[0], height=self.image_size[1]))
        try:
            self.augmentations = [augmentations[self.augm_index]]
        except IndexError:
            self.augmentations = []
        self.dataset = {
            'train': AlbumentationsDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/train')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            ),
            'val': AlbumentationsDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/val')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            ),
            'test': AlbumentationsDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/test')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            )
        }
        super(TinyImageNetDataLoaderImageClassification, self).__init__(self.dataset, config)


class CIFAR10DataLoaderImageClassification(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.augm_index = int(config['augmentation']['index'])
        try:
            self.augmentations = [augmentations[self.augm_index]]
        except IndexError:
            self.augmentations = []
        self.dataset = {
            'train': AlbumentationsDataset(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            ),
            'test': AlbumentationsDataset(
                dataset=datasets.CIFAR10(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
            )
        }
        super(CIFAR10DataLoaderImageClassification, self).__init__(self.dataset, config)
