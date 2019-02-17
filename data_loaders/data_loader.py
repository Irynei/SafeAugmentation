import os
from torchvision import datasets
from albumentations import Normalize, Resize
from base import (
    BaseDataLoader,
    AutoAugmentDataset,
    AlbumentationsDataset,
    AlbumentationsDatasetV2,
    AutoAugmentDatasetByGoogle
)
from augmentations.augmentation import (
    good_policies,
    get_good_policy_svhn,
    get_strong_augmentations,
    get_good_augmentations
)
from utils.util import download_and_unzip, create_val_folder
import augmentations.augmentation_autoaugment as augmentation_transforms


def get_dataloader_instance(dataloader_name, config):

    if dataloader_name == 'CIFAR10DataLoader':
        dataloader = CIFAR10DataLoader
    elif dataloader_name == 'SVHNDataLoader':
        dataloader = SVHNDataLoader
    elif dataloader_name == 'SVHNDataLoaderImageClassification':
        dataloader = SVHNDataLoaderImageClassification
    elif dataloader_name == 'CIFAR10DataLoaderImageClassification':
        dataloader = CIFAR10DataLoaderImageClassification
    elif dataloader_name == 'CIFAR10DataLoaderImageClassificationSafeAugment':
        dataloader = CIFAR10DataLoaderImageClassificationSafeAugment
    elif dataloader_name == 'CIFAR100DataLoaderImageClassification':
        dataloader = CIFAR100DataLoaderImageClassification
    elif dataloader_name == 'CIFAR10DataLoaderImageClassificationAutoAugmentByGoogle':
        dataloader = CIFAR10DataLoaderImageClassificationAutoAugmentByGoogle
    elif dataloader_name == 'SVHNDataLoaderImageClassificationAutoAugmentByGoogle':
        dataloader = SVHNDataLoaderImageClassificationAutoAugmentByGoogle
    elif dataloader_name == 'CIFAR100DataLoaderImageClassificationAutoAugmentByGoogle':
        dataloader = CIFAR100DataLoaderImageClassificationAutoAugmentByGoogle
    elif dataloader_name == 'TinyImageNetDataLoader':
        dataloader = TinyImageNetDataLoader
    elif dataloader_name == 'TinyImageNetDataLoaderImageClassification':
        dataloader = TinyImageNetDataLoaderImageClassification
    elif dataloader_name == 'TinyImageNetDataLoaderImageClassificationV2':
        dataloader = TinyImageNetDataLoaderImageClassificationV2
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
        create_val_folder(os.path.join(self.data_dir, 'tiny-imagenet-200'))
        self.dataset = {
            'train': AutoAugmentDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/train')),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                max_size=self.max_size
            ),
            'test': AutoAugmentDataset(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/val/images')),
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
        create_val_folder(os.path.join(self.data_dir, 'tiny-imagenet-200'))
        augmentations = get_good_augmentations(width=self.image_size[0], height=self.image_size[1])
        print('Using good augmentations')
        self.max_size = int(config['augmentation']['max_size'])
        self.base_transforms.append(Resize(width=self.image_size[0], height=self.image_size[1]))
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/train')),
                base_transforms=self.base_transforms,
                augmentations=augmentations,
                max_size=self.max_size
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/val/images')),
                base_transforms=self.base_transforms,
                augmentations=[],
            )
        }
        super(TinyImageNetDataLoaderImageClassification, self).__init__(self.dataset, config)


class TinyImageNetDataLoaderImageClassificationV2(BaseDataLoader):

    filename = "tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_transforms = [
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    def __init__(self, config):
        self.image_size = (64, 64)
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'tiny_imagenet')
        download_and_unzip(self.url, self.filename, self.data_dir)
        create_val_folder(os.path.join(self.data_dir, 'tiny-imagenet-200'))
        augmentations = get_strong_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.base_transforms.append(Resize(width=self.image_size[0], height=self.image_size[1]))
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/train')),
                base_transforms=self.base_transforms,
                augmentations=augmentations,
                max_size=self.max_size
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.ImageFolder(os.path.join(self.data_dir, 'tiny-imagenet-200/val/images')),
                base_transforms=self.base_transforms,
                augmentations=[],
            )
        }
        super(TinyImageNetDataLoaderImageClassificationV2, self).__init__(self.dataset, config)


class CIFAR10DataLoaderImageClassification(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        augmentations = get_good_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                augmentations=augmentations,
                max_size=self.max_size,
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR10(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                augmentations=[],
                train=False
            )
        }
        super(CIFAR10DataLoaderImageClassification, self).__init__(self.dataset, config)


class CIFAR10DataLoaderImageClassificationSafeAugment(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        self.augmentations = get_good_augmentations(width=self.image_size[0], height=self.image_size[1])
        #self.base_transforms.append(Resize(width=self.image_size[0], height=self.image_size[1]))
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                augmentations=self.augmentations,
                max_size=self.max_size,
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR10(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                augmentations=[],
                train=False
            )
        }
        super(CIFAR10DataLoaderImageClassificationSafeAugment, self).__init__(self.dataset, config)


class CIFAR100DataLoaderImageClassification(BaseDataLoader):
    """
    CIFAR100 data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        augmentations = get_good_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR100(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                augmentations=augmentations,
                max_size=self.max_size
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.CIFAR100(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                augmentations=[],
                train=False
            )
        }
        super(CIFAR100DataLoaderImageClassification, self).__init__(self.dataset, config)


class CIFAR10DataLoaderImageClassificationAutoAugmentByGoogle(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        policies = good_policies()
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar10')
        self.dataset = {
            'train': AutoAugmentDatasetByGoogle(
                dataset=datasets.CIFAR10(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                policies=policies
            ),
            'test': AutoAugmentDatasetByGoogle(
                dataset=datasets.CIFAR10(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                policies=[],
                train=False
            )
        }
        super(CIFAR10DataLoaderImageClassificationAutoAugmentByGoogle, self).__init__(self.dataset, config)

class CIFAR100DataLoaderImageClassificationAutoAugmentByGoogle(BaseDataLoader):
    """
    CIFAR100 data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        policies = good_policies()
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'cifar100')
        self.dataset = {
            'train': AutoAugmentDatasetByGoogle(
                dataset=datasets.CIFAR100(self.data_dir, train=True, download=True),
                base_transforms=self.base_transforms,
                policies=policies
            ),
            'test': AutoAugmentDatasetByGoogle(
                dataset=datasets.CIFAR100(self.data_dir, train=False, download=True),
                base_transforms=self.base_transforms,
                policies=[],
                train=False
            )
        }
        super(CIFAR100DataLoaderImageClassificationAutoAugmentByGoogle, self).__init__(self.dataset, config)

class SVHNDataLoaderImageClassificationAutoAugmentByGoogle(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    base_transforms = [
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        policies = get_good_policy_svhn()
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'SVHN')
        self.dataset = {
            'train': AutoAugmentDatasetByGoogle(
                dataset=datasets.SVHN(self.data_dir, split='train', download=True),
                base_transforms=self.base_transforms,
                policies=policies
            ),
            'test': AutoAugmentDatasetByGoogle(
                dataset=datasets.SVHN(self.data_dir, split='test', download=True),
                base_transforms=self.base_transforms,
                policies=[],
                train=False
            )
        }
        super(SVHNDataLoaderImageClassificationAutoAugmentByGoogle, self).__init__(self.dataset, config)

class SVHNDataLoaderImageClassification(BaseDataLoader):
    """
    SVHN data loader
    """
    base_transforms = [
        Normalize(augmentation_transforms.MEANS, augmentation_transforms.STDS),
    ]

    def __init__(self, config):
        self.image_size = (32, 32)
        augmentations = get_good_augmentations(width=self.image_size[0], height=self.image_size[1])
        self.data_dir = os.path.join(config['data_loader']['data_dir'], 'SVHN')
        self.max_size = int(config['augmentation']['max_size'])
        self.dataset = {
            'train': AlbumentationsDatasetV2(
                dataset=datasets.SVHN(self.data_dir, split='train', download=True),
                base_transforms=self.base_transforms,
                augmentations=augmentations,
                max_size=self.max_size,
            ),
            'test': AlbumentationsDatasetV2(
                dataset=datasets.SVHN(self.data_dir, split='test', download=True),
                base_transforms=self.base_transforms,
                augmentations=[],
                train=False
            )
        }
        super(SVHNDataLoaderImageClassification, self).__init__(self.dataset, config)

