import torch
import numpy as np
from torch.utils import data
from albumentations import Compose
from torchvision.transforms.functional import to_tensor


class AlbumentationsDataset(data.Dataset):
    """
    Dataset that works with transformations from albumentations lib

    """
    def __init__(self, dataset, base_transforms, augmentations):
        self.dataset = dataset
        self.base_transforms = base_transforms
        self.augmentations = augmentations

    def __getitem__(self, index):
        x, y = self.dataset[index]

        trfms = self.augmentations[:]
        trfms.extend(self.base_transforms)
        transforms = Compose(trfms)

        image_np = np.array(x)
        augmented = transforms(image=image_np)
        x = to_tensor(augmented['image'])

        return x, y

    def __len__(self):
        return len(self.dataset)
                                                                                            



class AutoAugmentDataset(data.Dataset):
    """
    Randomly applies subset of augmentations and set them as labels

    """
    def __init__(self, dataset, base_transforms, augmentations, max_size=7, train=True):
        self.dataset = dataset
        self.base_transforms = base_transforms
        self.augmentations = augmentations
        self.max_size = max_size
        self.train = train

    def __getitem__(self, index):
        x, y = self.dataset[index]

        trfms, labels = self.get_subset_of_transforms()
        trfms.extend(self.base_transforms)
        transforms = Compose(trfms)

        image_np = np.array(x)
        augmented = transforms(image=image_np)
        x = to_tensor(augmented['image'])
        y = torch.FloatTensor(labels)

        return x, y

    def __len__(self):
        return len(self.dataset)

    def get_subset_of_transforms(self):
        """
        in case of train dataset:
            Randomly get size of subset and then randomly choose subset of transformations
        in case of test dataset:
            Subset of transformations is always empty

        Returns:
            list of chosen transformations, one-hot-encoded labels

        """
        all_transforms_size = len(self.augmentations)

        if self.train:
            # size from 0 to max_size - 1
            subset_size = np.random.randint(0, self.max_size)
            all_transforms_idx = np.arange(all_transforms_size)
            # get random subset without duplicates
            np.random.shuffle(all_transforms_idx)
            transform_idx = all_transforms_idx[:subset_size]
            subset_transforms = [self.augmentations[i] for i in transform_idx]

            labels = np.zeros(all_transforms_size)
            labels[transform_idx] = 1
        else:
            # in case of test we do
            labels = np.zeros(all_transforms_size)
            subset_transforms = []
        return subset_transforms, labels
