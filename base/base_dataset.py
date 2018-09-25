import torch
import numpy as np
from torch.utils import data
from albumentations import Compose
from torchvision.transforms.functional import to_tensor


class AutoAugmentDataset(data.Dataset):
    """
    Randomly applies subset of all_augmentations and set them as labels

    """
    def __init__(self, dataset, base_transforms, augmentations):
        self.dataset = dataset
        self.base_transforms = base_transforms
        self.augmentations = augmentations

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
        Randomly get size of subset and then randomly choose subset of transformations

        Returns:
            list of chosen transformations, one-hot-encoded labels

        """
        all_transforms_size = len(self.augmentations)
        subset_transforms = []

        # size from 1 to all_transforms_size - 1
        subset_size = np.random.randint(1, all_transforms_size)
        transform_idx = np.random.choice(all_transforms_size, subset_size)
        for i in transform_idx:
            subset_transforms.append(self.augmentations[i])

        labels = np.zeros(all_transforms_size)
        labels[transform_idx] = 1
        return subset_transforms, labels
