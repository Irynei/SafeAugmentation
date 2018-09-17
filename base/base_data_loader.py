import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base data loader
    Handles train/validation/test split.
    """
    def __init__(self, dataset, config, **kwargs):

        # train and test datasets
        self.train_dataset = dataset['train']
        self.test_dataset = dataset['test']

        # train data params
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.validation_split = config['validation']['validation_split']
        self.shuffle = config['data_loader']['shuffle']
        self.train_n_samples = len(self.train_dataset)
        self.train_sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
        }
        self.init_kwargs.update(kwargs)
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def __len__(self):
        """
        Returns:
            total number of batches in train dataset
        """
        return self.train_n_samples // self.batch_size

    def _split_sampler(self, split):
        """
        Splits train dataset into train and validation based on split ration.

        Args:
            split: ratio of validation set

        Returns:
            train data sampler, validation data sampler

        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.train_n_samples)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_valid = int(self.train_n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        self.shuffle = False
        self.train_n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def get_validation_loader(self):
        """
        Get validation data loader if validation split
        Returns:
            validation data loader or None

        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_test_loader(self):
        """
        Get test data loader.
        Returns:
            test data loader

        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
