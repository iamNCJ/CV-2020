import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class DataModule(pl.LightningDataModule):
    def __init__(self, datasets, batch_size=32, split_ratio=0.7):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.model_dir = os.getcwd() + '/datasets'
        self.split_ratio = split_ratio

    def prepare_data(self):
        self.datasets(self.model_dir, train=True, download=True)
        self.datasets(self.model_dir, train=False, download=True)

    def setup(self, stage):
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # split dataset
        if stage == 'fit':
            training_data = self.datasets(self.model_dir, train=True, transform=transform)
            training_data_count = int(len(training_data) * self.split_ratio)
            validating_data_count = len(training_data) - training_data_count
            self.training_data, self.validating_data = random_split(training_data,
                                                                    [training_data_count, validating_data_count])
        if stage == 'test':
            self.testing_data = self.datasets(self.model_dir, train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        training_data = DataLoader(self.training_data, batch_size=self.batch_size)
        return training_data

    def val_dataloader(self):
        validating_data = DataLoader(self.validating_data, batch_size=self.batch_size)
        return validating_data

    def test_dataloader(self):
        testing_data = DataLoader(self.testing_data, batch_size=self.batch_size)
        return testing_data
