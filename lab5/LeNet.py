import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LeNet5(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Layer C1 is a convolution layer with six convolution kernels of 5x5
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            # Layer S2 is the subsampling/pooling layer that outputs 6 feature graphs of size 14x14
            nn.MaxPool2d(2, stride=2),
            # Layer C3 is a convolution layer with 16 5-5 convolution kernels
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # Layer S4 is similar to S2, with size of 2x2 and output of 16 5x5 feature graphs
            nn.MaxPool2d(2, stride=2),
            # Layer C5 is a convolution layer with 120 convolution kernels of size 5x5
            nn.Conv2d(16, 120, 5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            # F6 layer is fully connected to C5, and 84 feature graphs are output
            nn.Linear(3000, 84),
            nn.ReLU(),
            # Output
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        result = self.network(x)
        return result

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # split dataset
        if stage == 'fit':
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test':
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test


if __name__ == '__main__':
    dm = MNISTDataModule()

    # init model
    LeNet = LeNet5()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer(auto_select_gpus=True, max_epochs=10)
    trainer.fit(LeNet, dm)
    trainer.test(datamodule=dm)
