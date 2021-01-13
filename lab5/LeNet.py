import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class LeNet5(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),  # Layer C1 is a convolution layer with six convolution kernels of 5x5
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # Layer S2 is the subsampling/pooling layer that outputs 6 feature graphs of size 14x14
            nn.Conv2d(6, 16, 5),  # Layer C3 is a convolution layer with 16 5-5 convolution kernels
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 120, 5, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3000, 84),
            nn.ReLU(),
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


if __name__ == '__main__':
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

    # init model
    LeNet = LeNet5()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer()
    trainer.fit(LeNet, train_loader)
