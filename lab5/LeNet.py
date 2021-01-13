import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch import nn


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

    def compute(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
