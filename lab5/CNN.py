import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch import nn

from DataModule import DataModule
from torchvision.datasets import CIFAR10

class LeNet5(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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


if __name__ == '__main__':
    dm = DataModule(CIFAR10)
    LeNet = LeNet5()
    try:
        trainer = pl.Trainer(gpus=-1)
    except pl.utilities.exceptions.MisconfigurationException:
        trainer = pl.Trainer(gpus=0)
    trainer.fit(LeNet, dm)
    trainer.test(datamodule=dm)
