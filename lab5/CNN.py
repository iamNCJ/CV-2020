from typing import List, Union, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch import nn

from DataModule import DataModule
from torchvision.datasets import CIFAR10


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'])  #VGG8
        # make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], True)  # VGG19
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
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
    LeNet = VGGNet()
    try:
        trainer = pl.Trainer(gpus=-1)
    except pl.utilities.exceptions.MisconfigurationException:
        trainer = pl.Trainer(gpus=0)
    trainer.fit(LeNet, dm)
    trainer.test(datamodule=dm)
