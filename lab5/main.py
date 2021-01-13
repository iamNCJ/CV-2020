import pytorch_lightning as pl
from LeNet import LeNet5
from DataModule import DataModule

from torchvision.datasets import MNIST

if __name__ == '__main__':
    dm = DataModule(MNIST)

    # init model
    LeNet = LeNet5()
    try:
        trainer = pl.Trainer(gpus=-1)
    except pl.utilities.exceptions.MisconfigurationException:
        trainer = pl.Trainer(gpus=0)
    trainer.fit(LeNet, dm)
    trainer.test(datamodule=dm)
