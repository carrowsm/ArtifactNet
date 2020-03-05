"""
This script trains a cycleGAN neural network to remove dental artifacts (DA)
from RadCure CT image volumes.

"""



import os
from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config import get_args

from data.data_loader import load_img_names, RadiomicsDataset

from models.generators import UNet3D
from models.discriminators import PatchGAN_3D







""" MAIN PYTORCH-LIGHTNING MODULE """
class GAN(pl.LightningModule) :

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        ### Initialize Networks ###
        # generator_y maps X -> Y and generator_x maps Y -> X
        self.generator_y = self.UNet3D(in_channels=1, out_channels=1, init_features=64)
        self.generator_x = self.UNet3D(in_channels=1, out_channels=1, init_features=64)

        # One discriminator to identify real DA+ images, another for DA- images
        self.discriminator_y = self.PatchGAN_3D(input_channels=1, out_size=1, n_filters=n)
        self.discriminator_x = self.PatchGAN_3D(input_channels=1, out_size=1, n_filters=n)
        ### ------------------- ###


        # Get train and test data sets
        self.train_files, self.test_files = load_img_names(self.hparams.img_dir,
                                               data_augmentation_factor=hparams.augmentation_factor,
                                               test_size=0.1)


        # Define loss functions
        self.l1_loss  = nn.L1Loss(reduction="mean")
        self.bce_loss = nn.BCELoss()



    @pl.data_loader
    def train_dataloader(self):
        # Load transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])

        # Load the dataset. This is a list of file names
        dataset = RadiomicsDataset(self.hparams.img_dir,   # Path to images
                                   self.train_files,       # Ordered list of image pair file names
                                   train=True,
                                   transform=None,
                                   test_size=0.1)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=True, num_workers=10)   # Load 10 batches in paralllel

        return data_loader
