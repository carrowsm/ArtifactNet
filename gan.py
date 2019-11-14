"""
To run this template just do:
python gan.py
After a few epochs, launch tensorboard to see the images being generated at every batch.
tensorboard --logdir default
"""

"""
This GAN uses a U-Net generator to take a 2D CT image with metal artifact
streaks and tranforms it to an image with no artifacts.
The discriminator is a CNN which attempts to distinguish between a real
artifact-free image and the generated image.

This code is based off the Pytorch-Lightning GAN template:
https://github.com/williamFalcon/pytorch-lightning/blob/master/pl_examples/domain_templates/gan.py

"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config import get_args

from data_loader import load_img_names, RadiomicsDataset




class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # Initialize networks
        data_shape = (1, 1, 300, 300) # Put this in a config file
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        # Get train and test data sets
        train_files, test_files = load_img_names(self.hparams.img_dir,
                                               data_augmentation_factor=hparams.augmentation_factor)


    @pl.data_loader
    def train_dataloader(self):
        # Load transforms
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize([0.5], [0.5])])

        # Load the dataset. This is a list of file names
        dataset = RadiomicsDataset(self.hparams.img_dir,   # Path to images
                                   self.train_files,       # Ordered list of image pair file names
                                   train=True,
                                   transform=None,
                                   test_size=0.1))
        return DataLoader(dataset, batch_size=self.hparams.batch_size)



    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_i == 0:
            # Get image with artifacts
            z =
            # z = torch.randn(imgs.shape[0], self.hparams.latent_dim)

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self.forward(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_i == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []



    def on_epoch_end(self):
        z = torch.randn(8, self.hparams.latent_dim)
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer()

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()

    main(hparams)
