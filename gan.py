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

This script trains on paired 2D images of DA+ and DA- shepps-logan
CT phantoms.

To run the script, just login to H4H and run:
$ sbatch /cluster/home/carrowsm/config/artifact_net/remove_artifacts.sh
"""
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config import get_args

from data_loader import load_img_names, RadiomicsDataset

from models.pix2pix import Generator, Discriminator





class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # Initialize networks
        data_shape = (1, 1, 300, 300) # Put this in a config file
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(output_dim=1)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        # Get train and test data sets
        self.train_files, self.test_files = load_img_names(self.hparams.img_dir,
                                               data_augmentation_factor=hparams.augmentation_factor,
                                               test_size=0.1)

        self.criterion = nn.BCELoss()
        self.l1 = nn.L1Loss(reduction="mean")


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


    # Helper funtions for forward-pass

    def forward(self, z, network="g"):
        # Generator forward pass
        if network == "g" :
            return self.generator.forward(z)

        # Disciminiator forward pass
        elif network == "d" :
            return self.discriminator.forward(z)
        else :
            return None


    def adversarial_loss(self, y_hat, y):
        # return F.binary_cross_entropy(y_hat, y)
        return self.criterion(y_hat, y)
    def L1_loss(self, X_hat, X) :
        return self.l1(X_hat, X)


    def training_step(self, batch, batch_nb, optimizer_idx):

        self.zero_grad()

        a_img, no_a_img = batch

        batch_size = a_img.size(0)

        # Create ground truth result
        fake_labels = torch.zeros(batch_size)
        real_labels = torch.ones(batch_size)

        # Generate some artifact-free images
        gen_img = self(a_img, network="g")


        ### TRAIN DISCRIMINATOR ###
        if optimizer_idx == 1 :

            D_real = self(no_a_img, network="d").view(-1)          # Does it detect real images?
            D_fake = self(gen_img.detach(), network="d").view(-1)  # Does it detect fake images?


            # Put labels on correct GPU
            if self.on_gpu:
                fake_labels = fake_labels.cuda(D_fake.device.index)
                real_labels = real_labels.cuda(D_real.device.index)
            # ----------------------- #

            D_real_loss = self.adversarial_loss(D_real, real_labels)
            D_fake_loss = self.adversarial_loss(D_fake, fake_labels)

            D_loss = (D_real_loss + D_fake_loss)

            # Save the generator loss in a dictionary
            tqdm_dict = {'d_loss': D_loss}
            output = OrderedDict({'loss': D_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })
        ### TRAIN GENERATOR ###
        if optimizer_idx == 0 :
            D_fake = self(gen_img, network="d").view(-1)

            # Put labels on correct GPU
            if self.on_gpu:
                real_labels = real_labels.cuda(D_fake.device.index)
            # ----------------------- #

            G_loss = self.adversarial_loss(D_fake, real_labels)

            L_loss = self.L1_loss(gen_img, no_a_img)

            G_loss = L_loss + G_loss

            tqdm_dict = {'g_loss': G_loss}
            output = OrderedDict({'loss': G_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })

        self.last_imgs = [a_img, no_a_img]
        return output




    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        # opt_g = torch.optim.Adam([{'params': self.generator.parameters()}], lr=lr, betas=(b1, b2))
        # opt_d = torch.optim.Adam([{'params': self.discriminator.parameters()}], lr=lr, betas=(b1, b2))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


    def on_epoch_end(self):
        a_img, no_a_img = self.last_imgs[0], self.last_imgs[1]

        # log sampled images (with most recent batch img)
        gen_imgs = self.forward(a_img)
        grid1 = torchvision.utils.make_grid(a_img[0:3])
        grid2 = torchvision.utils.make_grid(no_a_img[0:3])
        grid3 = torchvision.utils.make_grid(gen_imgs[0:3])
        self.logger.experiment.add_image(f'imgs/epoch{self.current_epoch}/orig_w_artifact', grid1, 0)
        self.logger.experiment.add_image(f'imgs/epoch{self.current_epoch}/orig_n_artifact', grid2, 0)
        self.logger.experiment.add_image(f'imgs/epoch{self.current_epoch}/generated', grid3, 0)
        # pass


def dontloghparams(x) :
    """Hacky workaround to hparam logging being
    broken in pytorch ligthning"""
    # TO fix this, write custom logger
    return


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    logger  = TensorBoardLogger(hparams.logdir, name="DA_Reduction_GAN")
    logger.log_hyperparams = dontloghparams
    trainer = pl.Trainer(logger=logger, # At the moment, PTL breaks when using builtin logger
                         max_nb_epochs=100,
                         distributed_backend="dp",
                         gpus=[0]
                         )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()



    main(hparams)
