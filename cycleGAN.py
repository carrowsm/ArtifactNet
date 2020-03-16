"""
This script trains a cycleGAN neural network to remove dental artifacts (DA)
from RadCure CT image volumes.

"""



import os
import itertools
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

from data.data_loader import load_image_data_frame, load_img_names, UnpairedDataset

from models.generators import UNet3D
from models.discriminators import PatchGAN_3D



"""
This script trains a cycleGAN to remove dental artifacts from radcure images
using the pytorch-lightning framework. To run the script, just run python cycleGAN.py
on a GPU node on H4H.

Papers
-  https://junyanz.github.io/CycleGAN/
-  https://www.researchgate.net/publication/337386794_Three-dimensional_Generative_
   Adversarial_Nets_for_Unsupervised_Metal_Artifact_Reduction

CycleGAN Architecture:
---------------------
The aim of this netowrk is to map images from domain X (DA+) to images from
domain Y (DA-). Images from these domains are unpaired and do not have to be the
same length. For example:
X = {x_0, x_1, ..., x_N} and Y = {x_0, x_1, ..., x_M}
where x_i and y_i do not come from the same patient.

The network consits of two generators G_X and G_Y which attempt to perform the
mappings G_Y : X -> Y and G_X : Y -> X. Each of these is a 3D U-Net.

We also have two discriminators D_X and D_Y which try to real from fake images
in each class. For these we use the patchGAN discriminator architecture.
"""


""" MAIN PYTORCH-LIGHTNING MODULE """
class GAN(pl.LightningModule) :

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        ### Initialize Networks ###
        # generator_y maps X -> Y and generator_x maps Y -> X
        self.g_y = UNet3D(in_channels=1, out_channels=1, init_features=64)
        self.g_x = UNet3D(in_channels=1, out_channels=1, init_features=64)

        # One discriminator to identify real DA+ images, another for DA- images
        self.d_y = PatchGAN_3D(input_channels=1, out_size=1, n_filters=64)
        self.d_x = PatchGAN_3D(input_channels=1, out_size=1, n_filters=64)
        ### ------------------- ###


        # Get train and test data sets
        # Import CSV containing DA labels
        y_df, n_df = load_image_data_frame(hparams.csv_path)

        # Create train and test sets for each DA+ and DA- imgs
        files = load_img_names(hparams.img_dir,
                               y_da_df=y_df, n_da_df=n_df,
                               f_type="npy", suffix="_img",
                               data_augmentation_factor=hparams.augmentation_factor,
                               test_size=0.25)
        self.y_train, self.n_train, self.y_test, self.n_test = files
        """
        y == DA+, n == DA-
        y_train is a matrix with len N and two columns:
        y_train[:, 0] = z-index of DA in each patient
        y_train[:, 1] = full path to each patient's image file
        """



        # Define loss functions
        self.l1_loss  = nn.L1Loss(reduction="mean")
        self.mse_loss = nn.MSELoss()



    @pl.data_loader
    def train_dataloader(self):
        # Load transforms
        # transform = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize([0.5], [0.5])])


        # Test data loader
        dataset = UnpairedDataset(self.y_train[ :, 1],               # Paths to DA+ images
                                  self.n_train[ :, 1],               # Paths to DA- images
                                  file_type="npy",
                                  X_image_centre=self.y_train[:, 0], # DA slice index
                                  Y_image_centre=self.n_train[:, 0], # Mouth slice index
                                  image_size=[10, 500, 500],
                                  transform=None)

        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=True, num_workers=10)   # Load 10 batches in paralllel

        return data_loader



    """ Helper functions for forward pass """
    def forward(self, z, network="G_X"):
        # Perform a forward pass on the correct network
        if   network == "G_X" :
            return self.g_x.forward(z)
        elif network == "G_Y" :
            return self.g_y.forward(z)
        elif network == "D_X" :
            return self.d_x.forward(z)
        elif network == "D_Y" :
            return self.d_y.forward(z)
        else :
            raise ValueError("Argument 'netowrk' must be 'G_X', 'G_Y', 'D_X', or 'D_Y'.")


    """ Loss functions """
    def adv_loss(self, z_hat, z):
        return self.mse_loss(z_hat, z)

    def L1_loss(self, z_hat, z) :
        return self.l1_loss(z_hat, z)

    """ ############## """


    def training_step(self, batch, batch_nb, optimizer_idx) :
        self.zero_grad()            # Zero the gradients to prevent gradient accumulation

        # Get the images
        x, y = batch                # x == DA+ images, y == DA- images

        batch_size = x.size(0)

        # Create ground truth results
        zeros = torch.zeros(batch_size)
        ones = torch.ones(batch_size)


        # Generate some images
        gen_x = self.g_x(y)         # Generate fake DA+ images from some real DA- imgs
        gen_y = self.g_y(x)         # Generate fake DA- images from some real DA+ imgs

        ### TRAIN DISCRIMINATORS ###
        if optimizer_idx == 0 :
            # Forward pass through each discriminator
            # Run discriminator on generated images
            d_y_gen_y = self.d_y(gen_y.detach()).view(-1)
            d_x_gen_x = self.d_x(gen_x.detach()).view(-1)
            d_y_real = self.d_y(y).view(-1)
            d_y_real = self.d_x(x).view(-1)

            # Compute loss for each discriminator
            # Put labels on correct GPU
            if self.on_gpu:
                ones  =  ones.cuda(d_y_real.device.index)
                zeros = zeros.cuda(d_y_fake.device.index)
            # ----------------------- #
            loss_Dy = self.adv_loss(d_y_fake, zeros) + self.adv_loss(d_y_real, ones)

            # Put labels on correct GPU
            if self.on_gpu:
                ones  =  ones.cuda(d_x_real.device.index)
                zeros = zeros.cuda(d_x_fake.device.index)
            # ----------------------- #
            loss_Dx = self.adv_loss(d_x_fake, zeros) + self.adv_loss(d_x_real, ones)

            # Total discriminator loss is the sum of the two
            D_loss = loss_Dy + loss_Dx

            # Save the discriminator loss in a dictionary
            tqdm_dict = {'d_loss': D_loss}
            output = OrderedDict({'loss': D_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })

        ### TRAIN GENERATORS ###
        if optimizer_idx == 1 :
            # Run discriminator on generated images
            d_y_gen_y = self.d_y(gen_y).view(-1)
            d_x_gen_x = self.d_x(gen_x).view(-1)

            # Generate fake images from fake images (for cyclical loss)
            gen_x_gen_y = self.g_x(gen_y)
            gen_y_gen_x = self.g_y(gen_x)

            # Compute adversarial loss
            ones = ones.cuda(d_y_gen_y.device.index) if self.on_gpu else ones.cuda()
            loss_Gy = self.mse_loss(d_y_gen_y, ones)

            ones = ones.cuda(d_x_gen_x.device.index) if self.on_gpu else ones.cuda()
            loss_Gx = self.mse_loss(d_x_gen_x, ones)

            # Compute cyclic loss
            loss_cyc = self.l1_loss(gen_x_gen_y, x) + self.l1_loss(gen_y_gen_x, y)

            # Generator loss is the sum of these
            G_loss = loss_Gy + loss_Gx + loss_cyc

            # Save the discriminator loss in a dictionary
            tqdm_dict = {'d_loss': D_loss}
            output = OrderedDict({'loss': D_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })

        ### Log some sample images ###
        if batch_nb == 0 :
            grid = torchvision.utils.make_grid([x[0], y[0], gen_x[0], gen_y[0]])
            self.logger.experiment.add_image(f'imgs/epoch{self.current_epoch}', grid, 0)
        ### ---------------------- ###

        return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(itertools.chain(self.g_x.parameters(), self.g_y.parameters()), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(itertools.chain(self.g_x.parameters(), self.g_y.parameters()), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []







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
    logger  = TensorBoardLogger(hparams.log_dir, name="DA_Reduction_GAN")
    logger.log_hyperparams = dontloghparams
    trainer = pl.Trainer(logger=logger, # At the moment, PTL breaks when using builtin logger
                         max_nb_epochs=100,
                         distributed_backend="dp",
                         gpus=0
                         )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()



    main(hparams)
