"""
This script trains a cycleGAN neural network to remove dental artifacts (DA)
from RadCure CT image volumes.

To begin training this model on H4H, run
$ sbatch train_cycleGAN.sh
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

from config.options import get_args

from data.data_loader import load_image_data_frame, load_img_names, UnpairedDataset

from models.generators import UNet3D, ResNetK, UNet2D
from models.discriminators import PatchGAN_3D, CNN_3D, PatchGAN_NLayer, CNN_NLayer, CNN_2D, VGG2D
from util.helper_functions import set_requires_grad
from util.loggers import TensorBoardCustom

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

"""
This script trains a cycleGAN to remove dental artifacts from radcure images
using the pytorch-lightning framework. To run the script, just run python cycleGAN.py
on a GPU node on H4H.

Papers
-  https://junyanz.github.io/CycleGAN/
-  https://arxiv.org/pdf/1911.08105.pdf

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

    def __init__(self, hparams, image_size=[1, 256, 256]):
        super(GAN, self).__init__()
        self.hparams = hparams
        self.image_size = image_size

        ### Initialize Networks ###
        # generator_y maps X -> Y and generator_x maps Y -> X
        self.g_y = UNet2D(in_channels=image_size[0], out_channels=image_size[0], init_features=64)
        self.g_x = UNet2D(in_channels=image_size[0], out_channels=image_size[0], init_features=64)

        # One discriminator to identify real DA+ images, another for DA- images
        self.d_y = CNN_2D(in_channels=image_size[0], out_channels=1)
        self.d_x = CNN_2D(in_channels=image_size[0], out_channels=1)
        ### ------------------- ###

        # Put networks on GPUs
        self.gpu_check()

        # Get train and test data sets
        # Import CSV containing DA labels
        y_df, n_df = load_image_data_frame(hparams.csv_path)

        # Create train and test sets for each DA+ and DA- imgs
        files = load_img_names(hparams.img_dir,
                               y_da_df=y_df, n_da_df=n_df,
                               f_type="npy", suffix="",
                               data_augmentation_factor=hparams.augmentation_factor,
                               test_size=0.25)
        self.y_train, self.n_train, self.y_valid, self.n_valid = files
        """
        y == DA+, n == DA-
        y_train is a matrix with len N and two columns:
        y_train[:, 0] = z-index of DA in each patient
        y_train[:, 1] = full path to each patient's image file
        """

        # Define loss functions
        self.l1_loss  = nn.L1Loss(reduction="mean")
        # self.adv_loss = nn.MSELoss(reduction="mean")
        # self.adv_loss = nn.BCELoss()
        self.adv_loss = nn.BCEWithLogitsLoss() # Compatible with amp 16-bit

        # Define loss term coefficients
        self.lam = 10.0   # Coefficient for cycle consistency loss
        self.idt = 25.0   # Coefficient for identity loss




    def gpu_check(self) :
        if torch.cuda.is_available() :
            self.n_gpus = torch.cuda.device_count()
            print(f"{self.n_gpus} GPUs are available")

            for i in range(self.n_gpus) :
                device_name = torch.cuda.get_device_name(i)
                print(f"### Device {i}: ###")
                print("Name: ", device_name)
                nbytes = torch.cuda.memory_allocated(device=i)
                print("Memory allocated: ", nbytes)


    @pl.data_loader
    def train_dataloader(self):

        # Train data loader
        dataset = UnpairedDataset(self.y_train[ :, 1],           # Paths to DA+ images
                                  self.n_train[ :, 1],           # Paths to DA- images
                                  file_type="npy",
                                  # X_image_centre=self.y_train[:, 0], # DA slice index
                                  # Y_image_centre=self.n_train[:, 0], # Mouth slice index
                                  X_image_centre=None, # Imgs are preprocessed to be cropped
                                  Y_image_centre=None, # around DA
                                  image_size=self.image_size,
                                  transform=None,
                                  dim="2D"
                                  )

        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=True, num_workers=10, drop_last=True
                                 )
        self.dataset_size = len(dataset)

        return data_loader



    @pl.data_loader
    def val_dataloader(self) :
        # Validation data loader
        dataset = UnpairedDataset(self.y_valid[ :, 1],           # Paths to DA+ images
                                  self.n_valid[ :, 1],           # Paths to DA- images
                                  file_type="npy",
                                  # X_image_centre=self.y_valid[:, 0], # DA slice index
                                  # Y_image_centre=self.n_valid[:, 0], # Mouth slice index
                                  X_image_centre=None, # Imgs are preprocessed to be cropped
                                  Y_image_centre=None, # around DA
                                  image_size=self.image_size,
                                  transform=None,
                                  dim="2D"
                                  )
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=False, num_workers=10, drop_last=True
                                 )
        return data_loader


    """ Helper functions for forward pass """
    def forward(self, real_X, real_Y):
        # Perform a forward pass on the correct network
        """Run forward pass"""
        fake_Y = self.g_y(real_X)               # G_Y(X)
        cycl_X = self.g_x(fake_Y)               # G_X(G_Y(X))
        fake_X = self.g_x(real_Y)               # G_X(Y)
        cycl_Y = self.g_y(fake_X)               # G_Y(G_X(Y))
        return fake_Y, cycl_X, fake_X, cycl_Y


    def training_step(self, batch, batch_nb, optimizer_idx) :
        self.zero_grad()       # Zero the gradients to prevent gradient accumulation

        # Get the images
        x, y = batch           # x == DA+ images, y == DA- images

        batch_size = x.size(0)

        # Create ground truth results
        zeros = torch.zeros(batch_size)
        ones = torch.ones(batch_size)

        ### TRAIN GENERATORS ###
        if optimizer_idx == 0 :
            ### Calculate G_Y adversarial loss ###
            # Generate fake DA- images from real DA+ imgs
            gen_y = self.g_y(x)

            # Run discriminator on generated images
            d_y_gen_y = self.d_y(gen_y).view(-1)

            # Compute adversarial loss
            loss_adv_Y = self.adv_loss(d_y_gen_y, ones.to(d_y_gen_y.device))


            ### Calculate G_X adversarial loss ###
            # Generate fake DA- images from real DA+ imgs
            gen_x = self.g_x(y)

            # Run discriminator on generated images
            d_x_gen_x = self.d_x(gen_x).view(-1)

            # Compute adversarial loss
            loss_adv_X = self.adv_loss(d_x_gen_x, ones.to(d_x_gen_x.device))

            ### Compute Cycle Consistency Loss ###
            # Generate fake images from fake images (for cyclical loss)
            gen_x_gen_y = self.g_x(gen_y)     # fake DA+ from fake DA-
            gen_y_gen_x = self.g_y(gen_x)     # fake DA- from fake DA+

            # Compute cyclic loss
            loss_cyc_X = self.l1_loss(gen_x_gen_y, x) * self.lam
            loss_cyc_Y = self.l1_loss(gen_y_gen_x, y) * self.lam

            ### Compute Identidy loss ###
            loss_idt = (self.l1_loss(gen_y, x) + self.l1_loss(gen_x, y)) * self.idt

            # Generator loss is the sum of these
            G_loss = loss_adv_Y + loss_adv_X + loss_cyc_X + loss_cyc_Y + loss_idt



            # Save the discriminator loss in a dictionary
            tqdm_dict = {'g_loss': G_loss}
            output = OrderedDict({'loss': G_loss,
                                  'progress_bar': tqdm_dict,
                                  # 'log': tqdm_dict
                                  })
            self.logger.experiment.add_scalars(f'g_loss/', {'total': G_loss,
                                                           "idt": loss_idt,
                                                           "adv_Y": loss_adv_Y,
                                                           "adv_X": loss_adv_X,
                                                           "cyc_Y": loss_cyc_Y,
                                                           "cyc_X": loss_cyc_X},
                  batch_nb+(self.dataset_size*self.current_epoch // (batch_size*self.n_gpus)))

        ### TRAIN DISCRIMINATORS ###
        if optimizer_idx == 1 :
            # Generate some images
            gen_y = self.g_y(x) # fake DA- images from real DA+ imgs
            gen_x = self.g_x(y) # fake DA+ images from real DA- imgs

            # Forward pass through each discriminator
            # Run discriminator on generated images
            d_y_fake = self.d_y(gen_y.detach()).view(-1)
            d_x_fake = self.d_x(gen_x.detach()).view(-1)
            d_y_real = self.d_y(y).view(-1)
            d_x_real = self.d_x(x).view(-1)

            # Compute loss for each discriminator
            # Put labels on correct GPU
            ones  =  ones.to(d_y_real.device)
            zeros = zeros.to(d_y_fake.device)
            # ----------------------- #
            loss_Dy = self.adv_loss(d_y_fake, zeros) + self.adv_loss(d_y_real, ones)

            # Put labels on correct GPU
            ones  =  ones.to(d_x_real.device)
            zeros = zeros.to(d_x_fake.device)
            # ----------------------- #
            loss_Dx = self.adv_loss(d_x_fake, zeros) + self.adv_loss(d_x_real, ones)

            D_loss = (loss_Dy + loss_Dx)

            # Save the discriminator loss in a dictionary
            tqdm_dict = {'d_loss': D_loss}
            output = OrderedDict({'loss': D_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })

            # Plot loss every iteration
            self.logger.experiment.add_scalar(f'd_loss', D_loss,
            batch_nb+(self.dataset_size*self.current_epoch // (batch_size*self.n_gpus)))



        ### ---------------------- ###

        return output

    def validation_step(self, batch, batch_idx):
        # Get the images
        x, y = batch           # x == DA+ images, y == DA- images

        batch_size = x.size(0)

        # Create ground truth results
        zeros = torch.zeros(batch_size)
        ones = torch.ones(batch_size)

        gen_y = self.g_y(x)             # Generate DA- images from original DA+
        gen_x = self.g_x(y)             # Generate DA+ images from original DA-
        gen_x_gen_y = self.g_x(gen_y)   # Generate DA+ images from fake DA-
        gen_y_gen_x = self.g_y(gen_x)   # Generate DA- images from fake DA+

        ### DISCRIMINATOR LOSS ###
        # Forward pass through each discriminator
        # Run discriminator on generated images
        d_y_fake = self.d_y(gen_y).view(-1)
        d_x_fake = self.d_x(gen_x).view(-1)
        d_y_real = self.d_y(y).view(-1)
        d_x_real = self.d_x(x).view(-1)

        # Compute loss for each discriminator
        loss_Dy = self.adv_loss(d_y_fake, zeros.to(d_y_fake.device)) + \
                  self.adv_loss(d_y_real, ones.to(d_y_real.device))
        # ----------------------- #
        loss_Dx = self.adv_loss(d_x_fake, zeros.to(d_x_fake.device)) + \
                  self.adv_loss(d_x_real, ones.to(d_x_real.device))
        D_loss = (loss_Dy + loss_Dx)
        ### ------------------ ###

        ### -- GENERATOR LOSS -- ###
        # Compute adversarial loss
        loss_adv_X = self.adv_loss(d_x_fake, ones.to(d_x_fake.device))
        loss_adv_Y = self.adv_loss(d_y_fake, ones.to(d_y_fake.device))

        # Compute Cycle Consistency Loss
        loss_cyc_X = self.l1_loss(gen_x_gen_y, x) * self.lam
        loss_cyc_Y = self.l1_loss(gen_y_gen_x, y) * self.lam

        # Compute Identidy loss
        loss_idt = (self.l1_loss(gen_y, x) + self.l1_loss(gen_x, y)) * self.idt

        # Generator loss is the sum of these
        G_loss = loss_adv_Y + loss_adv_X + loss_cyc_X + loss_cyc_Y + loss_idt
        ### -- ------------- -- ###



        ### Log some sample images once per epoch ###
        if batch_idx == 0 :
            with torch.no_grad() :
                # Generate some fake images to plot
                images = [    x[0, self.image_size[0] // 2, :, :].cpu(),
                              y[0, self.image_size[0] // 2, :, :].cpu(),
                          gen_x[0, self.image_size[0] // 2, :, :].cpu(),
                          gen_y[0, self.image_size[0] // 2, :, :].cpu()]

            # Plot the image
            self.logger.add_mpl_img(f'imgs/epoch{self.current_epoch}', images, batch_idx)

        # Save the discriminator loss in a dictionary
        tqdm_dict = {'d_loss_val': D_loss,
                     'g_loss_val': G_loss}
        output = OrderedDict({'d_loss_val': D_loss,
                              'g_loss_val': G_loss,
                              'progress_bar': tqdm_dict,
                              'log': tqdm_dict
                              })

        # Plot loss every iteration
        b = self.current_epoch
        self.logger.experiment.add_scalar(f'd_loss_val', D_loss, b)
        self.logger.experiment.add_scalar(f'g_loss_val', G_loss, b)

        return output



    def configure_optimizers(self):
        """
        Define two optimizers (D & G), each with its own learning rate scheduler.
        """
        lr = self.hparams.lr
        G_lr = lr
        D_lr = lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(itertools.chain(self.g_x.parameters(),
                                self.g_y.parameters()), lr=G_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(itertools.chain(self.d_x.parameters(),
                                self.d_y.parameters()), lr=D_lr, betas=(b1, b2))

        # Decay generator learning rate by factor of 10 after 1 epoch
        # scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[2, 3, 4, 5, 6], gamma=0.5)
        # scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, 'min')
        # Increase discriminator learning rate by factor of 5 every epoch
        # scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[2, 3, 4, 5, 6], gamma=0.5)
        # scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, 'min')

        scheduler_g = {'scheduler': ReduceLROnPlateau(opt_g, 'min'),
                       'monitor': 'loss', # Default: val_loss
                       'interval': 'epoch',
                       'frequency': 1
                       }
        scheduler_d = {'scheduler': ReduceLROnPlateau(opt_d, 'min'),
                       'monitor': 'loss', # Default: val_loss
                       'interval': 'epoch',
                       'frequency': 1
                       }
        return [opt_g, opt_d], [scheduler_g, scheduler_d]



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams, image_size=[16, 256, 256])


    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # Custom logger defined in loggers.py
    logger = TensorBoardCustom(hparams.log_dir, name="16_256_256px_2D")

    # Main PLT training module
    trainer = pl.Trainer(logger=logger,
                         # accumulate_grad_batches=4,
                         # gradient_clip_val=0.9,
                         # max_nb_epochs=hparams.max_num_epochs,
                         val_percent_check=0.2,
                         amp_level='O1', precision=16, # Enable 16-bit presicion
                         gpus=hparams.n_gpus,
                         distributed_backend="ddp"
                         )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()



    main(hparams)
