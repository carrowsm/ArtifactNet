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
import time
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config import get_args

from data_loader import load_img_names, RadiomicsDataset

from models.pix2pix import Generator, Discriminator

from test_tube import Experiment




class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # Writer will output to ./runs/ directory by default
        self.logger = SummaryWriter(hparams.logdir)

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
                                   test_size=0.1)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                 shuffle=True, num_workers=10)   # Load 10 batches in paralllel

        return data_loader


    # Helper funtions for forward-pass
    # Generator forward pass
    def forward(self, z):
        return self.generator(z)
    # Disciminiator forward pass
    def discriminate(self, z) :
        return self.discriminator.forward(z)




    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        a_img, no_a_img = batch


        self.last_imgs = [a_img, no_a_img]

        ### TRAIN GENERATOR ###
        if optimizer_i == 0:
            # Get image and sinogram with artifacts
            # a_img    has shape (batch_size, channels (2 or 4), 300, 300)

            # generate images
            self.generated_imgs = self.forward(a_img)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            sample_imgs = [a_img[0], self.generated_imgs[0]]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.add_image(f'imgs/epoch{self.current_epoch}/batch_sample', grid, batch_nb)

            # Run the generated images through the discriminator to see how good they are
            gen_preds = self.discriminate(self.generated_imgs)

            # Create ground truth result (ie: all fake)
            valid = torch.ones(gen_preds.size(0), 1)
            # Put ground truth result on correct GPU
            if self.on_gpu:
                valid = valid.cuda(gen_preds.device.index)

            # Calculate Adversarial loss using BCE loss
            g_loss = self.adversarial_loss(gen_preds, valid)

            # Save the generator loss in a dictionary
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({'loss': g_loss,
                                  'progress_bar': tqdm_dict,
                                  'log': tqdm_dict
                                  })

            return output

        ### TRAIN DISCRIMINATOR ###
        if optimizer_i == 1:
            # Measure discriminator's ability to classify real from generated samples

            # Generate an image if previous there are none
            # (generator train step may have happened on another GPU)
            if self.generated_imgs is None :
                self.generated_imgs = self.forward(a_img)

            ### REAL IMAGE CLASSIFICATION ###
            # Give the discriminator the non-artifact imgs
            real_preds = self.discriminate(no_a_img)

            # Create ground truth result (ie: all fake)
            real = torch.ones(real_preds.size(0), 1)
            # Put ground truth result on correct GPU
            if self.on_gpu:
                real = real.cuda(real_preds.device.index)

            # Compute discriminator loss for real images
            real_loss = self.adversarial_loss(real_preds, real)
            ### ------------------------- ###

            ### FAKE IMAGE CLASSIFICATION ###
            fake_preds = self.discriminate(self.generated_imgs.detach()) # detach() detaches the
                                                                         # output from the computational
            # Create ground truth result (ie: all fake)                  # graph so we don't backprop
            fake = torch.zeros(fake_preds.size(0), 1)                    # through it.
            # Put ground truth result on correct GPU
            if self.on_gpu:
                fake = fake.cuda(fake_preds.device.index)

            # Compute discriminator loss for fake images
            fake_loss = self.adversarial_loss(fake_preds, fake)
            ### ------------------------- ###

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
        # a_img, no_a_img = self.last_imgs[0], self.last_imgs[1]
        #
        # # log sampled images (with most recent batch img)
        # gen_imgs = self.forward(a_img)
        # grid1 = torchvision.utils.make_grid(a_img[0:7])
        # grid2 = torchvision.utils.make_grid(no_a_img[0:7])
        # grid3 = torchvision.utils.make_grid(gen_imgs[0:7])
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_orig_w_artifact', grid1, 0)
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_orig_n_artifact', grid2, 0)
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_generated', grid3, 0)
        pass


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)
    exp = Experiment(save_dir=hparams.logdir)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = pl.Trainer(max_nb_epochs=10, gpus=hparams.ngpu, distributed_backend='ddp')
    # trainer = pl.Trainer(max_nb_epochs=10, , distributed_backend='ddp')
    trainer = pl.Trainer(exp, max_nb_epochs=100,
                         #amp_level='O2', use_amp=False,
                         distributed_backend='dp', gpus=[0, 1, 2, 3],
                         # distributed_backend='dp', gpus=[0, 1, 2, 3]
                         )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()



    main(hparams)
