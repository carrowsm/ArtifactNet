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

from models.pix2pix_homemade import Generator, Discriminator




class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # Writer will output to ./runs/ directory by default
        self.logger = SummaryWriter(hparams.logdir)

        # Initialize networks
        data_shape = (1, 1, 300, 300) # Put this in a config file
        self.generator = Generator(latent_dim=hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.generator.cuda()
        self.discriminator.cuda()

        self.discriminate = self.discriminator.forward



        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

        # Get train and test data sets
        self.train_files, self.test_files = load_img_names(self.hparams.img_dir,
                                               label_csv=self.hparams.label_path,
                                               data_type=self.hparams.data_type,
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
                                   test_size=0.1,
                                   data_type=self.hparams.data_type)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)



    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_i):
        a_img, no_a_img = batch

        # match gpu device (or keep as cpu)
        if self.on_gpu:
            print("ON GPU")
            a_img.cuda()
            no_a_img.cuda()


        self.last_imgs = [a_img, no_a_img]

        ### TRAIN GENERATOR ###
        if optimizer_i == 0:
            # Get image and sinogram without artifacts
            z = no_a_img   # has shape (batch_size, channels (2 or 4), 300, 300)

            # generate images with artifacts
            self.generated_imgs = self.forward(z)

            # log sampled images
            sample_imgs = [z[0], a_img[0], self.generated_imgs[0]]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.add_image(f'imgs/epoch{self.current_epoch}/batch_sample', grid, batch_nb)

            # ground truth result (ie: all fake)
            valid = torch.ones(no_a_img.size(0), 1).type(dtype=torch.cuda.FloatTensor)

            # Run the generated images through the discriminator to see how good they are
            gen_preds = self.discriminate(self.generated_imgs)

            # Calculate Adversarial loss using BCE loss
            print(min(gen_preds),max(gen_preds))
            g_loss = self.adversarial_loss(gen_preds, valid)

            # Add Generator loss to tensorboard
            self.logger.add_scalar("loss/generator", g_loss, batch_nb)

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

            # how well can it label as real?
            # Give the discriminator the artifact imgs
            valid = torch.ones(no_a_img.size(0), 1).type(dtype=torch.cuda.FloatTensor)
            real_preds = self.discriminate(a_img)
            real_loss = self.adversarial_loss(real_preds, valid)

            # how well can it label as fake?
            fake = torch.zeros(a_img.size(0), 1).type(dtype=torch.cuda.FloatTensor)
            fake_preds = self.discriminate(self.generated_imgs.detach()) # detach() detaches the
            fake_loss = self.adversarial_loss(fake_preds, fake)          # output from the computational
                                                                         # graph so we don't backprop
            # discriminator loss is the average of these                 # through it.
            d_loss = (real_loss + fake_loss) / 2

            # Add Discriminator loss to tensorboard
            self.logger.add_scalar("loss/discriminator", d_loss, batch_nb)


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
        a_img, no_a_img = self.last_imgs[0], self.last_imgs[1]

        # match gpu device (or keep as cpu)
        z = a_img
        if self.on_gpu:
            z = z.cuda(a_img.device.index)

        # log sampled images (with most recent batch img)
        # gen_imgs = self.forward(z)
        # grid1 = torchvision.utils.make_grid(a_img[0:4])
        # grid2 = torchvision.utils.make_grid(no_a_img[0:7])
        # grid3 = torchvision.utils.make_grid(gen_imgs[0:7])
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_orig_w_artifact', grid1, 0)
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_orig_n_artifact', grid2, 0)
        # self.logger.add_image(f'imgs/epoch{self.current_epoch}/epoch_end_generated', grid3, 0)



def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = pl.Trainer(max_nb_epochs=10, gpus=hparams.ngpu, distributed_backend='ddp')
    trainer = pl.Trainer(max_nb_epochs=50)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':

    # Get Hyperparameters and other arguments for training
    hparams, unparsed_args = get_args()



    main(hparams)
