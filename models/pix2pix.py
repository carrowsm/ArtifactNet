import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from generators import UNet2D, CNN_2D



class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        # Initialize UNet2D with 2 input channels for CT and sinogram domain
        self.network = UNet2D(in_channels=1, out_channels=1, init_features=64)

    def forward(self, z):
        img = self.network.forward(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, output_dim=1):
        super(Discriminator, self).__init__()

        """ Use architecture from Mattea's DA detection paper """
        # PyTorch's conv2d takes the following form:
        # initialization:   (channels_in, channels_out, kernel_size)
        # Input :           (batch_size, Channels_in, H, W)

        self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride)
        self.LRelu = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)

        self.avgPool = nn.AvgPool2d(2, 2)

        self.fc3 = nn.Linear(64 * 9 * 9, output_dim)

        self.softmax = torch.nn.Softmax(dim=1)

        self.sigmoid = torch.nn.Sigmoid()

        self.linear = nn.Linear(300*300, output_dim)
        self.linear.requires_grad = True

    def forward(self, X):
        X = self.pool(self.conv1_bn(self.LRelu(self.conv1(X))))
        X = self.pool(self.conv2_bn(self.LRelu(self.conv2(X))))
        X = self.pool(self.conv3_bn(self.LRelu(self.conv3(X))))
        X = self.pool(self.conv4_bn(self.LRelu(self.conv4(X))))
        X = self.conv5_bn(self.LRelu(self.conv5(X)))
        X = self.avgPool(X)

        # X.view(-1, Y) reshapes X to shape (batch_size, Y) for FC layer
        X = X.view(-1, 64 * 9 * 9)
        X = self.fc3(X)

        # Constrain output of model to (0, 1)
        X = self.sigmoid(X)


        return X
