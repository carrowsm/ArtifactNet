import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn




class CNN_2D(nn.Module):
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



class PatchGAN_3D(nn.Module) :
    """A 3D PatchGAN """
    def __init__(self, input_channels=1, out_size=1, n_filters=64):
        """
        Parameters:
            input_channels (int) : Number of input channels (default=1).
            out_size (int) :       The shape of the output tensor. The output
                                   will be cubic with shape
                                   (out_size, out_size, out_size). Default=1.
            n_filters :            The number of filters to use in the last
                                   concolutional layer (default=64).
        """
        super(PatchGAN_3D, self).__init__()

        # Parameters for convolutional layers
        ks = 4     # Kernel size
        pads = 1   # Padding size
        s = 2      # Convolution stride

        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=n_filters,
                               kernel_size=ks, stride=s, padding=pads)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv3d(in_channels=n_filters, out_channels=n_filters * 2,
                               kernel_size=ks, stride=s, padding=pads, bias=True)

        self.bnorm = nn.BatchNorm3d(n_filters * 2)

        self.conv3 = nn.Conv3d(in_channels=n_filters * 2, out_channels=1,
                               kernel_size=ks, stride=s, padding=pads, bias=True)

        self.convf = nn.Conv3d(in_channels=1, out_channels=1,
                               kernel_size=[8, 64, 64], stride=s, padding=0, bias=True)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X) : # Assume X.shape = (N, 1,   50, 512, 512)
        X = self.conv1(X)                   # (N, 64,  25, 256, 256)
        X = self.Lrelu(X)                   # (N, 64,  25, 256, 102)
        X = self.conv2(X)                   # (N, 128, 13, 128, 50)
        X = self.bnorm(X)                   # (N, 128, 13, 128, 50)
        X = self.Lrelu(X)                   # (N, 128, 13, 128, 50)
        X = self.conv3(X)                   # (N, 1,   7,  64,  64)

        X = self.convf(X)                   # (N, 1,   1,   1,   1)
        X = self.sigmoid(X)
        return X
