import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# conv = nn.Conv2d(
#     in_channels=1,
#     out_channels=32,
#     kernel_size=3,
#     padding=1,
#     bias=False)
# tensor = torch.randn((10, 1, 300, 300))
# np.shape(tensor)
#
# np.shape(conv(tensor))
#
# def get_pool_dims(Hin, Win, stride, kernel_size, padding=0, dilation=1):
#     # Wout = ( Hin+(2*padding) -dilation*(kernel_size-1) - 1)/stride  + 1
#     # return Wout, Wout
#     x = torch.randn(1, 1, Hin, Win)
#     pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
#     return np.shape(pool(x))
#
# def get_upconvdims(Hin, Win, stride, kernel_size, padding=0, dilation=1):
#     c = nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride)
#     x = torch.randn(1, 1, Hin, Win)
#     return np.shape(c(x))
#
# get_upconvdims(150, 150, kernel_size=2, stride=2)
# get_pool_dims(38, 38, kernel_size=2, stride=2, padding=0)
#
# x = torch.randn(1,1,38, 38)
# y = torch.randn(1,1,38, 38)
# np.shape(torch.cat((x, y), dim=1))




class UNet2D(nn.Module):

    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        ### ENCODER ###
        """ We use three blocks of convolutions, each followed by max pooling
        to reduce the dimensionality"""
        self.features = init_features
        self.encoder1 = self.conv_relu(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self.conv_relu(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self.conv_relu(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.encoder4 = self.conv_relu(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        ### ------- ###

        self.bottleneck = self.conv_relu(features * 8, features * 16, name="bottleneck")

        ### DECODER ###
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self.conv_relu((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=1, stride=2)
        self.decoder3 = self.conv_relu((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.conv_relu((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self.conv_relu(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        ### ------- ###

    def forward(self, x):
        enc1 = self.encoder1(x)                            # (N, 32, 300, 300)
        enc2 = self.encoder2(self.pool1(enc1))             # (N, 64, 150, 150)
        enc3 = self.encoder3(self.pool2(enc2))             # (N, 128, 75, 75)
        enc4 = self.encoder4(self.pool3(enc3))             # (N, 256, 38, 38)

        bottleneck = self.bottleneck(self.pool4(enc4))     # (N, 512, 19, 19)

        dec4 = self.upconv4(bottleneck)                    # (N, 256, 38, 38)
        dec4 = torch.cat((dec4, enc4), dim=1)              # (N, 512, 38, 38)
        dec4 = self.decoder4(dec4)                         # (N, 256, 38, 38)
        dec3 = self.upconv3(dec4)                          # (N, 128, 75, 75)
        dec3 = torch.cat((dec3, enc3), dim=1)              # (N, 256, 75, 75)
        dec3 = self.decoder3(dec3)                         # (N, 128, 75, 75)
        dec2 = self.upconv2(dec3)                          # (N, 64, 150, 150)
        dec2 = torch.cat((dec2, enc2), dim=1)              # (N, 128, 150, 150)
        dec2 = self.decoder2(dec2)                         # (N, 64, 150, 150)
        dec1 = self.upconv1(dec2)                          # (N, 32, 300, 300)
        dec1 = torch.cat((dec1, enc1), dim=1)              # (N, 64, 300, 300)
        dec1 = self.decoder1(dec1)                         # (N, 32, 300, 300)
        return torch.sigmoid(self.conv(dec1))              # (N, 1, 300, 300)

    @staticmethod
    def conv_relu(in_channels, features, name):
        '''Perform:
        1. 2d convolution, kernel=3, padding=1, so output_size=input_size
        2. Batch normalization
        3. Relu
        4. Another convolution, with same input and output size
        5. batch normalization
        6. Relu'''
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        # Initialize UNet2D with 2 input channels for CT and sinogram domain
        self.network = UNet2D(in_channels=1, out_channels=1, init_features=32)

    def forward(self, z):
        img = self.network.forward(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
