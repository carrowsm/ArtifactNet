import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn



class UNet2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet2D, self).__init__()

        features = init_features
        ### ENCODER ###
        """
            Use the original U-Net architecture from the original paper:
            https://arxiv.org/abs/1505.04597
            GitHub: https://github.com/milesial/Pytorch-UNet
        """

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








class UNet3D(nn.Module):
    """A 3D implementation of the original UNet"""
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet3D, self).__init__()

        features = init_features

        ### ENCODER ###
        """
            Use the original U-Net architecture from the original paper:
            https://arxiv.org/abs/1505.04597
            GitHub: https://github.com/milesial/Pytorch-UNet
        """

        self.features = init_features
        self.encoder1 = self.conv_relu(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self.conv_relu(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self.conv_relu(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=1, stride=2)

        self.encoder4 = self.conv_relu(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        ### ------- ###

        self.bottleneck = self.conv_relu(features * 8, features * 16, name="bottleneck")

        ### DECODER ###
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=[3,2,2], stride=2)
        self.decoder4 = self.conv_relu((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=1, stride=2)
        self.decoder3 = self.conv_relu((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self.conv_relu((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self.conv_relu(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


        ### ------- ###

    def forward(self, x):
        enc1 = self.encoder1(x)                            # (N, 64, 20, 300, 300)
        enc2 = self.encoder2(self.pool1(enc1))             # (N, 128, 10, 150, 150)
        enc3 = self.encoder3(self.pool2(enc2))             # (N, 256, 5, 75, 75)
        enc4 = self.encoder4(self.pool3(enc3))             # (N, 512, 3, 38, 38)

        bottleneck = self.bottleneck(self.pool4(enc4))     # (N, 1024, 1, 19, 19)

        dec4 = self.upconv4(bottleneck)                    # (N, 512, 3, 38, 38)
        dec4 = torch.cat((dec4, enc4), dim=1)              # (N, 1024, 3, 38, 38)
        dec4 = self.decoder4(dec4)                         # (N, 512, 3, 38, 38)

        dec3 = self.upconv3(dec4)                          # (N, 256, 5, 75, 75)
        dec3 = torch.cat((dec3, enc3), dim=1)              # (N, 512, 5, 75, 75)
        dec3 = self.decoder3(dec3)                         # (N, 256, 5, 75, 75)

        dec2 = self.upconv2(dec3)                          # (N, 128, 10, 150, 150)
        dec2 = torch.cat((dec2, enc2), dim=1)              # (N, 256, 10, 150, 150)
        dec2 = self.decoder2(dec2)                         # (N, 128, 10, 150, 150)

        dec1 = self.upconv1(dec2)                          # (N, 64, 20, 300, 300)
        dec1 = torch.cat((dec1, enc1), dim=1)              # (N, 128, 20, 300, 300)
        dec1 = self.decoder1(dec1)                         # (N, 64, 20, 300, 300)
        # return torch.sigmoid(self.conv(dec1))            # (N, 1, 20, 300, 300)
        return self.conv(dec1)


    @staticmethod
    def conv_relu(in_channels, features, name):
        '''Perform:
        1. 3d convolution, kernel=3, padding=1, so output_size=input_size
        2. Batch normalization
        3. Relu
        4. Another convolution, with same input and output size
        5. batch normalization
        6. Relu'''
        normfunc = nn.InstanceNorm3d
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", normfunc(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(0.2, inplace=True)),
                    # (
                    #     name + "conv2",
                    #     nn.Conv3d(
                    #         in_channels=features,
                    #         out_channels=features,
                    #         kernel_size=3,
                    #         padding=1,
                    #         bias=False,
                    #     ),
                    # ),
                    # (name + "norm2", normfunc(num_features=features)),
                    # (name + "relu2", nn.LeakyReLU(0.2, inplace=True)),
                ]
            )
        )


class ResNetK(nn.Module):
    """A generator following the ResNet architecture with K residual
    blocks."""
    def __init__(self, in_channels=1, out_channels=1, n_filters=64, n_layers=4, norm_layer=nn.InstanceNorm3d,
                 use_bias=True, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            n_filters (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(ResNetK, self).__init__()

        """c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3"""

        # Add first convolutional layer
        net = [nn.ReplicationPad3d(3),
                 nn.Conv3d(in_channels, n_filters, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(n_filters),
                 nn.ReLU(True)]

        # Add two downsampling layers
        for i in range(2):  # add downsampling layers
            mult = 2 ** i
            net += [nn.Conv3d(n_filters * mult, n_filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(n_filters * mult * 2),
                      nn.ReLU(True)]
        # Add K residual layers
        for i in range(n_blocks) :
            net += [ResNetBlock(n_filters * (2 ** 2), padding_type='zero',
                                norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]

        # Add two Upsampling layers
        for i in range(2):
            mult = 2 ** (2 - i)
            net += [nn.ConvTranspose3d(n_filters * mult, int(n_filters * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(n_filters * mult / 2)),
                      nn.ReLU(True)]

        # Add output block
        net += [nn.ReplicationPad3d(3)]
        net += [nn.Conv3d(n_filters, out_channels, kernel_size=7, padding=0)]
        net += [nn.Tanh()]

        self.net = nn.Sequential(*net)

    def forward(self, X) :
        out = self.net(X)
        return X + out


class ResNetBlock(nn.Module):
    """
    A single residual convolutional block. This roughly follows the
    structure by Zhu et al in cycleGAN (https://arxiv.org/pdf/1703.10593.pdf).
    Parameters:
        channels (int)      -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not
    """
    def __init__(self, channels, padding_type='zero', norm_layer=nn.InstanceNorm3d,
                 use_dropout=False, use_bias=True):
        super(ResNetBlock, self).__init__()

        # Create a list of convolutional blocks
        res_block = []

        # Create padding layer
        p_layer, p = self.get_padding(padding_type)

        # Add padding layer
        res_block += p_layer

        # Add convolutional layer
        res_block += [nn.Conv3d(channels, channels, kernel_size=3,
                                 padding=p, bias=use_bias),
                       norm_layer(channels), nn.ReLU(True)]

        # Add dropout, if needed
        if use_dropout:
            res_block += [nn.Dropout(0.5)]

        # Add another padding layer
        res_block += p_layer

        # Add another conv layer
        res_block += [nn.Conv3d(channels, channels, kernel_size=3,
                                 padding=p, bias=use_bias),
                       norm_layer(channels), nn.ReLU(True)]

        self.res_block = nn.Sequential(*res_block)

    def get_padding(self, padding_type):
        """ Takes the input padding type and returns the appropiate padding
        layer and the ammount of padding to use in convolutional layer"""
        p, p_layer = 0, []            # Ammount of padding and type of layer
        if padding_type == 'reflect':
            p_layer = [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            p_layer = [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':  # If using 'zero', don't add padding layer
            p = 1
        else:
            raise NotImplementedError(f"padding {padding_type} is not implemented")
        return p_layer, p

    def forward(self, X) :
        """ Add skip connection around the two convolutions"""
        # X = X + self.res_block(X)
        X = self.res_block(X)
        return X
