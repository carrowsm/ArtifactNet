import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn



def conv_out_shape(in_shape, kernel_size=[4,4,4], stride=1, padding=0, dilation=1) :
    """Calculate the output shape of a tensor passed to conv3d with
    these params"""
    # Cast lists to arrays
    in_shape = np.array(in_shape) if type(in_shape) is list else in_shape
    stride = np.array(stride) if type(stride) is list else stride

    out = ((in_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    return np.floor(out)




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



class CNN_3D(nn.Module) :
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
        super(CNN_3D, self).__init__()

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

        self.fc = torch.nn.Linear(in_features=1*2*37*37, out_features=1, bias=True)


    def forward(self, X) :
        # Assume batch_size = N and X.shape = (N,   1, 20, 300, 300)
        X = self.conv1(X)                   # (N,  64, 10, 150, 150)
        X = self.Lrelu(X)                   # (N,  64, 10, 150, 150)
        X = self.conv2(X)                   # (N, 128,  5,  75,  75)
        X = self.bnorm(X)                   # (N, 128,  5,  75,  75)
        X = self.Lrelu(X)                   # (N, 128,  5,  75,  75)
        X = self.conv3(X)                   # (N,   1,  2,  37,  37)
        X = self.fc(X.view(-1, 1*2*37*37))  # (N,   1,  1,   1,   1)
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
        ks = 4              # Kernel size
        pads = 1            # Padding size
        s = [1, 2, 2]       # Convolution stride
        use_bias = True     # Include learnable bias term
        normfunc = nn.InstanceNorm3d

        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=n_filters,
                               kernel_size=ks, stride=s, padding=pads)
        self.conv2 = nn.Conv3d(in_channels=n_filters, out_channels=n_filters * 2,
                               kernel_size=ks, stride=s, padding=pads, bias=use_bias)
        self.conv3 = nn.Conv3d(in_channels=n_filters * 2, out_channels=n_filters * 4,
                               kernel_size=ks, stride=s, padding=pads, bias=use_bias)
        self.conv4 = nn.Conv3d(in_channels=n_filters * 4, out_channels=n_filters * 8,
                               kernel_size=ks, stride=s, padding=pads, bias=use_bias)

        self.convf = nn.Conv3d(in_channels=n_filters * 8, out_channels=1,
                               kernel_size=[16, 18,  18], stride=s, padding=0, bias=True)

        self.inorm2 = normfunc(n_filters * 2, affine=False)
        self.inorm3 = normfunc(n_filters * 4, affine=False)
        self.inorm4 = normfunc(n_filters * 8, affine=False)

        self.Lrelu1 = nn.LeakyReLU(0.2, True)
        self.Lrelu2 = nn.LeakyReLU(0.2, True)
        self.Lrelu3 = nn.LeakyReLU(0.2, True)
        self.Lrelu4 = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X) :
        # Assume batch_size = N and X.shape = (N,   1, 20, 300, 300)
        # Layer 1
        X = self.conv1(X)                    # (N,  64, 19, 150, 150)
        X = self.Lrelu1(X)                   # (N,  64, 19, 150, 150)

        # Layer 2
        X = self.conv2(X)                    # (N, 128,  18, 75,  75)
        X = self.Lrelu2(X)                   # (N, 128,  18, 75,  75)
        X = self.inorm2(X)                   # (N, 128,  18, 75,  75)

        # Layer 3
        X = self.conv3(X)                    # (N, 256,  17, 37,  37)
        X = self.Lrelu3(X)                   # (N, 256,  17, 37,  37)
        X = self.inorm3(X)                   # (N, 256,  17, 37,  37)

        # Layer 4
        X = self.conv4(X)                    # (N, 512,  16, 18,  18)
        X = self.Lrelu4(X)                   # (N, 512,  16, 18,  18)
        X = self.inorm4(X)                   # (N, 512,  16, 18,  18)

        # Final convolutional layer to make scalar output
        X = self.convf(X)                   # (N,   1,  1,   1,   1)

        return X


class CNN_NLayer(nn.Module) :
    """A 3D CNN with variable depth"""
    def __init__(self, input_channels=1, out_size=1, n_filters=64, n_layers=4, norm="instance",
                 input_shape=[20, 300, 300]):
        """
        Parameters:
            input_channels (int) : Number of input channels (default=1).
            out_size (int) :       The shape of the output tensor. The output
                                   will be cubic with shape
                                   (out_size, out_size, out_size). Default=1.
            n_filters :            The number of filters to use in the last
                                   concolutional layer (default=64).
            n_layers :             The number of convolutional layers to use
                                   (default = 4).
            norm :                 Type of normalization to use (can be either
                                   'instance' or 'batch').
        """
        super(CNN_NLayer, self).__init__()

        # Parameters for convolutional layers
        ks = 4             # Kernel size
        pads = 1           # Padding size
        s = [1, 2, 2]      # Convolution stride
        use_bias = True
        normfunc = nn.InstanceNorm3d


        # Create first convolutional layer
        net_list = [nn.Conv3d(in_channels=input_channels, out_channels=n_filters,
                               kernel_size=ks, stride=s, padding=pads),
                    nn.LeakyReLU(0.2, True)]
        out_shape = conv_out_shape(input_shape, kernel_size=ks, stride=s, padding=pads)

        # Add middle layers
        for i in range(1, n_layers) :
            in_channels = n_filters * (2 ** (i - 1))
            out_filters = n_filters * (2 ** i)
            net_list += [nn.Conv3d(in_channels=in_channels, out_channels=out_filters,
                                   kernel_size=ks, stride=s, padding=pads),
                         normfunc(out_filters, affine=False),
                         nn.LeakyReLU(0.2, True)]
            out_shape = conv_out_shape(out_shape, kernel_size=ks, stride=s, padding=pads)


        # Add final conv layer
        self.fc_in_size = int(np.prod(out_shape))  # Length of flattened array from last conv layer
        self.fc = nn.Linear(in_features=self.fc_in_size, out_features=1, bias=use_bias)

        self.net = nn.Sequential(*net_list)

    def forward(self, X) :
        X = self.net(X)
        # X.view(-1, Y) reshapes X to shape (batch_size, Y) for FC layer
        X = X.view(-1, self.fc_in_size)
        X = self.fc(X)
        return X


class PatchGAN_NLayer(nn.Module) :
    """A 3D PatchGAN with variable depth"""
    def __init__(self, input_channels=1, out_size=1, n_filters=64, n_layers=4, norm="instance",
                 input_shape=[20, 300, 300]):
        """
        Parameters:
            input_channels (int) : Number of input channels (default=1).
            out_size (int) :       The shape of the output tensor. The output
                                   will be cubic with shape
                                   (out_size, out_size, out_size). Default=1.
            n_filters :            The number of filters to use in the last
                                   concolutional layer (default=64).
            n_layers :             The number of convolutional layers to use
                                   (default = 4).
            norm :                 Type of normalization to use (can be either
                                   'instance' or 'batch').
            input_shape :          Shape of the input tensor (input image).
        """
        super(PatchGAN_NLayer, self).__init__()

        # Parameters for convolutional layers
        ks = 4                    # Kernel size
        pads = 1                  # Padding size
        s = [1, 2, 2]             # Convolution stride
        use_bias = True
        normfunc = nn.InstanceNorm3d


        # Build up layers sequentially
        net_list = [nn.Conv3d(in_channels=input_channels, out_channels=n_filters,
                               kernel_size=ks, stride=s, padding=pads, bias=use_bias),
                    nn.LeakyReLU(0.2, True)]
        out_shape = conv_out_shape(input_shape, kernel_size=ks, stride=s, padding=pads)

        # Add middle layers
        for i in range(1, n_layers) :
            in_channels = n_filters * (2 ** (i - 1))
            out_filters = n_filters * (2 ** i)
            net_list += [nn.Conv3d(in_channels=in_channels, out_channels=out_filters,
                                   kernel_size=ks, stride=s, padding=pads, bias=use_bias),
                         normfunc(out_filters, affine=False),
                         nn.LeakyReLU(0.2, True)]
            # Calculate shape of output tensor from this layer
            out_shape = conv_out_shape(out_shape, kernel_size=ks, stride=s, padding=pads)

        # Add final conv layer
        net_list += [nn.Conv3d(in_channels=out_filters, out_channels=1,
                               kernel_size=[16, 18,  18], stride=s, padding=0, bias=use_bias)]
        out_shape = conv_out_shape(out_shape, kernel_size=ks, stride=s, padding=pads)



        self.net = nn.Sequential(*net_list)

    def forward(self, X) :
        X = self.net(X)
        return X
