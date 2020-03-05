import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn



class Generator_G(nn.Module):
    """
    A network representing a map G: X -> Y.
    - X is a set of images {x_1, x_2, ..., x_n} from one
      domain (e.g. with artifact) and Y is a set of images
      {y_1, y_2, ..., y_m} from another domain (without DA).
    - The generator uses a 5-layer 3D UNet architecture.
    Methods:
    - forward(x): Performs a forward pass through the network
      with the tensor x.
    """
    def __init__(self, input_channels=1):
        super(Generator_G, input_channels=1).__init__()
        self.in_channels = input_channels

        # Create an instance of a 3D UNet
        self.network = UNet3D(inchannels=1, out_channels=1, init_fetures=64)

    def forward(self, x) :
        y_hat = self.network.forward(x)
        return y_hat




class Generator_F(nn.Module):
        """
        A network representing a map F: Y -> X.
        - X is a set of images {x_1, x_2, ..., x_n} from one
          domain (e.g. with artifact) and Y is a set of images
          {y_1, y_2, ..., y_m} from another domain (without DA).
        - The generator uses a 5-layer 3D UNet architecture.
        Methods:
        - forward(x): Performs a forward pass through the network
          with the tensor x.
        """
    def __init__(self):
        super(Generator_F, input_channels=1).__init__()
        self.in_channels = input_channels

        # Create an instance of a 3D UNet
        self.network = UNet3D(inchannels=1, out_channels=1, init_fetures=64)

    def forward(self, y) :
        x_hat = self.network.forward(y)
        return x_hat


class
