"""
A script to generate and save images using one of the pre-trained generators
from cycleGAN for DA reduction.


"""

import os
import itertools
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from pytorch_lightning.callbacks import ModelCheckpoint

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from config.options import get_args

from data.data_loader import load_image_data_frame, UnpairedDataset, PairedDataset
from data.transforms import ToTensor, Normalize

from util.helper_functions import set_requires_grad
from util.loggers import TensorBoardCustom





def prepare_data(img_list_csv) :
    """
    """
    df = pd.read_csv(img_list_csv, dtype=str).set_index("patient_id")



def main() :
