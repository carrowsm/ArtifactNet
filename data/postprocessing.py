import io
import os
import numpy as np
import torch
from typing import Callable, Optional, Union, Sequence

import SimpleITK as sitk
from skimage.transform import resize

from preprocessing import resample_image, get_dicom_path


def insert_volume(subvolume: sitk.Image, full_image: sitk.Image) -> sitk.Image :
    """ Insert a subvolume into the original image, replacing those pixels.
    """
    return
