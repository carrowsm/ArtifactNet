import torch
import torch.utils.data as data

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aerts.src.settings import DATA_PATH_CLINICAL_PROCESSED, RANDOM_SEED, DATA_PATH_IMAGE_SIZE, IMAGE_ROTATIONS, IMAGE_AUGMENTATION_FACTOR
from aerts.src.data.dataset.radiomics import make_dataset, load_image_meta



class RadiomicsFolderBinary(data.Dataset):
    """Image dataset for binary classification following Aerts et al. (2018).

       Attributes
       ----------
       root : str
           Path to directory containing the images.
       extensions : list of str
           A list of allowed file extensions.
       train : bool
          Whether to return the training set or test set.
       transform : callable, optional
          Transformation that will be applied to the images on loading.
       test_size : float or int
          Number of samples in the test set (if int) or the proportion of
          samples to include in the test set (if float between 0. and 1.).
    """

    def __init__(self,
                 image_paths, # Ordered list of paths to images (len(image_paths) = number of patients*aug_factor)
                 targets,     # Ordered list of targets (HPV statuses)
                 train=True,
                 transform=None,
                 test_size=.3,
                 aug_factor=IMAGE_AUGMENTATION_FACTOR):

        self.image_paths = image_paths # List of npz file names
        self.image_dir_paths = os.path.dirname(image_paths[0])
        self.targets = targets
        # self.image_meta_data = load_image_meta(DATA_PATH_IMAGE_SIZE)

        self.transform = transform

        self.train = train
        self.test_size = test_size
        self.image_rotations = IMAGE_ROTATIONS
        self.aug_factor = aug_factor

        self.dir_size = len(self.image_paths) / self.aug_factor   # Number of npz files in image_paths
        self.dataset_length = len(self.image_paths)               # Total number of 3D images


    def normalize(self, img, MIN:float, MAX:float) :
        # Normalize the image (var = 1, mean = 0)
        img = (img - MIN) / (MAX - MIN)
        img = np.clip(img, MIN, MAX)
        img -= img.mean()
        img /= img.std()
        return img


    def get_rotation(self, image_aug_index) :
        '''Takes the index of the image within the npz file and finds
           key in the dictionary for the rotation it corresponds to.'''
        index = 0
        for i in range(self.image_rotations['x']) :                       # This function mimmics _get_rotations() in
            for j in range(self.image_rotations['y']) :                   # aerts.src.data.preprocess.structures
                for k in range(self.image_rotations['z']) :               # since this is how the rotations are defined.
                    if index == image_aug_index :                         #
                        name = "{:03}_{:03}_{:03}".format(i*90, j*90, k*90)
                        return name
                    else:
                        index = index + 1




    def __getitem__(self, index):
        '''This function gets called by the trainer and passes one input image to the model.
           Given the index of that specific image in the dataset, this function returns the pytorch
           tensor representing the image to train the model, as well as the correstponding label
           (HPV status) for that image.
           --------------
           The image_path initiated by this class takes the form:
               /path/to/image/patientID.npz_augmentationindex
           where:
               - /path/to/image/patientID.npz is the path and filename of the npz file represented as a string.
               - augmentationindex is a single number between 0 and aug_factor which represented the index of the
                 augmented image within the patient's image dictionary. This is interpreted by self.get_rotations().
           --------------
           The output of this function must be of the form:
               (image_tensor, target)
           where:
               - image_tensor is a pytorch tensor with shape: (batch_size, input_channels, depth, height, width)
               - target is a scalar representing the HPV status of that image
           --------------
        '''

        image_file, image_aug_index = self.image_paths[index].split("_")   # Location of the npz file in the directory


        # Each file in image_paths is a compressed collection of one patient's augmented images
        image_dict = np.load(image_file)                             # Dictionary of augmented 3D imgs for 1 patient
        image_key = self.get_rotation(int(image_aug_index))          # A key for the image rotation that our index corresponds to
        image = image_dict[image_key]                                # 3D array representing the specific image in question
        image = self.normalize(image, 0.64, 0.86)                      # Normalize the image

        # The image has to be wrapped in an array to give it an extra dimension of size 1 (representing 1 input channel).
        # This is because the conv3d functions in the model expect a 5D input vector.
        image_tensor = torch.from_numpy( np.array([image]) ).float()

        target = self.targets[index]  # For this to work, targets must be an ordered list of patient
                                      # outcomes (survival, HPV, whatever) with same length as
                                      # number of patients * aug_factor.

        # Convert the targets to a binary one-hot-encoded array
        # target = self.one_hot_encode(target)

        return np.array([image]), target   # The image has to be wrapped in an array to give it an extra dimension of size 1
                                           # This is because the conv3d functions in the model expect a 5D input vector.

    def __len__(self):
        return self.dataset_length
