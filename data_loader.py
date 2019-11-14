import torch
import torch.utils.data as data

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




class RadiomicsFolder(data.Dataset):
    """Image dataset for ArtifactNet artifact removal GAN.

       Attributes
       ---------
       train : bool
          Whether to return the training set or test set.
       dataset_length: int
          Number of images in the data set
       normalize: callable
          perform linear normalization on the data

    """

    def __init__(self,
                 image_root_dir,        # Path to images
                 image_names,           # Ordered list of image pair file names
                 train=True,
                 transform=None,
                 test_size=.3):


        self.root_dir = image_root_dir
        self.image_names = image_names # List of npy file names
        '''image_names should be an ordere list of npy file names with the
        following format:
            path_to_images/i.npy
        where :
            i is the patient index, corresponding to a file named "i.npy".
            .npy is the file extension.
        '''

        # Transform to be applied to the image
        self.transform = transform

        self.train = train
        self.test_size = test_size


        self.dataset_length = len(self.image_paths)   # Total number of image pairs (including augmentations)


    def normalize(self, img, MIN:float, MAX:float) :
        # Normalize the image (var = 1, mean = 0)
        img = (img - MIN) / (MAX - MIN)
        img = np.clip(img, MIN, MAX)
        img -= img.mean()
        img /= img.std()
        return img

    def transform(X) :
        """When we start using data augmentation we will call transform to
        apply random rotations, translations etc to the data"""
        return X


    def __getitem__(self, index):
        '''This function gets called by the trainer and passes four input images to the model.
           Given the index of that patient in the dataset, this function returns the pytorch
           tensor representing the image with artifacts, the image without artifacts, the
           sinogram with artifacts, and the sinogram without artifacts.
           --------------
           The image_path initiated by this class takes the form:
               /path/to/image/img_number.npz
           where:
               - /path/to/image/img_number.npz is the path and filename of the npz file represented as a string.

           --------------
           The output of this function must be of the form:
               (image_tensor, sinogram_tensor)
           where:
               - artifact_tensor is a pytorch tensor with shape: (2, height, width).
               - no_artifact_tensor is a pytorch tensor with shape: (2, height, width).
               - The first dimension (with size two) is for the models two input channels (image and sinogram)
               The stacking order is:
                    artifact_tensor[0, : , :] = image_with_artifact
                    artifact_tensor[1, : , :] = sinogram_with_artifact
               and
                    no_artifact_tensor[0, : , :] = image_without_artifact
                    no_artifact_tensor[1, : , :] = sinogram_without_artifact
           --------------
        '''

        image_file_name = self.image_names[i]
        image_path = os.path.join(self.root_dir, image_file_name)


        # Each file in image_paths is a compressed collection of one patient's augmented images
        image_stack = np.load(image_path)                     # 3D image with shape (4, height, width)

        # Perfom transformation and normalization
        # image_stack = transform(image_stack)
        # image = self.normalize(image, 0.64, 0.86)             # Normalize the image

        # Seperate the 3D image into two np arrays representing 4 images.
        artifact_arr     = image_stack[0:2, :, :]   # 3D image with shape (4, height, width)
        no_artifact_arr  = image_stack[2: , :, :]   # 3D image with shape (4, height, width)



        # The Pytorch model takes a tensor of shape (batch_size, in_Channels, height, width)
        # Reshape the arrays to add another dimension
        height, width   = np.shape(artifact_arr[0])
        artifact_arr    =    artifact_arr.reshape(1, 2, height, width)
        no_artifact_arr = no_artifact_arr.reshape(1, 2, height, width)

        # Convert the np.arrays to PyTorch tensors
        artifact_tensor    = torch.from_numpy( artifact_arr).float()
        no_artifact_tensor = torch.from_numpy( no_artifact_arr).float()



        return artifact_tensor, no_artifact_tensor

    def __len__(self):
        return self.dataset_length
