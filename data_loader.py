import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.utils.data as data






def load_img_names(dir, extensions=["npy"], data_augmentation_factor=1, test_size=0.25):
    """
    Finds all images in the specified directory and returns a list of the file names
    in that directory.  The data is also split into a training and test group, then
    this list is augmented according to the data augmentation factor and ordered
    so that the RadiomicsFolder data loader does not train on the same image twice.
    Parameters
        dir : str
            A root directory containing one npy file per patient. Each of these
            npy files should contain 4 arrays stacked together: one for the image
            with artifacts, sinogram with artifact, image without artifact, and
            sinogram without artifact, for one patient.
        extensions : list-like, containing str (Default=["npy"])
            A list of allowed file extensions.
        data_augmentation_factor : int (Default=1)
            Factor by which data will be augmented. data_augmentation_factor=1
            means no augmentations will be performed. data_augmentation_factor=k
            means data set size will increase by a factor of k.
        test_size : float (Default=0.25)
            Proportion of the data set to use for testing.

    Returns:
        Two lists: A list of paths for all the training images and another for test.
    """
    file_list = []

    # TODO Remove bad images from dataset.
    bad_imgs = []

    # Get list of all the patients in the directory
    for file in os.listdir(dir) :
        extension = file.split(".")[-1]
        if extension in extensions :

            if file not in bad_imgs :
                # This file can be used; add this file to the file_list
                file_list.append(file)

    # We now have a list of all the valid data files
    # Seperate them into train and test sets
    train_files, test_files = train_test_split(np.array(file_list), test_size=test_size)

    # Increase the size of each list of files by repeating each element
    # data_augmentation_factor times
    aug_train_files = np.repeat(train_files, data_augmentation_factor)
    aug_test_files  = np.repeat(test_files,  data_augmentation_factor)

    # Randomize the order of the arrays
    train = shuffle(aug_train_files, random_state=0)
    test  = shuffle(aug_test_files,  random_state=0)

    return train, test


class RadiomicsDataset(data.Dataset):
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
                 test_size=0.25):
        '''
        Parameters:
            image_root_dir : str
                The full absolute path to the directory containing the image data.
            image_names : List-like, str
                An ordered list of npy file names with the following format:
                    i.npy
                where i is the patient index, corresponding to a file named "i.npy".
        '''
        self.root_dir = image_root_dir
        self.image_names = image_names # List of npy file names

        # Transform to be applied to the image
        self.transform = transform

        self.train = train
        self.test_size = test_size

        self.dataset_length = len(self.image_names)   # Total number of image pairs (including augmentations)


    def normalize(self, img, MIN:float, MAX:float) :
        # Normalize the image (var = 1, mean = 0)
        # img = (img - MIN) / (MAX - MIN)
        # img = np.clip(img, MIN, MAX)
        # img -= img.mean()
        # img /= img.std()
        # return img
        pass

    def lin_norm(self, array, newMin=0., newMax=1.) :
        oldMin, oldMax = np.min(array), np.max(array)
        return (array - oldMin) * ((newMax - newMin)/(oldMax - oldMin)) + newMin

    def nonlin_norm(self, X, newMin=0, newMax=1) :
        oldMin, oldMax = 0., 1.
        B, a = 0.02, 0.005# B=intensity center, a=intensity gaussian width
        return ( (newMax - newMin) / (1+np.exp(-(X-B)/a)) ) + newMin

    def transform(self, X) :
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
           Returns:
               image_tensor, sinogram_tensor
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

        image_file_name = self.image_names[index]
        image_path = os.path.join(self.root_dir, image_file_name)


        # Each file in image_paths is a compressed collection of one patient's augmented images
        image_stack = np.load(image_path)                       # 3D image with shape (4, height, width)
                                                                # or (2, height, width) if no sinograms available

        # Get the size of the image stack
        z, height, width   = np.shape(image_stack)

        # Perfom transformation and normalization
        # image_stack = transform(image_stack, z=z, x=height, y=width)
        image_stack = self.lin_norm(image_stack)             # Normalize the image

        # Seperate the 3D image into two np arrays representing 4 (or 2) images.
        if z == 4 :
            # We have the images and sinograms.
            artifact_arr     = image_stack[0:2, :, :]   # 3D image with shape (2, height, width)
            no_artifact_arr  = image_stack[2: , :, :]   # 3D image with shape (2, height, width)

            # The Pytorch model takes a tensor of shape (batch_size, in_Channels, height, width)
            # Reshape the arrays to add another dimension
            artifact_arr    =    artifact_arr.reshape(2, height, width)
            no_artifact_arr = no_artifact_arr.reshape(2, height, width)

        elif z == 2 :
            # We only have the images, no sinograms
            artifact_arr     = image_stack[0, :, :]     # 2D image with shape (height, width)
            no_artifact_arr  = image_stack[1, :, :]     # 2D image with shape (height, width)

            # The Pytorch model takes a tensor of shape (batch_size, in_Channels, height, width)
            # Reshape the arrays to add another dimension
            artifact_arr    =    artifact_arr.reshape(1, height, width)
            no_artifact_arr = no_artifact_arr.reshape(1, height, width)

        else :
            # The images are not paired
            raise ValueError(f"Image shape ({z}, {width}, {height}) not accepted by the model.\nData must have z-size 2 or 4.")




        # Convert the np.arrays to PyTorch tensors
        artifact_tensor    = torch.tensor( artifact_arr, dtype=torch.float32)
        no_artifact_tensor = torch.tensor( no_artifact_arr, dtype=torch.float32)


        return artifact_tensor, no_artifact_tensor

    def __len__(self):
        return self.dataset_length


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ### Unit testing ###

    # Test make_dataset
    dir = "/home/colin/Documents/BHKLab/data/Artifact_Net_Training/trg_data"
    train, test = load_img_names(dir, data_augmentation_factor=3)


    train[0]

    # Test RadiomicsFolder
    data_set = RadiomicsDataset(dir,        # Path to images
                               test,           # Ordered list of image pair file names
                                train=False,
                                transform=None,
                                test_size=.3)

    art_image, nonart_img = data_set[100]

    print(np.shape(art_image))

    plt.figure()
    plt.title("Has Artifact")
    plt.imshow(art_image[ 0, :, :])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title("No Artifact")
    plt.imshow(nonart_img[ 0, :, :])
    plt.colorbar()
    plt.show()
