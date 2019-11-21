import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.utils.data as data

import random

from skimage.transform import rotate




def load_img_names(dir, label_csv=None, extensions=["npy"],
                   data_type="phantom", data_augmentation_factor=1, test_size=0.25):
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
        label_csv : str or path-like
            Full path to the CSV containing the artifact labels and patient IDs
        extensions : list-like, containing str (Default=["npy"])
            A list of allowed file extensions.
        data_type : str (Default='phantom')
            The type of data the model will use. Either 'phantom' or 'patient'.
        data_augmentation_factor : int (Default=1)
            Factor by which data will be augmented. data_augmentation_factor=1
            means no augmentations will be performed. data_augmentation_factor=k
            means data set size will increase by a factor of k.
        test_size : float (Default=0.25)
            Proportion of the data set to use for testing.

    Returns:
        if data_type=='phantom':
            Two lists: A list of file names for all the training images and another for test.
        if data_type=='patient':
            Two arrays: A list of tuples of paired artifact and non-artifact patient file names
            these will each have the shape: (2, data_set_size*aug_factor, 2). And
                patients_with_artifacts     = aug_train_files[0, :, :]
                patients_without_artifacts  = aug_train_files[1, :, :]
                patient_IDs                 = aug_train_files[:, :, 0]
                artifact_slice_indices      = aug_train_files[:, :, 1]
    """
    if data_type == "phantom" :
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

    elif data_type == "patient" :
        # Get the patient IDs for each class
        patient_df = pd.read_csv(label_csv, index_col="p_index",
                                  dtype=str, na_values=['nan', 'NaN', ''])

        a_list, no_a_list = [], []
        # Get a list of patients with artifacts and another list without
        for index, row in patient_df.iterrows() :
            if row["has_artifact"] == '1' or row["has_artifact"] == '2' :
                # This patient has an artifact. add it to the a_list
                # Each item in the list is a patient's ID and the location of the artifact
                a_list.append( tuple([row["patient_id"]+"_img.npy", int(row['a_slice'])]) )
            elif row["has_artifact"] == '0' :
                # This patient has no artifacts. add it to the no_a_list
                no_a_list.append( tuple([row["patient_id"]+"_img.npy", int(row['a_slice'])]) )
            else :
                # Patient is probably not labelled. just skip and continue loop.
                continue



        # Now we have a list for each patients with and without artifacts.
        # Augment the smaller data set so that they are the same length
        ratio = len(a_list) / len(no_a_list)
        if len(a_list) > len(no_a_list) : # tile repeats the array in order
            no_a_list = np.tile(no_a_list, (int(np.ceil(ratio)), 1))[0 : len(a_list)]
        elif len(a_list) < len(no_a_list) :
            a_list = np.tile(a_list, (int(np.ceil(1./ratio)), 1))[0 : len(no_a_list)]

        # We now have two lists of equal length for each class.
        # Seperate the data into train and test sets
        a_train, a_test, no_a_train, no_a_test = train_test_split(a_list,
                                                                  no_a_list,
                                                                  test_size=test_size)
        # Group together the atifact and non artifact sets
        train_files = np.stack((a_train, no_a_train), axis=0)
        test_files  = np.stack((a_test, no_a_test),   axis=0)




    # Increase the size of each list of files by repeating each element
    # data_augmentation_factor times
    aug_train_files = np.tile(train_files, (data_augmentation_factor, 1))
    aug_test_files  = np.tile(test_files,  (data_augmentation_factor, 1))

    # Randomize the order of the arrays
    # train = shuffle(aug_train_files, random_state=0)
    # test  = shuffle(aug_test_files,  random_state=0)

    return aug_train_files, aug_test_files


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
                 data_type="phantom",
                 transform=True,
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
        # self.transform = transform

        self.train = train
        self.test_size = test_size

        self.data_type = data_type

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
        # array = array - oldMin    # Set new min to zero
        # oldMax = oldMax - oldMin  # Bump up old max
        return (array - oldMin) * ((newMax - newMin)/(oldMax - oldMin)) + newMin

    def nonlin_norm(self, X, newMin=0, newMax=1) :
        oldMin, oldMax = 0., 1.
        B, a = 0.02, 0.005# B=intensity center, a=intensity gaussian width
        return ( (newMax - newMin) / (1+np.exp(-(X-B)/a)) ) + newMin

    def transform(self, X, stack_size=2) :
        """When we start using data augmentation we will call transform to
        apply random rotations, translations etc to the data"""
        # Generate a random angle between -30 and 30 degrees
        angle = random.uniform(-10, 10)

        for i in range(stack_size) :
            # Rotate the image by that angle
            X[i, :, :] = rotate(X[i, :, :], angle)

        X = self.lin_norm(X)
        return X

    def parse_phantom(self, img_data) :
        '''Function to take npy image stack of phantom images and seperate them
        into two arrays, one for with artifact and one for without_artifact'''
        z, height, width   = np.shape(img_data)
        image_stack = img_data

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
            no_artifact_arr  = image_stack[1 , :, :]    # 2D image with shape (height, width)

            # The Pytorch model takes a tensor of shape (batch_size, in_Channels, height, width)
            # Reshape the arrays to add another dimension
            artifact_arr    =    artifact_arr.reshape(1, height, width)
            no_artifact_arr = no_artifact_arr.reshape(1, height, width)

        else :
            # The images are not paired
            raise ValueError(f"Image shape ({z}, {width}, {height}) not accepted by the model.\nData must have z-size 2 or 4.")

        return artifact_arr, no_artifact_arr



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

        if self.data_type == "phantom" :
            image_file_name = self.image_names[index]
            image_path = os.path.join(self.root_dir, image_file_name)

            # Each file in image_paths is a compressed collection of one patient's augmented images
            image_stack = np.load(image_path)                       # 3D image with shape (4, height, width)
                                                                    # or (2, height, width) if no sinograms available
            # Seperate the array into the with and without artifact images
            if self.data_type == "phantom" :
                artifact_arr, no_artifact_arr = self.parse_phantom(image_stack)

        elif self.data_type == "patient" :
            a_file, a_slice       = self.image_names[0, index, 0], self.image_names[0, index, 1]
            no_a_file, no_a_slice = self.image_names[1, index, 0], self.image_names[1, index, 1]


            a_img = np.load(os.path.join(self.root_dir, a_file))[int(a_slice), : , :]
            no_a_img = np.load(os.path.join(self.root_dir, no_a_file))[int(no_a_slice), : , :]

            # Stack the two images together and crop them
            image_stack = np.stack((a_img, no_a_img))[:, 0:300, 106 : -106]


        # Perfom transformation and normalization and resize
        z, height, width   = np.shape(image_stack)
        image_stack = self.transform(image_stack, stack_size=z)

        image_stack = image_stack.reshape(1, 2, height, width)

        # Seperate the arrays again
        artifact_arr    = image_stack[:, 0, :, :]
        no_artifact_arr = image_stack[:, 1, :, :]

        # Convert the np.arrays to PyTorch tensors
        artifact_tensor    = torch.from_numpy( artifact_arr   ).float()
        no_artifact_tensor = torch.from_numpy( no_artifact_arr).float()


        return artifact_tensor.cuda(), no_artifact_tensor.cuda()



    def __len__(self):
        if self.data_type == "phantom" :
            # Total number of image pairs (including augmentations)
            self.dataset_length = len(self.image_names)
        elif self.data_type == "patient" :
            # Total number of image pairs (including augmentations)
            self.dataset_length = len(self.image_names[0, :, 0])
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
                                test,       # Ordered list of image pair file names
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
