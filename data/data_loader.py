import os
import sys
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.filters import threshold_otsu
from scipy import ndimage

import torch
from torch.utils.data import Dataset
import torchvision

from data.sitk_processing import read_dicom_image, resample_image, read_nrrd_image
from data.augmentations import affine_transform



def load_image_data_frame(path, img_X, img_Y, label_col="has_artifact") :
    """ Load data Frame containing the DA label and location of each patient
    Parameters :
    ------------
    path (str) :    Full path to the CSV containing the image IDs and corresponding
                    labels.
    X_label (list) : The CSV label for images in domain X. These labels are in the
                    column col_name in the CSV.
    Y_label (list) : The CSV label for images in domain Y.
    label_col (str): The CSV column name containing the image labels. Default is
                    'has_artifact'.
    Returns :
    ---------
    One DF for each image domain with the z-index of the DA slice for each patient.
    """
    df = pd.read_csv(path,
                     usecols=["patient_id", label_col, "DA_z"],
                     dtype=str, na_values=['nan', 'NaN', ''])
    df.set_index("patient_id", inplace=True)
    df["DA_z"] = df["DA_z"].astype(int)

    X_df = df[df[label_col].isin(img_X)]  # Patients in domain X (DA+)
    Y_df = df[df[label_col].isin(img_Y)]  # Patients in domain X (DA-)

    print(X_df["has_artifact"])
    print(Y_df["has_artifact"])

    return X_df, Y_df



class BaseDataset(Dataset):
    """Dataset class used to load images (paired or unpaired) to dental artifact
    reduction GANs.

    DICOM images are loaded using SimpleITK and cached in NRRD or NPY format for
    faster loading during training.
    """
    def __init__(self,
                 X_df: pd.DataFrame, Y_df: pd.DataFrame,
                 img_dir: str,
                 cache_dir: str,
                 file_type: str = "DICOM",
                 image_size=[8, 256, 256],
                 dim=3,
                 transform: Optional[Callable] = None,
                 num_workers: int = 1,
                 patient_id_col: str = "patient_id",
                 da_size_col: str = "has_artifact",
                 da_slice_col: str = "a_slice") :
        """ Initialize the class.

        Get the full paths to all images. Load and preprocess if the files if
        the cache does not already exist.

        Parameters
        ----------
        X_df : pandas.DataFrame
            A pandas dataframe containing the DA z-index, indexed by patient ID,
            of each image in domain X.
        Y_df : pandas.DataFrame
            A pandas dataframe containing the DA z-index, indexed by patient ID,
            of each image in domain Y.
        img_dir : str
            The path to the directory containing the raw image files.
        cache_dir :
            The path to the directory in which to cache preprocessed images.
        file_type: str, (default: "nrrd")
            The file type to load. Can be either "npy" or "nrrd" or "dicom".
        image_size : int, list, None
            The size of the image to use (same for both domains X and Y).
            - If None, use the entire image (centre_pix will be ignored).
            - If int=a, each image will be cropped to form a cube of size
              a^3 aroud centre_pix.
            - If list, it should have len=3 and represent the (z, y, x) size to
              crop all images to.
        dim : int
            The dimension of the output image. If 2, images will have shape
            If 2, the images will have shape(batch_size, z_size, y_size, x_size).
            If 3, the images will have shape
            (batch_size, 1, z_size, y_size, x_size).
        transform
            Callable used to transform the images after loading and preprocessing.
        num_workers : int
            The number of parallel processes to use for image preprocessing.
        patient_id_col: str (default: "patient_id")
            The name of the column lif containing the patient_id.
        da_size_col: str (default: "has_artifact")
            The name of the column containing the magnitude of the DA.
        da_slice_col: str  (default: a_slice)
            The name of the column containing the z-index of the DA.
        """
        self.X_df, self.Y_df = X_df, Y_df
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.file_type = file_type
        self.use_raw = use_raw
        self.img_size = np.array(img_size)
        self.dim = dim
        self.transform = transform
        self.num_workers = num_workers
        self.patient_id_col = patient_id_col
        self.da_size_col = da_size_col
        self.da_slice_col = da_slice_col

        # Get the correct function to load the raw image type
        self.load_img = self.get_img_loader()

        # Create a cache if needed. Check if cache already exists
        sample_path = os.path.join(self.cache_dir, X_df.index[0] + ".nrrd")
        if os.path.exists(sample_path) :
            # The sample file exists. Now check that the size is right
            sample_file = sitk.GetArrayFromImage(read_nrrd_image(sample_path))
            if np.array(sample_file.shape) != self.img_size :
                # The images are not correctly cached. Process them again
                self._prepare_data()
        else :
            # The images are not cached. Process them.
            os.makedirs(self.cache_dir)
            self._prepare_data()


    def _get_img_loader(self) :
        if self.file_type == "npy" :
            return np.load
        elif self.file_type == "nrrd" :
            return read_nrrd_image
        elif self.file_type == "dicom" :
            return read_dicom_image
        else :
            raise Exception(f"file_type {self.file_type} not accepted.")


    def _prepare_data(self) :
        """Preprocess and cache the dataset."""
        self.full_df = pd.concat([self.X_df, self.Y_df])
        Parallel(n_jobs=self.num_workers)(
            delayed(self._preprocess_image)(patient_id)
            for patient_id in self.full_df.index)


    def _preprocess_image(self, patient_id: str) :
        """Preprocess and cache a single image."""
        # Load image and DA index in original voxel spacing
        path = os.path.join(self.img_dir, patient_id)
        image = read_dicom_image(path)
        da_idx = int(self.full_df.at[patient_id, self.da_slice_col])

        # Resample image and DA slice to [1,1,1] voxel spacing
        da_coords = image.TransformIndexToPhysicalPoint([150, 150, da_idx]) # Physical coords of DA
        image = resample_image(image, new_spacing=[1, 1, 1])
        da_z = image.TransformPhysicalPointToIndex(da_coords)[2] # Index of DA in isotropic spacing

        ### Cropping ###
        # Get the centre of the head in the DA slice
        slice = sitk.GetArrayFromImage(image[:, :, da_z])
        t = threshold_otsu(np.clip(slice, -1000, 1000))
        slice = np.array(slice > t, dtype=int)
        com  = ndimage.measurements.center_of_mass(slice)
        y, x = int(com[0]) - 25, int(com[1])
        crop_centre = np.array([x, y, da_z])

        # Crop to required size around this point
        _min = np.floor(crop_centre - self.img_size / 2).astype(np.int64)
        _max = np.floor(crop_centre + self.img_size / 2).astype(np.int64)
        image = image[_min[0] : _max[0], _min[1] : _max[1], _min[2] : _max[2]]
        ### --------- ###

        # Save the image
        sitk.WriteImage(image, os.path.join(self.cache_path, f"{patient_id}.nrrd))


    def __getitem__(self, index) :
        raise NotImplementedError

    def __len__(self) :
        raise NotImplementedError







def load_img_names(dir, y_da_df=None, n_da_df=None, f_type="npy", suffix="", data_augmentation_factor=1, test_size=0.25):
    """
    Finds all images in the specified directory and returns a list of the file names
    in that directory.  The data is also split into a training and test group, then
    this list is augmented according to the data augmentation factor and ordered
    so that the RadiomicsFolder data loader does not train on the same image twice.
    Parameters
        dir : str
            A root directory containing one file per patient. Each of these
            npy files should contain 4 arrays stacked together: one for the image
            with artifacts, sinogram with artifact, image without artifact, and
            sinogram without artifact, for one patient.
        patient_ids : A list of patient IDs to use to look for files.
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

    suffix = suffix + "." + f_type

    if y_da_df is None or y_da_df is None:
        file_list = []
        # Get list of all the patients in the directory
        for file in os.listdir(dir) :
            if file.endswith(f_type) :
                # This file can be used; add this file to the file_list
                file_list.append(file)

        # We now have a list of all the valid data files
        # Seperate them into train and test sets
        train_files, test_files = train_test_split(np.array(file_list), test_size=test_size)

        # Increase the size of each list of files by repeating each element
        # data_augmentation_factor times
        aug_train_files = np.repeat(train_files, data_augmentation_factor)
        aug_test_files  = test_files
        # aug_test_files  = np.repeat(test_files,  data_augmentation_factor)

        # Randomize the order of the arrays
        train = shuffle(aug_train_files, random_state=0)
        test  = shuffle(aug_test_files,  random_state=0)
        return train, test
    else :
        y_da_df, n_da_df = y_da_df.to_frame(), n_da_df.to_frame()
        y_da_df["paths"] = dir + "/" + y_da_df.index.values + suffix   # Paths to DA+ images
        n_da_df["paths"] = dir + "/" + n_da_df.index.values + suffix   # Paths to DA- images

        y = y_da_df.loc[:, "DA_z" : "paths"].values                    # np array with one col
        n = n_da_df.loc[:, "DA_z" : "paths"].values                    # for paths, one for z-index

        y_train, y_test = train_test_split(y, test_size=test_size) # y_train, test
        n_train, n_test = train_test_split(n, test_size=test_size) # has 2 cols, first
                                                                   # for paths, 2nd for z-indeces

        y_aug_train = np.repeat(y_train, data_augmentation_factor, axis=0)
        n_aug_train = np.repeat(n_train, data_augmentation_factor, axis=0)

        y_train = shuffle(y_aug_train, random_state=0)
        n_train = shuffle(n_aug_train, random_state=0)
        y_test  = shuffle(y_test,  random_state=0)
        n_test  = shuffle(n_test,  random_state=0)

        # Returns an array with two cols, one for img path one for img DA z-index
        return y_train, n_train, y_test, n_test





class UnpairedDataset(Dataset):
    """
    Image dataset for ArtifactNet artifact removal GAN.

    Parameters:
    -----------
        X_image_names : array, shape(N, 1)
            A column vector were each row is the full path to the image file of
            a patient from domain X.
        Y_image_names : array, shape(N, 1)
            A column vector were each row is the full path to the image file of
            a patient from domain X.
        X_image_centre : list (shape (N,) or (N, 3)), None
            A list of the pixel index to use for each image from domain X.
            - If None, each image will be cropped around its centre.
            - If the list has shape (N,) (a 1D vector of len N), it will
              be assumed that each element is the z-index of the slice of
              interest and any cropping will occur around the centre of that slice.
            - If the list has shape (N, 3) (a matrix of len N with 3 columns)
              the columns represents the (z, y, x) coordinates of the pixel around
              which to crop and rotate. Each row is the z,y,x pixel for a patient.
            - List must have same order as X_image_names.
        Y_image_centre : list (shape (M,) or (M, 3)), None
            A list of the pixel index to use for each image from domain Y.
            - Must have the same number of columns as X_image_centre, but can
              have different length.
            - Treated exactly the same as X_image_centre.
        image_size : int, list, None
            The size of the image to use (same for both domains X and Y).
            - If None, use the entire image (centre_pix will be ignored).
            - If int=a, each image will be cropped to form a cube of size
              a^3 aroud centre_pix.
            - If list, it should have len=3 and represent the (z, y, x) size to
              crop all images to.
        file_type: str, (default: "nrrd")
            - The file type to load. Can be either "npy" or "nrrd" or "dicom".
        aug_factor : int (default: 1)
            The number of times to reuse each image, with transformations.
        dim : int
            The dimension of the output image. If 2, images will have shape
            If 2, the images will have shape(batch_size, z_size, y_size, x_size).
            If 3, the images will have shape
            (batch_size, 1, z_size, y_size, x_size).

    Attributes :
    ------------

    Methods :
    ---------


    """

    def __init__(self,
                 X_image_names,
                 Y_image_names,
                 X_image_centre=None,
                 Y_image_centre=None,
                 file_type="nrrd",
                 image_size=None,
                 aug_factor=1,
                 dim=3):

        self.dim          = dim
        self.x_img_paths  = X_image_names # Paths to images in domain X
        self.y_img_paths  = Y_image_names # Paths to images in domain Y
        self.file_type    = file_type
        self.x_img_centre = X_image_centre
        self.y_img_centre = Y_image_centre
        self.image_size   = image_size
        self.aug_factor   = aug_factor

        # Total number of images (max from either class)
        self.x_size = len(X_image_names)
        self.y_size = len(Y_image_names)
        self.dataset_length = self.x_size

        # Get the correct function to load the image type
        self.load_img = self.get_img_loader()

        # Get the cropping function appropriate for the format of centre_pix
        self.crop_img = self.get_cropper()

        self.count = 0


    def get_img_loader(self) :
        if self.file_type == "npy" :
            return np.load
        elif self.file_type == "nrrd" :
            return read_nrrd_image
        elif self.file_type == "dicom" :
            return read_dicom_image
        else :
            raise Exception(f"file_type {self.file_type} not accepted.")


    """ IMAGE CROPPING FUNTIONS """
    def get_cropper(self) :
        """ Identify which cropping function should be used """
        if self.image_size is None :                         # No cropping
            self.x_img_centre = np.zeros((self.x_size, 3))  # A dummy array that the cropping fn accepts
            self.y_img_centre = np.zeros((self.y_size, 3))
            return lambda X, size, p : X                # return original image


        else :
            if type(self.image_size) == int :
                i = self.image_size                     # Make image size into a list
                self.image_size = [i, i, i]             # of size in each axis

            if self.x_img_centre is None or self.y_img_centre is None :
                print("no image centres given")
                # No img centre given. Use centre cropping. Create dummy array for function to accept
                self.x_img_centre = np.zeros((self.x_size, 3))
                self.y_img_centre = np.zeros((self.y_size, 3))
                return self.centre_crop                 # crop around image centre
            else :
                # self.image_centre = np.array(self.image_centre)
                if self.x_img_centre.ndim == 1 :            # Only slice index given,
                    return self.slice_crop                  # crop around centre of slice
                elif self.x_img_centre.ndim == 2 and self.x_img_centre.shape[-1] == 3 :
                    return self.point_crop                  # Image centre given,
                                                            # Crop around it
                else :
                    raise Exception("This format of image_centre cannot be used.")

    def centre_crop(self, z, size=None, p=None) :
        """ Crop an image arond its centre """
        p = np.array(z.shape) // 2   # Centre of image
        return self.crop(z, p, size)

    def slice_crop(self, z, size=None, p:int=None) :
        """ Crop an image z around the centre of a particular slice p.  """
        py, px = np.array(z[p, :, :].shape) // 2   # Centre of slice
        return self.crop(z, [p, py, px], size)

    def point_crop(self, z, size=None, p:int=list) :
        """ Crop an image around a point, given as p = [z, y, x] """
        return self.crop(z, p, size)

    def crop(self, Z, p, size) :
        # Get size of image volume in each direction
        zs, ys, xs = size[0] // 2, size[1] // 2, size[2] // 2
        # Get coordinates of pixel around which to crop
        z, y, x = p[0], p[1], p[2]
        if self.image_size[0] == 1 :
            return Z[z, y-ys : y+ys, x-xs : x+xs]
        else :
            return Z[z-zs : z+zs, y-ys : y+ys, x-xs : x+xs]
    """ ########################## """


    def transform(self, X) :
        """When we start using data augmentation we will call transform to
        apply random rotations, translations etc to the data
        Parameters :
        ------------
        X (np.ndarray) :
            The untransformed image in the form of a numpy ndarray.

        Returns :
        ----------
        The transformed image as a pytorch 32-bit tensor.
        """

        if self.aug_factor > 1 :
            # Apply random rotation
            X = affine_transform(X, angle=30.0, pixels=(20, 20))
            X = torch.tensor(X, dtype=torch.float32)
        else :
            X = torch.tensor(X, dtype=torch.float32)

        # Apply intensity windowing and scaling
        min_val = -1000.0
        max_val =  1000.0
        X = torch.clamp(X, min=min_val, max=max_val)# Make range (-1000, 1000)
        X = X / 1000.0                              # Make range (-1, 1)

        return X


    def __getitem__(self, index):
        '''
        Returns a pair of DA+ and DA- images from that dataset and applies
        transforms. Image i from DA+ images will be taken and a random DA-
        image will be taken. This method is called by calling dataset[index]
        in the pytorch lightning module.
        '''
        # Randomize index for images in Y domain to avoid pairs
        x_index = index
        y_index = np.random.randint(0, self.y_size - 1)

        # Load the image from each class
        X = self.load_img(self.x_img_paths[x_index], mmap_mode="r")
        Y = self.load_img(self.y_img_paths[y_index], mmap_mode="r")

        # Make datatype ints (not unsigned ints)
        X, Y = X.astype(np.int16), Y.astype(np.int16)

        # Transform the image (augmentation)
        X = self.transform(X)
        Y = self.transform(Y)

        # Crop the image
        X = self.crop_img(X, size=self.image_size, p=self.x_img_centre[x_index])
        Y = self.crop_img(Y, size=self.image_size, p=self.y_img_centre[y_index])

        # The Pytorch model takes a tensor of shape (batch_size, in_Channels, depth, height, width)
        # or if the input image and model is 2D :   (batch_size, depth, height, width)
        # Reshape the arrays to add another dimension
        if self.dim == 2 :
            X = X.reshape(self.image_size[0], self.image_size[1], self.image_size[2])
            Y = Y.reshape(self.image_size[0], self.image_size[1], self.image_size[2])
        else :
            X = X.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])
            Y = Y.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])

        return X, Y




    def __len__(self):
        return self.dataset_length








class PairedTestset(t_data.Dataset):
    """
    Image dataset for ArtifactNet test images where we have pairs.

    Parameters:
    -----------
        X_image_dir : str
            The path to the directory containing the images from domain X. Each
            file name should be unique and correspond to a paired image with the
            same name in Y_image_dir. Each image should be centered around its DA
            slice.
        Y_image_dir : str
            The path to the directory containing the images from domain Y.
        image_size : int, list, None
            The size of the image to use (same for both domains X and Y).
            - If None, use the entire image.
            - If int=a, each image will be cropped to form a cube of size a^3
            aroud the central pixel of the image.
            - If list, it should have len=3 and represent the (z, y, x) size to
              crop all images to, around the centre.
        file_type: str, (default: "nrrd")
            - The file type to load. Can be either "npy" or "nrrd" or "dicom".
        dim : int
            The dimension of the output image. If 2, images will have shape
            If 2, the images will have shape(batch_size, z_size, y_size, x_size).
            If 3, the images will have shape
            (batch_size, 1, z_size, y_size, x_size).

    Attributes :
    ------------

    Methods :
    ---------


    """

    def __init__(self,
                 X_image_dir,
                 Y_image_dir,
                 file_type="nrrd",
                 image_size=None,
                 aug_factor=1,
                 dim=3):

        self.x_image_dir  = X_image_dir # Path to images in domain X
        self.y_image_dir  = Y_image_dir # Path to images in domain Y
        self.file_type    = file_type
        self.image_size   = image_size
        self.dim          = dim

        # Get a list of all images
        self.img_files = os.listdir(X_image_dir)

        # Total number of images
        self.dataset_length = len(self.img_files)

        # Get the correct function to load the image type
        self.load_img = self.get_img_loader()


    def get_img_loader(self) :
        if self.file_type == "npy" :
            return np.load
        elif self.file_type == "nrrd" :
            return read_nrrd_image
        elif self.file_type == "dicom" :
            return read_dicom_image
        else :
            raise Exception(f"file_type {self.file_type} not accepted.")


    def centre_crop(img) :
        """ Crop an image arond its centre """
        z, y, x    = np.array(img.shape) // 2   # Centre of image
        zs, ys, xs = np.array(self.image_size) // 2

        if self.image_size[0] == 1 :
            return img[z, y-ys : y+ys, x-xs : x+xs]
        else :
            return img[z-zs : z+zs, y-ys : y+ys, x-xs : x+xs]

    def transform(X) :
        X = torch.tensor(X, dtype=torch.float32)

        # Apply intensity windowing and scaling
        min_val = -1000.0
        max_val =  1000.0
        X = torch.clamp(X, min=min_val, max=max_val)# Make range (-1000, 1000)
        X = X / 1000.0                              # Make range (-1, 1)
        return X


    def __getitem__(self, index) :
        # Get the name of the image file at index
        file_name = self.img_files[index]

        # Load the image from each domain
        X = self.load_img(os.path.join(self.x_image_dir), file_name)
        Y = self.load_img(os.path.join(self.y_image_dir), file_name)

        # Make datatype ints (not unsigned ints)
        X, Y = X.astype(np.int16), Y.astype(np.int16)

        # Transform the image (augmentation)
        X = self.transform(X)
        Y = self.transform(Y)

        # Crop the image
        X = self.centre_crop(X)
        Y = self.centre_crop(Y)

        # The Pytorch model takes a tensor of shape (batch_size, in_Channels, depth, height, width)
        # or if the input image and model is 2D :   (batch_size, depth, height, width)
        # Reshape the arrays to add another dimension
        if self.dim == 2 :
            X = X.reshape(self.image_size[0], self.image_size[1], self.image_size[2])
            Y = Y.reshape(self.image_size[0], self.image_size[1], self.image_size[2])
        else :
            X = X.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])
            Y = Y.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])

        return X, Y








class PairedDataset(t_data.Dataset):
    """
    Image dataset for ArtifactNet artifact removal GAN.

    Attributes:
        image_root_dir : str
            The absolute path to the directory containing the image data.
        image_names : list
            A list of file names. These can be tuples, each containing the paths
            for both paired images. Alternatively they can be a 1D list of paths
            where each (npy/nrrd) file contains boths images to make the pair.
        centre_pix : list (shape (N, 2) or (3, N, 2), None
            A list of the pixel index to use for each image.
            - If None, the image will be cropped around its centre to image_size.
            - If the list has shape (N, 2), it will be assumed that each element is
              the z-index of the slice of interest with each column corresponding to
              one of the two image classes and any cropping will occur around the
              centre of that slice.
            - If the list has shape (N, 3) each element represents the (z, y, x)
              coordinates of the pixel around which to crop and rotate.
            - List must have same order as image_names.
        image_size : int, list, None
            The size of the image to use.
            - If None, use the entire image (centre_pix will be ignored).
            - If int=a, each image will be cropped to form a cube of size
              a^3 aroud centre_pix.
            - If list, it must have shape (N, 3) and each element represents the
            (z, y, x) size of each image, cropped around each image's centre pix.
        transform : list
            - A list containing the names of the random transforms to be applied.
            - Accepted names are ["rotatate", "flip" ]
        file_type: str, (default: "nrrd")
            - The file type to load. Can be either "npy" or "nrrd"
    """
    def __init__(self, arg):
        super(PairedDataset, self).__init__()
        self.arg = arg

    def __getitem__(self, index):
        '''

        '''
        # If we are accessing one image at a time, get the image

        image_file_name = self.image_names[index]
        image_path = os.path.join(self.root_dir, image_file_name)



        # If data is paired,

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
        artifact_tensor    = torch.tensor( artifact_arr,    dtype=torch.float32)
        no_artifact_tensor = torch.tensor( no_artifact_arr, dtype=torch.float32)


        return artifact_tensor, no_artifact_tensor

    def __len__(self):
        return self.dataset_length
