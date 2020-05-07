import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.utils.data as t_data
import torchvision.transforms as transforms


from data.sitk_processing import read_dicom_image, resample_image, read_nrrd_image



def load_image_data_frame(path) :
    """ Load data Frame containing the DA label and location of each patient"""
    df = pd.read_csv(path,
                     # index_col="patient_id",
                     usecols=["patient_id", "has_artifact", "a_slice"],
                     dtype=str, na_values=['nan', 'NaN', ''])
    df.set_index("patient_id", inplace=True)
    df["DA_z"] = df["a_slice"].astype(int)

    da_plus  = df[df["has_artifact"] == "2"]
    da_minus = df[df["has_artifact"] == "0"]

    return da_plus["DA_z"], da_minus["DA_z"]



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
    if len(suffix) > 0  :
        suffix = suffix + "." + f_type

    if y_da_df is None or y_da_df is None:
        file_list = []
        # Get list of all the patients in the directory
        for file in os.listdir(dir) :
            extension = file.split(".")[-1]
            if extension == f_type :
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
    else :
        y_da_df, n_da_df = y_da_df.to_frame(), n_da_df.to_frame()
        y_da_df["paths"] = dir + "/" + y_da_df.index.values + suffix   # Paths to DA+ images
        n_da_df["paths"] = dir + "/" + n_da_df.index.values + suffix   # Paths to DA- images

        y = y_da_df.loc[:, "DA_z" : "paths"].values           # np array with one col
        n = n_da_df.loc[:, "DA_z" : "paths"].values           # for paths, one for z-index

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





class UnpairedDataset(t_data.Dataset):
    """
    Image dataset for ArtifactNet artifact removal GAN.

    Attributes:

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
        transform : list
            - A list containing the names of the random transforms to be applied.
            - Accepted names are ["rotatate", "flip" ].
        file_type: str, (default: "nrrd")
            - The file type to load. Can be either "npy" or "nrrd" or "dicom".


    Methods :


    """

    def __init__(self,
                 X_image_names,
                 Y_image_names,
                 X_image_centre=None,
                 Y_image_centre=None,
                 file_type="nrrd",
                 image_size=None,
                 transform=None,
                 dim="3D"):

        self.dim          = dim
        # self.root_dir     = image_root_dir
        self.x_img_paths  = X_image_names # Paths to images in domain X
        self.y_img_paths  = Y_image_names # Paths to images in domain Y
        self.file_type    = file_type
        self.x_img_centre = X_image_centre
        self.y_img_centre = Y_image_centre
        self.image_size   = image_size
        self.transforms   = transforms.Compose([
                             transforms.Normalize([-814.7], [615.9])
                             ])


        # Total number of images (max from either class)
        self.x_size = len(X_image_names)
        self.y_size = len(Y_image_names)
        self.dataset_length = np.max([self.x_size, self.y_size])

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
            self.x_img_centre = np.zeros((self.dataset_length, 3))  # A dummy array that the cropping fn accepts
            self.y_img_centre = np.zeros((self.dataset_length, 3))
            return lambda X, size, p : X                # return original image
        else :
            if type(self.image_size) == int :
                i = self.image_size                     # Make image size into a list
                self.image_size = [i, i, i]             # of size in each axis

            if self.x_img_centre is None :              # No centre pixel given,
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
        if self.dim == "2D" :
            return Z[z, y-ys : y+ys, x-xs : x+xs]
        else :
            return Z[z-zs : z+zs, y-ys : y+ys, x-xs : x+xs]
    """ ########################## """



    def lin_norm(self, array, newMin=0., newMax=1.) :
        oldMin, oldMax = np.min(array), np.max(array)
        return (array - oldMin) * ((newMax - newMin)/(oldMax - oldMin)) + newMin

    def transform(self, X) :
        """When we start using data augmentation we will call transform to
        apply random rotations, translations etc to the data"""
        min_val = -1000.0
        max_val =  1500.0
        X = np.clip(X, min_val, max_val) - min_val  # Make min 0
        X = torch.tensor(X, dtype=torch.float32)
        # X = self.transforms(X)                    # Apply augmentations
        X = X / (max_val - min_val)                 # Make range (0, 1)
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
        X = self.load_img(self.x_img_paths[x_index])
        Y = self.load_img(self.y_img_paths[y_index])

        # Make datatype ints (not unsigned ints)
        X, Y = X.astype(np.int16), Y.astype(np.int16)


        # Crop the image
        X = self.crop_img(X, size=self.image_size, p=self.x_img_centre[x_index])
        Y = self.crop_img(Y, size=self.image_size, p=self.y_img_centre[y_index])

        # Check image has the right size
        # self.check_size(X)
        # self.check_size(Y)

        # Transform the image (augmentation)
        X_tensor = self.transform(X)
        Y_tensor = self.transform(Y)

        # The Pytorch model takes a tensor of shape (batch_size, in_Channels, depth, height, width)
        # Reshape the arrays to add another dimension
        try :
            if self.dim == "2D" :
                print(X_tensor.shape)
                X_tensor = X_tensor.reshape(1, self.image_size[1], self.image_size[2])
                print(X_tensor.shape)
                Y_tensor = Y_tensor.reshape(1, self.image_size[1], self.image_size[2])
            else :
                X_tensor = X_tensor.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])
                Y_tensor = Y_tensor.reshape(1, self.image_size[0], self.image_size[1], self.image_size[2])
        except :
            print("image not found")
            i = np.random.randint(0, self.x_size - 1)
            return self[i]


        return X_tensor, Y_tensor


    def __len__(self):
        return self.dataset_length




class PairedDataset(object):
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







if __name__ == '__main__':
    ### Unit testing ###
    import matplotlib as mpl
    mpl.use("Qt5Agg")
    import matplotlib.pyplot as plt

    def plt_imgs(X, Y, fname="") :
        fig, ax = plt.subplots(ncols=2, nrows=1)
        ax[0].imshow(x[x.shape[0]//2, :, :])
        ax[1].imshow(y[x.shape[0]//2, :, :])
        ax[0].set_title("X")
        ax[1].set_title("Y")
        plt.savefig(fname)


    # Import CSV containing DA labels
    csv_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
    dir = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"
    y_df, n_df = load_image_data_frame(csv_path)


    # Create train and test sets for each DA+ and DA- imgs
    files = load_img_names(dir, y_da_df=y_df, n_da_df=n_df,
                           f_type="npy", suffix="_img",
                           data_augmentation_factor=1,
                           test_size=0.25)

    y_train, n_train, y_test, n_test = files

    """
    y_train is a matrix with len N and two columns:
    y_train[:, 0] = z-index of DA in each patient
    y_train[:, 1] = full path to each patient's image file
    """


    print(y_train.shape, n_train.shape)
    print(y_train[:, 0].shape)


    # Test data loader
    dataset = UnpairedDataset(y_train[ :, 1],
                              n_train[ :, 1],
                              file_type="npy",
                              X_image_centre=y_train[:, 0],
                              Y_image_centre=n_train[:, 0],
                              image_size=[50, 500, 500],
                              transform=None)
    # Get a test image set
    x, y = dataset[2]

    print(x.shape, y.shape)

    # Save the image
    plt_imgs(x, y, fname="1.png")
