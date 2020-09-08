import os
import sys
from typing import Callable, Optional, Tuple, Sequence
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy import ndimage
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset
import torchvision

from data.preprocessing import read_dicom_image, resample_image
from data.transforms import AffineTransform


def load_image_data_frame(path, img_X: Sequence[str], img_Y: Sequence[str],
                          label_col="has_artifact") :
    """ Load data Frame containing the DA label and location of each patient
    Parameters :
    ------------
    path (str) :    Full path to the CSV containing the image IDs and corresponding
                    labels.
    img_X (list) :  The CSV label for images in domain X. Must be a list of labels
                    as strings. The union of all labels in the list will be used.
    img_Y (list) :  The CSV label for images in domain Y.
    label_col (str):The CSV column name containing the image labels. Default is
                    'has_artifact'.
    Returns :
    ---------
    One DF for each image domain with the z-index of the DA slice for each patient.
    """
    df = pd.read_csv(path,
                     usecols=["patient_id", label_col, "DA_z", "a_slice"],
                     dtype=str, na_values=['nan', 'NaN', ''])
    df.set_index("patient_id", inplace=True)
    df["DA_z"] = df["DA_z"].astype(int)

    X_df = df[df[label_col].isin(img_X)]  # Patients in domain X (DA+)
    Y_df = df[df[label_col].isin(img_Y)]  # Patients in domain X (DA-)

    return X_df, Y_df



class BaseDataset(Dataset):
    """Dataset class used to load images (paired or unpaired) to dental artifact
    reduction GANs.

    DICOM images are loaded using SimpleITK and cached in NRRD or NPY format for
    faster loading during training.
    """
    def __init__(self,
                 X_df: pd.DataFrame, Y_df: pd.DataFrame,
                 image_dir: str,
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
        self.img_dir = image_dir
        self.cache_dir = cache_dir
        self.file_type = file_type
        self.img_size = np.array(image_size)
        self.dim = dim
        self.transform = transform
        self.num_workers = num_workers
        self.patient_id_col = patient_id_col
        self.da_size_col = da_size_col
        self.da_slice_col = da_slice_col

        # Get the number of images in each domain
        self.x_size, self.y_size = len(X_df), len(Y_df)
        self.x_ids, self.y_ids = self.X_df.index, self.Y_df.index

        # Create a cache if needed. Check if cache already exists
        sample_path = os.path.join(self.cache_dir, self.x_ids[0] + ".nrrd")
        print("")
        if os.path.exists(sample_path) :
            # The sample file exists. Now check that the size is right
            sample_file = sitk.GetArrayFromImage(sitk.ReadImage(sample_path))
            if (np.array(sample_file.shape) != self.img_size).any() :
                print(f"Image {sample_file.shape} is different from expected {self.img_size}")
                # The images are not correctly cached. Process them again
                self._prepare_data()
        else :
            print(f"cache {sample_path} does not exist")
            # The images are not cached. Process them.
            os.makedirs(self.cache_dir, exist_ok=True)
            self._prepare_data()
        print("Data successfully cached\n")


    def _get_img_loader(self) :
        if self.file_type == "nrrd" :
            return sitk.ReadImage
        elif self.file_type == "DICOM" :
            return read_dicom_image
        else :
            raise Exception(f"file_type {self.file_type} not accepted.")


    def _prepare_data(self) :
        """Preprocess and cache the dataset."""
        # Get the correct function to load the raw image type
        self.load_img = self._get_img_loader()
        self.full_df = pd.concat([self.X_df, self.Y_df])
        self.full_df["img_center"] = np.nan # col for keeping track of subvolume centre

        print(f"Preprocessing {len(self.full_df)} images. This may take a moment.")
        Parallel(n_jobs=self.num_workers)(
            delayed(self._preprocess_image)(patient_id)
            for patient_id in self.full_df.index)


    def _preprocess_image(self, patient_id: str) :
        """Preprocess and cache a single image."""
        # Load image and DA index in original voxel spacing
        path = os.path.join(self.img_dir, patient_id)
        image = read_dicom_image(path)
        da_idx = int(self.full_df.at[patient_id, self.da_slice_col])

        # Resample image and DA slice to [1,1,1] mm voxel spacing
        da_coords = image.TransformIndexToPhysicalPoint([150, 150, da_idx])
        image = resample_image(image, [1.0, 1.0, 1.0])
        da_z = image.TransformPhysicalPointToIndex(da_coords)[2] # DA index in 1mm spacing

        ### Cropping ###
        # Get the centre of the head in the DA slice
        slice = sitk.GetArrayFromImage(image[:, :, da_z])
        t = threshold_otsu(np.clip(slice, -1000, 1000))
        slice = np.array(slice > t, dtype=int)
        com  = ndimage.measurements.center_of_mass(slice)
        y, x = int(com[0]) - 25, int(com[1])
        crop_centre = np.array([x, y, da_z])
        crop_size   = self.img_size[::-1] # Reverse array to comply with sitk indexing

        # Save the location of the centre of the image
        phys_centre = image.TransformIndexToPhysicalPoint((x, y, da_z))
        self.full_df.at[patient_id, "img_center"] = phys_centre

        # Crop to required size around this point
        _min = np.floor(crop_centre - crop_size / 2).astype(np.int64)
        _max = np.floor(crop_centre + crop_size / 2).astype(np.int64)
        image = image[_min[0] : _max[0], _min[1] : _max[1], _min[2] : _max[2]]
        ### --------- ###

        # Save the image
        sitk.WriteImage(image, os.path.join(self.cache_dir, f"{patient_id}.nrrd"))




    def __getitem__(self, index) :
        raise NotImplementedError

    def __len__(self) :
        """Return the number of samples in the dataset.

        The 'size' of the dataset is the number of images in domain X. For every
        image in domain X a random image from Y will be selected. It will take
        more than one epoch to sample all images in Y if len(df_X) < len(df_Y).
        """
        return self.x_size






class UnpairedDataset(BaseDataset):
    """Dataset class used to load unpaired images from two domains X and Y.

    Since this is a subclass of BaseDataset, the images will be cached if needed
    and read from the cache during training.
    """
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor] :
        """ Get a the image at index from domain X and get an accompanying
        random image from domain Y. Assume the images are preprocessed (sized)
        and cached.

        Parameters
        ----------
        index (int)
            The index of the image to take from domain X.
        """

        # Randomize index for images in Y domain to avoid pairs
        x_index = index
        y_index = np.random.randint(0, self.y_size - 1)
        x_patient_id = self.x_ids[x_index]
        y_patient_id = self.y_ids[y_index]

        # Load the sitk image from each class
        X = sitk.ReadImage(os.path.join(self.cache_dir, f"{x_patient_id}.nrrd"))
        Y = sitk.ReadImage(os.path.join(self.cache_dir, f"{y_patient_id}.nrrd"))

        # Apply random transforms
        if self.transform is not None:
            X, Y = self.transform((X, Y)) # Apply the same transform to both images

        if self.dim == 2 : # Use the channels as third dimension
            X = X.reshape(self.img_size[0], self.img_size[1], self.img_size[2])
            Y = Y.reshape(self.img_size[0], self.img_size[1], self.img_size[2])

        return X, Y


class PairedDataset(BaseDataset):
    """Dataloader for a PairedDataset."""
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        # Check that the two dataframes are the same length
        if self.x_size != self.y_size :
            raise ValueError("Paired datasets must have the same size.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor] :
        """ Get a the image at index from domain X and get an accompanying
        random image from domain Y. Assume the images are preprocessed (sized)
        and cached.

        Parameters
        ----------
        index (int)
            The index of the image in both domains.
        """
        x_patient_id = self.x_ids[index]
        y_patient_id = self.y_ids[index]

        # Load the sitk image from each class
        X = sitk.ReadImage(os.path.join(self.cache_dir, f"{x_patient_id}.nrrd"))
        Y = sitk.ReadImage(os.path.join(self.cache_dir, f"{y_patient_id}.nrrd"))

        # Apply random transforms
        if self.transform is not None: # Apply the same transform to both images
            X, Y = self.transform((X, Y))

        if self.dim == 2 : # Use the channels as third dimension
            X = X.reshape(self.img_size[0], self.img_size[1], self.img_size[2])
            Y = Y.reshape(self.img_size[0], self.img_size[1], self.img_size[2])

        return X, Y
