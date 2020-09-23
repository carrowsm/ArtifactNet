import io
import os
import numpy as np
import torch
from typing import Callable, Optional, Union, Sequence

import SimpleITK as sitk
from skimage.transform import resize

from data.preprocessing import resample_image, read_dicom_image, read_nrrd_image




class PostProcessor :
    """PostProcessor class to take the ouput tensor from a model and save it as
    a full-size NRRD or DICOM.
    """
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 output_spacing: Union[Sequence, str] = "orig",
                 input_file_type: str = "DICOM",
                 output_file_type: str = "nrrd"):
        """ Initialize the class
        Parameters
        ----------
        input_dir (str)
            The directory containing the original images. Each patient's image
            should be contained in a subdirectory named with the patient's ID.
        output_dir (str)
            The directory in which to save the generated images.
        output_spacing (Callable, str)
            The spacing of the output SITK file. Expected to be [x, y, z]. If
            'orig', the original spacing will be used.
        input_file_type (str)
            The file type of the original images. Can be 'DICOM' or 'nrrd'.
        output_file_type (str)
            The file type to save the output files. Can be 'DICOM' or 'nrrd'.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_spacing = output_spacing
        self.input_file_type = input_file_type
        self.output_file_type = output_file_type

        # Get the correct function to with which to load images
        if self.input_file_type == 'nrrd' :
            self.read_original_img = read_nrrd_image
        elif self.input_file_type == 'DICOM' :
            self.read_original_img = read_dicom_image
        else :
            raise NotImplementedError("input_file_type must be 'DICOM' or 'nrrd'.")

        # Get the correct function to with which to save images
        if self.output_file_type == 'nrrd' :
            self.save_output_img = sitk.WriteImage
        elif self.output_file_type == 'DICOM' :
            raise NotImplementedError(
            "Saving output as DICOM is not currently supported. Please use 'nrrd'.")
        else :
            raise NotImplementedError("output_file_type must be 'dicom' or 'nrrd'.")



    def __call__(self, model_output: torch.Tensor, patient_id: str, img_centre: Sequence) :
        """ Process a single image from the output of a deep learning generator.
        Convert the image to an SITK image, combine it with the original
        full-size DICOM, and save the image.

        Parameters
        ----------
        model_output (torch.Tensor)
            The direct output from the generator model. This will be used to
            replace pixels in the original image.
        patient_id (str)
            The patient ID or MRN of the current patient. This will be used to
            access the original DICOM and save the output.
        img_centre (Sequence, [x, y, z])
            The location of the centre of the subvolume in physical coordinates.
        """
        if len(model_output.shape) > 3 :
            model_output = model_output[0, 0, :, :, :]

        # Convert model output image back to HU
        model_output = model_output * 1000.0 # Make range (-1000, 1000)

        # Convert the tensor image to SITK (with 1mm voxel spacing)
        sub_img = sitk.GetImageFromArray(model_output.numpy())

        # Load original (uncorrected) image
        orig_path = os.path.join(self.input_dir, f"{patient_id}.{self.input_file_type}")
        full_img = self.read_original_img(orig_path)
        orig_spacing = full_img.GetSpacing()
        full_img = sitk.Clamp(full_img, lowerBound=-1000.0, upperBound=1000.0)

        # Resample sub image to the same spacing as full image
        sub_img_size = np.array(sub_img.GetSize())
        # full_img = resample_image(full_img, [1.0, 1.0, 1.0])
        sub_img = resample_image(sub_img, orig_spacing)

        # Get the index of the subvolume center
        sub_img_center = np.array(full_img.TransformPhysicalPointToIndex(img_centre))

        # Insert the subvolume pixels into the full original image
        _min = np.floor(sub_img_center - sub_img_size / 2).astype(int)
        _max = np.floor(sub_img_center + sub_img_size / 2).astype(int)
        full_img = sitk.Paste(full_img, sub_img, sub_img.GetSize(),
                              destinationIndex=_min.tolist() )

        # Resample the resulting image to the required spacing
        if self.output_spacing != 'orig' :
            full_img = resample_image(full_img, self.output_spacing)

        # Save the image
        file_name = f"{patient_id}.{self.output_file_type}"
        self.save_output_img(full_img, os.path.join(self.output_dir, file_name))
