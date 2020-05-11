import io
import os
import numpy as np


import SimpleITK as sitk
from skimage.transform import resize

def resample_image(image, new_spacing=[1,1,1]):

    #Resample image to common resolution of 1x1x1
    # new_spacing = [1,1,1]

    #Set up SitK resampling image filter
    rif = sitk.ResampleImageFilter()
    rif.SetOutputSpacing(new_spacing)
    rif.SetOutputDirection(image.GetDirection())

    #Get original image size and spacing
    orig_size = np.array(image.GetSize(), dtype = np.int)
    orig_spacing = np.array(image.GetSpacing())

    #Calculate new image size based on current size and desired spacing.
    new_size = np.ceil(orig_size*(orig_spacing/new_spacing)).astype(np.int)
    new_size = [int(s) for s in new_size]

    #Set up SitK resampling parameters
    rif.SetSize(new_size)
    rif.SetOutputOrigin(image.GetOrigin())
    rif.SetOutputPixelType(image.GetPixelID())
    rif.SetInterpolator(sitk.sitkLinear)

    #Resample image and generate numpy array from image
    resampledImage = rif.Execute(image)
    imageArray = sitk.GetArrayFromImage(resampledImage)

    return imageArray



def read_nrrd_image(nrrd_file_path) :
    image = sitk.ReadImage(nrrd_file_path)

    # Resize the image
    resamp_img = resample_image(image, new_spacing=[1,1,1])

    imageArray = sitk.GetArrayFromImage(image)

    return imageArray



def read_dicom_image(dicom_path) :
    """Return SITK image given the absolute path
    to a DICOM series."""
    reader = sitk.ImageSeriesReader()
    # path is the directory of the dicom folder
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Resize the image
    resamp_img = resample_image(image, new_spacing=[1,1,1])

    # Comvert image to np array
    imageArray = sitk.GetArrayFromImage(resamp_img)

    return imageArray
