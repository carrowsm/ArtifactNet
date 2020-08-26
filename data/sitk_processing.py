import io
import os
import numpy as np


import SimpleITK as sitk
from skimage.transform import resize


def get_dicom_path(path) :
    """ Return the path to a .DICOM series that exists in some
    subdirectory of a directory 'path'."""
    for root, dirs, files in os.walk(path, topdown=True):
       for name in dirs:
          if os.path.join(root, name).endswith(".DICOM") :
              return os.path.join(root, name)


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

    # Resample the image
    image = resample_image(image, new_spacing=[1,1,1])

    # image = sitk.GetArrayFromImage(image)

    return image



def read_dicom_image(path) :
    """Return SITK image given the path to a directory containing
    a dicom series"""
    if os.path.exists(path) and path.endswith(".DICOM") :
        dicom_path = path
    else :
        dicom_path = get_dicom_path(path)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Resize the image
    # image = resample_image(image, new_spacing=[1,1,1])

    # Comvert image to np array
    # imageArray = sitk.GetArrayFromImage(resamp_img)

    return image


class Cropper:
    """docstring for Cropper."""

    def __init__(self, image_size, ):
        super(Cropper, self).__init__()
        self.arg = arg

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
