import io
import os
import numpy as np
from typing import Callable, Optional, Tuple, Union, Sequence

import SimpleITK as sitk
from skimage.transform import resize


def get_dicom_path(path) :
    """ Return the path to a .DICOM series that exists in some
    subdirectory of a directory 'path'."""
    for root, dirs, files in os.walk(path, topdown=True):
       for name in dirs:
          if os.path.join(root, name).endswith(".DICOM") :
              return os.path.join(root, name)


def resample_image(image: sitk.Image,
             spacing: Union[float, Sequence[float], np.ndarray],
             interpolation: str = "linear",
             anti_alias: bool = True,
             anti_alias_sigma: Optional[float] = None,
             transform: Optional[sitk.Transform] = None,
             output_size: Optional[Sequence[float]] = None) -> sitk.Image:
    """Resample image to a given spacing, optionally applying a transformation.

    Parameters
    ----------
    image
        The image to be resampled.

    spacing
        The new image spacing. If float, assumes the same spacing in all
        directions. Alternatively, a sequence of floats can be passed to
        specify spacing along each dimension. Passing 0 at any position will
        keep the original spacing along that dimension (useful for in-plane
        resampling).

    interpolation, optional
        The interpolation method to use. Valid options are:
        - "linear" for bi/trilinear interpolation (default)
        - "nearest" for nearest neighbour interpolation
        - "bspline" for order-3 b-spline interpolation

    anti_alias, optional
        Whether to smooth the image with a Gaussian kernel before resampling.
        Only used when downsampling, i.e. when `spacing < image.GetSpacing()`.
        This should be used to avoid aliasing artifacts.

    anti_alias_sigma, optional
        The standard deviation of the Gaussian kernel used for anti-aliasing.

    transform, optional
        Transform to apply to input coordinates when resampling. If None,
        defaults to identity transformation.

    output_size, optional
        Size of the output image. If None, it is computed to preserve the
        whole extent of the input image.

    Returns
    -------
    sitk.Image
        The resampled image.
    """
    INTERPOLATORS = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }

    try:
        interpolator = INTERPOLATORS[interpolation]
    except KeyError:
        raise ValueError(
            f"interpolator must be one of {list(INTERPOLATORS.keys())}, got {interpolator}."
        )

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if isinstance(spacing, (float, int)):
        new_spacing = np.repeat(spacing,
                                len(original_spacing)).astype(np.float64)
    else:
        spacing = np.asarray(spacing)
        new_spacing = np.where(spacing == 0, original_spacing, spacing)

    if not output_size:
        new_size = np.floor(original_size * original_spacing / new_spacing).astype(np.int)
    else:
        new_size = np.asarray(output_size)

    rif = sitk.ResampleImageFilter()
    rif.SetOutputOrigin(image.GetOrigin())
    rif.SetOutputSpacing(new_spacing)
    rif.SetOutputDirection(image.GetDirection())
    rif.SetSize(new_size.tolist())

    if transform is not None:
        rif.SetTransform(transform)

    downsample = new_spacing > original_spacing
    if downsample.any() and anti_alias:
        if not anti_alias_sigma:
            # sigma computation adapted from scikit-image
            # https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py
            anti_alias_sigma = np.maximum(1e-11, (original_spacing / new_spacing - 1) / 2)
        sigma = np.where(downsample, anti_alias_sigma, 1e-11)
        image = sitk.SmoothingRecursiveGaussian(image, sigma)

    rif.SetInterpolator(interpolator)
    resampled_image = rif.Execute(image)

    return resampled_image



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
