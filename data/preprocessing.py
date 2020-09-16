import io
import os
import numpy as np
from typing import Callable, Optional, Union, Sequence

import SimpleITK as sitk
from skimage.transform import resize


def get_dicom_path(path) :
    """ Return the path to a .DICOM series that exists in some
    subdirectory of a directory 'path'."""
    for root, dirs, files in os.walk(path, topdown=True):
       for name in dirs:
          if os.path.join(root, name).endswith(".DICOM") :
              return os.path.join(root, name)



def read_dicom_image(path, pixel_type=sitk.sitkFloat32) :
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
    image = sitk.Cast(image, pixel_type) # Change image pixel type

    return image


def read_nrrd_image(path, pixel_type=sitk.sitkFloat32) :
    """Return SITK image given the path to an NRRD file."""
    image = sitk.ReadImage(path)
    image = sitk.Cast(image, pixel_type) # Change image pixel type
    return image




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
        resampling). If list, assumes format [x, y, z].

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
