import numpy as np
import scipy.ndimage
import torch
import SimpleITK as sitk
from typing import Tuple




def affine_transform(a, angle=15.0, pixels=(20, 20), fill_mode='nearest') :
    """ Apply a random rotation and translation to a 3D numpy array in the x-y plane.
    Parameters :
    ------------
    a : ndarray
        A 3D numpy array to transform.
    angle : float
        The bounds of the random rotation. The actual rotation angle will be a
        randomly generated number in the range (-angle, angle) in the x-y plane.
    pixels : tuple of 2 ints
        The number of pixels by which to randomly translate the image in the x
        and y planes. The actual translation will be a random number of pixels
        along the x-axis in the range (-pixels[0], pixels[0]) and along the
        y-axis in the range (-pixels[1], pixels[1]).
    fill_mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        Method for filling in new values after image rotation. This string is
        given to scipy.ndimage.rotate. See documentation here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html

    Returns :
        The transformed 3D array.
    """

    # Get angle for rotation, and both translations
    theta = np.random.uniform(-angle, angle)
    x = np.random.randint(-pixels[0], pixels[0])
    y = np.random.randint(-pixels[1], pixels[1])

    # Perform rotation
    a = scipy.ndimage.rotate(a, theta, axes=(1, 2), reshape=False, mode=fill_mode)

    # Perform translation
    a = scipy.ndimage.shift(a, [0, y, x], mode=fill_mode)

    return a





class AffineTransform:
    """Apply an affine transform to a sitk image
    """

    def __init__(self,
                 max_angle: float = 20.0,
                 max_pixels=[20, 20],
                 fill_value: float = -1050.0) :
        """Initialize the transform class.

        Parameters
        ----------
        max_angle
            The maximum absolute angle by which to rotate the image from its
            original position. The actual rotation angle is uniformly randomly
            chosen from the range [-max_angle, max_angle].
        max_pixels
            The number of pixels by which to translate the image in the x and y
            plane. The actualy translation will be randomly selected from the
            range [-max_pixels[0], max_pixels[0]]; [-max_pixels[1], max_pixels[1]]
            in the x and y axes respectively.
        fill_value
            The pixel value to fill in the rotations/translations.
        """
        self.max_angle = max_angle * (np.pi / 180.0) # Convert to radians
        self.max_pixels = max_pixels
        self.fill_value = fill_value

    def __call__(self, image_xy : Tuple[sitk.Image, sitk.Image]) -> Tuple[sitk.Image, sitk.Image] :
        """Apply the transform.

        Parameters
        ----------
        image_xy
            A tuple containing the image from domain X and Y to transform

        Returns
        -------
        Tuple[sitk.Image, sitk.Image]
            The transformed images from domain X and Y.
        """
        image_x, image_y = image_xy
        angle = -self.max_angle + 2 * self.max_angle * torch.rand(1).item()
        rotation_centre = np.array(image_x.GetSize()) / 2
        rotation_centre = image_x.TransformContinuousIndexToPhysicalPoint(rotation_centre)

        max_pixel = torch.Tensor([self.max_pixels[0], self.max_pixels[1]])
        pixel = (-max_pixel + 2 * max_pixel * torch.rand(2)).numpy().astype(np.float64)

        rotation = sitk.Euler3DTransform(
            rotation_centre,
            0,      # the angle of rotation around the x-axis, in radians -> coronal rotation
            0,      # the angle of rotation around the y-axis, in radians -> saggittal rotation
            angle,  # the angle of rotation around the z-axis, in radians -> axial rotation
            (pixel[0], pixel[1], 0.0)  # translation
        )
        image_x = sitk.Resample(image_x, image_x, rotation, sitk.sitkLinear, self.fill_value)
        image_y = sitk.Resample(image_y, image_y, rotation, sitk.sitkLinear, self.fill_value)

        return image_x, image_y

    def __repr__(self):
        return f"{self.__class__.__name__}(max_angle={self.max_angle}, fill_value={self.fill_value})"





class ToTensor:
    """Convert a SimpleITK image to torch.Tensor."""
    def __call__(self, image_xy : Tuple[sitk.Image, sitk.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the transform.

        Parameters
        ----------
        image_xy
            A tuple containing the image from domain X and Y to transform.

        Returns
        -------
        Tuple[sitk.Image, sitk.Image]
            The transformed images from domain X and Y.
        """
        image_x, image_y = image_xy
        X, Y = sitk.GetArrayFromImage(image_x), sitk.GetArrayFromImage(image_y)
        X = torch.from_numpy(X).unsqueeze(0).float()
        Y = torch.from_numpy(Y).unsqueeze(0).float()

        return X, Y

    def __repr__(self):
        return f"{self.__class__.__name__}()"





class Normalize :
    """ Normalize the pixel intensities in the image"""
    def __init__(self, min_hu: float = -1000.0, max_hu: float = 1000.0) :
        self.f = sitk.ClampImageFilter()
        self.f.SetLowerBound(min_hu)
        self.f.SetUpperBound(max_hu)
        self.scale = 1000.0

    def __call__(self, image_xy : Tuple[sitk.Image, sitk.Image]) -> Tuple[sitk.Image, sitk.Image]:
        """Apply the transform.

        Parameters
        ----------
        image_xy
            A tuple containing the image from domain X and Y to transform.

        Returns
        -------
        Tuple[sitk.Image, sitk.Image]
            The transformed images from domain X and Y.
        """
        image_x, image_y = image_xy
        image_x = (self.f.Execute(image_x)) / self.scale
        image_y = (self.f.Execute(image_y)) / self.scale

        return image_x, image_y




class HorizontalFlip:
    """Flip the image about the vertical axis.
    """
    def __call__(self, image_xy : Tuple[sitk.Image, sitk.Image]) -> Tuple[sitk.Image, sitk.Image] :
        """Apply the transform.

        Parameters
        ----------
        image_xy
            A tuple containing the image from domain X and Y to transform

        Returns
        -------
        Tuple[sitk.Image, sitk.Image]
            The transformed images from domain X and Y.
        """
        x, y = image_xy

        # Randomly perform the flip (50% of the time)
        if np.random.random() > 0.5 :
            x = sitk.Flip(x, [True, False, False])
            y = sitk.Flip(y, [True, False, False])
        return x, y
