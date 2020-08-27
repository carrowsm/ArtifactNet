import numpy as np
import scipy.ndimage


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

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Qt5Agg")
    import pandas as pd

    # Test on some images
    # Load trg data
    trg_path = "/cluster/home/carrowsm/ArtifactNet/datasets/train_labels.csv"
    img_path = "/cluster/projects/radiomics/Temp/colin/isotropic_npy/images"

    df = pd.read_csv(trg_path, index_col="p_index", dtype=str)

    # Load an image
    X = np.load(os.path.join(img_path, df.at[0, "patient_id"])+".npy")
    z, y, x = X.shape

    plt.ion()

    plt.figure(1)
    plt.imshow(X[z//2, : , :])
    plt.show()

    X = affine_transform(X, angle=45, pixels=(20, 20) )

    plt.ioff()

    plt.figure(2)
    plt.imshow(X[z//2, : , :])
    plt.show()
