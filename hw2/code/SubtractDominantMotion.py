import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import matplotlib.pyplot as plt

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    mask = np.ones(image1.shape, dtype=bool)

    # LK
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    # warp
    im1_approx = affine_transform(image2,M)

    # error img
    error_img = im1_approx-image1
    error_img = image1-image2

    # binary img
    mask[error_img>tolerance] = 1
    mask[error_img<tolerance] = 0

    # erode and dilate
    mask = binary_erosion(mask)
    mask = binary_dilation(mask, iterations=20)
    mask = binary_erosion(mask)

    # plt.subplot(2, 2, 1)
    # plt.imshow(im1_approx)
    # plt.subplot(2, 2, 2)
    # plt.imshow(image1)
    # plt.subplot(2, 2, 3)
    # plt.imshow(error_img)
    # plt.subplot(2, 2, 4)
    # plt.imshow(mask_dilated)
    # plt.show()

    return mask.astype(bool)
