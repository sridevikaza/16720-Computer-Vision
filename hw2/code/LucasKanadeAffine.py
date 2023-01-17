import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
from scipy.ndimage import sobel
import cv2
import matplotlib.pyplot as plt

# debugging display function
def disp_img(img, title):
    img = np.array(img)
    cv2.imshow(title, img)
    cv2.waitKey()

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dp = np.ones((6,1))
    p = np.zeros(6)
    count = 0

    # get x and y vals
    rows,cols = It.shape
    x = np.arange(0,rows)
    y = np.arange(0,cols)
    xx, yy = np.meshgrid(x,y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # interpolation
    interp_It = RectBivariateSpline(x,y,It)
    interp_It1 = RectBivariateSpline(x,y,It1)
    T = interp_It.ev(xx,yy).T
    It1_interp = interp_It1.ev(xx,yy).T

    # gradient
    It1_x = interp_It1.ev(xx,yy,dx=1,dy=0)
    It1_y = interp_It1.ev(xx,yy,dx=0,dy=1)

    while(np.linalg.norm(dp) > threshold and count<num_iters):
        count+=1

        # warp
        warped_img = affine_transform(It1_interp,M)

        # fill in missing areas
        T_temp = np.copy(T)
        zero_indices = np.where(warped_img == 0)
        T_temp[zero_indices] = 0

        # error img
        error_img = warped_img-T_temp

        # warp gradient
        warped_img_x = affine_transform(It1_x,M)
        warped_img_y = affine_transform(It1_y,M)
        grad_x = warped_img_x.flatten()
        grad_y = warped_img_y.flatten()

        # steepest descent
        descent = np.array([grad_y*xx_flat,grad_y*yy_flat,grad_y,grad_x*xx_flat,grad_x*yy_flat,grad_x]).T

        # hessian
        H = np.dot(descent.T,descent)

        # calculate dp
        b = np.dot(descent.T, error_img.flatten())
        dp = np.dot(np.linalg.inv(H),b)
        p += dp

        # update M
        M[0,0] = 1+p[0]
        M[0,1] = p[1]
        M[0,2] = p[2]
        M[1,0] = p[3]
        M[1,1] = 1+p[4]
        M[1,2] = p[5]

    return M
