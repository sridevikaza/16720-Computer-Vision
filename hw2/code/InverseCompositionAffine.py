import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dp = np.ones((6,1))
    dM = np.ones((3,3))
    count = 0

    # get x and y vals
    rows,cols = It.shape
    n = rows*cols
    x = np.arange(0,rows)
    y = np.arange(0,cols)
    xx, yy = np.meshgrid(x,y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # interpolation
    interp_It = RectBivariateSpline(x,y,It)
    interp_It1 = RectBivariateSpline(x,y,It1)
    It_interp = interp_It.ev(xx,yy).T
    It1_interp = interp_It1.ev(xx,yy).T

    # gradient
    It_x = interp_It.ev(xx,yy,dx=1,dy=0).T
    It_y = interp_It.ev(xx,yy,dx=0,dy=1).T

    # warp gradient
    warped_img_x = affine_transform(It_x,M0)
    warped_img_y = affine_transform(It_y,M0)
    grad_x = warped_img_x.flatten()
    grad_y = warped_img_y.flatten()

    # steepest descent
    descent = np.array([grad_y*xx_flat,grad_y*yy_flat,grad_y,grad_x*xx_flat,grad_x*yy_flat,grad_x]).T

    # hessian
    H = np.dot(descent.T,descent)

    while(np.linalg.norm(dp) > threshold and count<num_iters):
        count+=1

        # warp
        warped_img = affine_transform(It1_interp,np.linalg.inv(M0))

        # fill in missing areas
        It1_temp = np.copy(It1)
        zero_indices = np.where(warped_img == 0)
        It1_temp[zero_indices] = 0
        error_img = It1_temp-warped_img

        # calculate dp
        b = np.dot(descent.T, error_img.flatten())
        dp = np.dot(np.linalg.inv(H),b)

        # calculate M
        dM[0,0] = 1+dp[0]
        dM[0,1] = dp[1]
        dM[0,2] = dp[2]
        dM[1,0] = dp[3]
        dM[1,1] = 1+dp[4]
        dM[1,2] = dp[5]
        dM[2,0] = 0
        dM[2,1] = 0
        dM[2,2] = 1
        M0 = np.dot(M0, np.linalg.inv(dM))

    return M0
