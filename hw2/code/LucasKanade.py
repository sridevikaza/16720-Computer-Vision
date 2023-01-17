import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

import matplotlib.pyplot as plt


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    count = 0
    p = np.copy(p0)
    dp = np.array([[10],[10]])

    # rect corners
    x1 = rect[0]
    x2 = rect[2]
    y1 = rect[1]
    y2 = rect[3]

    # get shapes
    It_rows = It.shape[0]
    It_cols = It.shape[1]
    It1_rows = It1.shape[0]
    It1_cols = It1.shape[1]
    rect_rows = y2-y1
    rect_cols = x2-x1

    # jacobian
    J = np.array([[1,0],[0,1]])

    # interpolation
    x_It = np.arange(0, It_cols)
    y_It = np.arange(0, It_rows)
    x_It1 = np.arange(0, It1_cols)
    y_It1 = np.arange(0, It1_rows)
    interp_It = RectBivariateSpline(y_It,x_It,It)
    interp_It1 = RectBivariateSpline(y_It1,x_It1,It1)

    # T
    r = np.arange(y1, y2)
    c = np.arange(x1, x2)
    rr, cc = np.meshgrid(r, c)
    T = interp_It.ev(rr, cc)

    while((dp[0]**2+dp[1]**2) > threshold and count<num_iters):
        count+=1

        # warp img
        warp_rows = np.arange(y1,y2)+p[1]
        warp_cols = np.arange(x1,x2)+p[0]
        rr, cc = np.meshgrid(warp_rows,warp_cols)
        warped_Img = interp_It1.ev(rr,cc)

        # error img
        error_img = T - warped_Img

        # gradient
        interp_It1_x = interp_It1.ev(rr,cc,dx=1,dy=0)
        interp_It1_y = interp_It1.ev(rr,cc,dx=0,dy=1)
        gradient = np.array([interp_It1_y.flatten(),interp_It1_x.flatten()]).T

        # steepest descent
        descent = np.dot(gradient,J)

        # hessian
        H = np.dot(descent.T,descent)

        # calculate dp
        b = np.dot(gradient.T, error_img.flatten())
        dp = np.dot(np.linalg.inv(H),b)

        # update p
        p += dp

    return p
