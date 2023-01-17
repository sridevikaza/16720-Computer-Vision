import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
import math

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2



# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on
        the results/expected number of inliners. You can also define your own metric.
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values

'''
def ransacF(pts1, pts2, M, nIters=100, tol=10):
    n = pts1.shape[0]
    most_inliers = 0
    ones_arr = np.ones((n, 1))
    pts1 = np.hstack((pts1, ones_arr))
    pts2 = np.hstack((pts2, ones_arr))

    for i in range(nIters):
        print(i)
        # pick random indices
        points_idxs = random.sample(range(0, n), 8)
        rand_pts1 = pts1[points_idxs]
        rand_pts2 = pts2[points_idxs]

        # compute F
        F = eightpoint(rand_pts1[:,0:2], rand_pts2[:,0:2], M)

        # compute num inliers
        err = calc_epi_error(pts1, pts2, F)
        inliers = err < tol
        num_inliers = len(np.where(inliers)[0])

        # check for best F
        if num_inliers >= most_inliers:
            most_inliers = num_inliers
            idx = np.where(inliers)
            inlier_pts1 = pts1[idx]
            inlier_pts2 = pts2[idx]

    F = eightpoint(inlier_pts1[:,0:2], inlier_pts2[:,0:2], M)

    return F, inliers



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    u = r/theta
    I = np.eye(3)
    u1 = u[0,0]
    u2 = u[1,0]
    u3 = u[2,0]
    ux = np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]])
    uut = np.matmul(u, u.T)
    R = I*math.cos(theta) + (1-math.cos(theta))*uut + ux*math.sin(theta)
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R-R.T)/2
    a32 = A[2,1]
    a13 = A[0,2]
    a21 = A[1,0]
    p = np.array([a32,a13,a21]).T
    s = np.linalg.norm(p)
    r11 = R[0,0]
    r22 = R[1,1]
    r33 = R[2,2]
    c = (r11+r22+r33-1)/2
    u = p/s
    theta = np.arctan2(s,c)
    r = u*theta
    return r.T


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    w = x[0:-6].reshape((-1,3))
    r2 = x[-6:-3].reshape((3,1))
    t2 = x[-3:].reshape((3,1))
    n = w.shape[0]
    w_h = np.hstack((w, np.ones((n, 1))))

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    p1_hat = np.matmul(C1, w_h.T)
    p2_hat = np.matmul(C2, w_h.T)
    p1_hat = p1_hat / p1_hat[-1, :]
    p2_hat = p2_hat / p2_hat[-1, :]
    p1_hat = p1_hat[0:2,:].T
    p2_hat = p2_hat[0:2,:].T

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual.
        You can try different (method='..') in scipy.optimize.minimize for best results.
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):

    obj_start = obj_end = 0

    R_init = M2_init[:, :3]
    t_init = M2_init[:, 3:]
    r2_init = invRodrigues(R_init)
    x = np.concatenate([P_init[:, :3].reshape((-1, 1)), r2_init.reshape((-1, 1)), t_init]).reshape((-1, 1))

    fun = lambda x: np.sum(rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2)

    t = scipy.optimize.minimize(fun,x)
    f = t.x

    P = f[:-6].reshape((-1,3))
    r2 = f[-6:-3].reshape((3,1))
    t2 = f[-3:].reshape((3,1))
    R2 = rodrigues(r2).reshape((3, 3))
    M2 = np.hstack((R2, t2))

    obj_end = t.fun

    return M2, P, obj_start, obj_end


if __name__ == "__main__":

    # np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    M = np.max([*im1.shape, *im2.shape])
    # F = eightpoint(noisy_pts1, noisy_pts2, M)
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M)
    idx = np.where(inliers)
    pts1 = noisy_pts1[idx]
    pts2 = noisy_pts2[idx]
    print("Num Inliers: ",pts2.shape[0])
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)

    # YOUR CODE HERE
    M2, C2, P_before = findM2(F, pts1, pts2, intrinsics)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    M2, P_after, obj_start, obj_end = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_before)
    print(P_after.shape)
    print(P_before.shape)
    plot_3D_dual(P_before, P_after)

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)