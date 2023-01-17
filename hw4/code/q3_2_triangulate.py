import math

import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point
    (5) You do not need to follow the exact procedure above.
'''
def triangulate(C1, pts1, C2, pts2):
    n = pts1.shape[0]
    X = np.ones((n, 3))
    error = 0

    P1_1 = C1[0, :]
    P1_2 = C1[1, :]
    P1_3 = C1[2, :]
    P2_1 = C2[0, :]
    P2_2 = C2[1, :]
    P2_3 = C2[2, :]

    for i in range(n):
        # get 2D image points
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]

        # make A matrix
        A1 = y1*P1_3-P1_2
        A2 = P1_1-x1*P1_3
        A3 = y2*P2_3-P2_2
        A4 = P2_1-x2*P2_3
        A = np.vstack((A1,A2,A3,A4))

        # SVD to get 3D point
        D, V = np.linalg.eig(np.dot(A.T, A))
        idx = np.argmin(D)
        pt3D = V[:, idx]
        pt3D = pt3D/pt3D[3]
        X[i,:] = pt3D[0:3]

        # project back to 2D to get error
        img1_pt = np.matmul(C1,pt3D.T)
        img2_pt = np.matmul(C2,pt3D.T)
        img1_pt = (img1_pt/img1_pt[2])[0:2]
        img2_pt = (img2_pt/img2_pt[2])[0:2]

        # sum error
        e1 = (np.linalg.norm(img1_pt - pts1[i, :]))**2
        e2 = (np.linalg.norm(img2_pt - pts2[i, :]))**2
        error += (e1+e2)

    return X, error

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    '''
    error = math.inf
    best_idx = 0
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.matmul(K1, M1)

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2,M2)
        P, e = triangulate(C1, pts1, C2, pts2)
        if e<error and np.all(P[:, -1] > 0):
            best_idx = i
            error = e

    M2 = M2s[:, :, best_idx]
    C1 = np.matmul(K1,M1)
    C2 = np.matmul(K2,M2)
    P, e = triangulate(C1, pts1, C2, pts2)
    # np.savez(filename, M2=M2, C2=C2, P=P)
    return M2, C2, P
    # return M1, M2, C1, C2, P


if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert(err < 500)
