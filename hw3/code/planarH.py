import numpy as np
import cv2
import random


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    n = x1.shape[0]
    A = np.zeros((2*n,9))
    for i in range(n):
        xs = x1[i,0]
        ys = x1[i,1]
        xd = x2[i,0]
        yd = x2[i,1]
        A[2*i] = [xd,yd,1,0,0,0,-xs*xd,-xs*yd,-xs]
        A[2*i+1] = [0,0,0,xd,yd,1,-ys*xd,-ys*yd,-ys]
    D,V = np.linalg.eig(np.dot(A.T,A))
    idx = np.argmin(D)
    H = np.reshape(V[:,idx], (3,3))
    return H


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    mean1 = np.mean(x1,axis=0)
    u1 = mean1[0]
    v1 = mean1[1]
    mean2 = np.mean(x2,axis=0)
    u2 = mean2[0]
    v2 = mean2[1]

    #Shift the origin of the points to the centroid
    x1_shift = x1-mean1
    x2_shift = x2-mean2

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_scale = np.sqrt(2) / np.max(np.linalg.norm(x1_shift,axis=1),axis=0)
    x2_scale = np.sqrt(2) / np.max(np.linalg.norm(x2_shift,axis=1),axis=0)
    x1_norm = x1_shift*x1_scale
    x2_norm = x2_shift*x2_scale

    #Similarity transform 1
    T1 = x1_scale*np.array([[1,0,-u1],[0, 1, -v1],[0, 0, 1/x1_scale]])

    #Similarity transform 2
    T2 = x2_scale*np.array([[1,0,-u2],[0, 1, -v2],[0, 0, 1/x2_scale]])

    #Compute homography
    Hnorm = computeH(x1_norm, x2_norm)

    #Denormalization
    H2to1 = np.dot(np.dot(np.linalg.inv(T1),Hnorm), T2)

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    n = len(locs1)
    most_inliers = 0
    best_H = []
    inliers = []
    ones_arr = np.ones((n,1))
    locs2_with1 = np.hstack((locs2, ones_arr))

    for i in range(max_iters):
        # pick 4 random points
        points_idxs = random.sample(range(0, n), 4)

        # compute H norm
        x1 = locs1[points_idxs]
        x2 = locs2[points_idxs]
        H = computeH_norm(x1, x2)

        # apply H
        locs2_H = np.dot(H, locs2_with1.T).T

        # make last column 1
        locs2_H_0 = np.reshape(locs2_H[:,0] / locs2_H[:,2] , (n,1))
        locs2_H_1 = np.reshape(locs2_H[:,1] / locs2_H[:,2] , (n,1))
        locs2_H = np.hstack((locs2_H_0,locs2_H_1))

        # calculate inliers
        diffs = np.linalg.norm(locs2_H-locs1,axis=1)
        inliers = diffs<inlier_tol
        num_inliers = len(np.where(inliers)[0])

        # check if best_H
        if num_inliers >= most_inliers:
            best_H = H
            most_inliers = num_inliers

    return best_H, inliers



def compositeH(H2to1, template, img):

    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image

    hp_desk = cv2.warpPerspective(img, np.linalg.inv(H2to1), dsize=(template.shape[1],template.shape[0]))

    black_idxs = np.where(hp_desk==0)
    vals = template[black_idxs]
    hp_desk[black_idxs] = vals

    return hp_desk
