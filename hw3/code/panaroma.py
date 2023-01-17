import numpy as np
import cv2
from cpselect.cpselect import cpselect
from planarH import computeH_ransac
from opts import get_opts

# Import necessary functions


# Q4
opts = get_opts()
pano_left = '/Users/sridevikaza/desktop/IMG_4632.jpeg'
pano_right = '/Users/sridevikaza/desktop/IMG_4631.jpeg'
pano_left_img = cv2.imread(pano_left)
pano_right_img = cv2.imread(pano_right)
pano_left_img = cv2.resize(pano_left_img, dsize=(pano_right_img.shape[1],pano_right_img.shape[0]))

# get matches
controlpointlist = cpselect(pano_left,pano_right)
n = len(controlpointlist)
x1 = np.zeros((n,2))
x2 = np.zeros((n,2))
for i in range(n):
    d = controlpointlist[i]
    x1[i,0] = d['img1_x']
    x1[i,1] = d['img1_y']
    x2[i,0] = d['img2_x']
    x2[i,1] = d['img2_y']

#get H
H2to1, inliers = computeH_ransac(x1, x2, opts)

# apply H
left_warp = cv2.warpPerspective(pano_left_img, np.linalg.inv(H2to1), dsize=(pano_right_img.shape[1],pano_right_img.shape[0]))

# stitch together
black_idxs = np.where(left_warp==0)
vals = pano_right_img[black_idxs]
left_warp[black_idxs] = vals

# display panorama
cv2.imshow('panorama', left_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
