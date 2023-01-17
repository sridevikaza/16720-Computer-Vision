import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
import matplotlib


# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Q2.2.4
def warpImage(opts):
    # read images and resize
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    hp_cover = cv2.resize(hp_cover, dsize=(cv_cover.shape[1],cv_cover.shape[0]))

    # find matches
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    locs1 = np.fliplr(locs1)
    locs2 = np.fliplr(locs2)
    x1 = locs1[matches[:,0]]
    x2 = locs2[matches[:,1]]

    # compute H (RANSAC)
    H2to1, inliers = computeH_ransac(x1, x2, opts)

    # create composite image
    composite_img = compositeH(H2to1, cv_desk, hp_cover)

    # display image
    cv2.imshow('a', composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_opts()
    warpImage(opts)
