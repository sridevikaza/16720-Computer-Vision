import numpy as np
import cv2
from matchPics import matchPics
from displayMatch import displayMatched
import skimage.color
from opts import get_opts
import scipy.ndimage
import matplotlib.pyplot as plt

#Q2.1.6
def rotTest(opts):

    #Read the image
    img = cv2.imread('../data/cv_cover.jpg')
    hist_counts = []
    x_vals = np.arange(0,36)

    for i in range(36):
        #Rotate Image
        angle = i*10
        rot_img = scipy.ndimage.rotate(img,angle)
        #Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img,rot_img,opts)
        hist_counts.append(len(matches))

    plt.bar(x_vals,hist_counts)
    plt.show()


if __name__ == "__main__":
    opts = get_opts()
    rotTest(opts)
