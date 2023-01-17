import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """

        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # Convert Images to GrayScale
        I1 = skimage.color.rgb2gray(I1)
        I2 = skimage.color.rgb2gray(I2)

        # Detect Features in Both Images
        locs1 = corner_detection(I1, sigma)
        locs2 = corner_detection(I2, sigma)

        # Obtain descriptors for the computed feature locations
        desc1, brief_locs1 = computeBrief(I1, locs1)
        desc2, brief_locs2 = computeBrief(I2, locs2)

        # Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)

        return matches, brief_locs1, brief_locs2
