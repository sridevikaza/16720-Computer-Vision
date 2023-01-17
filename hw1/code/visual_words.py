import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """
    # make sure image has 3 channels
    if(len(img.shape)<3):
        img = np.dstack((img,img,img))
    if(img.shape[2] == 4):
        img = np.delete(img, 3, 2)
    if(img.shape[2] == 1):
        img = np.dstack((img,img,img))
    # convert to lab
    img = skimage.color.rgb2lab(img)
    filter_scales = opts.filter_scales
    filter_responses = ()
    c1 = img[:,:,0]
    c2 = img[:,:,1]
    c3 = img[:,:,2]

    for scale in filter_scales: # loop through each scale

        gaussian_1 = scipy.ndimage.gaussian_filter(input=c1,sigma=scale,mode='constant')
        gaussian_2 = scipy.ndimage.gaussian_filter(input=c2,sigma=scale,mode='constant')
        gaussian_3 = scipy.ndimage.gaussian_filter(input=c3,sigma=scale,mode='constant')
        gaussian = np.dstack((gaussian_1,gaussian_2,gaussian_3))

        laplacian_1 = scipy.ndimage.gaussian_laplace(input=c1,sigma=scale,mode='constant')
        laplacian_2 = scipy.ndimage.gaussian_laplace(input=c2,sigma=scale,mode='constant')
        laplacian_3 = scipy.ndimage.gaussian_laplace(input=c3,sigma=scale,mode='constant')
        laplacian = np.dstack((laplacian_1,laplacian_2,laplacian_3))

        x_deriv_1 = scipy.ndimage.gaussian_filter(input=c1,sigma=scale,order=(0,1),mode='constant')
        x_deriv_2 = scipy.ndimage.gaussian_filter(input=c2,sigma=scale,order=(0,1),mode='constant')
        x_deriv_3 = scipy.ndimage.gaussian_filter(input=c3,sigma=scale,order=(0,1),mode='constant')
        x_deriv = np.dstack((x_deriv_1,x_deriv_2,x_deriv_3))

        y_deriv_1 = scipy.ndimage.gaussian_filter(input=c1,sigma=scale,order=(1,0),mode='constant')
        y_deriv_2 = scipy.ndimage.gaussian_filter(input=c2,sigma=scale,order=(1,0),mode='constant')
        y_deriv_3 = scipy.ndimage.gaussian_filter(input=c3,sigma=scale,order=(1,0),mode='constant')
        y_deriv = np.dstack((y_deriv_1,y_deriv_2,y_deriv_3))

        filter_responses += (gaussian,laplacian,x_deriv,y_deriv)

    filter_responses = np.dstack(filter_responses)
    return filter_responses



def compute_dictionary_one_image(opts, file):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    out_dir = opts.out_dir
    alpha = opts.alpha
    img_path = join(opts.data_dir, file)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    filter_responses = extract_filter_responses(opts, img) # (H,W,3F)
    H = filter_responses.shape[0]
    W = filter_responses.shape[1]
    three_F = filter_responses.shape[2]
    alpha_responses = np.ones((alpha,three_F))

    for i in range(alpha): # loop through pixels
        h_rand = np.random.randint(0,H-1)
        w_rand = np.random.randint(0,W-1)
        one_pixel_response = filter_responses[h_rand,w_rand,:]
        alpha_responses[i,:] = one_pixel_response
    return alpha_responses
    # np.save(join(out_dir, file), alpha_responses)


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    K = opts.K
    F = len(opts.filter_scales)*4
    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    T = len(train_files)
    filter_responses = ()

    for f in range(len(train_files)): # loop through training images
        file = train_files[f]
        img_responses = compute_dictionary_one_image(opts, file)
        filter_responses += (img_responses,)

    filter_responses = np.vstack(filter_responses) #alphaT x 3F
    kmeans = KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """
    filter_size = len(opts.filter_scales)*4*3
    H = img.shape[0]
    W = img.shape[1]
    wordmap = np.ones((H,W))
    filter_responses = extract_filter_responses(opts, img)
    for h in range(H):
        for w in range(W):
            pixel_values = filter_responses[h,w,:]
            pixel_values = np.reshape(pixel_values,(1,filter_size))
            distances = scipy.spatial.distance.cdist(pixel_values, dictionary, metric='euclidean')
            wordmap[h,w] = np.argmin(distances)
    return wordmap
