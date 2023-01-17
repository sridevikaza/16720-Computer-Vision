import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """
    K = opts.K
    hist = np.histogram(wordmap,range(0,K+1))[0]
    return hist/np.sum(hist)


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.
    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)
    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """
    final = []
    K = opts.K
    L = opts.L
    height = wordmap.shape[0]
    width = wordmap.shape[1]
    # resize wordmap to be divisible by 4
    if (height % 4) != 0:
        height = height - (height % 4)
    if (width % 4) != 0:
        width = width - (width % 4)
    wordmap = wordmap[0:height,0:width]
    l = L
    # loop through layers (starting at finest layer)
    while (l >= 0):
        n = 2**l
        cell_h = height/n
        cell_w = width/n
        # set weight based on layer
        if (l==0 or l==1):
            weight = 2**(-L)
        else:
            weight = 2**(l-L-1)
        #finest layer
        if l == L:
            hist_array = np.ones((n,n,K))
            for w in range(n):
                for h in range(n):
                    starth = int(h*cell_h)
                    endh = int((h+1)*cell_h)
                    startw = int(w*cell_w)
                    endw = int((w+1)*cell_w)
                    cell_image = wordmap[starth:endh,startw:endw]
                    hist = get_feature_from_wordmap(opts,cell_image) #hist for one cell
                    hist_array[h,w,:] = hist
        # all other layers
        else:
            hist_array = np.ones((n,n,K))
            for w in range(n):
                for h in range(n):
                    startw = int(2*w)
                    endw = int(2*w+1)
                    starth = int(2*h)
                    endh = int(2*h+1)
                    add_hists = np.ones((4,K))
                    add_hists[0,:] = prev_hist_array[starth,startw,:]
                    add_hists[1,:] = prev_hist_array[starth,endw,:]
                    add_hists[2,:] = prev_hist_array[endh,startw,:]
                    add_hists[3,:] = prev_hist_array[endh,endw,:]
                    add_hists = np.sum(add_hists,axis=0) # combine 4 hists (add together)
                    normalized_add_hists = add_hists/np.sum(add_hists)
                    hist_array[h,w,:] = normalized_add_hists
        hist_array = hist_array/np.sum(hist_array) #normalize at layer level
        final.append(hist_array.flatten()*weight) #flatten layer level of hists and append to final
        prev_hist_array = hist_array
        l -= 1
    # flatten 3 arrays in final to 1 array
    flat_list = []
    for sublist in final:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    hist_size = int(opts.K * (4**(SPM_layer_num+1)-1) / 3)

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # get list of SPM histograms for all training images
    N = len(train_files)
    features = np.ones((N,hist_size))
    for i in range(N):
        img_path = join(opts.data_dir, train_files[i])
        features[i,:] = get_image_feature(opts, img_path, dictionary)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    sim = np.sum( np.minimum(word_hist,histograms)  , axis=1)
    dist = np.ones(len(sim)) - sim
    return dist



def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]
    features = trained_system["features"]
    labels = trained_system["labels"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    confusion = np.ones((8,8))

    N = len(test_files)
    predicted_labels = np.zeros((N))
    for i in range(N):
        img_path = join(opts.data_dir,test_files[i])
        word_hist = get_image_feature(opts, img_path, dictionary)
        distances = similarity_to_set(word_hist, features)
        min_index = np.argmin(distances)
        predicted_labels[i] = int(labels[min_index])
        print(i)
        confusion[ int(predicted_labels[i]),test_labels[i] ] += 1
    accuracy = confusion.trace() / np.sum(confusion)
    return confusion, accuracy


def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass
