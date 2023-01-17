import os
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import string
import cv2
import skimage.io
from q4 import *
import numpy as np
from nn import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def get_crop(bbox, bw, image_it):
    y1, x1, y2, x2 = bbox
    image = bw[y1:y2, x1:x2]
    w = x2 - x1
    h = y2 - y1
    if h > w:
        padding = int(h / 2)
        image = np.pad(image, (padding, padding), "constant")
    elif w > h:
        padding = int(w / 2)
        image = np.pad(image, (padding, padding), "constant")
    idx_0 = np.where(image == 0)
    idx_1 = np.where(image == 1)
    image[idx_0] = 1
    image[idx_1] = 0
    if image_it == 0:
        erosion_amt = 30
    elif image_it == 1:
        erosion_amt = 17
    elif image_it == 2:
        erosion_amt = 17
    else:
        erosion_amt = 7
    image = cv2.erode(image.astype(float), np.ones((erosion_amt, erosion_amt)), iterations=1)
    image = cv2.resize(image.T, (32, 32)).flatten()
    return image

def sort_boxes(box_arr):
    x_arr = []
    for i in box_arr:
        x_arr.append(i[1])
    x_arr.sort()
    res = []
    for i in x_arr:
        for j in box_arr:
            if i == j[1]:
                res.append(j)
    return res

img_count = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    rows = []
    bboxes_in_row = [bboxes[0]]
    for i in range(1,len(bboxes)):
        bbox = bboxes[i]
        y1, x1, y2, x2 = bbox
        yc = (y1 + y2) / 2
        bbox_first = bboxes_in_row[-1]
        if yc < bbox_first[0] or yc > bbox_first[2]:
            rows.append(bboxes_in_row)
            bboxes_in_row = []
            bboxes_in_row.append(bbox)
        else:
            bboxes_in_row.append(bbox)
    rows.append(bboxes_in_row)

    # crop the bounding boxes
    # note: before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    for r in range(len(rows)):
        rows[r] = sort_boxes(rows[r])
        for character in range(len(rows[r])):
            rows[r][character] = get_crop(rows[r][character],bw,img_count)

    # load the weights
    # run the crops through your neural network and print them out
    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    map_char = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
                30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
    for r in rows:
        r = np.vstack(r)
        h1 = forward(r, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        letters_in_row = ""
        for i in range(probs.shape[0]):
            pred = np.argmax(probs[i, :])
            letters_in_row += map_char[pred]
        print(letters_in_row)
    print("\n")
    img_count+=1

