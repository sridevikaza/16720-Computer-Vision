import numpy as np
import cv2
from PIL import Image
from helper import loadVid
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac, compositeH
import multiprocessing

# load videos/img
n_cores = multiprocessing.cpu_count()
ar_vid = loadVid('../data/ar_source.mov')
book_vid = loadVid('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

# find values for cropping
img = ar_vid[0,:,:,:]
y_nonzero, x_nonzero, _ = np.nonzero(img > 20)
h,w,_ = cv_cover.shape
ar = w/h
f = ar_vid.shape[0]
opts = get_opts()
composite_imgs = []

# loop through frames
for i in range(f):
    # get images
    print(i)
    book_img = book_vid[i,:,:,:]
    book_img = book_img[:, :, [2, 1, 0]]
    mov_img = ar_vid[i,:,:,:]
    mov_img = mov_img[:, :, [2, 1, 0]]

    # remove black border
    mov_img = mov_img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    # crop image w/ aspect ratio
    mov_h = mov_img.shape[0]
    mov_w = mov_h*ar
    center = int(mov_img.shape[1]/2)
    left = center-int(mov_w/2)
    right = center+int(mov_w/2)
    mov_img = mov_img[:,left:right,:]

    # get matches
    matches, locs1, locs2 = matchPics(cv_cover, book_img, opts)
    locs1 = np.fliplr(locs1)
    locs2 = np.fliplr(locs2)
    x1 = locs1[matches[:,0]]
    x2 = locs2[matches[:,1]]

    # compute H
    H2to1, inliers = computeH_ransac(x1, x2, opts)

    # composite img
    mov_img = cv2.resize(mov_img, dsize=(cv_cover.shape[1],cv_cover.shape[0]))
    black_parts = np.where(mov_img==0)
    mov_img[black_parts] = 1
    composite_img = compositeH(H2to1,book_img,mov_img)
    index = str(i)
    im = Image.fromarray(composite_img)
    im.save('../data/pics/'+index+".jpeg")
    composite_imgs.append(composite_img)

# make video
final_vid = np.stack(composite_imgs,axis=0)
frameSize = (final_vid.shape[2],final_vid.shape[1])
final_vid = final_vid[:, :, :, [2, 1, 0]]
fps = 20
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout = cv2.VideoWriter()
success = vout.open('output.mov',fourcc,fps,frameSize,True)
for i in range(f):
    img = final_vid[i,:,:,:]
    vout.write(img)
vout.release()
