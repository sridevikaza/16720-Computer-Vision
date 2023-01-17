import skimage
import skimage.color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    image = skimage.color.rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(10))
    cleared = clear_border(bw)
    label_image = label(cleared)
    bboxes = []
    for region in regionprops(label_image):
        if region.area >= 300:
            bbox = region.bbox
            bboxes.append(bbox)
    return bboxes, bw