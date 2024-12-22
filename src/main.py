import cv2 as cv
from contours import *
from corners import *


img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

find_contours(bilateral_blur)
find_corners()
split_edges()
