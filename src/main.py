import cv2 as cv
from contours import *
from corners import *


img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

find_contours(bilateral_blur)
find_corners()
all_contours = split_edges()
print(len(all_contours))
counter = 0
for piece_contour in all_contours:
    rotate_contour(piece_contour, 0, img, counter)
    translate_contour(piece_contour, 200, 200, img, counter)
    counter += 1