import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

# filter 
median_blur = cv.medianBlur(img, 3)  
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

# canny edge detection
edges = cv.Canny(img, 100, 200)
edges_medianblur = cv.Canny(median_blur, 100, 200)
edges_bilateralblur = cv.Canny(bilateral_blur, 100, 200)

cv.imshow("Original", img)
cv.imshow("Edges", edges)
cv.imshow("Edges_MedianBlur", edges_medianblur)
cv.imshow("Edges_BilateralBlur", edges_bilateralblur)
cv.imshow("Median Filter", median_blur)
cv.imshow("Bilateral Filter", bilateral_blur)
cv.waitKey(0)
cv.destroyAllWindows()