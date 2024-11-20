import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('../data/10b.jpg')         
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(img,100,200)
 
cv.imshow("Edges", edges)
cv.imshow("Original", img)
cv.imshow("Gray", gray)
cv.waitKey(0)
cv.destroyAllWindows()