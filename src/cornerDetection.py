import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"
img_harris = cv.imread('../data/10b.jpg') 
img_tomashi = cv.imread('../data/10b.jpg') 
img_approx = cv.imread('../data/10b.jpg')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# filter 
median_blur = cv.medianBlur(img, 3)  
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

# canny edge detection
edges = cv.Canny(img, 100, 200)
edges_medianblur = cv.Canny(median_blur, 100, 200)
edges_bilateralblur = cv.Canny(bilateral_blur, 100, 200)

#shi-tomasi corner detection
corners = cv.goodFeaturesToTrack(edges_bilateralblur,maxCorners=150, qualityLevel=0.001, minDistance=50)
corners = np.int0(corners)

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes


for i in corners:
    x,y = i.ravel()
    cv.circle(img_tomashi,(x,y),3,255,-1)

#harris corner detection
dst = cv.cornerHarris(edges_medianblur,2,3,0.04)
dst = cv.dilate(dst,None)
img_harris[dst>0.01*dst.max()]=[0,0,255]


#approxPolyDP
gray = cv.cvtColor(img_approx, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
epsilon = 0.02 * cv.arcLength(contours[0], True)
approx = cv.approxPolyDP(contours[0], epsilon, True)

for point in approx:
    x, y = point.ravel()
    cv.circle(img_approx, (x, y), 5, (0, 255, 0), -1)
 
#cv.imshow("Original", img)
cv.imshow("tomashi", img_tomashi)
cv.imshow("Puzzle Pieces", img)
#cv.imshow("Harris Corner Detection", img_harris)
#cv.imshow("ApproxPoly", img_approx)
cv.waitKey(0)
cv.destroyAllWindows()