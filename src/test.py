import glob
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def extract_features(piece_image):
    # Use keypoint detection or shape descriptors
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(piece_image, None)
    return keypoints, descriptors


img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

#median_blur = cv.medianBlur(img, 3) 
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)
gray = cv.cvtColor(bilateral_blur, cv.COLOR_BGR2GRAY)
#ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY) 

#Kanten detection
edges_medianblur = cv.Canny(gray, 100, 200)
contours, _ = cv.findContours(edges_medianblur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
edges_bilateralblur_float = np.float32(edges_medianblur)

puzzleAmount = len(contours)
print(contours)


cv.drawContours(img, contours, -1, (0, 255, 0), 2)
#cv.drawContours(edges_bilateralblur_float, contours, -1, (0, 255, 0), 2)
#shi-tomasi corner detection
corners = cv.goodFeaturesToTrack(edges_bilateralblur_float,maxCorners=puzzleAmount*8, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

cv.imshow("Puzzle Pieces", img)
cv.waitKey(0)
cv.destroyAllWindows()