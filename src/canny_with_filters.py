import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob

img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

# filter 
median_blur = cv.medianBlur(img, 3)  
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

# canny edge detection
edges = cv.Canny(img, 100, 200)
edges_medianblur = cv.Canny(median_blur, 100, 200)
edges_bilateralblur = cv.Canny(bilateral_blur, 100, 200)

# find contours
contours, _ = cv.findContours(edges_bilateralblur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# filter contours by area
filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 50]

# save edge_images in this folder
output_folder = 'output_edges_contours'

# delete existing data in the folder
files = glob.glob(os.path.join(output_folder, '*.png'))
for f in files:
    os.remove(f)

# extract and save edges
for i, cnt in enumerate(filtered_contours):
    # create an empty image and draw the contour
    edge_image = np.zeros_like(edges_bilateralblur)
    cv.drawContours(edge_image, [cnt], -1, 255, 1)
    
    # save and visualize the edge image
    output_filename = os.path.join(output_folder, f'output_edges_contour_{i}.png')
    cv.imwrite(output_filename, edge_image)
    cv.imshow(f'Contour {i}', edge_image)

cv.imshow("Original", img)
cv.imshow("Edges", edges)
cv.imshow("Edges_MedianBlur", edges_medianblur)
cv.imshow("Edges_BilateralBlur", edges_bilateralblur)
cv.imshow("Median Filter", median_blur)
cv.imshow("Bilateral Filter", bilateral_blur)
cv.waitKey(0)
cv.destroyAllWindows()