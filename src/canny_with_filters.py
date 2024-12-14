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
    #cv.imshow(f'Contour {i}', edge_image)


input_folder = 'output_edges_contours'
output_folder = 'output_corners'
input_images = glob.glob(os.path.join(input_folder, '*.png'))

# harris corner detection
'''for i, image_path in enumerate(input_images):
    edge_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    edge_image_float = np.float32(edge_image)
    
    corners = cv.cornerHarris(edge_image_float, blockSize=2, ksize=3, k=0.04)
    corners_dilated = cv.dilate(corners, None)

    corners_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR)
    corners_image[corners_dilated > 0.1 * corners_dilated.max()] = [0, 0, 255]  # Rot für Ecken

    output_filename = os.path.join(output_folder, f'corners_{i}.png')
    cv.imwrite(output_filename, corners_image)'''

# shi-tomasi corner detection
for i, image_path in enumerate(input_images):
    edge_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    corners = cv.goodFeaturesToTrack(edge_image, maxCorners=6, qualityLevel=0.01, minDistance=20)

    if corners is not None:
        corners = np.int32(corners)  

        corners_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR)
        
        # mark the corners with red circles
        for corner in corners:
            x, y = corner.ravel()
            cv.circle(corners_image, (x, y), 3, (0, 0, 255), -1)

        # save the image with the marked corners
        output_filename = os.path.join(output_folder, f'corners_{i}.png')
        cv.imwrite(output_filename, corners_image)


#cv.imshow("Original", img)
#cv.imshow("Edges", edges)
#cv.imshow("Edges_MedianBlur", edges_medianblur)
#cv.imshow("Edges_BilateralBlur", edges_bilateralblur)
#cv.imshow("Median Filter", median_blur)
#cv.imshow("Bilateral Filter", bilateral_blur)
#cv.imshow("Harris Corners", edges_with_corners)
cv.waitKey(0)
cv.destroyAllWindows()