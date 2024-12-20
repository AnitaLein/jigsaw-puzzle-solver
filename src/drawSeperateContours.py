import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob
import itertools

def sort_corners(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in points], [p[0] - center[0] for p in points])
    sorted_points = [p for _, p in sorted(zip(angles, points))]
    return sorted_points

img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

# filter 
median_blur = cv.medianBlur(img, 3)  
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

# canny edge detection
edges = cv.Canny(img, 100, 200)
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

files = glob.glob(os.path.join(output_folder, '*.png'))
for f in files:
    os.remove(f)

# shi-tomasi corner detection
for i, image_path in enumerate(input_images):
    edge_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    corners = cv.goodFeaturesToTrack(edge_image, maxCorners=4, qualityLevel=0.01, minDistance=100, blockSize=7)

    if corners is not None:
        corners = np.int32(corners)  

        #extract corner points
        corner_points = [tuple(corner.ravel()) for corner in corners]

        # sort the corner points
        sorted_corners = sort_corners(corner_points)    


        # Create a blank image to draw the contours
        height, width = edge_image.shape
        output_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR)

        # Draw the detected corners for visualization
        for corner in sorted_corners:
            cv.circle(output_image, corner, radius=10, color=(0, 0, 0), thickness=-1)
        # save the image with the marked corners
        output_filename = os.path.join(output_folder, f'corners_{i}.png')
        cv.imwrite(output_filename, output_image)

input_folder = 'output_corners'
output_folder = 'corner_edges'
input_images = glob.glob(os.path.join(input_folder, '*.png'))

files = glob.glob(os.path.join(output_folder, '*.png'))
for f in files:
    os.remove(f)


for i, image_path in enumerate(input_images):
    edge_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # find contours
    contours, _ = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #draw the contours
    output_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR)
    cv.drawContours(output_image, contours, 0, (0, 255, 0), 2)
    cv.drawContours(output_image, contours, 1, (255, 0, 0), 2)
    cv.drawContours(output_image, contours, 2, (0, 0, 255), 2)
    cv.drawContours(output_image, contours, 3, (255, 255, 0), 2)
    output_filename = os.path.join(output_folder, f'corners_{i}.png')
    cv.imwrite(output_filename, output_image)

    

cv.waitKey(0)
cv.destroyAllWindows()