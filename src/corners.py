import os
import cv2 as cv
import numpy as np
import glob


def sort_corners(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in points], [p[0] - center[0] for p in points])
    sorted_points = [p for _, p in sorted(zip(angles, points))]
    return sorted_points

def find_corners():
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

