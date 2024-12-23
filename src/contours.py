
import cv2 as cv
import numpy as np
import os
import glob

def find_contours(filtered_img):

    # canny edge detection
    #edges = cv.Canny(img, 100, 200)
    edges_blur = cv.Canny(filtered_img, 100, 200)

    # find contours
    contours, _ = cv.findContours(edges_blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
        edge_image = np.zeros_like(edges_blur)
        cv.drawContours(edge_image, [cnt], -1, 255, 1)
        
        # save and visualize the edge image
        output_filename = os.path.join(output_folder, f'output_edges_contour_{i}.png')
        cv.imwrite(output_filename, edge_image)
        #cv.imshow(f'Contour {i}', edge_image)

def split_edges():
    input_folder = 'output_corners'
    output_folder = 'output_split_edges'
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

