
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
    all_contours = []
    for i, image_path in enumerate(input_images):
        edge_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        # find contours
        contours, _ = cv.findContours(edge_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
        #draw the contours
        output_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR)
        cv.drawContours(output_image, contours, 0, (0, 255, 0), 2)
        cv.drawContours(output_image, contours, 1, (255, 0, 0), 2)
        cv.drawContours(output_image, contours, 2, (0, 0, 255), 2)
        cv.drawContours(output_image, contours, 3, (255, 255, 0), 2)
        output_filename = os.path.join(output_folder, f'corners_{i}.png')
        cv.imwrite(output_filename, output_image)
    return all_contours

def rotate_contour(contours, angle, img, counter ):
    output_folder = 'output_rotated_contours'
    output_image = np.zeros_like(img)

    for i, cnt in enumerate(contours):
        # get the bounding box of the contour
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(output_image, (x, y), (x+w, y+h), (255,0, 0), 2)  

        # get the center of the bounding box
        center = (x + w//2, y + h//2)

        # get the rotation matrix
        M = cv.getRotationMatrix2D(center, angle, 1)

        # rotate the contour
        rotated_cnt = cv.transform(cnt, M)

        # draw the rotated contour
        cv.drawContours(output_image, [rotated_cnt], 0, (0, 255, 0), 2)
        # draw w and h of the bounding box
        cv.putText(output_image, f'w: {w}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv.putText(output_image, f'h: {h}', (x, y-30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # save the rotated contour
    output_filename = os.path.join(output_folder, f'rotated_contour_{counter}.png')
    cv.imwrite(output_filename, output_image)

def translate_contour(contour, pos_x, pos_y, img, counter):
    output_folder = 'output_translated_contours'
    output_image = np.zeros_like(img)
    
    for i, cnt in enumerate(contour):
        # get the bounding box of the contour
        x, y, w, h = cv.boundingRect(cnt)

        # translate the contour
        translated_cnt = cnt + np.array([pos_x, pos_y])

        # draw the translated contour
        cv.drawContours(output_image, [translated_cnt], -1, (0, 255, 0), 2)

    output_filename = os.path.join(output_folder, f'translated_contour_{counter}.png')
    cv.imwrite(output_filename, output_image)
        

 
