import cv2 as cv
from contours import *


def filter_contour_intersection(cnt1, cnt2, img, i , j, x , y):
    """
    Check if two contours intersect
    """
    output_folder = 'output_intersection'
    output_img = np.zeros_like(img)

    center_cnt1 = translate_one_specific_contour(cnt1, img, 1)
    center_cnt2 = translate_one_specific_contour(cnt2, img, 2)

    # draw bounding boxes
    x1, y1, w1, h1 = cv.boundingRect(center_cnt1)
    x2, y2, w2, h2 = cv.boundingRect(center_cnt2)

    # draw bounding boxes
    cv.drawContours(output_img, [center_cnt1], -1, (0, 255, 0), 2)
    cv.drawContours(output_img, [center_cnt2], -1, (0, 255, 0), 2)
    cv.rectangle(output_img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
    cv.rectangle(output_img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
    output_filename = os.path.join(output_folder, f'intersection_contours_{i,x}_{j,y}.png')
    cv.imwrite(output_filename, output_img)

    # check if the width and height of the bounding boxes are equal
    if w1 == w2 or h1 == h2:
        print(f'cnt {i},{x}, {w1, h1}and cnt {j},{y}{w2, h2} intersect!')
        calculate_intersection(center_cnt1, center_cnt2, img, i, j)
    elif w1 == h2 or h1 == w2:
        return
        #print('rotated')
        #print(f'cnt {i},{x}{w1, h1} and cnt {j},{y} {w2, h2} intersect!')
        #rotated_cnt2 = rotate_contour(center_cnt2, 45, img, 'rotated')
        #calculate_intersection(center_cnt1, rotated_cnt2)

def calculate_intersection(cnt1, cnt2, img, i, j):
    """
    Calculate the intersection of two contours
    """
    output_folder = 'output_intersection_mask'
    #generate mask for the two contours
    mask1 = np.zeros_like(img)
    mask2 = np.zeros_like(img)
    cv.drawContours(mask1, [cnt1], -1, (255, 255, 255), -1)
    cv.drawContours(mask2, [cnt2], -1, (255, 255, 255), -1)

    # find the intersection of the two contours
    intersection = cv.bitwise_and(mask1, mask2)

    # generate exlusive masks 
    mask1_excl = cv.bitwise_xor(mask1, intersection)

    #draw the intersection of the masks
    output_filename = os.path.join(output_folder, f'intersection_mask{i,j}.png')
    cv.imwrite(output_filename, intersection)





    
