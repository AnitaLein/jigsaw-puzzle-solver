import cv2 as cv
from contours import *


def check_countour_intersection(cnt1, cnt2, img, i , j, x , y):
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
    elif w1 == h2 or h1 == w2:
        print(f'cnt {i},{x}{w1, h1} and cnt {j},{y} {w2, h2} intersect!')
    else:
        print(f'cnt {i},{x} {w1, h1} and cnt {j},{y} {w2, h2} do not intersect')

    
