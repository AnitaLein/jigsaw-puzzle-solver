import cv2 as cv
from contours import *


def filter_contour_intersection(cnt1, cnt2, img, i , j, x , y):
    """
    Check if two contours intersect
    """
    output_folder = 'output_intersection'
    output_img = np.zeros_like(img)

    center_cnt1 = np.array(translate_one_specific_contour(np.array(cnt1), img, 1))
    center_cnt2 = np.array(translate_one_specific_contour(np.array(cnt2), img, 2))

    # draw bounding boxes
    x1, y1, w1, h1 = cv.boundingRect(center_cnt1)
    x2, y2, w2, h2 = cv.boundingRect(center_cnt2)

    # draw bounding boxes
    cv.polylines(output_img, [center_cnt1], False, (0, 255, 0), 2)
    cv.polylines(output_img, [center_cnt2], False, (0, 255, 0), 2)
    cv.rectangle(output_img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
    cv.rectangle(output_img, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
    output_filename = os.path.join(output_folder, f'intersection_contours_{i,x}_{j,y}.png')
    cv.imwrite(output_filename, output_img)
    intersection_area = 0
    # check if the width and height of the bounding boxes are equal
    if w1 == w2 or h1 == h2:
        if x == y:
            rotatec_cnt = rotate_one_specific_contour(center_cnt2, 180, img, {j,y})
            #print(f'cnt rotated {i},{x}, {w1, h1}and cnt {j},{y}{w2, h2} intersect!')
            intersection_area = calculate_intersection(center_cnt1, rotatec_cnt, img, i, j, x, y)
        else:
            #print(f'cnt {i},{x}, {w1, h1}and cnt {j},{y}{w2, h2} intersect!')
            intersection_area = calculate_intersection(center_cnt1, center_cnt2, img, i, j, x, y)
    elif w1 == h2 or h1 == w2:
       if ((x == 0 and y == 2) or (x == 1 and y == 0) or (x== 3 and y==1) or (x==2 and y==3)):
            rotatec_cnt = rotate_one_specific_contour(center_cnt2, 270, img, {j,y})
            #print(f'cnt rotated {i},{x}, {w1, h1}and cnt {j},{y}{w2, h2} intersect!')
            intersection_area = calculate_intersection(center_cnt1, rotatec_cnt, img, i, j, x, y)
       elif ((x == 0 and y ==1) or (x == 1 and y == 3) or (x== 3 and y==2) or (x==2 and y==0)):
            rotatec_cnt = rotate_one_specific_contour(center_cnt2, -270, img, {j,y})
            #print(f'cnt rotated {i},{x}, {w1, h1}and cnt {j},{y}{w2, h2} intersect!')
            intersection_area = calculate_intersection(center_cnt1, rotatec_cnt, img, i, j, x, y)

    return intersection_area
            
        

def calculate_intersection(cnt1, cnt2, img, i, j, x, y):
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

    intersection = cv.cvtColor(intersection, cv.COLOR_BGR2GRAY) if len(intersection.shape) == 3 else intersection
  
    intersection_area = cv.countNonZero(intersection)
    print(f'intersection area : {intersection_area}')

    #draw the intersection of the masks
    output_filename = os.path.join(output_folder, f'intersection_mask{i,x}_{j,y}.png')
    cv.imwrite(output_filename, intersection)
    return intersection_area






    
