import cv2 as cv
from contours import *
from corners import *
from intersection import *


img = cv.imread('../data/10b.jpg')     
assert img is not None, "file could not be read, check with os.path.exists()"

bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

piece_contour = find_contours(bilateral_blur)
#rotated_pieces = rotate_contour(piece_contour, 45, img, 'output_rotated_contours')
#translated_pieces = translate_contours(piece_contour, img, 'translated_contours')
find_corners()
all_contours = split_edges()
#print(len(all_contours))
# cnt 7,1|2 h and cnt 4,1|2 h should intersect
# cnt 8,1|2 h and cnt 2,0|3 w should intersect

# compare all the contours
for i in range(0, len(all_contours)):
    for j in range(i, len(all_contours)):
        if(i == j):
            print('skipped')
            continue
        for x in range(0, 4):
            for y in range(0, 4):
                check_countour_intersection(all_contours[i][x], all_contours[j][y], img, i, j, x, y)

print("Done")
'''print(len(all_contours))
counter = 0
for piece_contour in all_contours:
    rotate_contour(piece_contour, 45, img, counter)
    translate_contour(piece_contour, img, counter)
    counter += 1'''