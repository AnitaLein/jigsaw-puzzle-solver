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

# compare all the contours
for i in range(0, len(all_contours)):
    for j in range(i, len(all_contours)):
        if(i == j):
            print('skipped')
            continue
        for x in range(0, 4):
            for y in range(0, 4):
                filter_contour_intersection(all_contours[i][x], all_contours[j][y], img, i, j, x, y)
print("Done")

#calculate_intersection(all_contours[0][0], all_contours[1][0], img)

# uncomment to rotate and translate contours
'''
counter = 0
for piece_contour in all_contours:
    rotate_contour(piece_contour, 45, img, counter)
    translate_contour(piece_contour, img, counter)
    counter += 1'''