import cv2 as cv
from contours import *
from corners import *
from intersection import *
from piece_classification import *
from edge_matching import *


img = cv.imread('../data/misc/eda_black_merged.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)     
assert img is not None, "file could not be read, check with os.path.exists()"

all_output_folders = [
    'output_rotated_contours', 'translated_contours', 'output_intersection', 
    'output_intersection_mask', 'output_corners', 'output_edges_contours', 
    'output_edges', 'output_split_edges', 'output_intersection_drawing',
    'output_created_piece']
for folder in all_output_folders:
    files = glob.glob(os.path.join(folder, '*.png'))
    for f in files:
        os.remove(f)

bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

piece_contour = find_contours(bilateral_blur)


puzzle_pieces = classify_piece(img, gray)
for x in range(0, len(puzzle_pieces)): 
    for i in range(0, 4):
        puzzle_piece1 = puzzle_pieces[x]
        if puzzle_piece1.edges[i].type == EdgeType.Gerade:
            continue
        best_fit = 0
        matching_puzzle_piece = -1
        matching_index = -1

        for y in range(x,len(puzzle_pieces)):
            puzzle_piece2 = puzzle_pieces[y]    
            for j in range(0, 4):
                edge1 = puzzle_piece1.edges[i]
                edge2 = puzzle_piece2.edges[j]
                if x == y or edge1.type == edge2.type or  edge2.type == EdgeType.Gerade:
                    print('skipped')
                    continue
                intersection_area = filter_contour_intersection(edge1.points, edge2.points, img, 0, 1, i, j)
                if intersection_area > best_fit:
                    best_fit = intersection_area
                    matching_puzzle_piece = y
                    matching_index = j
        print(f'Best fit for {i} is {matching_index} {best_fit}')
        #draw the puzzles that intersect
        output_folder = 'output_intersection_drawing'
        output_img = np.zeros_like(img)
        cv.polylines(output_img, [np.array(puzzle_piece1.edges[0].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_piece1.edges[1].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_piece1.edges[2].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_piece1.edges[3].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_piece1.edges[i].points)], False, (0, 255, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_pieces[matching_puzzle_piece].edges[0].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_pieces[matching_puzzle_piece].edges[1].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_pieces[matching_puzzle_piece].edges[2].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_pieces[matching_puzzle_piece].edges[3].points)], False, (255, 0, 0), 2)
        cv.polylines(output_img, [np.array(puzzle_pieces[matching_puzzle_piece].edges[matching_index].points)], False, (0, 255, 0), 2)

        #cv.drawContours(output_img, puzzle_piece2.edges,  edge_index, (255, 0, 0), 2)
        cv.imwrite(os.path.join(output_folder, f'intersection_contours_{i}_{matching_index}.png'), output_img)

        print("done")

        #compare_edges(edge1, edge2)

        
        
#split_edges(basic_classified_pieces)
#rotated_pieces = rotate_contour(piece_contour, 45, img, 'output_rotated_contours')
#translated_pieces = translate_contours(piece_contour, img, 'translated_contours')
"""find_corners()
all_contours = 
# compare all the contours
for i in range(0, len(all_contours)): 
    for x in range(0,4):
        best_fit = 0
        puzzle_piece = -1
        puzzle_piece_contour = -1
        for j in range(i, len(all_contours)):
            if(i == j):
                print('skipped')
                continue
            for y in range(0, 4):
                intersection_area= filter_contour_intersection(all_contours[i][x], all_contours[j][y], img, i, j, x, y)
                if intersection_area > best_fit:
                    best_fit = intersection_area
                    puzzle_piece = j
                    puzzle_piece_contour = y
        
        print(f'Best fit for {i,x} is {puzzle_piece,puzzle_piece_contour} {best_fit}')
        #draw the puzzles that intersect
        output_folder = 'output_intersection_drawing'
        output_img = np.zeros_like(img)
        cv.drawContours(output_img, all_contours[i], -1, (0, 255, 0), 2)
        cv.drawContours(output_img, all_contours[i], x, (255, 0, 0), 2)
        cv.drawContours(output_img, all_contours[puzzle_piece], -1, (0, 255, 0), 2)
        cv.drawContours(output_img, all_contours[puzzle_piece], puzzle_piece_contour, (255, 0, 0), 2)
        cv.imwrite(os.path.join(output_folder, f'intersection_contours_{i,x}_{puzzle_piece,puzzle_piece_contour}.png'), output_img)
print("Done")"""

#calculate_intersection(all_contours[0][0], all_contours[1][0], img)

# uncomment to rotate and translate contours
'''
counter = 0
for piece_contour in all_contours:
    rotate_contour(piece_contour, 45, img, counter)
    translate_contour(piece_contour, img, counter)
    counter += 1'''
