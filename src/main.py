import cv2 as cv
from Puzzle import *
from arrange_pieces import *
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

#piece_contour = find_contours(bilateral_blur)

puzzle_pieces = classify_piece(img, gray)
grid = [[None for _ in range(len(puzzle_pieces))] for _ in range(len(puzzle_pieces))]
similarity_matrix= compute_similarity_matrix(puzzle_pieces)
corner_pieces = load_puzzle_pieces_from_csv('corner_pieces')
#grid = solvePuzzle(puzzle_pieces, similarity_matrix, corner_pieces)
grid, connected_pieces, rotated_pieces = place_corner_piece(grid, corner_pieces, puzzle_pieces)
grid =place_edge_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces, rotated_pieces)
save_grid_to_csv(grid)

print('done')

