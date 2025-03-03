from extract_pieces import main as extract_pieces_main
from classify_piece import main as classify_piece_main
from compute_similarities import main as compute_similarities_main
from Puzzle import *

data_dir = "../data"
work_dir = "../work"
puzzle_name = "eda"
'''scans = ["1", "2"]

for scan in scans:
    extract_pieces_main(data_dir, puzzle_name, scan, work_dir)

    for piece_number in range(0, 6):
        classify_piece_main(puzzle_name, scan + "_" + str(piece_number), work_dir)'''

'''extract_pieces_main(data_dir, puzzle_name, '1', work_dir)
extract_pieces_main(data_dir, puzzle_name, '2', work_dir)
#extract_pieces_main(data_dir, puzzle_name, '3', work_dir)

for piece_number in range(0, 24):
    classify_piece_main(puzzle_name, '1_' + str(piece_number), work_dir)

for piece_number in range(0, 25):
    classify_piece_main(puzzle_name, '2_' + str(piece_number), work_dir)

compute_similarities_main(puzzle_name, work_dir)'''

# read similarity matrix
similarity_matrix = []
with open(f'{work_dir}/{puzzle_name}/similarities/matrix.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    row = [float(x) if x != "inf" else float('inf') for x in line.strip().split(",")]
    similarity_matrix.append(row)

# read piece order and save it to a list
f = open(f'{work_dir}/{puzzle_name}/similarities/piece_order.txt', 'r')
puzzle_pieces = f.read().split(', ')
f.close()

# initialize the grid
grid = [[None for _ in range(len(puzzle_pieces))] for _ in range(len(puzzle_pieces))]
grid, appended = setFirstPiece(grid, puzzle_pieces)
grid = iterateOverAppended(appended, grid, puzzle_pieces, similarity_matrix)
save_grid_to_csv(grid)
'''corner_pieces = load_puzzle_pieces_from_csv('corner_pieces')
grid = solvePuzzle(puzzle_pieces, similarity_matrix, corner_pieces)
grid, connected_pieces, rotated_pieces = place_corner_piece(grid, corner_pieces, puzzle_pieces)
grid = place_edge_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces, rotated_pieces)

'''
print('done')

