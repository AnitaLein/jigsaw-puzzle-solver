from extract_pieces import main as extract_pieces_main
from classify_piece import main as classify_piece_main
from compute_similarities import main as compute_similarities_main

data_dir = "../data"
work_dir = "../work"
puzzle_name = "eda"
scans = ["1", "2"]

for scan in scans:
    extract_pieces_main(data_dir, puzzle_name, scan, work_dir)

    for piece_number in range(0, 6):
        classify_piece_main(puzzle_name, scan + "_" + str(piece_number), work_dir)

compute_similarities_main(puzzle_name, work_dir)

#corner_pieces = load_puzzle_pieces_from_csv('corner_pieces')
##grid = solvePuzzle(puzzle_pieces, similarity_matrix, corner_pieces)
#grid, connected_pieces, rotated_pieces = place_corner_piece(grid, corner_pieces, puzzle_pieces)
#grid = place_edge_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces, rotated_pieces)
#save_grid_to_csv(grid)

print('done')

