

from arrange_pieces import *
from csv_saving import load_puzzle_pieces_from_csv
from puzzle_types import EdgeType


def place_corner_piece(grid, corner_pieces, puzzle_pieces):
    rotated_pieces = []
    rotation = -1
    connected_pieces = []

    straight_edges = []

    for x in range(4):
        if puzzle_pieces[0].edges[x].type == EdgeType.Gerade:
            straight_edges.append(x)

    if len(corner_pieces) == 4:  # Top-left (0,0)
        rotation = corner_rotation(straight_edges)
        rotated_piece = rotate_piece(corner_pieces[0], rotation)
        rotated_pieces.append(rotated_piece)
        save_puzzle_piece_to_grid(grid, corner_pieces[0], rotation, 0, 0)
        connected_pieces.append(corner_pieces[0].number)
    else:
        print("Not enough corner pieces")    
    print('connected', connected_pieces)
    return grid, connected_pieces, rotated_pieces

def place_edge_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces, rotated_pieces):
    grid_size = len(grid)

    def place_piece(grid, row, col, piece_index, edge_index, rotation, is_row):
        piece = puzzle_pieces[piece_index]
        match_results = get_matches(piece, edge_index, similarity_matrix)
        
        print(match_results)
        for best_match in match_results:
            match_piece = puzzle_pieces[best_match[1] // 4]
            
            
            if grid_size > 8 and (match_piece.type == PieceType.Center or 
                                  (match_piece.type == PieceType.Corner and piece.type == PieceType.Corner)):
                print('Skipped due to type constraints')
                continue
            
            if match_piece.number in connected_pieces:
                print("Piece already connected")
                continue
            
            new_index = (edge_index - int(rotation)) % 4
            expected_index = 1 if is_row else 2
            
            if new_index == expected_index:
                rotation = piece_rotation(best_match[1] % 4, new_index)
                rotated_piece = rotate_piece(match_piece, rotation)
                
                edge_check_index = 0 if is_row else 3
                if rotated_piece.edges[edge_check_index].type != EdgeType.Gerade:
                    print('Skipped due to incorrect rotation')
                    continue
                
                new_row, new_col = (row, col + 1) if is_row else (row + 1, col)
                grid = save_puzzle_piece_to_grid(grid, match_piece, rotation, new_row, new_col)
                rotated_pieces.append(rotated_piece)
                connected_pieces.append(match_piece.number)
                print('Piece placed successfully')
                break
            else:
                print('Skipped due to index mismatch')
    
        else:
            print("No suitable match found")
    
    # Place the first row
    for i in range(grid_size):
        if grid[0][i] is None:
            break
        current_piece, current_rotation = grid[0][i]
        current_piece = int(current_piece)
        
        for x in range(4):
            place_piece(grid, 0, i, current_piece, x, current_rotation, is_row=True)
    
    # Place the first column
    for i in range(grid_size):
        if grid[i][0] is None:
            break
        current_piece, current_rotation = grid[i][0]
        current_piece = int(current_piece)
        
        for x in range(4):
            place_piece(grid, i, 0, current_piece, x, current_rotation, is_row=False)
    #place last row
    '''for i in range(3):
        if grid[end_of_col][i] == None:
            break
        current_piece2, current_rotation2 = grid[end_of_col][i]
        current_piece2 = int(current_piece2)
        for x in range(4):
            place_row_piece(grid, end_of_col, i, current_piece2, x, current_rotation2)
    
    #place last column
    for i in range(3):
        if grid[i][end_of_row] == None:
            break
        current_piece2, current_rotation2 = grid[i][end_of_row]
        current_piece2 = int(current_piece2)
        for x in range(4):
            place_column_piece(grid, i, end_of_row, current_piece2, x, current_rotation2)'''
        
        
    return grid

    
def rotate_piece(piece, rotation):
    new_piece = PuzzlePiece(piece.number, piece.type, edges=[])
    for i in range(4):
        new_piece.edges.append(piece.edges[(i + rotation) % 4])

    return new_piece

def corner_rotation(a):
    if a == [0, 1]:
        return 3
    elif a == [1, 2]:
        return 2
    elif a == [2, 3]:
        return 1
    elif a == [3,0]:
        return 0

#how many times needs to be rotated
def piece_rotation(a, b):
    return (a - b + 2) % 4

