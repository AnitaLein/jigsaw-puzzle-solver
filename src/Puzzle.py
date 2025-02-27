

#from arrange_pieces import *
from csv_saving import *
from puzzle_types import EdgeType
from puzzle_types import *
import csv


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

    end_of_row = 0
    end_of_col = 0

    # Place the first row
    for i in range(grid_size):
        if grid[0][i] is None:
            end_of_row += i - 1 
            break
        current_piece, current_rotation = grid[0][i]
        current_piece = int(current_piece)
        
        for x in range(4):
            piece = puzzle_pieces[current_piece]
            match_results = get_matches(piece, x, similarity_matrix)
    
    # Place the first column
    for i in range(grid_size):
        if grid[i][0] is None:
            end_of_col += i - 1
            break
        current_piece, current_rotation = grid[i][0]
        current_piece = int(current_piece)
        
        for x in range(4):
            place_piece(grid, i, 0, current_piece, x, current_rotation, is_row=False)
    
    #place last row
    for i in range(end_of_row):
        print('grid' , grid[end_of_col][i], end_of_row, end_of_col)
        if grid[end_of_col][i] == None:
            break
        current_piece, current_rotation = grid[end_of_col][i]
        current_piece = int(current_piece)
        
        for x in range(4):
            place_piece(grid, end_of_col, i, current_piece, x, current_rotation, is_row=True)
    
    #place last column
    for i in range(end_of_col):
        print('grid' , grid[i][end_of_row], end_of_row, end_of_col)
        if grid[i][end_of_row] == None:
            break
        current_piece, current_rotation = grid[i][end_of_row]
        current_piece = int(current_piece)

        for x in range(4):
            place_piece(grid, i, end_of_row, current_piece, x, current_rotation, is_row=False)
        
        
    return grid

def place_center_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces, rotated_pieces):
    grid_size = len(grid)

    for j in range(grid_size):
        for i in range(grid_size):
            if grid[j][i] is not None and grid[j - 1][i + 1] is not None:
                left_piece, left_rotation = grid[j][i]
                top_piece, top_rotation = grid[j - 1][i + 1]
                for x in range(4):
                    left_piece = puzzle_pieces[left_piece]
                    top_piece = puzzle_pieces[top_piece]
                    left_match_results = get_matches(left_piece, x, similarity_matrix)
                    top_match_results = get_matches(top_piece, x, similarity_matrix)

                    left_match_results = [x[1] // 4 for x in left_match_results]
                    top_match_results = [x[1] // 4 for x in top_match_results]

                    # if there is an element in left_match_results and in top_match_results
                    


                    

    
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


def find_best_match(puzzle_piece, edge_index, similarity_matrix):
    best_match_i = None
    min_dist = float('inf')
    i = int(puzzle_piece.number)

    row = similarity_matrix[i+edge_index]
    for j in range(len(row)):
        if row[j] < min_dist:
            min_dist = row[j]
            best_match_i = j
    return (min_dist, best_match_i)

def get_matches(puzzle_piece, edge_index, similarity_matrix):
    i = int(puzzle_piece.number)
    row = similarity_matrix[i + edge_index]  # Get the corresponding row

    # Collect all non-infinity values along with their original indices
    matches = [(row[j], j) for j in range(len(row)) if row[j] != float('inf')]

    # Sort matches by similarity score (lower is better)
    matches.sort()

    return matches
