

from arrange_pieces import *
from csv_saving import load_puzzle_pieces_from_csv
from puzzle_types import EdgeType


def place_corner_pieces(grid, corner_pieces, puzzle_pieces):

    grid_size = len(grid)
    rotation = -1
    connected_pieces = []

    for i in range(len(corner_pieces)):

        index = int(corner_pieces[i].number)
        straight_edges = []

        for x in range(4):
            if puzzle_pieces[index].edges[x].type == EdgeType.Gerade:
                straight_edges.append(x)

        if i == 0:  # Top-left (0,0)
            rotation = corner_rotation(straight_edges)
            print(rotation)
            save_puzzle_piece_to_grid(grid, corner_pieces[i], rotation, 0, 0)
            connected_pieces.append(index)
        elif i == 1:  # Top-right (0, N-1)
            #add 1 to all elements in list
            straight_edges[0] = (straight_edges[0] + 1) % 4
            straight_edges[1] = (straight_edges[1] + 1) % 4
            rotation = corner_rotation(straight_edges)
            save_puzzle_piece_to_grid(grid, corner_pieces[i], rotation, 0, grid_size - 1)
            connected_pieces.append(index)

        elif i == 2:  # Bottom-left (N-1, 0)
            straight_edges[0] = (straight_edges[0] + 3) % 4
            straight_edges[1] = (straight_edges[1] + 3) % 4
            rotation = corner_rotation(straight_edges)
            save_puzzle_piece_to_grid(grid, corner_pieces[i], rotation, grid_size - 1,0)
            connected_pieces.append(index)

        elif i == 3:  # Bottom-right (N-1, N-1)
            straight_edges[0] = (straight_edges[0] + 2) % 4
            straight_edges[1] = (straight_edges[1] + 2) % 4
            rotation = corner_rotation(straight_edges)
            save_puzzle_piece_to_grid(grid, corner_pieces[i], rotation, grid_size - 1, grid_size - 1)
            connected_pieces.append(index)

    return grid, connected_pieces

def place_edge_pieces(grid, puzzle_pieces, similarity_matrix, connected_pieces):
    grid_size = len(grid)

    def place_piece(grid, row, col, piece_index, edge_index, rotation):
        """Helper function to find and place the best matching piece."""
        #edge_index = (neighbor_rotation + edge_offset) % 4
        piece = puzzle_pieces[piece_index]
        match_result = get_matches(piece, edge_index, similarity_matrix)

        for best_match in match_result:
            match_piece_i = best_match[1] // 4
            match_piece = puzzle_pieces[match_piece_i]

            if match_piece.number in connected_pieces:
                print("Piece already connected")
                continue

            else:
                new_index = (edge_index - int(rotation)) % 4
                if new_index == 0:
                    rotation = piece_rotation(best_match[1] % 4, new_index)
                    grid = save_puzzle_piece_to_grid(grid, puzzle_pieces[match_piece_i], rotation, row, col - 1)
                    print(edge_index)
                if new_index == 1:
                    rotation = piece_rotation(best_match[1] % 4, new_index)
                    grid = save_puzzle_piece_to_grid(grid, puzzle_pieces[match_piece_i], rotation, row + 1 , col)
                    print(edge_index)
                elif new_index == 2:
                    rotation = piece_rotation(best_match[1] % 4, new_index)
                    grid = save_puzzle_piece_to_grid(grid, puzzle_pieces[match_piece_i], rotation, row, col + 1)
                    print(edge_index)
                elif new_index == 3:
                    rotation = piece_rotation(best_match[1] % 4, new_index)
                    grid = save_puzzle_piece_to_grid(grid, puzzle_pieces[match_piece_i], rotation, row - 1, col)
                connected_pieces.append(match_piece.number) 
        else:
            print("No match found")

    # Step 1: Start at (0,0) and extend right and down
    top_left_piece, top_left_rotation = grid[0][0]
    top_left_piece = int(top_left_piece)

    # Find and place the best matching edge piece for the right side of (0,0)
    place_piece(grid, 0, 0 , top_left_piece, 0, top_left_rotation)   # Right edge of (0,0)
    place_piece(grid, 0, 0 , top_left_piece, 3, top_left_rotation) 
    
    current_piece, current_rotation = grid[0][1]
    current_piece = int(current_piece)
    #place_piece(grid, 0, 1 , current_piece, 0, current_rotation)
    place_piece(grid, 0, 1 , current_piece, 1, current_rotation)
    place_piece(grid, 0, 1 , current_piece, 2, current_rotation)
    place_piece(grid, 0, 1 , current_piece, 3, current_rotation)
    print(connected_pieces)

    return grid

''' for col in range(grid_size - 1):  # Filling the top row
        current_piece, current_rotation = grid[0][col]
        current_piece = int(current_piece)
        for x in range(4):
            place_piece(grid, 0, col, current_piece, x, current_rotation)

        # Step 3: Iterate over the last column to fill the right edge
    for row in range(grid_size - 1):
        current_piece, current_rotation = grid[row][-1]
        current_piece = int(current_piece)
        for x in range(4):
            place_piece(row, grid_size - 1, current_piece, x, current_rotation)  # Match bottom edge

    # Step 4: Iterate over the last row to fill the bottom edge
    for col in range(grid_size - 1, 0, -1):
        current_piece, current_rotation = grid[-1][col]
        current_piece = int(current_piece)
        for x in range(4):
            place_piece(grid_size - 1, col, current_piece, x, current_rotation)  # Match left edge

    # Step 5: Iterate over the first column to fill the left edge
    for row in range(grid_size - 1, 0, -1):
        current_piece, current_rotation = grid[row][0]
        current_piece = int(current_piece)
        for x in range(4):
            place_piece(row, 0, current_piece, x, current_rotation)
'''
    


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
    if a == b:
        return 0 
    elif b > a:
        if b == (a + 1) % 4:
            return 3
        if b == (a + 2) % 4:
            return 2
        if b == (a + 3) % 4:
            return 1
    elif a > b:
        if a == (b + 1) % 4:
            return 1
        if a == (b + 2) % 4:
            return 2
        if a == (b + 3) % 4:
            return 3

