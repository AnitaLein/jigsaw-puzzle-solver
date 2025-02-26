import csv
import os
import pandas as pd
import cv2 as cv
from puzzle_types import *


def arrange(puzzle_pieces, similarity_matrix, corner_pieces, grid):

    for piece in corner_pieces:
        # Find the top-left corner piece
        # Problem: if the scan has rotated puzzle piece
        if piece.edges[0].type == EdgeType.Gerade and piece.edges[3].type == EdgeType.Gerade:
            start_piece = piece
            break
    save_puzzle_piece_to_grid(grid, start_piece,-1, 0, 0)
    for x in range(4):
        i = start_piece.number
        min_dist, best_match_i = find_best_match(puzzle_pieces[i], x, similarity_matrix)
        if best_match_i == None:
            continue
        if x % 2 == 0:
            save_puzzle_piece_to_grid(grid, puzzle_pieces[best_match_i//4], best_match_i %4, 0, 1)
        else:
            save_puzzle_piece_to_grid(grid, puzzle_pieces[best_match_i//4], best_match_i %4, 1, 0)
        

# rotation to the right
def save_puzzle_piece_to_grid(grid, piece, rotation, row, col):
    grid[row][col] = (piece.number, rotation)



def save_grid_to_csv(grid):
    with open('puzzle_grid', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid)
    print("Puzzle grid saved to 'puzzle_grid.csv'")



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
