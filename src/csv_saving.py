

import csv

from puzzle_types import PuzzlePiece


def save_puzzle_pieces_to_csv(puzzle_pieces, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for piece in puzzle_pieces:
            writer.writerow([piece.number, piece.type, piece.edges])  # CSV Row

    print(f"Saved {len(puzzle_pieces)} puzzle pieces to {filename}.")

def load_puzzle_pieces_from_csv(filename):
    puzzle_pieces = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            number, type, edges = row
            puzzle_piece = PuzzlePiece(number, type, edges)
            puzzle_pieces.append(puzzle_piece)
    return puzzle_pieces