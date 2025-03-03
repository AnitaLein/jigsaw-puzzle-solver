import sys
from pathlib import Path
import numpy as np
import csv
import time
import random

class Vec2(tuple):
    def __new__(cls, *args):
        return tuple.__new__(cls, args)
    def __add__(self, other):
        return Vec2(*([sum(x) for x in zip(self, other)]))
    def __sub__(self, other):
        return self.__add__(-i for i in other)


def main(puzzle_name, work_dir, random_walks = 10_000, max_random_walk_length = 3):
    input_dir = Path(work_dir, puzzle_name, "similarities")
    solution_output_dir = Path(work_dir, puzzle_name, "solution")

    # create output directory if it does not exist
    Path(solution_output_dir).mkdir(parents = True, exist_ok = True)

    similarity_matrix = read_similarity_matrix(Path(input_dir, "matrix.txt"))

    with open(Path(input_dir, "piece_order.txt"), "r") as file:
        piece_order = file.readline()[:-1].split(", ")

    print("solving puzzle")
    t0 = time.time()
    solution = solve_puzzle(similarity_matrix, piece_order, random_walks, max_random_walk_length, print_progress = True)
    t1 = time.time()
    print()

    # find bounding box of solution
    top = min(pos[1] for pos in solution)
    bottom = max(pos[1] for pos in solution)
    left = min(pos[0] for pos in solution)
    right = max(pos[0] for pos in solution)

    width = right - left + 1
    height = bottom - top + 1

    partial_solution = len(solution) < len(similarity_matrix) / 4 or width * height > len(similarity_matrix) / 4

    benchmark = True
    if partial_solution:
        if benchmark:
            print(f"puzzle could not be solved, time elapsed: {t1 - t0:.3f} seconds")
        else:
            print("puzzle could not be solved")
    else:
        if benchmark:
            print(f"puzzle solved in {t1 - t0:.3f} seconds")
        else:
            print("puzzle solved")

    if partial_solution:
        print("writing partial solution")
    else:
        print("writing solution")

    output_matrix = [[None for _ in range(width)] for _ in range(height)]
    for pos, (piece, rotation) in solution.items():
        output_matrix[pos[1] - top][pos[0] - left] = (piece, rotation)

    for row in output_matrix:
        for i, piece in enumerate(row):
            if piece:
                row[i] = f"({piece[0]}, {piece[1]})"
            else:
                row[i] = "None"

    max_length = max(len(piece) for row in output_matrix for piece in row)
    for row in output_matrix:
        for i, piece in enumerate(row):
            row[i] = piece.ljust(max_length)

    with open(Path(solution_output_dir, "solution.txt"), "w") as file:
        for row in output_matrix:
            file.write(", ".join(row) + "\n")


def read_similarity_matrix(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        similarity_matrix = list(reader)

    for row in similarity_matrix:
        for i, similarity in enumerate(row):
                row[i] = float(similarity)

    return similarity_matrix


def solve_puzzle(similarity_matrix, piece_order, random_walks = 10_000, max_random_walk_length = 3, print_progress = False):
    rng = np.random.default_rng()
    directions = [Vec2(0, 1), Vec2(1, 0), Vec2(0, -1), Vec2(-1, 0)]

    grid = {Vec2(0, 0): (0, 0)}
    used = {0}

    while len(grid) < len(similarity_matrix) / 4:
        # find pieces that are missing a neighbor
        frontier = []
        for pos, (piece, rotation) in grid.items():
            for i in range(4):
                if pos + directions[(i + rotation) % 4] not in grid and has_matches(piece, i, similarity_matrix):
                    frontier.append(pos)
                    break

        best_tentative = None
        best_score = float("inf")
        for i in range(random_walks):
            # pick starting point for random walk
            pos = random.choice(frontier)

            tentative = random_walk(pos, grid, used, similarity_matrix, piece_order, rng, max_random_walk_length)
            if not tentative:
                continue

            score = score_random_walk(grid, tentative, similarity_matrix)
            if score < best_score:
                best_tentative = tentative
                best_score = score

        if not best_tentative and print_progress:
            print("no valid random walk found")
            break

        if print_progress:
            print(f"random walk found, length: {len(best_tentative)}, score: {best_score:.3f}")
            print("new pieces: " + ", ".join([piece_order[piece[0]] for piece in best_tentative.values()]))

        # commit best tentative to grid
        for pos, (piece, rotation) in best_tentative.items():
            grid[pos] = (piece, rotation)
            used.add(piece)

    return grid


def random_walk(pos, grid, used, similarity_matrix, piece_order, rng, max_random_walk_length = 3):
    tentative = {}
    tentative_used = set()
    while len(tentative) < max_random_walk_length:
        # pick an available direction
        edge, next_pos = pick_next_direction(pos, grid, tentative, similarity_matrix, rng)
        if edge is None:
            break

        possible_matches = enumerate(similarity_matrix[4 * (grid[pos][0] if pos in grid else tentative[pos][0]) + edge])
        piece_not_used = lambda piece: piece not in used and piece not in tentative_used
        possible_matches = [(piece_edge, similarity) for piece_edge, similarity in possible_matches if similarity != float("inf") and piece_not_used(piece_edge // 4)]
        if not possible_matches:
            break

        # prepare probabilitiy distribution based on similarity, lower similarity means higher probability
        total = sum(1 / similarity for piece_edge, similarity in possible_matches)
        probabilities = [1 / similarity / total for piece_edge, similarity in possible_matches]

        # pick a piece based on the probability distribution
        next_piece_edge = random.choices(possible_matches, weights = probabilities)[0][0]
        next_piece = next_piece_edge // 4
        next_edge = next_piece_edge % 4

        tentative[next_pos] = (next_piece, (edge + (grid[pos][1] if pos in grid else tentative[pos][1]) - next_edge + 2) % 4)
        tentative_used.add(next_piece)

        pos = next_pos

    return tentative


def pick_next_direction(pos, grid, tentative, similarity_matrix, rng):
    directions = [Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0)]

    if pos in grid:
        piece, rotation = grid[pos]
    else:
        piece, rotation = tentative[pos]

    for edge in rng.permutation(4):
        edge = int(edge)
        next_pos = pos + directions[(edge + rotation) % 4]
        if next_pos not in grid and next_pos not in tentative and is_near_grid(next_pos, grid) and has_matches(piece, edge, similarity_matrix):
            return (edge, next_pos)

    return (None, None)


def score_random_walk(grid, tentative, similarity_matrix):
    directions = [Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0)]

    # sum edges from tentative to grid
    sum_to_grid = 0
    num_edges_to_grid = 0
    for pos, (piece, rotation) in tentative.items():
        for edge in range(4):
            adjacent_pos = pos + directions[(edge + rotation) % 4]
            if adjacent_pos in grid:
                sum_to_grid += similarity_matrix[4 * piece + edge][4 * grid[adjacent_pos][0] + (edge + rotation + 2 - grid[adjacent_pos][1]) % 4]
                num_edges_to_grid += 1

    # sum edges from tentative to tentative
    sum_to_tentative = 0
    num_edges_to_tentative = 0
    for pos, (piece, rotation) in tentative.items():
        for edge in range(4):
            adjacent_pos = pos + directions[(edge + rotation) % 4]
            if adjacent_pos in tentative:
                sum_to_tentative += similarity_matrix[4 * piece + edge][4 * tentative[adjacent_pos][0] + (edge + rotation + 2 - tentative[adjacent_pos][1]) % 4]
                num_edges_to_tentative += 1

    sum_to_tentative /= 2
    num_edges_to_tentative /= 2

    score = (sum_to_grid + sum_to_tentative) / (num_edges_to_grid + num_edges_to_tentative)

    avg_edges_per_piece = (num_edges_to_grid + num_edges_to_tentative) / len(tentative) # 1 to 4
    edges_per_piece_factor = 1 + (-avg_edges_per_piece + 4) / (3 * 4) # 1 to 1.25

    imbalance = abs(num_edges_to_grid - num_edges_to_tentative) / (num_edges_to_grid + num_edges_to_tentative) # 0 to 1
    imbalance_factor = 1 + imbalance / 2 # 1 to 1.5

    #length_factor = (-len(tentative) + 6) / 5 + 1 # 1 to 2

    return score * edges_per_piece_factor * imbalance_factor# * length_factor


def has_matches(piece, edge, similarity_matrix):
    return any(similarity != float("inf") for similarity in similarity_matrix[4 * piece + edge])


def is_near_grid(pos, grid):
    directions = [Vec2(0, -1), Vec2(1, 0), Vec2(0, 1), Vec2(-1, 0), Vec2(1, 1), Vec2(1, -1), Vec2(-1, 1), Vec2(-1, -1)]
    return any(pos + direction in grid for direction in directions)


if __name__ == "__main__":
    main(*sys.argv[1:])
