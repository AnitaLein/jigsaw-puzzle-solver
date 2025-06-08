import sys
from dataclasses import dataclass
from typing import List
from pathlib import Path
import numpy as np
from statistics import mean
import time
from scipy.spatial import cKDTree
from scipy import stats
from puzzle_types import Edge, EdgeType
from compute_similarities import read_edges_file, transforms_equal, transform_points, find_transformation_lsq

@dataclass
class PuzzlePieceShape:
    name: str
    edges: List[Edge]


def main(puzzle_name, work_dir):
    input_dir_edges = Path(work_dir, puzzle_name, "edges")
    input_path_solution = Path(work_dir, puzzle_name, "solution", "solution.txt")
    placements_output_dir = Path(work_dir, puzzle_name, "placements")

    # create output directory if it does not exist
    Path(placements_output_dir).mkdir(parents = True, exist_ok = True)

    # read solution matrix
    solution_matrix = read_solution_matrix(input_path_solution)

    # read all puzzle pieces referenced in the solution matrix
    puzzle_pieces = {}
    for row in solution_matrix:
        for piece_name, _ in row:
            if piece_name == None:
                continue
            puzzle_pieces[piece_name] = read_edges_file(Path(input_dir_edges, f"{piece_name}.txt"))

    print(f"read solution matrix with {len(puzzle_pieces)} puzzle pieces")

    print("computing placements")
    t0 = time.time()
    placements = compute_placements(solution_matrix, puzzle_pieces, print_progress = True)
    t1 = time.time()

    benchmark = True
    if benchmark:
        print(f"computed placements in {t1 - t0:.3f} seconds")

    print("writing placements")
    with open(Path(placements_output_dir, "placements.txt"), mode = "w") as file:
        for piece_name in placements:
            t, w = placements[piece_name]
            file.write(f"{piece_name}: {t[0]} {t[1]} {w}\n")


def read_solution_matrix(file_path):
    matrix = []
    with open(file_path, "r") as f:
        for line in f:
            for c in [" ", "(", ")"]:
                line = line.replace(c, "")

            row = []
            for piece in line.split(";"):
                if piece == "None":
                    row.append((None, None))
                    continue

                name, rot = piece.split(",")
                row.append((name, int(rot)))
            matrix.append(row)

    return matrix


def compute_placements(solution_matrix, puzzle_pieces, print_progress = False):
    # negate the y coordinates of all puzzle piece points
    for piece_name, edges in puzzle_pieces.items():
        for edge in edges:
            edge.points[:, 1] = -edge.points[:, 1]

    neighbor_transformations = {}
    for row_idx, row in enumerate(solution_matrix):
        for col_idx, (piece_name, rotation) in enumerate(row):
            if piece_name is None:
                continue

            neighbors = find_neighbors(solution_matrix, row_idx, col_idx)
            neighbor_transformations[piece_name] = {}
            for neighbor_direction, neighbor_name, neighbor_rotation in neighbors:
                edge_idx = (neighbor_direction - rotation) % 4
                edge_neighbor_idx = (neighbor_direction + 2 - neighbor_rotation) % 4  # opposite direction

                edge = puzzle_pieces[piece_name][edge_idx]
                edge_neighbor = puzzle_pieces[neighbor_name][edge_neighbor_idx]

                neighbor_transformations[piece_name][neighbor_name] = fit_edge(edge_neighbor, edge, (np.array([0, 0]), 0))

    # calculate average piece width and height
    piece_dimensions = np.empty((0, 2))
    for edges in puzzle_pieces.values():
        # concatenate all edge points
        points = np.concatenate([edge.points for edge in edges], axis = 0)

        # calculate width and height
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])

        piece_dimensions = np.append(piece_dimensions, [[width, height]], axis = 0)

    avg_dimension = np.mean(piece_dimensions, axis = 0)

    # initialize placements
    placements = {}
    for row_idx, row in enumerate(solution_matrix):
        for col_idx, (piece_name, rotation) in enumerate(row):
            if piece_name is None:
                continue

            # set initial transformation for the piece
            x = col_idx * avg_dimension[0]
            y = -row_idx * avg_dimension[1]
            if rotation == 1:
                x += avg_dimension[1]
            elif rotation == 2:
                x += avg_dimension[0]
                y -= avg_dimension[1]
            elif rotation == 3:
                y -= avg_dimension[0]
            placements[piece_name] = (np.array([x, y]), rotation * np.pi / 2)

    #zero_zero_piece = solution_matrix[0][0][0]
    #placements[zero_zero_piece] = optimize_placement(puzzle_pieces, solution_matrix, placements, 0, 0)
    #optimize_placement(puzzle_pieces, solution_matrix, placements, 0, 0)

    # iteratively optimize placements
    for i in range(500):
        if print_progress:
            print(f"iteration {i}")

        # optimize placements for each piece
        for row_idx, row in enumerate(solution_matrix):
            for col_idx, (piece_name, rotation) in enumerate(row):
                if piece_name is None:
                    continue

                placements[piece_name] = optimize_placement(puzzle_pieces, solution_matrix, placements, neighbor_transformations, row_idx, col_idx)

        # average new and old placements
        for piece_name, (t, w) in placements.items():
            if piece_name in placements:
                old_t, old_w = placements[piece_name]
                new_t = (t + old_t) / 2
                new_w = stats.circmean([w, old_w], low = 0, high = 2 * np.pi)
                placements[piece_name] = (new_t, new_w)

    # align the upper edge of the puzzle
    top_left_piece_name, top_left_rotation = solution_matrix[0][0]
    top_right_piece_name, top_right_rotation = solution_matrix[0][-1]
    if top_left_piece_name and top_right_piece_name:
        top_left_edge = puzzle_pieces[top_left_piece_name][(-top_left_rotation) % 4]
        top_right_edge = puzzle_pieces[top_right_piece_name][(-top_right_rotation) % 4]

        point_left = transform_points(top_left_edge.points, placements[top_left_piece_name])[-1]
        point_right = transform_points(top_right_edge.points, placements[top_right_piece_name])[0]

        alignment_transformation = find_transformation_lsq(np.array([[0, 0], [1000, 0]]), np.array([point_left, point_right]))
        for piece_name in placements:
            placements[piece_name] = transform_transform(placements[piece_name], alignment_transformation)

    # calculate top left, move transformations to origin # todo: fix
    top_left = np.array([np.inf, np.inf])
    for piece_name, (t, w) in placements.items():
        top_left = np.minimum(top_left, t)
    for piece_name, (t, w) in placements.items():
        placements[piece_name] = (t - top_left, w)


    # negate the y coordinates of all placements
    for piece_name, (t, w) in placements.items():
        placements[piece_name] = (np.array([t[0], -t[1]]), w)

    return placements


def optimize_placement(puzzle_pieces, solution_matrix, placements, neighbor_transformations, row_idx, col_idx):
    piece_name, rotation = solution_matrix[row_idx][col_idx]
    piece_edges = puzzle_pieces[piece_name]

    # find neighbors and fit edges
    neighbors = find_neighbors(solution_matrix, row_idx, col_idx)

    transformations = []
    for neighbor_direction, neighbor_name, neighbor_rotation in neighbors:
        neighbor_edges = puzzle_pieces[neighbor_name]

        direction = neighbor_direction
        direction_neighbor = (neighbor_direction + 2) % 4  # opposite direction

        edge = piece_edges[(direction - rotation) % 4]
        edge_neighbor = neighbor_edges[(direction_neighbor - neighbor_rotation) % 4]

        # fit edge to neighbor edge
        transformations.append(transform_transform(neighbor_transformations[piece_name][neighbor_name], placements[neighbor_name]))

    if not transformations:
        # no neighbors, return initial placement
        return placements[piece_name]

    # normalize angles
    transformations = [(t, w % (2 * np.pi)) for t, w in transformations]

    # average transformations
    t_avg = np.mean([t for t, _ in transformations], axis = 0)
    w_avg = stats.circmean([w for _, w in transformations], low = 0, high = 2 * np.pi)

    return (t_avg, w_avg)


def find_neighbors(solution_matrix, row_idx, col_idx):
    neighbors = []
    rows = len(solution_matrix)
    cols = len(solution_matrix[0])

    # check top neighbor
    if row_idx > 0:
        top_piece, top_rotation = solution_matrix[row_idx - 1][col_idx]
        if top_piece is not None:
            neighbors.append((0, top_piece, top_rotation))

    # check right neighbor
    if col_idx < cols - 1:
        right_piece, right_rotation = solution_matrix[row_idx][col_idx + 1]
        if right_piece is not None:
            neighbors.append((1, right_piece, right_rotation))

    # check bottom neighbor
    if row_idx < rows - 1:
        bottom_piece, bottom_rotation = solution_matrix[row_idx + 1][col_idx]
        if bottom_piece is not None:
            neighbors.append((2, bottom_piece, bottom_rotation))

    # check left neighbor
    if col_idx > 0:
        left_piece, left_rotation = solution_matrix[row_idx][col_idx - 1]
        if left_piece is not None:
            neighbors.append((3, left_piece, left_rotation))

    return neighbors


# finds transformation that optimally fits edge b to edge a
def fit_edge(a, b, a_transform):
    a_transformed = transform_points(a.points, a_transform)

    transform = find_transformation_lsq(
        np.array([a_transformed[0], a_transformed[-1]]),
        np.array([b.points[-1], b.points[0]])
    )

    for i in range(10):
        matches_a, matches_b = find_matches_closest(a_transformed, b, transform)
        new_transform = find_transformation_lsq(matches_a, matches_b)

        if transforms_equal(transform, new_transform) and i > 0:
            break
        transform = new_transform

    return transform


def find_matches_closest(a, b, transform):
    b_transformed = transform_points(b.points, transform)
    _, indices = cKDTree(a).query(b_transformed)

    return (a[indices], b.points)


def transform_transform(a, b):
    t_a, w_a = a
    t_b, w_b = b

    return (transform_points(np.array([t_a]), b)[0], w_a + w_b)


if __name__ == "__main__":
    main(*sys.argv[1:])
