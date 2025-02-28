import sys
from dataclasses import dataclass
from typing import List
from pathlib import Path
import numpy as np
from statistics import mean
import math
from puzzle_types import Edge, EdgeType
from scipy.spatial import cKDTree
import csv
import time

@dataclass
class PuzzlePieceShape:
    name: str
    edges: List[Edge]


def main(puzzle_name, work_dir):
    input_dir = Path(work_dir, puzzle_name, "edges")
    similarities_output_dir = Path(work_dir, puzzle_name, "similarities")

    # create output directory if it does not exist
    Path(similarities_output_dir).mkdir(parents = True, exist_ok = True)

    # read all puzzle pieces from the input directory
    files = input_dir.glob('*.txt')
    files = [f for f in files if f.is_file()]

    puzzle_pieces = []
    for file in files:
        piece_name = file.stem
        edges = read_edges_file(file)
        puzzle_pieces.append(PuzzlePieceShape(piece_name, edges))

    print(f"read {len(puzzle_pieces)} puzzle pieces")

    print("computing similarity matrix")
    t0 = time.time()
    similarity_matrix = compute_similarity_matrix(puzzle_pieces, print_progress = True)
    t1 = time.time()
    print()

    benchmark = True
    if benchmark:
        print(f"computed similarity matrix in {t1 - t0:.3f} seconds")

    print("writing similarity matrix")
    with open(Path(similarities_output_dir, "matrix.txt"), mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerows(similarity_matrix)

    # write the puzzle piece names
    with open(Path(similarities_output_dir, "piece_order.txt"), mode = "w", newline = "") as file:
        file.write(", ".join([piece.name for piece in puzzle_pieces]))


def read_edges_file(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            type, points = line.split(": ")
            edge_type = EdgeType[type]
            points = points[1:-2].split("), (")
            points = [point.split(", ") for point in points]
            points = [(int(x), int(y)) for x, y in points]
            edges.append(Edge(edge_type, np.array(points)))

    assert len(edges) == 4

    return edges


def compute_similarity_matrix(puzzle_pieces, print_progress = False):
    # calculate edge lengths
    for piece in puzzle_pieces:
        for edge in piece.edges:
            edge.length = np.sum(np.linalg.norm(np.diff(edge.points, axis = 0), axis = 1))

    # precompute the kdtrees
    for piece in puzzle_pieces:
        for edge in piece.edges:
            edge.kdtree = cKDTree(edge.points)

    similarity_matrix = []
    for a in puzzle_pieces:
        for i in range(4):
            similarities = []
            for b in puzzle_pieces:
                for j in range(4):
                    if a is b:
                        similarity = float("inf")

                    # check if length differs by more than 3%
                    elif abs(a.edges[i].length - b.edges[j].length) > 0.03 * mean([a.edges[i].length, b.edges[j].length]):
                        similarity = float("inf")

                    # check flat edge continuity
                    elif ((a.edges[(i + 1) % 4].type == EdgeType.Flat) != (b.edges[(j + 3) % 4].type == EdgeType.Flat) or
                          (a.edges[(i + 3) % 4].type == EdgeType.Flat) != (b.edges[(j + 1) % 4].type == EdgeType.Flat)):
                        similarity = float("inf")
                    else:
                        # ensure reflexivity
                        similarity = compare_edges(a.edges[i], b.edges[j]) + compare_edges(b.edges[j], a.edges[i])

                    similarities.append(similarity)

            similarity_matrix.append(similarities)

            if print_progress:
                print(".", end = "", flush = True)

    return np.array(similarity_matrix)


def compare_edges(a, b):
    if a.type == b.type or a.type == EdgeType.Flat or b.type == EdgeType.Flat:
        return float("inf")

    transform = find_transformation_lsq(
        np.array([a.points[0], a.points[-1]]),
        np.array([b.points[-1], b.points[0]])
    )

    for i in range(10):
        matches_a, matches_b = find_matches_closest(a, b, transform)
        new_transform = find_transformation_lsq(matches_a, matches_b)

        if transforms_equal(transform, new_transform) and i > 0:
            break
        transform = new_transform

        if i == 0:
            sum_sq_dist = np.sum(np.linalg.norm(matches_a - transform_points(matches_b, transform), axis = 1) ** 2)
            if sum_sq_dist > 10_000:
                return sum_sq_dist

    return np.sum(np.linalg.norm(matches_a - transform_points(matches_b, transform), axis = 1) ** 2)


def transforms_equal(a, b):
    return np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def find_matches_closest(a, b, transform):
    b_transformed = transform_points(b.points, transform)
    _, indices = a.kdtree.query(b_transformed)

    return (a.points[indices], b.points)


def transform_points(points, transform):
    t, w = transform
    rotation_matrix = np.array([
        [np.cos(w), -np.sin(w)],
        [np.sin(w), np.cos(w)]
    ])

    return np.dot(points, rotation_matrix) + t


def find_transformation_lsq(a, b):
    a_mean = np.mean(a, axis = 0)
    b_mean = np.mean(b, axis = 0)

    a = a - a_mean
    b = b - b_mean

    a_b = a * b
    a_rb = a * b[:, ::-1]

    s_xx, s_yy = np.sum(a_b, axis = 0)
    s_xy, s_yx = np.sum(a_rb, axis = 0)

    w = np.arctan2(s_xy - s_yx, s_xx + s_yy)
    rotation_matrix = np.array([
        [np.cos(w), -np.sin(w)],
        [np.sin(w), np.cos(w)]
    ])
    t = a_mean - np.dot(b_mean, rotation_matrix)

    return (t, w)


if __name__ == "__main__":
    main(*sys.argv[1:])
