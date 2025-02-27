import sys
from dataclasses import dataclass
from typing import List
from pathlib import Path
import numpy as np
from puzzle_types import Edge, EdgeType
from scipy.spatial import cKDTree
import csv

@dataclass
class PuzzlePieceShape:
    name: str
    edges: List[Edge]


def main(puzzle_name, work_dir):
    input_dir = Path(work_dir, puzzle_name, "edges")

    # read all puzzle pieces in input_dir

    files = input_dir.glob('*.txt')
    files = [f for f in files if f.is_file()]

    puzzle_pieces = []
    for file in files:
        piece_name = file.stem
        edges = read_edges_file(file)
        puzzle_pieces.append(PuzzlePieceShape(piece_name, edges))

    print(f"read {len(puzzle_pieces)} puzzle pieces")

    print("computing similarity matrix")
    similarity_matrix = compute_similarity_matrix(puzzle_pieces, print_progress = True)
    print()

    print("writing similarity matrix")
    with open(Path(work_dir, puzzle_name, "similarity_matrix.txt"), mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerows(similarity_matrix)


def read_edges_file(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            type, points = line.split(": ")
            edge_type = EdgeType[type]
            points = points[1:-2].split("), (")
            points = [point.split(", ") for point in points]
            points = [(int(x), int(y)) for x, y in points]
            edges.append(Edge(edge_type, points))

    assert len(edges) == 4

    return edges


def compute_similarity_matrix(puzzle_pieces, print_progress = False):
    similarity_matrix = []

    for i in range(len(puzzle_pieces)):
        for x in range(4):
            similarities = []

            for j in range(len(puzzle_pieces)):
                for y in range(4):
                    if(puzzle_pieces[i].edges[x].type == EdgeType.Flat or puzzle_pieces[j].edges[y].type == EdgeType.Flat):
                        similarity = float("inf")
                    # Check edge continuity
                    elif ((puzzle_pieces[i].edges[(x + 1) % 4].type == EdgeType.Flat) !=
                        (puzzle_pieces[j].edges[(y + 3) % 4].type == EdgeType.Flat) or
                        (puzzle_pieces[i].edges[(x + 3) % 4].type == EdgeType.Flat) !=
                        (puzzle_pieces[j].edges[(y + 1) % 4].type == EdgeType.Flat)):
                        similarity = float("inf")
                    else:
                        similarity = compare_edges(puzzle_pieces[i].edges[x], puzzle_pieces[j].edges[y])

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
        matches_a, matches_b = find_matches_closest(a.points, b.points, transform)
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
    b_transformed = transform_points(b, transform)

    tree = cKDTree(a)
    _, indices = tree.query(b_transformed)

    return ([a[idx] for idx in indices], b)


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
    t = a_mean - np.dot(rotation_matrix, b_mean)

    return (t, w)


if __name__ == "__main__":
    main(*sys.argv[1:])
