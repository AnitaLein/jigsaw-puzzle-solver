import sys
from pathlib import Path
from joblib import Parallel, delayed
from extract_pieces import main as extract_pieces_main
from classify_piece import main as classify_piece_main
from compute_similarities import main as compute_similarities_main
from solve_puzzle import main as solve_puzzle_main

def main(puzzle_name = "horse300", data_dir = "../data", work_dir = "../work"):
    extract_pieces(puzzle_name, data_dir, work_dir)
    classify_pieces(puzzle_name, work_dir)
    compute_similarities_main(puzzle_name, work_dir)
    solve_puzzle_main(puzzle_name, work_dir, random_walks = 10_000, max_random_walk_length = 3)

    print("done")


def extract_pieces(puzzle_name, data_dir, work_dir):
    scan_dir = Path(data_dir, puzzle_name)
    scans = scan_dir.glob("*b.jpg")
    scans = [f.stem[:-1] for f in scans if f.is_file()]

    for scan in scans:
        extract_pieces_main(data_dir, puzzle_name, scan, work_dir)


def classify_pieces(puzzle_name, work_dir):
    contour_dir = Path(work_dir, puzzle_name, "contours")
    pieces = contour_dir.glob("*.txt")
    pieces = [f.stem for f in pieces if f.is_file()]

    Parallel(n_jobs = 16)(delayed(classify_piece_main)(puzzle_name, piece, work_dir) for piece in pieces)


if __name__ == "__main__":
    main(*sys.argv[1:])
