import sys
from pathlib import Path
import cv2
from joblib import Parallel, delayed
from extract_pieces import main as extract_pieces_main
from classify_piece import main as classify_piece_main
from compute_similarities import main as compute_similarities_main
from solve_puzzle import main as solve_puzzle_main
from place_pieces import main as place_pieces_main
from Fourier import compute_2d_fft
sys.path.append('../ui')
from mainWindow import main as main_window_main

def main(puzzle_name = "eda", data_dir = "../data", work_dir = "../work", random_walks = 10_000, max_random_walk_length = 3, workers = 12):
    extract_pieces(puzzle_name, data_dir, work_dir)
    classify_pieces(puzzle_name, work_dir, workers)
    compute_similarities_main(puzzle_name, work_dir, workers)
    solve_puzzle_main(puzzle_name, work_dir, random_walks, max_random_walk_length, workers)
    place_pieces_main(puzzle_name, work_dir)
    main_window_main(puzzle_name, 400)
    print("done")


def extract_pieces(puzzle_name, data_dir, work_dir):
    scan_dir = Path(data_dir, puzzle_name)
    scans = scan_dir.glob("*b.jpg")
    scans = [f.stem[:-1] for f in scans if f.is_file()]

    for scan in scans:
        extract_pieces_main(data_dir, puzzle_name, scan, work_dir)


def classify_pieces(puzzle_name, work_dir, workers):
    contour_dir = Path(work_dir, puzzle_name, "contours")
    pieces = contour_dir.glob("*.txt")
    pieces = [f.stem for f in pieces if f.is_file()]

    Parallel(n_jobs = workers)(delayed(classify_piece_main)(puzzle_name, piece, work_dir) for piece in pieces)


def fourier_transform(puzzle_name, work_dir):
    # read from pieces directory
    image_dir = Path(work_dir, puzzle_name, "pieces")
    pieces = image_dir.glob("*.png")
    pieces = [f.stem for f in pieces if f.is_file()]

    for piece in pieces:
        image = cv2.imread(Path(image_dir, f"{piece}.png"), cv2.IMREAD_UNCHANGED)
        image[:, :, 3] = 255
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        compute_2d_fft(work_dir, puzzle_name, image, piece)


if __name__ == "__main__":
    main(*sys.argv[1:])
