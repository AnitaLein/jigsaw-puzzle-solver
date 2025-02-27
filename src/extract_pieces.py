import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import statistics

@dataclass
class UnclassifiedPuzzlePiece:
    name: str
    image: np.ndarray
    contour: np.ndarray


def main(data_dir, puzzle_name, scan_name, work_dir):
    input_dir = Path(data_dir, puzzle_name)
    output_dir = Path(work_dir, puzzle_name)
    image_output_dir = Path(output_dir, "pieces")
    contour_output_dir = Path(output_dir, "contours")

    # create output directories if they do not exist
    Path(image_output_dir).mkdir(parents = True, exist_ok = True)
    Path(contour_output_dir).mkdir(parents = True, exist_ok = True)

    input_path = Path(input_dir, scan_name + "b.jpg")
    scan = cv2.imread(input_path)
    assert scan is not None, f"Could not read image at {input_path}"

    puzzle_pieces = extract_pieces(scan, scan_name)

    # write puzzle pieces to disk
    for puzzle_piece in puzzle_pieces:
        debug = False
        if debug:
            # restore background
            #puzzle_piece.image[:, :, 3] = 255
            pass

        # write cropped image
        cv2.imwrite(Path(image_output_dir, f"{puzzle_piece.name}.png"), puzzle_piece.image)

        # write contour
        with open(Path(contour_output_dir, f"{puzzle_piece.name}.txt"), "w") as file:
            points = ", ".join([f"({p[0]}, {p[1]})" for p in puzzle_piece.contour])
            file.write(f"{points}\n")

    print("extracted", len(puzzle_pieces), "puzzle pieces")


def extract_pieces(scan, scan_name):
    contours = find_contours(scan, False)

    puzzle_pieces = []
    for i, contour in enumerate(contours):
        cropped_image = crop_contour(scan, contour)

        # move contour to origin
        bounding_box = cv2.boundingRect(contour)
        contour -= bounding_box[:2]

        puzzle_pieces.append(UnclassifiedPuzzlePiece(f"{scan_name}_{i}", cropped_image, contour))

    return puzzle_pieces


def find_contours(image, b):
    preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = cv2.medianBlur(preprocessed_image, 3)

    # otsu threshold
    _, preprocessed_image = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fill small holes in the background and foreground
    if b:
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        preprocessed_image = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, kernel)
        preprocessed_image = cv2.morphologyEx(preprocessed_image, cv2.MORPH_CLOSE, kernel)

    # segment image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
    contours = [cnt.reshape(-1, 2) for cnt in contours]

    # find average contour height
    average_height = statistics.mean([cv2.boundingRect(contour)[3] for contour in contours])

    # sort by contour center
    def grid_order(contour):
        moments = cv2.moments(contour)
        center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        return center[1] * image.shape[1] * 2 / average_height + center[0]

    contours.sort(key = grid_order)

    # reverse all contours to get clockwise orientation
    for contour in contours:
        contour[:] = contour[::-1]

    return contours


def crop_contour(scan, contour):
    crop_region = cv2.boundingRect(contour)

    # move contour to origin
    contour = contour.copy()
    contour -= crop_region[:2]

    # create a new image with an alpha channel
    cropped_image = np.zeros((crop_region[3], crop_region[2], 4), dtype=np.uint8)

    # set the alpha channel to the filled contour
    mask = np.zeros((crop_region[3], crop_region[2]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
    cropped_image[:, :, 3] = mask

    # copy the actual image data
    cropped_image[:, :, :3] = scan[crop_region[1]:crop_region[1] + crop_region[3], crop_region[0]:crop_region[0] + crop_region[2]]

    return cropped_image


if __name__ == "__main__":
    main(*sys.argv[1:])
