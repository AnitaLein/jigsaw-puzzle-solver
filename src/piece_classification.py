import sys
from pathlib import Path
import cv2
import numpy as np
import statistics
from itertools import product
from puzzle_types import PuzzlePiece, Edge, EdgeType

def process_scan(scan, scan_name):
    contours = find_contours(scan, False)

    puzzle_pieces = []
    for i, contour in enumerate(contours):
        crop_region = cv2.boundingRect(contour)

        # move contour to origin
        contour -= crop_region[:2]

        # create a new image with an alpha channel
        cropped_image = np.zeros((crop_region[3], crop_region[2], 4), dtype=np.uint8)

        # set the alpha channel to the filled contour
        mask = np.zeros((crop_region[3], crop_region[2]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
        cropped_image[:, :, 3] = mask

        # copy the actual image data
        cropped_image[:, :, :3] = scan[crop_region[1]:crop_region[1] + crop_region[3], crop_region[0]:crop_region[0] + crop_region[2]]

        # get initial classification parameters
        params = classify_piece(contour)
        synthetic_contour = create_puzzle_piece(cv2.boundingRect(contour), *params)

        # refine synthesized contour to get more accurate corner points
        synthetic_contour = refine_contour(synthetic_contour, contour)

        # find corner indices in the original contour
        corner_indices = get_corner_indices(synthetic_contour, contour, params[:4])

        # split the contour into edges
        edge_paths = split_contour(contour, corner_indices)
        edges = [Edge(edge_type, edge_path) for edge_type, edge_path in zip(params[:4], edge_paths)]

        puzzle_pieces.append(PuzzlePiece(f"{scan_name}_{i}", cropped_image, edges))

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
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def classify_piece(contour):
    bounding_box = cv2.boundingRect(contour)
    mask = np.zeros((bounding_box[3], bounding_box[2]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

    def genParams():
        for edge_types in product(EdgeType, repeat = 4):
            nonFlatOffsets = np.linspace(-0.1, 0.1, 3)
            possible_offsets = [([0] if val == 0 else nonFlatOffsets) for val in edge_types]

            for offsets in product(*possible_offsets):
                yield *edge_types, *offsets

    def diffToReference(params):
        test = np.zeros_like(mask)
        test_contour = create_puzzle_piece(cv2.boundingRect(contour), *params)
        cv2.drawContours(test, [test_contour], 0, 255, cv2.FILLED)

        # calculate the difference between the actual puzzle piece and the generated one
        difference = cv2.absdiff(mask, test)
        return cv2.countNonZero(difference)

    return min(genParams(), key = diffToReference)


def create_puzzle_piece(bounding_box, top, right, bottom, left, top_offset, right_offset, bottom_offset, left_offset):
    tab_height = 0.3
    tab_width = 0.25
    expansion = 0  # 0.03

    points = []
    points.append([0.0, 0.0])

    if top != 0:
        points.append([(1 - tab_width) / 2 + top_offset, 0.0])
        points.append([(1 - tab_width) / 2 + top_offset - expansion, -top * tab_height])
        points.append([(1 + tab_width) / 2 + top_offset + expansion, -top * tab_height])
        points.append([(1 + tab_width) / 2 + top_offset, 0.0])
    points.append([1.0, 0.0])

    if right != 0:
        points.append([1.0, (1 - tab_width) / 2 + right_offset])
        points.append([1 + right * tab_height, (1 - tab_width) / 2 + right_offset - expansion])
        points.append([1 + right * tab_height, (1 + tab_width) / 2 + right_offset + expansion])
        points.append([1.0, (1 + tab_width) / 2 + right_offset])
    points.append([1.0, 1.0])

    if bottom != 0:
        points.append([(1 + tab_width) / 2 + bottom_offset, 1.0])
        points.append([(1 + tab_width) / 2 + bottom_offset + expansion, 1 + bottom * tab_height])
        points.append([(1 - tab_width) / 2 + bottom_offset - expansion, 1 + bottom * tab_height])
        points.append([(1 - tab_width) / 2 + bottom_offset, 1.0])
    points.append([0.0, 1.0])

    if left != 0:
        points.append([0.0, (1 + tab_width) / 2 + left_offset])
        points.append([-left * tab_height, (1 + tab_width) / 2 + left_offset + expansion])
        points.append([-left * tab_height, (1 - tab_width) / 2 + left_offset - expansion])
        points.append([0.0, (1 - tab_width) / 2 + left_offset])

    points = np.array(points)
    min_x, min_y = np.min(points, axis = 0)
    max_x, max_y = np.max(points, axis = 0)

    width = max_x - min_x
    height = max_y - min_y

    # normalize to a unit square
    points[:, 0] = (points[:, 0] - min_x) / width
    points[:, 1] = (points[:, 1] - min_y) / height

    # scale and translate to output bounding box
    puzzle_piece = []
    for point in points:
        x = round(bounding_box[0] + bounding_box[2] * point[0])
        y = round(bounding_box[1] + bounding_box[3] * point[1])
        puzzle_piece.append([x, y])

    return np.array(puzzle_piece)


def refine_contour(contour, reference):
    bounding_box = cv2.boundingRect(reference)
    mask = np.zeros((bounding_box[3], bounding_box[2]), dtype=np.uint8)
    cv2.drawContours(mask, [reference], 0, 255, cv2.FILLED)

    directions = [
        (0, -1), (1, 0), (0, 1), (-1, 0),
        (-1, -1), (1, 1), (-1, 1), (1, -1),
        (0, -2), (2, 0), (0, 2), (-2, 0)
    ]

    for _ in range(100):
        for i, point in enumerate(contour):
            test = np.zeros_like(mask)
            cv2.drawContours(test, [contour], 0, 255, cv2.FILLED)
            difference = cv2.absdiff(mask, test)
            best_diff = cv2.countNonZero(difference)
            best_dir = (0, 0)

            for dir in directions:
                new_point = point + dir
                if 0 <= new_point[0] < mask.shape[1] and 0 <= new_point[1] < mask.shape[0]:
                    test = np.zeros_like(mask)
                    contour[i] = new_point
                    cv2.drawContours(test, [contour], 0, 255, cv2.FILLED)
                    difference = cv2.absdiff(mask, test)
                    diff = cv2.countNonZero(difference)
                    if diff < best_diff:
                        best_diff = diff
                        best_dir = dir

                    contour[i] = point

            contour[i] += best_dir

    return contour


def get_corner_indices(synthetic_contour, reference_contour, edge_types):
    # extract approximate corners from generated contour
    corners = []
    i = 0
    for edge_type in edge_types:
        corners.append(synthetic_contour[i])
        i += 1 if edge_type == EdgeType.Flat else 5

    # slightly move corners diagonally outwards to improve accuracy
    directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    corners = [corner + direction for corner, direction in zip(corners, directions)]

    # find the indices of the nearest contour points
    return [min(range(len(reference_contour)), key = lambda i: np.linalg.norm(corner - reference_contour[i])) for corner in corners]


def split_contour(contour, corner_indices):
    paths = []

    for i in range(len(corner_indices)):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % 4]

        path = []
        j = start_idx
        while j != end_idx:
            path.append(contour[j])
            j = (j + 1) % len(contour)

        path.append(contour[end_idx])
        paths.append(np.array(path))

    return paths


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    puzzle_name = sys.argv[2]
    scan_name = sys.argv[3]
    work_dir = Path(sys.argv[4])

    input_dir = Path(data_dir, puzzle_name)
    output_dir = Path(work_dir, puzzle_name)
    img_output_dir = Path(output_dir, "pieces")
    edge_output_dir = Path(output_dir, "edges")

    # create output directories if they do not exist
    Path(img_output_dir).mkdir(parents = True, exist_ok = True)
    Path(edge_output_dir).mkdir(parents = True, exist_ok = True)

    input_path = Path(input_dir, scan_name + "b.jpg")
    scan = cv2.imread(input_path)
    assert scan is not None, f"Could not read image at {input_path}"

    puzzle_pieces = process_scan(scan, scan_name)

    # write puzzle pieces to disk
    for puzzle_piece in puzzle_pieces:
        debug = False
        if debug:
            # restore background
            #puzzle_piece.image[:, :, 3] = 255

            # draw edges
            for edge in puzzle_piece.edges:
                if edge.type == EdgeType.Flat:
                    color = (0, 255, 255, 255)
                elif edge.type == EdgeType.Tab:
                    color = (0, 255, 0, 255)
                elif edge.type == EdgeType.Blank:
                    color = (0, 0, 255, 255)

                cv2.polylines(puzzle_piece.image, [edge.points], False, color, 2)

            # draw corners
            for edge in puzzle_piece.edges:
                cv2.circle(puzzle_piece.image, tuple(edge.points[0]), 5, (255, 0, 0, 255), cv2.FILLED)

        # write cropped images
        cv2.imwrite(Path(img_output_dir, f"{puzzle_piece.name}.png"), puzzle_piece.image)

        # write edges
        with open(Path(edge_output_dir, f"{puzzle_piece.name}.txt"), 'w') as f:
            for edge in puzzle_piece.edges:
                points = ", ".join([f"({p[0]}, {p[1]})" for p in edge.points])
                f.write(f"{edge.type.name}: {points}\n")

    print("classfication done")
