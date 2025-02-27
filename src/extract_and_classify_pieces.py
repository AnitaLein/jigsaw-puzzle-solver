import sys
from pathlib import Path
import cv2
from piece_classification import process_scan

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

scan = cv2.imread(Path(input_dir, scan_name + "b.jpg"))
assert scan is not None, f"Could not read image at {path}"

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
