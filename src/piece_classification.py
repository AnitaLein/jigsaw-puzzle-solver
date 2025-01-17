import cv2
import numpy as np


def preprocess_image(image, b):
    preprocessed_image = cv2.medianBlur(image, 3)

    # Otsu threshold
    _, preprocessed_image = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill small holes in the background and foreground
    if b:
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        preprocessed_image = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, kernel)
        preprocessed_image = cv2.morphologyEx(preprocessed_image, cv2.MORPH_CLOSE, kernel)

    # Segment image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_image = np.zeros_like(image, dtype=np.uint8)
    for contour in contours:
        if len(contour) > 150:
            cv2.drawContours(preprocessed_image, [contour], 0, 255, cv2.FILLED)

    return preprocessed_image


def create_puzzle_piece(bounding_box, top, top_offset, right, right_offset, bottom, bottom_offset, left, left_offset):
    tab_height = 0.3
    tab_width = 0.25
    expansion = 0  # 0.03

    points = []
    points.append((0.0, 0.0))

    if top != 0:
        points.append(((1 - tab_width) / 2 + top_offset, 0.0))
        points.append(((1 - tab_width) / 2 + top_offset - expansion, -top * tab_height))
        points.append(((1 + tab_width) / 2 + top_offset + expansion, -top * tab_height))
        points.append(((1 + tab_width) / 2 + top_offset, 0.0))
    points.append((1.0, 0.0))

    if right != 0:
        points.append((1.0, (1 - tab_width) / 2 + right_offset))
        points.append((1 + right * tab_height, (1 - tab_width) / 2 + right_offset - expansion))
        points.append((1 + right * tab_height, (1 + tab_width) / 2 + right_offset + expansion))
        points.append((1.0, (1 + tab_width) / 2 + right_offset))
    points.append((1.0, 1.0))

    if bottom != 0:
        points.append(((1 + tab_width) / 2 + bottom_offset, 1.0))
        points.append(((1 + tab_width) / 2 + bottom_offset + expansion, 1 + bottom * tab_height))
        points.append(((1 - tab_width) / 2 + bottom_offset - expansion, 1 + bottom * tab_height))
        points.append(((1 - tab_width) / 2 + bottom_offset, 1.0))
    points.append((0.0, 1.0))

    if left != 0:
        points.append((0.0, (1 + tab_width) / 2 + left_offset))
        points.append((-left * tab_height, (1 + tab_width) / 2 + left_offset + expansion))
        points.append((-left * tab_height, (1 - tab_width) / 2 + left_offset - expansion))
        points.append((0.0, (1 - tab_width) / 2 + left_offset))

    rect = [0, 0, 1, 1]  # x, y, width, height
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    rect_width = max_x - min_x
    rect_height = max_y - min_y

    # Normalize to a unit square
    points[:, 0] = (points[:, 0] - min_x) / rect_width
    points[:, 1] = (points[:, 1] - min_y) / rect_height

    # Scale and translate to output bounding box
    puzzle_piece = []
    for point in points:
        x = round(bounding_box[0] + bounding_box[2] * point[0])
        y = round(bounding_box[1] + bounding_box[3] * point[1])
        puzzle_piece.append((x, y))

    return np.array(puzzle_piece, dtype=np.int32)


def main():
    # Main processing
    image = cv2.imread("../data/10b.jpg", cv2.IMREAD_GRAYSCALE)
    preprocessed_image = preprocess_image(image, False)
    preprocessed_image2 = preprocess_image(image, True)

    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_puzzle_pieces = np.zeros(preprocessed_image.shape, dtype=np.uint8)

    for contour in contours:
        bounding_box = cv2.boundingRect(contour)
        roi = image[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

        #contour = [point - bounding_box[:2] for point in contour]
        for point in contour:
            point -= bounding_box[:2]

        mask = np.zeros((bounding_box[3], bounding_box[2]), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

        best_params = None
        best_white = float('inf')

        for top in range(-1, 2):
            for right in range(-1, 2):
                for bottom in range(-1, 2):
                    for left in range(-1, 2):
                        for top_offset in np.arange(-0.1, 0.11, 0.1):
                            for right_offset in np.arange(-0.1, 0.11, 0.1):
                                for bottom_offset in np.arange(-0.1, 0.11, 0.1):
                                    for left_offset in np.arange(-0.1, 0.11, 0.1):
                                        test = np.zeros_like(mask)
                                        puzzle_piece = create_puzzle_piece(cv2.boundingRect(contour), top, top_offset, right, right_offset, bottom, bottom_offset, left, left_offset)
                                        cv2.drawContours(test, [puzzle_piece], 0, 255, cv2.FILLED)

                                        difference = cv2.absdiff(mask, test)
                                        white = cv2.countNonZero(difference)

                                        if white < best_white:
                                            best_params = (top, top_offset, right, right_offset, bottom, bottom_offset, left, left_offset)
                                            best_white = white

        puzzle_piece = create_puzzle_piece(cv2.boundingRect(contour), *best_params)
        print(best_params)
        directions = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (-1, -1), (1, 1), (-1, 1), (1, -1),
            (0, -2), (2, 0), (0, 2), (-2, 0)
        ]


        for _ in range(100):
            for point_idx, point in enumerate(puzzle_piece):
                test = np.zeros_like(mask)
                cv2.drawContours(test, [puzzle_piece], 0, 255, cv2.FILLED)
                difference = cv2.absdiff(mask, test)
                best_diff = cv2.countNonZero(difference)
                best_dir = (0, 0)

                for direction in directions:
                    new_point = point + direction
                    if 0 <= new_point[0] < mask.shape[1] and 0 <= new_point[1] < mask.shape[0]:
                        test = np.zeros_like(mask)
                        puzzle_piece[point_idx] = new_point
                        cv2.drawContours(test, [puzzle_piece], 0, 255, cv2.FILLED)
                        difference = cv2.absdiff(mask, test)
                        diff = cv2.countNonZero(difference)
                        if diff < best_diff:
                            best_diff = diff
                            best_dir = direction

                        puzzle_piece[point_idx] = point

                puzzle_piece[point_idx] += best_dir

        puzzle_piece += bounding_box[:2]
        cv2.drawContours(found_puzzle_pieces, [puzzle_piece], 0, 255, cv2.FILLED)

        

    # Compute difference for visualization
    diff = cv2.absdiff(preprocessed_image, found_puzzle_pieces)
    cv2.namedWindow("Difference", cv2.WINDOW_NORMAL)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
