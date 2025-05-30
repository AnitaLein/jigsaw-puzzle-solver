from enum import Enum
import random
import re
import sys
import qdarkstyle
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsPixmapItem
from PySide6.QtGui import QColor, QColorConstants, QPixmap
from PySide6.QtCore import Qt, QObject, QEvent

# pyside6-uic mainWindow.ui -o ui_mainWindow.py
from ui_mainWindow import Ui_MainWindow

sys.path.append("../src")
from compute_similarities import read_edges_file, transform_points

def main(puzzle_name, grid_spacing):
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api = "pyside6"))
    widget = MainWindow(puzzle_name, int(grid_spacing))
    widget.show()
    sys.exit(app.exec())


class MainWindow(QMainWindow):
    def __init__(self, puzzle_name, grid_spacing, parent = None):
        super().__init__(parent)

        self.puzzle_name = puzzle_name
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()

        matrix_path = f"../work/{self.puzzle_name}/solution/solution.txt"
        matrix = []
        with open(matrix_path, "r") as f:
            for line in f:
                for c in [" ", "(", ")"]:
                    line = line.replace(c, "")

                row = []
                for piece in line.split(";"):
                    if piece == "None":
                        row.append((None, None))
                        continue

                    img_idx, rot = piece.split(",")
                    row.append((img_idx, int(rot)))
                matrix.append(row)

        edge_dict = {}
        for row in matrix:
            for piece_name, _ in row:
                if piece_name == None:
                    continue
                edge_dict[piece_name] = read_edges_file(f"../work/{puzzle_name}/edges/{piece_name}.txt")

        # initialize position matrix to np.array([0, 0])
        pos_matrix = np.zeros((len(matrix), len(matrix[0]), 2))

        for i in range(len(matrix)):
            if matrix[i][0][1] == 1 or matrix[i][0][1] == 2:
                edges = edge_dict[matrix[i][0][0]]
                # transform all edges to the correct orientation
                points = []
                for j in range(4):
                    edge = edges[j]
                    edge_t = transform_points(edge.points, (np.array([0, 0]), matrix[i][0][1] * -np.pi / 2))
                    points.append(edge_t)

                # compute bounding box of all edges
                min_x = min([min(edge[:, 0]) for edge in points])
                max_x = max([max(edge[:, 0]) for edge in points])
                min_y = min([min(edge[:, 1]) for edge in points])
                max_y = max([max(edge[:, 1]) for edge in points])

                # add width of piece
                pos_matrix[i][0][0] = (max_x - min_x) if matrix[i][0][1] == 1 else (max_x - min_x)

        for j in range(len(matrix[0])):
            if matrix[0][j][1] == 2 or matrix[0][j][1] == 3:
                edges = edge_dict[matrix[0][j][0]]
                # transform all edges to the correct orientation
                points = []
                for i in range(4):
                    edge = edges[i]
                    edge_t = transform_points(edge.points, (np.array([0, 0]), matrix[0][j][1] * -np.pi / 2))
                    points.append(edge_t)

                # compute bounding box of all edges
                min_x = min([min(edge[:, 0]) for edge in points])
                max_x = max([max(edge[:, 0]) for edge in points])
                min_y = min([min(edge[:, 1]) for edge in points])
                max_y = max([max(edge[:, 1]) for edge in points])

                # add height of piece
                pos_matrix[0][j][1] = (max_y - min_y) if matrix[0][j][1] == 2 else (max_y - min_y)

        # vertical positioning
        for i in range(len(matrix) - 1):
            for j in range(len(matrix[i])):
                if matrix[i][j][0] == None:
                    continue

                edge_down_orig = edge_dict[matrix[i][j][0]][(2 - matrix[i][j][1]) % 4].points.copy()
                edge_up_orig = edge_dict[matrix[i + 1][j][0]][(0 - matrix[i + 1][j][1]) % 4].points.copy()

                edge_down = transform_points(edge_down_orig, (pos_matrix[i][j], matrix[i][j][1] * -np.pi / 2))
                edge_up = transform_points(edge_up_orig, (pos_matrix[i + 1][j], matrix[i + 1][j][1] * -np.pi / 2))

                dist_left_y = edge_down[-1][1] - edge_up[0][1]
                dist_right_y = edge_down[0][1] - edge_up[-1][1]

                avg_dist = (dist_left_y + dist_right_y) / 2
                pos_matrix[i + 1][j][1] += avg_dist

        # horizontal positioning
        for i in range(len(matrix)):
            for j in range(len(matrix[i]) - 1):
                if matrix[i][j][0] == None:
                    continue

                edge_right_orig = edge_dict[matrix[i][j][0]][(1 - matrix[i][j][1]) % 4].points.copy()
                edge_left_orig = edge_dict[matrix[i][j + 1][0]][(3 - matrix[i][j + 1][1]) % 4].points.copy()

                edge_right = transform_points(edge_right_orig, (pos_matrix[i][j], matrix[i][j][1] * -np.pi / 2))
                edge_left = transform_points(edge_left_orig, (pos_matrix[i][j + 1], matrix[i][j + 1][1] * -np.pi / 2))

                dist_top_x = edge_right[-1][0] - edge_left[0][0]
                print(edge_right[-1][0], edge_left[0][0])
                dist_bottom_x = edge_right[0][0] - edge_left[-1][0]
                print(edge_right[0][0], edge_left[-1][0])

                avg_dist = (dist_top_x + dist_bottom_x) / 2
                print(pos_matrix[i][j + 1][0])
                pos_matrix[i][j + 1][0] += avg_dist


        # Image loading base path
        image_base_path = f"../work/{puzzle_name}/pieces/"

        # Loop through the matrix and place images
        for row_idx, row in enumerate(matrix):
            for col_idx, (piece_name, rotation) in enumerate(row):
                if piece_name == None:
                    continue
                print(row_idx, col_idx, piece_name, rotation)
                image_path = f"{image_base_path}{piece_name}.png"
                pixmap = QPixmap(image_path)

                if not pixmap.isNull():
                    item = QGraphicsPixmapItem(pixmap)

                    # Positioning with offset
                    #item.setPos(col_idx * grid_spacing - pixmap.width() / 2, row_idx * grid_spacing - pixmap.height() / 2)
                    item.setPos(pos_matrix[row_idx][col_idx][0], pos_matrix[row_idx][col_idx][1])
                    # set origin point in center of image
                    #item.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
                    item.setRotation(rotation * 90)
                    item.setFlag(QGraphicsPixmapItem.ItemIsSelectable)
                    item.setFlag(QGraphicsPixmapItem.ItemIsMovable)

                    self.scene.addItem(item)

        # make scene rect very large to allow for scrolling
        self.scene.setSceneRect(-32000, -32000, 64000, 64000)

        self.ui.puzzleView.setScene(self.scene)

        # center on the scene
        #self.ui.puzzleView.centerOn(0, 0)


if __name__ == "__main__":
    main("presentation", 340)
