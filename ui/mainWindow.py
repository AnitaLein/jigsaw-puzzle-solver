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

        placements_path = f"../work/{self.puzzle_name}/placements/placements.txt"
        placements = {}
        with open(placements_path, "r") as f:
            for line in f:
                piece_name, transformation = line.split(": ")
                x, y, w = transformation.split(" ")
                placements[piece_name] = (np.array([float(x), float(y)]), float(w.strip()))

        pieces = {}
        for piece_name in placements:
            pieces[piece_name] = read_edges_file(f"../work/{puzzle_name}/edges/{piece_name}.txt")

        # Image loading base path
        image_base_path = f"../work/{puzzle_name}/pieces/"

        # Loop through the matrix and place images
        for piece_name, (pos, rotation) in placements.items():
            image_path = f"{image_base_path}{piece_name}.png"
            pixmap = QPixmap(image_path)

            if not pixmap.isNull():
                item = QGraphicsPixmapItem(pixmap)

                # Positioning with offset
                #item.setPos(col_idx * grid_spacing - pixmap.width() / 2, row_idx * grid_spacing - pixmap.height() / 2)
                item.setPos(pos[0], pos[1])
                # set origin point in center of image
                #item.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
                item.setRotation(rotation * 180 / np.pi)
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
