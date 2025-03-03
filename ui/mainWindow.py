from enum import Enum
import random
import re
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsPixmapItem
from PySide6.QtGui import QColor, QColorConstants, QPixmap
from PySide6.QtCore import Qt, QObject, QEvent

# pyside6-uic form.ui -o ui_form.py
from ui_mainWindow import Ui_MainWindow

rotation = {
    0 : 0,
    1 : 90,
    2 : 180,
    3 : 270
}


class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()

        matrix_path = '../work/eda/solution/solution.txt'
        matrix = []
        with open(matrix_path, "r") as f:
            for line in f:
                row = []
                # Convert matches to list of tuples, ensuring numbers are converted to int
                elements = line.strip().split("; ")  # Split based on "), ("
                print(elements)
                for elem in elements:
                    elem = elem.strip("() ")  # Remove parentheses and spaces
                    parts = elem.rsplit(", ", 1)  # Split only at the last comma to preserve "2_0"
                    if len(parts) == 2:
                        img_idx, rot = parts[0], int(parts[1])  # Convert rotation to int
                        row.append((img_idx, rot))
                matrix.append(row)


        # Image loading base path
        image_base_path = '../work/eda/pieces/'

        # Grid settings
        grid_spacing = 400  # Adjust as needed  

        # Loop through the matrix and place images
        for row_idx, row in enumerate(matrix):
            for col_idx, (image_index, rotation_count) in enumerate(row):
                print(row_idx, col_idx, image_index, rotation_count)
                image_path = f"{image_base_path}{image_index}.png"
                pixmap = QPixmap(image_path)

                if not pixmap.isNull():  # Ensure the image loads correctly
                    item = QGraphicsPixmapItem(pixmap)

                    # Positioning with offset
                    #item.setRotation(rotation_count * 90)  # Convert count to degrees
                    #setOriginPoint in center of image
                    item.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
                    item.setRotation(rotation[rotation_count])
                    item.setPos(col_idx * grid_spacing, row_idx * grid_spacing)
                    item.setFlag(QGraphicsPixmapItem.ItemIsSelectable)
                    item.setFlag(QGraphicsPixmapItem.ItemIsMovable)

                    self.scene.addItem(item)

        # make scene rect very large to allow for scrolling
        self.scene.setSceneRect(-32000, -32000, 64000, 64000)

        self.ui.puzzleView.setScene(self.scene)

        # center on the scene
        #self.ui.puzzleView.centerOn(0, 0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
