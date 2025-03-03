from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QGridLayout, QWidget, QLabel, QGraphicsScene
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTransform
import sys

# pyside6-uic form.ui -o ui_form.py
from ui_mainWindow import Ui_MainWindow

rotation = {
    0 : 0,
    1 : 90,
    2 : 180,
    3 : 270
}

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()

        matrix_path = '../work/horse208/solution/solution.txt'
        matrix = []
        with open(matrix_path, "r") as f:
            for line in f:
                row = []
                elements = line.strip().split("; ")  # Split based on "), ("
                for elem in elements:
                    if elem == None:
                        continue
                    elem = elem.strip("() ")  # Remove parentheses and spaces
                    parts = elem.rsplit(", ", 1)  # Split only at the last comma to preserve "2_0"
                    if len(parts) == 2:
                        img_idx, rot = parts[0], int(parts[1])  # Convert rotation to int
                        row.append((img_idx, rot))
                matrix.append(row)

        # Image loading base path
        image_base_path = '../work/horse208/pieces/'

        # Create a grid layout for the main window
        grid_layout = QGridLayout()

        # Define a zoom-out factor (e.g., 0.5 for 50% size)
        zoom_factor = 0.225

        horizontal_spacing = 5  # Space between columns
        vertical_spacing = 5  # Space between rows

        # Set horizontal and vertical spacing between the items
        grid_layout.setHorizontalSpacing(horizontal_spacing)  # Set horizontal spacing
        grid_layout.setVerticalSpacing(vertical_spacing)  # Set vertical spacing
        grid_layout.setContentsMargins(5, 5, 5, 5)  # Optional: Set margins around the grid

        # Loop through the matrix and create labels for each image
        for row_idx, row in enumerate(matrix):
            for col_idx, (image_index, rotation_count) in enumerate(row):
                image_path = f"{image_base_path}{image_index}.png"
                pixmap = QPixmap(image_path)

                if not pixmap.isNull():  # Ensure the image loads correctly
                    # Apply zoom factor to resize the image
                    new_width = int(pixmap.width() * zoom_factor)
                    new_height = int(pixmap.height() * zoom_factor)
                    pixmap = pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio)

                    # Create a QTransform for rotating the pixmap
                    transform = QTransform()
                    transform.rotate(rotation_count * 90)  # Rotate based on the rotation_count
                    pixmap = pixmap.transformed(transform)

                    # Create QLabel for each image
                    label = QLabel()
                    label.setPixmap(pixmap)

                    # Add the QLabel to the grid layout at the correct row and column
                    grid_layout.addWidget(label, row_idx, col_idx)

        # Create a QWidget to hold the grid layout
        central_widget = QWidget(self)
        central_widget.setLayout(grid_layout)

        # Set central widget to the main window
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
