import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PySide6.QtGui import QColor, QColorConstants
from PySide6.QtCore import Qt, QObject, QEvent

# pyside6-uic form.ui -o ui_form.py
from ui_mainWindow import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.scene = QGraphicsScene()
        # add a grid of rectangles to the scene
        for i in range(10):
            for j in range(10):
                rect = self.scene.addRect(i * 40, j * 40, 30, 30)
                rect.setBrush(QColor(i * 25, j * 25, 0)) # QColorConstants.Red

                # make the rectangles selectable and movable
                rect.setFlag(QGraphicsRectItem.ItemIsSelectable)
                rect.setFlag(QGraphicsRectItem.ItemIsMovable)

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
