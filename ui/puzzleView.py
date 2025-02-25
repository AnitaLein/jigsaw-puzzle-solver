from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PySide6.QtGui import QColor, QColorConstants, QTransform
from PySide6.QtCore import Qt, QObject, QEvent

class PuzzleView(QGraphicsView):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.panning = False
        self.rubberBanding = False
        self.dragged_items = []
        self.zoom_levels = [0.25, 0.5, 1, 2, 4, 8]
        self.zoom_index = 2

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.panning = True
            self.panStart = event.pos()
            self.updateCursor(event.pos())

        elif event.button() == Qt.LeftButton:
            super().mousePressEvent(event)

            self.dragged_items = self.scene().selectedItems()
            for item in self.dragged_items:
                item.setZValue(1)
            self.rubberBanding = not self.dragged_items

            self.updateCursor(event.pos())

        elif event.button() == Qt.RightButton:
            pass

        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.panStart
            self.translate(delta.x() / self.transform().m11(), delta.y() / self.transform().m22())
            self.panStart = event.pos()

        self.updateCursor(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.updateCursor(event.pos())

        elif event.button() == Qt.LeftButton:
            self.rubberBanding = False
            for item in self.dragged_items:
                # stack before colliding items
                for colliding_item in self.scene().collidingItems(item):
                    if colliding_item.zValue() == 0:
                        colliding_item.stackBefore(item)

            for item in self.dragged_items:
                item.setZValue(0)

            self.dragged_items = []
            self.updateCursor(event.pos())
            super().mouseReleaseEvent(event)

        elif event.button() == Qt.RightButton:
            pass

        else:
            super().mouseReleaseEvent(event)

    def updateCursor(self, pos):
        if self.panning:
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
        elif not self.rubberBanding and not self.dragged_items and self.itemAt(pos) is not None:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.dragged_items:
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self.viewport().unsetCursor()

    def wheelEvent(self, event):
        if self.panning:
            return

        if event.angleDelta().y() > 0:
            self.zoom_index += 1
        else:
            self.zoom_index -= 1
        self.zoom_index = max(0, min(self.zoom_index, len(self.zoom_levels) - 1))

        zoom_level = self.zoom_levels[self.zoom_index]
        #print(zoom_level)
        self.setTransform(QTransform.fromScale(zoom_level, zoom_level))
