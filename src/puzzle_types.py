from enum import Enum
from dataclasses import dataclass
import numpy as np
import math
import cv2
from typing import List, Tuple

# Define the EdgeType class
class EdgeType(Enum):
    Gerade = "Gerade"
    Nase = "Nase"
    Loch = "Loch"

# Define the BasicEdge class
@dataclass
class BasicEdge:
    type: EdgeType
    offset: float

# Define the BasicPuzzlePiece class
@dataclass
class BasicPuzzlePiece:
    edges: List[BasicEdge]  # List of BasicEdge objects

# Define the Edge class
@dataclass
class Edge:
    type: EdgeType
    points: List[np.ndarray]  # List of points as np.array (cv2.Point2d equivalent)

@dataclass
class PuzzlePiece:
    edges: List[Edge]  # List of Edge objects

@dataclass

class Transform:
    def __init__(self, t=np.array([0.0, 0.0]), w=0.0):
        self.t = np.array(t, dtype=np.float64)
        self.w = w
    
    def __eq__(self, other):
        return np.allclose(self.t, other.t) and np.isclose(self.w, other.w)
    
    def __call__(self, p):
        p = np.squeeze(p)  # Ensure p is a 1D array of shape (2,)
        cos_w, sin_w = np.cos(self.w), np.sin(self.w)
        return np.array([
            p[0] * cos_w - p[1] * sin_w + self.t[0],
            p[0] * sin_w + p[1] * cos_w + self.t[1]
        ])

