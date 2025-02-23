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
    def __init__(self, t=None, w=0.0):
        self.t = np.array(t if t is not None else [0.0, 0.0])
        self.w = w
    
    def __eq__(self, other):
        return np.allclose(self.t, other.t) and math.isclose(self.w, other.w)
    
    def __call__(self, p):
        x, y = p
        return np.array([
            x * math.cos(self.w) - y * math.sin(self.w) + self.t[0],
            x * math.sin(self.w) + y * math.cos(self.w) + self.t[1]
        ])
