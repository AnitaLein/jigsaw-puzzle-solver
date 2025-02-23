from enum import Enum
from dataclasses import dataclass
import numpy as np
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
