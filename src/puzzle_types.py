from enum import Enum
from dataclasses import dataclass
#from ibm_db import num_rows
import numpy as np
import math
import cv2
from typing import List, Tuple, Dict


class EdgeType(int, Enum):
    Flat = 0
    Tab = 1
    Blank = -1

class PieceType(Enum):
    Corner = "Corner"
    Edge = "Edge"
    Center = "Center"

@dataclass
class Edge:
    type: EdgeType
    points: np.ndarray


@dataclass
class PuzzlePiece:
    name: str
    image: np.ndarray
    edges: List[Edge]
