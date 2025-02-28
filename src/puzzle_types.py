from enum import Enum
from dataclasses import dataclass
import numpy as np
from typing import List

class EdgeType(int, Enum):
    Flat = 0
    Tab = 1
    Blank = -1

@dataclass
class Edge:
    type: EdgeType
    points: np.ndarray

@dataclass
class PuzzlePiece:
    name: str
    image: np.ndarray
    edges: List[Edge]
