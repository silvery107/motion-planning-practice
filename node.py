import numpy as np
from utils import EPSILON

class AbcNode:
    def __init__(self, config, parent=None) -> None:
        self.config = config
        self.parent = parent
    
    def __eq__(self, other:object) -> bool:
        return np.linalg.norm(self.config - other.config) < EPSILON

    def __hash__(self) -> int:
        return hash(tuple(self.config))
    
    def __repr__(self) -> str:
        return f"[Node: {self.config}]"
