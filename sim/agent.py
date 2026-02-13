import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix


class Agent(object):
    def __init__(self):
        self.current_pos: np.ndarray = np.array([0, 0], dtype=np.int64)
        self.target_lot: np.ndarray = np.array([0, 0], dtype=np.int64)
        self.destination: int = 15

    def find_lot(self, A: csr_matrix):
        # Tcsr = sparse.csgraph.breadth_first_order()
        ...

    def step(self):
        ...

    def act(self):
        ...

    def reset(self):
        ...

    def finished(self):
        ...
