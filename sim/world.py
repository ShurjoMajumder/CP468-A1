import copy
import iplotx as ipx
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sim import gen_matrix


class World(object):
    def __init__(self):
        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=42)
        self.parking_lot_locs: np.ndarray = np.array([0, 0, 0], dtype=np.int64)
        for i in range(len(self.parking_lot_locs)):
            self.parking_lot_locs[i] = np.random.choice(np.arange(16, dtype=np.int64))

    def draw_street_graph(self):
        G = nx.from_scipy_sparse_array(self._street_graph, create_using=nx.DiGraph, edge_attribute="weight")
        layout = nx.spring_layout(G)
        ipx.network(G, layout=layout, show=True)

    def get_map(self) -> csr_matrix:
        return copy.copy(self._street_graph)

    def is_parking_lot(self, i) -> bool:
        return i in self.parking_lot_locs
