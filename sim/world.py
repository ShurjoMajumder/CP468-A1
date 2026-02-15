import copy
import datetime

import iplotx as ipx
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sim import gen_matrix


class World(object):
    def __init__(self, seed=None):
        print(f"[{datetime.datetime.now().astimezone()}] Initializing world...")
        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=seed)
        self.parking_lot_locs: np.ndarray = np.random.choice(np.arange(0, 16, dtype=np.int32), size=3)
        print(f"[{datetime.datetime.now().astimezone()}] Parking lots: {self.parking_lot_locs}")

    def draw_street_graph(self):
        G = nx.from_scipy_sparse_array(self._street_graph, create_using=nx.DiGraph, edge_attribute="weight")
        layout = nx.spring_layout(G)
        ipx.network(G, layout=layout, show=True)

    def update_street_graph(self):
        print(f"[{datetime.datetime.now().astimezone()}] Updating street graph...")
        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=None)  # update the code so a new graph is generated

    def get_map(self) -> csr_matrix:
        return copy.copy(self._street_graph)

    def is_parking_lot(self, i) -> bool:
        return i in self.parking_lot_locs
