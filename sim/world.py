import copy
import iplotx as ipx
import networkx as nx
from scipy.sparse import csr_matrix
from sim import gen_matrix


class World(object):
    def __init__(self):
        self._street_graph = gen_matrix.weighted_directed_grid4x4(seed=42)

    def draw_street_graph(self):
        G = nx.from_scipy_sparse_array(self._street_graph, create_using=nx.DiGraph, edge_attribute="weight")
        layout = nx.spring_layout(G)
        ipx.network(G, layout=layout, show=True)

    def get_map(self) -> csr_matrix:
        return copy.copy(self._street_graph)
