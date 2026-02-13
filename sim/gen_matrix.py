from typing import Any

import numpy as np
import networkx as nx
import iplotx as ipx

from numpy import dtype, ndarray
from scipy import sparse


HEX_LABELS = [*list("0123456789ABCDEF")]

def _grid4x4_nodes_labels():
    """Returns nodes as strings: ['0','1',...,'9','A',...,'F']"""
    return HEX_LABELS.copy()

def _grid4x4_undirected_edges():
    """
    Returns undirected edges for the 4x4 grid as pairs of node indices (u,v), u < v.
    Layout (row-major):
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15
    """
    edges = []
    for r in range(4):
        for c in range(4):
            u = r * 4 + c
            # right neighbor
            if c < 3:
                v = u + 1
                edges.append((u, v))
            # down neighbor
            if r < 3:
                v = u + 4
                edges.append((u, v))
    return edges


def weighted_directed_grid4x4(low=0.0, high=1.0, seed=None) -> sparse.csr_matrix:
    """
    Produces a directed weighted adjacency for the grid:
      - For every undirected neighbor pair {u,v}, creates TWO directed edges u->v and v->u
      - Weights are in (low, high) open interval (default (0,1))
    :param low: lower bound
    :param high: upper bound
    :param seed: random seed
    """
    rng = np.random.default_rng(seed)

    if not (low < high):
        raise ValueError("Require low < high.")
    # draw strictly inside ]low, high[
    def draw_weight():
        # rng.random() is in [0,1[; this maps to [low, high[ (practically open on floats)
        w = low + (high - low) * float(rng.random())
        # avoid exact low (vanishingly unlikely), bump if desired
        if w <= low:
            w = np.nextafter(low, high)
        return w

    undirected = _grid4x4_undirected_edges()

    A: ndarray[tuple[int, int], dtype[np.float64]] = np.zeros((16, 16), dtype=np.float64)

    for u, v in undirected:
        w_uv = draw_weight()
        w_vu = draw_weight()

        A[u, v] = w_uv
        A[v, u] = w_vu

    np.fill_diagonal(A, 0.0)

    return sparse.csr_matrix(A)


# --- Example ---
if __name__ == "__main__":
    A = weighted_directed_grid4x4(seed=42)
    G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph, edge_attribute="weight")
    layout = nx.layout.spring_layout(G)
    ipx.network(G, layout=layout, show=True)
    print(A)
    print(G)
    # print("Nodes:", A.get)
    # print("First 10 directed edges:", edges[:10])
    # print("Adjacency matrix shape:", A.shape)
