import numpy as np

from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

from sim.world import World


class Agent(object):
    def __init__(self):
        self.current_pos: int = 0
        self.target_lot: int = 0
        self.destination: int = 15

    def find_lot(self, world: World, max_walking_dist=2) -> int:
        bfo = csgraph.breadth_first_order(world.get_map(), 0, directed=True)[0]
        dist_mat = csgraph.floyd_warshall(world.get_map(), directed=True)
        l = [i for i in bfo if world.is_parking_lot(i) and dist_mat[0, i] < max_walking_dist]

        if not l:
            print("No suitable parking lot found.")
            return -1

        self.target_lot = l[0]

        return self.target_lot

    def step(self):
        ...

    def act(self):
        ...

    def reset(self):
        ...

    def finished(self):
        ...
