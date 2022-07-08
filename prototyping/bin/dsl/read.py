from typing import List, Dict, Generator

import numpy as np
from arc import Riddle
from arc.utils.dataset import get_riddles, load_riddle_from_file
from scipy.spatial import KDTree
from sknetwork.topology import get_connected_components
from scipy.sparse import csr_matrix
from collections import defaultdict


import networkx as nx
import matplotlib.pyplot as plt

# riddles = map(load_riddle_from_file, get_riddles(['training']).values())
riddle = load_riddle_from_file('/home/ANT.AMAZON.COM/leoveac/.arc/cache/dataset/evaluation/009d5c81.json')



def binarize(img: np.ndarray) -> np.ndarray:
    return (img != 0).astype(int)


def make_matrix_square(img: np.ndarray) -> np.ndarray:
    delta = img.shape[0] - img.shape[1]
    pad_width = ((0, abs(delta)), (0, 0)) if delta < 0 else ((0, 0), (0, delta))
    return np.pad(img, pad_width, constant_values=0)


def find_shapes(img: np.ndarray) -> List[np.ndarray]:
    clusters = list(strongly_connected_components(img, enforce_square=True))


Graph = Dict[int, List[int]]
def construct_graph(img: np.ndarray, enforce_square: bool) -> Dict[int, Graph]:
    if enforce_square:
        img = make_matrix_square(img)

    # graph construction based on color
    g = nx.from_numpy_matrix(img)
    g_info = defaultdict(lambda: defaultdict(set))
    edges_data = g.edges.data()
    colors = set(map(lambda x: x[2]['weight'], edges_data))
    for color in colors:
        edges = filter(lambda x: x[2]['weight'] == color, edges_data)
        for edge in edges:
            idx = zip(*np.where(img==color))
            # we need to peform this check because g.edges.data() method doesn't sort the returned indices
            # basically, an edge from (1, 8) != (8, 1) because img[1, 8] != img[8, 1]
            if edge[:2] in idx:
                g_info[color][edge[0]].add(edge[1])
                continue
            g_info[color][edge[1]].add(edge[0])
    return g_info


def strongly_connected_components(graph: Graph):
    vertices = graph.keys()
    edges = graph.copy()
    identified = set()
    stack = []
    index = {}
    boundaries = []

    for v in vertices:
        if v not in index:
            to_do = [('VISIT', v)]
            while to_do:
                operation_type, v = to_do.pop()
                if operation_type == 'VISIT':
                    index[v] = len(stack)
                    stack.append(v)
                    boundaries.append(index[v])
                    to_do.append(('POSTVISIT', v))
                    to_do.extend(reversed([('VISITEDGE', w) for w in edges[v]]))
                elif operation_type == 'VISITEDGE':
                    if v not in index:
                        to_do.append(('VISIT', v))
                    elif v not in identified:
                        while index[v] < boundaries[-1]:
                            boundaries.pop()
                else:
                    if boundaries[-1] == index[v]:
                        boundaries.pop()
                        scc = set(stack[index[v]:])
                        del stack[index[v]:]
                        identified.update(scc)
                        yield scc


img = riddle.train[0].input.np
g = construct_graph(img, True)
for x in strongly_connected_components(g[8]): print(x)
breakpoint()
