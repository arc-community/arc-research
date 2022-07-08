from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np
from arc.utils.dataset import load_riddle_from_file
from more_itertools import flatten


Graph = Dict[int, List[int]]


def binarize(img: np.ndarray) -> np.ndarray:
    return (img != 0).astype(int)


def make_matrix_square(img: np.ndarray) -> np.ndarray:
    delta = img.shape[0] - img.shape[1]
    pad_width = ((0, abs(delta)), (0, 0)) if delta < 0 else ((0, 0), (0, delta))
    return np.pad(img, pad_width, constant_values=0)


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


def test(riddle_path: str):
    riddle = load_riddle_from_file(riddle_path)
    img = riddle.train[0].input.np
    g = construct_graph(img, True)
    idx1 = tuple(zip(*flatten(map(lambda i: [(i, j) for j in g[1][i]], g[1].keys()))))
    idx8 = tuple(zip(*flatten(map(lambda i: [(i, j) for j in g[8][i]], g[8].keys()))))
    assert all(img[idx1] == 1)
    assert all(img[idx8] == 8)


if __name__ == '__main__':
    test('/Users/xmachine/.arc/cache/dataset/evaluation/009d5c81.json')
