from collections import defaultdict
from itertools import tee

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.topology import WeisfeilerLehman
from sknetwork.visualization import svg_graph

from arc import Riddle
from arc.utils.dataset import get_riddles, load_riddle_from_file


def graphs_from_riddle(riddle: Riddle):
    graphs = defaultdict(dict)
    for i, board in enumerate(riddle.train):
        graphs[i]['input'] = nx.from_numpy_matrix(binarize(board.input.np))
        graphs[i]['output'] = nx.from_numpy_matrix(binarize(board.output.np))
    return graphs


def binarize(x: np.ndarray):
    return (x != 0).astype(int)


def graphs_to_svg(riddle: Riddle):
    graphs = defaultdict(dict)
    for i, board in enumerate(riddle.train):
        weisfeiler_lehman = WeisfeilerLehman()

        inp = csr_matrix(riddle.train[i].input.np)
        out = csr_matrix(riddle.train[i].output.np)

        labels_inp = weisfeiler_lehman.fit_transform(inp)
        labels_out = weisfeiler_lehman.fit_transform(out)

        graphs[i]['input'] = svg_graph(inp, labels=labels_inp)
        graphs[i]['output'] = svg_graph(out, labels=labels_out)
    return graphs


def isomorphism_test(g: nx.classes.graph.Graph,
                     w: nx.classes.graph.Graph,
                     is_subgraph: bool):
    ismags = nx.algorithms.isomorphism.ISMAGS(g, w)
    return ismags.subgraph_is_isomorphic() if is_subgraph else nx.is_isomorphic(g, w)


def square_boards(riddle: Riddle):
    check_inp = all([riddle.train[i].input.num_rows == riddle.train[i].input.num_cols for i in range(len(riddle.train))])
    check_out = all([riddle.train[i].output.num_rows == riddle.train[i].output.num_cols for i in range(len(riddle.train))])
    check_size = all([np.prod(riddle.train[i].input.shape) < 400 for i in range(len(riddle.train))])
    return check_inp and check_out and check_size


def run_stats(dataset):
    # load the riddles
    riddles = map(load_riddle_from_file, get_riddles([dataset]).values())

    # keep square matrices only
    riddles_with_sq_boards = tee(filter(square_boards, riddles), 2)

    # filter riddles
    same_shape_riddles = list(filter(lambda x: all([x.train[i].input.shape == x.train[i].output.shape for i in range(len(x.train))]), riddles_with_sq_boards[0]))
    diff_shape_riddles = list(filter(lambda x: all([x.train[i].input.shape != x.train[i].output.shape for i in range(len(x.train))]), riddles_with_sq_boards[1]))

    # build graphs
    same_shape_graphs = map(graphs_from_riddle, same_shape_riddles)
    diff_shape_graphs = map(graphs_from_riddle, diff_shape_riddles)

    print('Same board shapes')
    print('='*25)
    for i, graphs in enumerate(same_shape_graphs):
        for k in graphs:
            g = graphs[k]['input']
            w = graphs[k]['output']
            if isomorphism_test(g, w, False):
                print(f'@id {same_shape_riddles[i].riddle_id} @train[{k}]: True')

    print('\n')
    print('Different board shapes')
    print('='*20)
    for i, graphs in enumerate(diff_shape_graphs):
        for k in graphs:
            g = graphs[k]['input']
            w = graphs[k]['output']
            if len(g) > len(w):
                if isomorphism_test(g, w, True):
                    print(f'@id {diff_shape_riddles[i].riddle_id} @train[{k}] subgraph(output) is isomorphic: True')
                    continue
            if isomorphism_test(w, g, True):
                print(f'@id {diff_shape_riddles[i].riddle_id} @train[{k}] subgraph(input) is isomorphic: True')

if __name__ == '__main__':
    run_stats('training')
    run_stats('evaluation')
