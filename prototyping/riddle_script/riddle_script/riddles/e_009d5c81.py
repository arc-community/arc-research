from collections import OrderedDict
import json
import random
from typing import List, Tuple
from riddle_script import Image, Point, PartSampler, color_shape_const, random_colors, empty, compose, compose_list


def check_overlap(objs: List[Image], margin: int = 0):
    m = margin
    remain = list(objs)
    while len(remain) > 1:
        a = remain.pop()
        for b in remain:
            if a.x < b.x + b.w + m and a.x + a.w + m > b.x and a.y < b.y + b.h + m and a.y + a.h + m > b.y:
                return True
    return False


def sample_non_overlapping_positions(sz: Point, objs: List[Image], margin: int = 0):
    while True:
        # randomize object positions
        for o in objs:
            o.x = random.randint(0, sz.x - o.w)
            o.y = random.randint(0, sz.y - o.h)
        if not check_overlap(objs, margin=margin):
            break


def riddle_to_json(train_pairs: List[Tuple[Image, Image]], test_pairs: List[Tuple[Image, Image]]) -> str:
    assert len(train_pairs) > 0 and len(test_pairs) > 0

    def pairs_to_json(p):
        return [OrderedDict(input=x[0].np.tolist(), output=x[1].np.tolist()) for x in p]

    riddle = OrderedDict(train=pairs_to_json(train_pairs), test=pairs_to_json(test_pairs))
    return json.dumps(riddle)


def print_riddle(train_pairs: List[Tuple[Image, Image]], test_pairs: List[Tuple[Image, Image]]):
    def print_pairs(ps):
        for i,p in enumerate(ps):
            print(f'[{i}] Input:')
            print(p[0].fmt(True))
            print(f'[{i}] Output:')
            print(p[1].fmt(True))
        
    print('Training pairs:')
    print_pairs(train_pairs)
    print('Test pairs:')
    print_pairs(test_pairs)


def generate_009d5c81():
    """
    https://github.com/arc-community/arc/wiki/Riddle_Evaluation_009d5c81

    Concept: Symbol specifies color transition of other object.

    Input and output board sizes are equal. Background is black.
    The input boards contain two distinct objects with two distinct colors placed at random positions.
    The same two colors are used on all input boards.
    One of the objects acts as a symbol that determines by its appearance which color the other object will have on the output board.
    The symbol object is not copied to the output board.
    The object whose color changes stays at the same positon and only all its pixel colors change.
    In the original riddle the symbol objects are of size 3x3 pixels.
    """

    # We generate two examples for each symbol color transition. One of the examples can then be used as test-example.

    # board sizes (not too small), original is 14x14
    w = random.randint(10, 18)
    h = random.randint(10, 18)

    # number of color-change example-pairs (3 is a reasonable number)
    num_color_pairs = 3

    # pick colors: two colors in input + target color for each symbol
    color_table = random_colors(2 + num_color_pairs)

    # color symbols
    ps = PartSampler()
    color_symbols = ps.distinct_symbols_sized(num_color_pairs, (3, 3))

    # objects to be colored
    objects_to_color = ps.distinct_symbols_size_range(
        num_color_pairs * 2, w / 2 - 1, h / 2 - 1, w - 4, h - 4, min_area=12
    )

    board_pairs = []

    # non-overlapping positions
    for i, obj in enumerate(objects_to_color):
        target_color_index = i // 2
        sym = color_symbols[target_color_index]

        in_sym = color_shape_const(sym, color_table[0])
        in_obj = color_shape_const(obj, color_table[1])
        out_obj = color_shape_const(obj, color_table[2 + target_color_index])

        sample_non_overlapping_positions(Point(w, h), [in_sym, in_obj], margin=1)
        out_obj.p = in_obj.p

        input_board = compose_list([empty(Point(w, h)), in_sym, in_obj])
        output_board = compose(empty(Point(w, h)), out_obj)

        # generate board-pair
        board_pairs.append((input_board, output_board))

    random.shuffle(board_pairs)
    train = board_pairs[:-1]
    test = board_pairs[-1:]

    return train, test


if __name__ == "__main__":
    train, test = generate_009d5c81()
    print_riddle(train, test)
    s = riddle_to_json(train, test)
    print(s)
