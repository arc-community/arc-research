import random
from typing import List, Tuple
from riddle_script import Image, Point, PartSampler, color_shape_const, random_colors, empty, compose, compose_list, sample_non_overlapping_positions, print_riddle, riddle_to_json


def generate_009d5c81(min_w=10, min_h=10, max_w=18, max_h=18) -> Tuple[List[Tuple[Image]]]:    
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

    assert min_w >= 10 and min_h >= 10 and min_w <= max_w and min_h <= max_h

    # board sizes (not too small), original is 14x14
    w = random.randint(min_w, max_w)
    h = random.randint(min_h, max_h)
    
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


from pathlib import Path
from tqdm import tqdm

def generate_10x10_dataset():
    overwrite = False
    output_dir = Path('./e_009d5c81_10x10_250k/')
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    total_riddles = 250000
    with tqdm(total=total_riddles) as pbar:
        while i < total_riddles:
            train, test = generate_009d5c81(10, 10, 10, 10)
            h = hash(tuple(train + test)) & 0xFFFFFFFF
            fn = f"{h:08x}.json"
            fn = output_dir / fn
            if not overwrite and fn.exists():
                print(f"File {fn} already exists.")
                continue
            j = riddle_to_json(train, test)
            fn.write_text(j)
            i += 1
            pbar.update(1)


if __name__ == "__main__":
    # generate_10x10_dataset()
    train, test = generate_009d5c81()
    print_riddle(train, test)
