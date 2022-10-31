import math
import random
from typing import List, Tuple
from riddle_script import Image, Point, random_colors, empty, compose_list, sample_non_overlapping_positions, print_riddle, square, outer_border


def generate_00dbd492() -> Tuple[List[Tuple[Image]]]:
    """
    https://github.com/arc-community/arc/wiki/Riddle_Evaluation_00dbd492
    """

    num_examples = 4

    num_squares = random.randint(2, 4)
    inside_length = [2*i + 1 for i in range(1, 5)]
    random.shuffle(inside_length)
    inside_length = inside_length[:num_squares]

    # colors: square border and center pixel color and fill colors
    color_table = random_colors(1 + num_squares)

    # generate square objects
    square_parts = []
    for i,l in enumerate(inside_length):
        in_square = outer_border(square(l, color=0), color_table[0])
        out_square = outer_border(square(l, color=color_table[1+i]), color_table[0])
        in_square[in_square.h//2, in_square.w//2] = color_table[0]
        out_square[in_square.h//2, in_square.w//2] = color_table[0]
        square_parts.append((in_square, out_square))

    example_sets = [[i % num_squares] for i in range(num_examples)]
    for s in example_sets:
        add_squares = random.randint(0, 3)
        s.extend([random.randint(0, num_squares-1) for j in range(add_squares)])

    example_sets.append(list(range(num_squares)))  # generate test example with all sizes 

    # generate board pairs
    board_pairs = []
    for s in example_sets:
        src_squares = [square_parts[i][0].copy() for i in s]
        dst_squares = [square_parts[i][1].copy() for i in s]

        # determine board size
        w = math.ceil(sum(s.w for s in src_squares)) + len(src_squares)
        if max(s.w for s in src_squares) == 11:
             w = min(w, 26)
        else:
             w = min(w, 22)

        board_size = Point(w, w)

        sample_non_overlapping_positions(board_size, src_squares, margin=1)
        
        # copy positions for squares on output board
        for a,b in zip(src_squares, dst_squares):
            b.p = a.p

        input_board = compose_list([empty(board_size)] + src_squares)
        output_board = compose_list([empty(board_size)] + dst_squares)

        board_pairs.append((input_board, output_board))

    train = board_pairs[:-1]
    test = board_pairs[-1:]

    return train, test


if __name__ == "__main__":
    train, test = generate_00dbd492()
    print_riddle(train, test)
