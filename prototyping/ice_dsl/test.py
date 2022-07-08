from typing import List
from arc.interface import Riddle, Board
from arc.utils import dataset
from image import (
    Point,
    Image,
    compress2,
    compress3,
    empty,
    majority_color,
    makeBorder2,
    sub_image,
    split_colors,
    invert,
    filter_color,
    broadcast,
    full,
    compress,
    fill,
    border,
    interior,
    interior2,
    rigid,
    get_regular,
    my_stack,
    wrap,
    extend,
    outer_product_is,
    outer_product_si,
    replace_colors,
    repeat,
    mirror,
    split_all,
    compose,
    compose_list,
    smear,
    make_border,
)

import typer


def board_to_image(board: Board) -> Image:
    return Image(Point(), Point(board.num_rows, board.num_cols), board.flat)


def image_to_board(img: Image) -> Board:
    data = [[img[(i, j)] for j in range(img.w)] for i in range(img.h)]
    return Board.parse_obj(data)


def print_image(img):
    typer.echo(image_to_board(img).fmt(True))


def main():
    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    for i in range(1):
        id = eval_riddle_ids[i]
        riddle = dataset.load_riddle_from_id(id)

        b = riddle.train[0].output
        img = board_to_image(b)
        img = sub_image(img, Point(2, 2), Point(4, 4))
        print("maj col:", majority_color(img))

        print("img", img)

        b2 = image_to_board(img)

        print("broadcast")
        y = broadcast(img, full((0, 0), (11, 11)))
        typer.echo(image_to_board(y).fmt(colored=True))
        typer.echo(image_to_board(img).fmt(colored=True))

        a = split_colors(img)
        for x in a:
            typer.echo(image_to_board(x).fmt(True))
            print()
            typer.echo(image_to_board(invert(x)).fmt(True))
            print()
            typer.echo(image_to_board(filter_color(x, 2)).fmt(True))

        print(a)
        typer.echo(b2.fmt(colored=True))

        # typer.echo(riddle.fmt(colored=True, with_test_outputs=False))

    print("compress")
    x = full((1, 1), (10, 10))
    x[5, 5] = 1
    x[7, 6] = 1
    y = compress(x)
    print_image(y)
    print(y.sz)

    print("fill:")
    print_image(fill(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])))
    print("border")
    print_image(border(Image((0, 0), (4, 4), [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1])))

    print("interior")
    print_image(interior(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1])))

    print("interior2")
    print_image(interior2(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1])))

    p0 = Image((0, 0), (5, 4), [1, 1, 1, 1, 4, 1, 0, 2, 0, 1, 1, 2, 0, 2, 1, 5, 1, 1, 1, 6])

    for i in range(9):
        x = rigid(p0, i)
        print(f"rigid{i}:", i)
        print_image(x)

    p1 = Image(
        (0, 0),
        (5, 5),
        [
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            7,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
        ],
    )
    print("getRegular")
    x = get_regular(p1)
    print_image(x)

    print("myStack")
    x = my_stack(p0, p1, 2)
    print_image(x)

    print("wrap")
    p3 = Image(
        (0, 0),
        (10, 1),
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
    )
    x = wrap(p3, full((0, 0), (3, 4)))
    print_image(x)

    print("extend")
    x = extend(p1, full((0, 0), (10, 10)))
    print_image(x)

    print("outerProductIS")
    x = outer_product_is(p1, p3)
    print_image(x)

    print("replaceCols")
    x = replace_colors(p0, Image((0, 0), (5, 1), [0, 0, 0, 2, 1]))
    print_image(x)

    print("repeat")
    x = repeat(p0, empty((0, 0), (15, 15)), 1)
    print_image(x)

    print("mirror")
    x = mirror(p0, empty((0, 0), (15, 15)), 1)
    print_image(x)

    print("smear")
    p4 = Image(
        (0, 0),
        (4, 4),
        [
            0,
            0,
            0,
            0,
            0,
            2,
            3,
            0,
            0,
            4,
            5,
            0,
            0,
            0,
            0,
            0,
        ],
    )
    x = smear(p4, 7)
    print_image(x)

    print("make_border")
    x = make_border(p4, 1)
    print_image(x)

    print("makeBorder2")
    x = makeBorder2(p4)
    print_image(x)

    print("split_colors")
    xs = split_colors(p4)
    print("original:")
    print_image(p4)
    print("split:")
    for x in xs:
        print_image(x)

    print("split_all")
    xs = split_all(p0)
    print("original:")
    print_image(p0)
    print("split:")
    for x in xs:
        print_image(x)
        print(x.sz)

    print("compress2")
    print_image(p4)
    x = compress(p4)
    print_image(x)

    p5 = Image(
        (0, 0),
        (4, 4),
        [
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            3,
            3,
            3,
            0,
            3,
            3,
            3,
        ],
    )

    print("compress3")
    x = compress3(p5)
    print_image(x)


if __name__ == "__main__":
    main()
