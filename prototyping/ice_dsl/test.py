from typing import List
from arc.interface import Riddle, Board
from arc.utils import dataset
from image import (
    Point,
    Image,
    compose_growing,
    compress2,
    compress3,
    connect,
    cut,
    cut_image,
    empty,
    gravity,
    inside_marked,
    majority_color,
    makeBorder2,
    my_stack_list,
    pick_maxes,
    pick_not_maxes,
    pick_unique,
    stack_line,
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


def print_image(img: Image):
    typer.echo(img.fmt(True))


def main():
    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    for i in range(1):
        id = eval_riddle_ids[i]
        riddle = dataset.load_riddle_from_id(id)

        b = riddle.train[0].output
        img = Image.from_board(b)
        img = sub_image(img, Point(2, 2), Point(4, 4))
        print("maj col:", majority_color(img))

        print("img", img)

        b2 = img.to_board()

        print("broadcast")
        y = broadcast(img, full((0, 0), (11, 11)))
        print_image(y)
        print_image(img)

        a = split_colors(img)
        for x in a:
            print_image(x)
            print()
            print_image(invert(x))
            print()
            print_image(filter_color(x, 2))

        print(a)
        print_image(b2)

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

    print(x.np)

    print("connect")

    p6 = Image(
        (0, 0),
        (4, 4),
        [
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
        ],
    )
    for i in range(3):
        print(i)
        x = connect(p6, i)
        print_image(x)

    print("cut_image")
    print("in:")
    print_image(p3)
    xs = cut_image(p3)
    print("out:")
    for i, x in enumerate(xs):
        print(f"{i}:")
        print_image(x)

    print("inside_marked")
    print("in:")
    print_image(p6)
    xs = inside_marked(p6)
    print("out:")
    for i, x in enumerate(xs):
        print(f"{i}: p: {x.p}")
        print_image(x)

    print("gravity")
    print("in:")
    print_image(p6)
    xs = gravity(p6, 2)
    print("out:")
    for i, x in enumerate(xs):
        print(f"{i}: p: {x.p}")
        print_image(x)
    print("composed:")
    print_image(compose_growing(xs))

    print("pick_unique")
    p7 = Image(
        (0, 0),
        (5, 5),
        [
            1,
            2,
            0,
            3,
            4,
            2,
            3,
            0,
            2,
            1,
            0,
            0,
            0,
            0,
            0,
            8,
            8,
            0,
            3,
            4,
            8,
            8,
            0,
            2,
            1,
        ],
    )

    xs = cut_image(p7)
    for x in xs:
        print_image(x)
    x = pick_unique(xs)
    print_image(p7)
    print_image(x)

    print("stack_line")
    x = stack_line([p1, p3, p4, p5])
    print_image(x)

    print("my_stack_list")
    x = my_stack_list([p1, p3, p4, p5], 1)
    print_image(x)

    print("pick_maxes")
    xs = pick_maxes([p1, p3, p4, p5], 4)
    print_image(xs[0])

    print("pick_not_maxes")
    xs = pick_not_maxes([p1, p3, p4, p5], 4)
    print(len(xs))
    

if __name__ == "__main__":
    main()
