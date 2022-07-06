from typing import List
from arc.interface import Riddle, Board
from arc.utils import dataset
from image import (
    Point,
    Image,
    empty,
    majorityCol,
    subImage,
    splitCols,
    invert,
    filterCol,
    broadcast,
    full,
    compress,
    fill,
    border,
    interior,
    interior2,
    rigid,
    getRegular,
    myStack,
    wrap,
    extend,
    outerProductIS,
    outerProductSI,
    replaceCols,
    repeat,
    mirror,
    splitAll,
    compose,
    compose_list,
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
        img = subImage(img, Point(2, 2), Point(4, 4))
        print("maj col:", majorityCol(img))

        print("img", img)

        b2 = image_to_board(img)

        print("broadcast")
        y = broadcast(img, full((0, 0), (11, 11)))
        typer.echo(image_to_board(y).fmt(colored=True))
        typer.echo(image_to_board(img).fmt(colored=True))

        a = splitCols(img)
        for x in a:
            typer.echo(image_to_board(x).fmt(True))
            print()
            typer.echo(image_to_board(invert(x)).fmt(True))
            print()
            typer.echo(image_to_board(filterCol(x, 2)).fmt(True))

        print(a)
        typer.echo(b2.fmt(colored=True))

        # typer.echo(riddle.fmt(colored=True, with_test_outputs=False))

    x = full((1, 1), (10, 10))
    x[5, 5] = 1
    x[7, 6] = 1
    y = compress(x)
    print(y.sz)

    print("fill:")
    typer.echo(image_to_board(fill(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]))).fmt(True))
    print("border")
    typer.echo(
        image_to_board(border(Image((0, 0), (4, 4), [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1]))).fmt(True)
    )
    print("interior")
    typer.echo(
        image_to_board(interior(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1]))).fmt(True)
    )
    print("interior2")
    typer.echo(
        image_to_board(interior2(Image((0, 0), (4, 4), [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1]))).fmt(True)
    )

    def print_image(img):
        typer.echo(image_to_board(img).fmt(True))

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
    x = getRegular(p1)
    print_image(x)

    print("myStack")
    x = myStack(p0, p1, 2)
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
    x = outerProductIS(p1, p3)
    print_image(x)

    print("replaceCols")
    x = replaceCols(p0, Image((0, 0), (5, 1), [0, 0, 0, 2, 1]))
    print_image(x)

    print("repeat")
    x = repeat(p0, empty((0, 0), (15, 15)), 1)
    print_image(x)

    print("mirror")
    x = mirror(p0, empty((0, 0), (15, 15)), 1)
    print_image(x)


if __name__ == "__main__":
    main()
