from arc.interface import Riddle, Board
from arc.utils import dataset

from image import (
    Point,
    Image,
    majorityCol,
    subImage,
    splitCols,
    invert,
    filterCol,
    broadcast,
    full,
    compress,
)

import typer


def board_to_image(board: Board) -> Image:
    return Image(Point(), Point(board.num_rows, board.num_cols), board.flat)


def image_to_board(img: Image) -> Board:
    data = [[img[(i, j)] for j in range(img.w)] for i in range(img.h)]
    return Board.parse_obj(data)


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


if __name__ == "__main__":
    main()
