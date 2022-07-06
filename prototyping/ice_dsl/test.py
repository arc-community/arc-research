from email.mime import base
from typing import List
from arc.interface import Riddle, Board
from arc.utils import dataset
from enum import Enum
import random
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
    compose_list
)

import typer


class ParameterType:
    Image = 1
    ImageList = 2


class Function:
    def __init__(self, name: str, fn, returnType: ParameterType, parameterTypes: List[ParameterType]):
        self.name = name
        self.fn = fn
        self.returnType = returnType
        self.parameterTypes = parameterTypes

    def evaluate(self, args):
        return self.fn(*args)


class Node:
    def __init__(self, return_type: ParameterType, name: str, id: int):
        self.return_type = return_type
        self.name = name
        self.id = id

    def evaluate(self, input):
        pass

    def fmt(self):
        pass


class InputNode(Node):
    def __init__(self, id=0):
        super().__init__(ParameterType.Image, "input", id)

    def evaluate(self, input):
        return input

    def fmt(self):
        return self.name


class FunctionNode(Node):
    def __init__(self, fn: Function, id: int):
        super().__init__(fn.returnType, fn.name, id)
        self.fn = fn
        self.input_nodes = [None] * len(fn.parameterTypes)

    def evaluate(self, input):
        inputs = [n.evaluate(input) for n in self.input_nodes]
        return self.fn.evaluate(inputs)

    def connect_input(self, index: int, source_node: Node):
        if index < 0 or index >= len(self.fn.parameterTypes):
            raise IndexError("Index out of range")
        if source_node.return_type != self.fn.parameterTypes[index]:
            raise ValueError("Source node type does not match input parameter type")
        self.input_nodes[index] = source_node

    def fmt(self):
        args = (f"x{a.id}" for a in self.input_nodes)
        s = self.name + "(" + ", ".join(args) + ")"
        return s


class NodeFactory:
    def __init__(self):
        self.functions = {}
        self.next_id = 1

    def register(self, name: str, fn, returnType: ParameterType, parameterTypes: List[ParameterType]):
        self.functions[name] = Function(name, fn, returnType, parameterTypes)

    def create_node(self, function_name) -> FunctionNode:
        fn = self.functions[function_name]
        id = self.next_id
        self.next_id += 1
        return FunctionNode(fn, id)

    def sample_function_name(self):
        fs = list(self.functions.keys())
        return random.choice(fs)

    def create_random_node(self) -> FunctionNode:
        fn = self.sample_function_name()
        return self.create_node(fn)


class NodeGraph:
    def __init__(self):
        self.nodes = [InputNode()]

    def add(self, node: Node):
        self.nodes.append(node)

    def __len__(self):
        return len(self.nodes)

    def filter_nodes_by_type(self, param_type: ParameterType):
        return [n for n in self.nodes if n.return_type == param_type]

    def fmt(self) -> str:
        s = ""
        for n in self.nodes:
            s += f"x{n.id} = {n.fmt()}\n"
        return s

    def evaluate(self, input):
        return [n.evaluate(input) for n in self.nodes]


def register_functions(f: NodeFactory):
    f.register("majorityCol", lambda img: majorityCol(img), ParameterType.Image, [ParameterType.Image])
    for i in range(8):
        f.register(f"rigid{i}", lambda img: rigid(img, i), ParameterType.Image, [ParameterType.Image])
    f.register("invert", invert, ParameterType.Image, [ParameterType.Image])
    f.register(
        "mirror", lambda a, b: mirror(a, b, pad=0), ParameterType.Image, [ParameterType.Image, ParameterType.Image]
    )
    f.register(
        "mirror_pad",
        lambda a, b: mirror(a, b, pad=1),
        ParameterType.Image,
        [ParameterType.Image, ParameterType.Image],
    )
    f.register(
        "replaceCols",
        lambda a, b: replaceCols(a, b),
        ParameterType.Image,
        [ParameterType.Image, ParameterType.Image],
    )
    f.register("wrap", wrap, ParameterType.Image, [ParameterType.Image, ParameterType.Image])
    for i in range(4):
        f.register(
            f"myStack{i}",
            lambda a, b: myStack(a, b, i),
            ParameterType.Image,
            [ParameterType.Image, ParameterType.Image],
        )
    f.register("compress", lambda x: compress(x), ParameterType.Image, [ParameterType.Image])
    f.register("border", border, ParameterType.Image, [ParameterType.Image])
    f.register("splitAll", splitAll, ParameterType.ImageList, [ParameterType.Image])
    for i in range(5):
        f.register(
            f"compose{i}",
            lambda a,b: compose(a,b, i), 
            ParameterType.Image, [ParameterType.Image, ParameterType.Image]
        )
        f.register(
            f"compose_list{i}",
            lambda xs: compose_list(xs, i), 
            ParameterType.Image, [ParameterType.ImageList]
        )
    f.register("fill", fill, ParameterType.Image, [ParameterType.Image])
    return f


def try_connect_inputs(n: FunctionNode, g: NodeGraph) -> bool:
    input_types = set(n.fn.parameterTypes)
    nodes_by_type = {t: g.filter_nodes_by_type(t) for t in input_types}
    for i, t in enumerate(n.fn.parameterTypes):
        if len(nodes_by_type[t]) == 0:
            return False
        source_node = random.choice(nodes_by_type[t])
        n.connect_input(i, source_node)

    return True


def generate_random_graph(f: NodeFactory, node_count: int = 5) -> NodeGraph:
    g = NodeGraph()
    while len(g) < node_count:
        n = f.create_random_node()
        if try_connect_inputs(n, g):
            g.add(n)
    return g


def board_to_image(board: Board) -> Image:
    return Image(Point(), Point(board.num_rows, board.num_cols), board.flat)


def image_to_board(img: Image) -> Board:
    data = [[img[(i, j)] for j in range(img.w)] for i in range(img.h)]
    return Board.parse_obj(data)


def print_image(img):
    typer.echo(image_to_board(img).fmt(True))


def main2():
    f = NodeFactory()
    register_functions(f)
    g = generate_random_graph(f, 10)
    print(g.fmt())

    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    for i in [22]:
        id = eval_riddle_ids[i]
        riddle = dataset.load_riddle_from_id(id)

        b = riddle.train[0].input
        img = board_to_image(b)

        print_image(img)

        # outputs = g.evaluate(img)
        # for o in outputs:
        #     print_image(o)


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
    main2()
