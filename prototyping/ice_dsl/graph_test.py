from typing import List
from arc.interface import Riddle, Board
from arc.utils import dataset
from enum import Enum
import random
from functools import partial
from image import (
    Point,
    Image,
    center,
    color_shape_const,
    compress2,
    compress3,
    count,
    empty,
    erase_color,
    filter_color,
    get_pos,
    get_size,
    get_size0,
    half,
    hull,
    hull0,
    majority_color,
    majority_color_image,
    make_border,
    makeBorder2,
    move,
    smear,
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
    to_origin,
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
    Pos
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

    def evaluate(self, input: Image):
        pass

    def fmt(self) -> str:
        pass

    @property
    def depth(self) -> int:
        pass


class InputNode(Node):
    def __init__(self, id=0):
        super().__init__(ParameterType.Image, "input", id)

    def evaluate(self, input: Image):
        return input

    def fmt(self):
        return self.name

    @property
    def depth(self) -> int:
        return 0


class FunctionNode(Node):
    def __init__(self, fn: Function, id: int):
        super().__init__(fn.returnType, fn.name, id)
        self.fn = fn
        self.input_nodes = [None] * len(fn.parameterTypes)

    def evaluate(self, input: Image):
        inputs = [n.evaluate(input) for n in self.input_nodes]
        return self.fn.evaluate(inputs)

    def connect_input(self, index: int, source_node: Node):
        if index < 0 or index >= len(self.fn.parameterTypes):
            raise IndexError("Index out of range")
        if source_node.return_type != self.fn.parameterTypes[index]:
            raise ValueError("Source node type does not match input parameter type")
        self.input_nodes[index] = source_node

    @property
    def depth(self) -> int:
        return max(n.depth for n in self.input_nodes) + 1

    def fmt(self):
        args = (f"x{a.id}" for a in self.input_nodes)
        s = self.name + "(" + ", ".join(args) + ")"
        return s


class NodeFactory:
    def __init__(self):
        self.functions = {}
        self.next_id = 1

    def register(self, name: str, fn, returnType: ParameterType, parameterTypes: List[ParameterType]):
        assert returnType in (ParameterType.Image, ParameterType.ImageList)
        assert parameterTypes != None and len(parameterTypes) > 0
        self.functions[name] = Function(name, fn, returnType, parameterTypes)

    def register_unary(self, name: str, fn):
        self.register(name, fn, returnType=ParameterType.Image, parameterTypes=[ParameterType.Image])

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

    for i in range(10):
        f.register_unary(f"filter_color{i}", partial(filter_color, id=i))
    for i in range(10):
        f.register_unary(f"erase_color{i}", partial(erase_color, col=i))
    for i in range(10):
        f.register_unary(f"color_shape_const{i}", partial(color_shape_const, id=i))

    f.register_unary("compress", compress)
    f.register_unary("get_pos", get_pos)
    f.register_unary("get_size", get_size)
    f.register_unary("get_size0", get_size0)
    f.register_unary("hull", hull)
    f.register_unary("hull0", hull0)
    f.register_unary("to_origin", to_origin)
    f.register_unary("fill", fill)
    f.register_unary("interior", interior)
    f.register_unary("interior2", interior2)
    f.register_unary("border", border)
    f.register_unary("center", center)
    f.register_unary("majority_color_image", majority_color_image)

    for i in range(1,9):
        f.register_unary(f"rigid{i}", partial(rigid, id=i))

    for a in range(3):
        for b in range(3):
            f.register_unary("count_{a}_{b}", partial(count, id=a, out_type=b))

    for i in range(15):
        f.register_unary(f"smear_{i}", partial(smear, id=i))

    f.register_unary("make_border", partial(make_border, bcol=1))

    for b in (False, True):
        f.register_unary("make_border2", partial(makeBorder2, usemaj=b))


    f.register_unary("compress2", compress2)
    f.register_unary("compress3", compress3)

    for i in range(4):
        f.register_unary(f"half{i}", partial(half, id=i))

    for dy in range(-2,3,1):
        for dx in range(-2,3,1):
            f.register_unary(f"move_{dx}_{dy}", partial(move, p=Pos(dx,dy)))


    f.register("invert", invert, ParameterType.Image, [ParameterType.Image])
    f.register(
        "mirror", mirror, ParameterType.Image, [ParameterType.Image, ParameterType.Image]
    )
    f.register(
        "mirror_pad",
        lambda a, b: mirror(a, b, pad=1),
        ParameterType.Image,
        [ParameterType.Image, ParameterType.Image],
    )
    f.register(
        "replaceCols",
        replace_colors,
        ParameterType.Image,
        [ParameterType.Image, ParameterType.Image],
    )
    f.register("wrap", wrap, ParameterType.Image, [ParameterType.Image, ParameterType.Image])

    for i in range(4):
        f.register(
            f"myStack{i}",
            partial(my_stack, orient=i),
            ParameterType.Image,
            [ParameterType.Image, ParameterType.Image],
        )
    f.register("border", border, ParameterType.Image, [ParameterType.Image])
    f.register("splitAll", split_all, ParameterType.ImageList, [ParameterType.Image])

    for i in range(6):
        f.register(
            f"compose{i}",
            partial(compose, id=i),
            ParameterType.Image,
            [ParameterType.Image, ParameterType.Image],
        )
        f.register(f"compose_list{i}", partial(compose_list, id=i), ParameterType.Image, [ParameterType.ImageList])
    f.register("fill", fill, ParameterType.Image, [ParameterType.Image])
    f.register("filterCol", lambda a,b: filterCol(a, b), ParameterType.Image, [ParameterType.Image, ParameterType.Image])
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
    print(f'p: {img.p.x},{img.p.y}; sz: {img.sz.x}x{img.sz.y};')


def main():
    random.seed(42)

    f = NodeFactory()
    register_functions(f)
    g = generate_random_graph(f, 5)
    print(g.fmt())

    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    for i in [100]:
        id = eval_riddle_ids[i]
        riddle = dataset.load_riddle_from_id(id)

        b = riddle.train[0].input
        img = board_to_image(b)

        print_image(img)

        outputs = g.evaluate(img)
        for i, o in enumerate(outputs):
            print(f"x{i}:")
            print('depth: ', g.nodes[i].depth)
            print_image(o)


        # configure DSL operations to include in graph
        # cache node outputs
        # output must be at least 1x1 and not larger than max-side-length
        # each operation in graph should make a difference


if __name__ == "__main__":
    main()
