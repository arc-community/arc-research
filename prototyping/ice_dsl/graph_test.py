from enum import Enum
from typing import Dict, List, Tuple, Union
from arc.interface import Board
from arc.utils import dataset
import random
from functools import partial
from image import (
    Point,
    Image,
    center,
    color_shape_const,
    compose_growing,
    compress2,
    compress3,
    connect,
    count,
    cut_image,
    embed,
    erase_color,
    filter_color,
    filter_color_palette,
    get_pos,
    get_size,
    get_size0,
    gravity,
    half,
    hull,
    hull0,
    inside_marked,
    majority_color_image,
    make_border,
    make_border2,
    move,
    my_stack_list,
    pick_max,
    pick_maxes,
    pick_not_maxes,
    pick_unique,
    smear,
    split_columns,
    split_rows,
    spread_colors,
    stack_line,
    split_colors,
    filter_color,
    broadcast,
    compress,
    fill,
    border,
    interior,
    interior2,
    rigid,
    get_regular,
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
    Pos,
)

import typer


CacheDict = Dict[int, Union[Image, List[Image]]]


class ParameterType(Enum):
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

    def evaluate(self, input: CacheDict):
        pass

    def fmt(self) -> str:
        pass

    @property
    def depth(self) -> int:
        pass


class InputNode(Node):
    def __init__(self, id=0):
        super().__init__(ParameterType.Image, "input", id)

    def evaluate(self, cache: CacheDict):
        return cache[self.id]

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

    def evaluate(self, cache: CacheDict):
        x = cache.get(self.id)
        if x is None:
            inputs = [n.evaluate(cache) for n in self.input_nodes]
            x = self.fn.evaluate(inputs)
            cache[self.id] = x
        return x

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

    def register(self, name: str, fn, returnType: ParameterType, parameterTypes: List[ParameterType]) -> None:
        assert returnType in (ParameterType.Image, ParameterType.ImageList)
        assert parameterTypes != None and len(parameterTypes) > 0
        self.functions[name] = Function(name, fn, returnType, parameterTypes)

    def register_unary(self, name: str, fn) -> None:
        self.register(name, fn, returnType=ParameterType.Image, parameterTypes=[ParameterType.Image])

    def register_binary(self, name: str, fn) -> None:
        self.register(
            name, fn, returnType=ParameterType.Image, parameterTypes=[ParameterType.Image, ParameterType.Image]
        )

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
        self.input_node = InputNode()
        self.nodes = [self.input_node]

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

    def evaluate(self, input: Image) -> CacheDict:
        cache = {self.input_node.id: input}
        for n in self.nodes:
            n.evaluate(cache)
        return cache


def register_functions(f: NodeFactory):
    # unary
    for i in range(10):
        f.register_unary(f"filter_color_{i}", partial(filter_color, id=i))
    for i in range(10):
        f.register_unary(f"erase_color_{i}", partial(erase_color, col=i))
    for i in range(10):
        f.register_unary(f"color_shape_const_{i}", partial(color_shape_const, id=i))

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

    for i in range(1, 9):
        f.register_unary(f"rigid_{i}", partial(rigid, id=i))

    for a in range(3):
        for b in range(3):
            f.register_unary(f"count_{a}_{b}", partial(count, id=a, out_type=b))

    for i in range(15):
        f.register_unary(f"smear_{i}", partial(smear, id=i))

    f.register_unary("make_border", partial(make_border, bcol=1))

    for b in (False, True):
        f.register_unary("make_border2", partial(make_border2, usemaj=b))

    f.register_unary("compress2", compress2)
    f.register_unary("compress3", compress3)

    for id in range(3):
        f.register_unary(f"connect_{id}", partial(connect, id=id))

    for skipmaj in (False, True):
        f.register_unary(f"spread_colors_{1 if skipmaj else 0}", partial(spread_colors, skipmaj=skipmaj))

    for i in range(4):
        f.register_unary(f"half_{i}", partial(half, id=i))

    for dy in range(-2, 3, 1):
        for dx in range(-2, 3, 1):
            if dy != 0 or dx != 0:
                f.register_unary(f"move_{dx}_{dy}", partial(move, p=Pos(dx, dy)))

    # binary
    f.register_binary("embed", embed)
    f.register_binary("wrap", wrap)
    f.register_binary("broadcast", partial(broadcast, include0=True))
    f.register_binary("repeat_0", partial(repeat, pad=0))
    f.register_binary("repeat_1", partial(repeat, pad=1))
    f.register_binary("mirror_0", partial(mirror, pad=0))
    f.register_binary("mirror_1", partial(mirror, pad=1))

    # split
    f.register("cut_image", cut_image, returnType=ParameterType.ImageList, parameterTypes=[ParameterType.Image])
    f.register(
        "split_colors",
        partial(split_colors, include0=False),
        returnType=ParameterType.ImageList,
        parameterTypes=[ParameterType.Image],
    )
    f.register("split_all", split_all, returnType=ParameterType.ImageList, parameterTypes=[ParameterType.Image])
    f.register("split_columns", split_columns, returnType=ParameterType.ImageList, parameterTypes=[ParameterType.Image])
    f.register("split_rows", split_rows, returnType=ParameterType.ImageList, parameterTypes=[ParameterType.Image])
    f.register("inside_marked", inside_marked, returnType=ParameterType.ImageList, parameterTypes=[ParameterType.Image])
    for d in range(4):
        f.register(
            f"gravity_{d}",
            partial(gravity, d=d),
            returnType=ParameterType.ImageList,
            parameterTypes=[ParameterType.Image],
        )

    # join
    for id in range(14):
        f.register(
            f"pick_max_{id}",
            partial(pick_max, id=id),
            returnType=ParameterType.Image,
            parameterTypes=[ParameterType.ImageList],
        )
    f.register(f"pick_unique", pick_unique, returnType=ParameterType.Image, parameterTypes=[ParameterType.ImageList])
    f.register(
        "compose_growing", compose_growing, returnType=ParameterType.Image, parameterTypes=[ParameterType.ImageList]
    )
    f.register("stack_line", stack_line, returnType=ParameterType.Image, parameterTypes=[ParameterType.ImageList])
    for id in range(4):
        f.register(
            f"my_stack_list_{id}",
            partial(my_stack_list, orient=id),
            returnType=ParameterType.Image,
            parameterTypes=[ParameterType.ImageList],
        )

    # vector
    for id in range(14):
        f.register(
            f"pick_maxes_{id}",
            partial(pick_maxes, id=id),
            returnType=ParameterType.ImageList,
            parameterTypes=[ParameterType.ImageList],
        )
        f.register(
            f"pick_not_maxes_{id}",
            partial(pick_not_maxes, id=id),
            returnType=ParameterType.ImageList,
            parameterTypes=[ParameterType.ImageList],
        )
    
    # f.register(
    #     "replace_colors",
    #     replace_colors,
    #     ParameterType.Image,
    #     [ParameterType.Image, ParameterType.Image],
    # )

    # for i in range(6):
    #     f.register(
    #         f"compose{i}",
    #         partial(compose, id=i),
    #         ParameterType.Image,
    #         [ParameterType.Image, ParameterType.Image],
    #     )
    #     f.register(f"compose_list{i}", partial(compose_list, id=i), ParameterType.Image, [ParameterType.ImageList])
    # f.register(
    #     "filter_color_palette", filter_color_palette, ParameterType.Image, [ParameterType.Image, ParameterType.Image]
    # )
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


def print_image(img):
    typer.echo(img.fmt(True))
    print(f"p: {img.p.x},{img.p.y}; sz: {img.sz.x}x{img.sz.y};")


def main():
    random.seed(42)

    f = NodeFactory()
    register_functions(f)
    print("Number of functions:", len(f.functions))

    g = generate_random_graph(f, 5)
    print(g.fmt())

    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    for i in [100]:
        id = eval_riddle_ids[i]
        riddle = dataset.load_riddle_from_id(id)

        b = riddle.train[0].input
        img = Image.from_board(b)

        print_image(img)

        outputs = g.evaluate(img)
        print(outputs)
        for k, v in outputs.items():
            print(k, v)
            if type(v) == Image:
                print_image(v)


if __name__ == "__main__":
    main()
