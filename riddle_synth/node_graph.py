from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Sequence, Set, Tuple, Union
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
    def __init__(self, name: str, fn, return_type: ParameterType, parameter_types: List[ParameterType]):
        self.name = name
        self.fn = fn
        self.return_type = return_type
        self.parameter_types = parameter_types

    def evaluate(self, args):
        return self.fn(*args)


class Node(ABC):
    def __init__(self, return_type: ParameterType, name: str, id: int):
        self.return_type = return_type
        self.name = name
        self.id = id

    @abstractmethod
    def evaluate(self, input: CacheDict):
        pass

    @abstractmethod
    def fmt(self) -> str:
        pass

    @property
    @abstractmethod
    def inputs(self) -> List[Node]:
        pass

    def ancestors(self, include_self: bool = False) -> Set[Node]:
        a = set()
        if include_self:
            a.add(self)

        def add_inputs(n):
            for i in n.inputs:
                if i is not None and i not in a:
                    a.add(i)
                    add_inputs(i)

        add_inputs(self)
        return a

    @property
    @abstractmethod
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
    def inputs(self) -> List[Node]:
        return []

    @property
    def depth(self) -> int:
        return 0


class FunctionNode(Node):
    def __init__(self, fn: Function, id: int):
        super().__init__(fn.return_type, fn.name, id)
        self.fn = fn
        self.input_nodes = [None] * len(fn.parameter_types)

    def evaluate(self, cache: CacheDict):
        x = cache.get(self.id)
        if x is None:
            inputs = [n.evaluate(cache) for n in self.input_nodes]
            x = self.fn.evaluate(inputs)
            cache[self.id] = x
        return x

    def connect_input(self, index: int, source_node: Node):
        if index < 0 or index >= len(self.fn.parameter_types):
            raise IndexError("Index out of range")
        if source_node.return_type != self.fn.parameter_types[index]:
            raise ValueError("Source node type does not match input parameter type")
        self.input_nodes[index] = source_node

    @property
    def depth(self) -> int:
        return max(n.depth for n in self.input_nodes) + 1

    def fmt(self):
        args = (f"x{a.id}" for a in self.input_nodes)
        s = self.name + "(" + ", ".join(args) + ")"
        return s

    @property
    def inputs(self) -> List[Node]:
        return self.input_nodes

    @property
    def is_unary_image(self) -> bool:
        return (
            len(self.input_nodes) == 1
            and self.fn.parameter_types[0] == ParameterType.Image
            and self.fn.return_type == ParameterType.Image
        )

    @property
    def is_binary_image(self) -> bool:
        return (
            len(self.input_nodes) == 2
            and self.fn.parameter_types[0] == ParameterType.Image
            and self.fn.parameter_types[1] == ParameterType.Image
            and self.fn.return_type == ParameterType.Image
        )


class NodeFactory:
    def __init__(self):
        self.functions = {}
        self.next_id = 1

    def register(self, name: str, fn, return_type: ParameterType, parameter_types: List[ParameterType]) -> None:
        assert return_type in (ParameterType.Image, ParameterType.ImageList)
        assert parameter_types != None and len(parameter_types) > 0
        if name in self.functions:
            raise ValueError(f"A function with name '{name}' was already registered.")
        self.functions[name] = Function(name, fn, return_type, parameter_types)

    def register_unary(self, name: str, fn) -> None:
        self.register(name, fn, return_type=ParameterType.Image, parameter_types=[ParameterType.Image])

    def register_binary(self, name: str, fn) -> None:
        self.register(
            name, fn, return_type=ParameterType.Image, parameter_types=[ParameterType.Image, ParameterType.Image]
        )

    def create_node(self, function_name) -> FunctionNode:
        fn = self.functions[function_name]
        id = self.next_id
        self.next_id += 1
        return FunctionNode(fn, id)

    def sample_function_name(self, function_names: List[str] = None):
        fs = function_names if function_names is not None else list(self.functions.keys())
        return random.choice(fs)

    def create_random_node(self, function_names: List[str] = None) -> FunctionNode:
        fn = self.sample_function_name(function_names)
        return self.create_node(fn)


class NodeGraph:
    def __init__(self):
        self.input_node = InputNode(id=0)
        self.nodes = [self.input_node]

    @classmethod
    def from_ancestors(cls, n: Node):
        """create new graph that only contains the nodes which are directly or indireltcy referenced by n"""
        nodes = list(n.ancestors(include_self=True))
        g = cls()
        g.input_node = next(filter(lambda x: isinstance(x, InputNode), nodes))
        g.nodes = nodes
        g.sort_nodes()
        return g

    def sort_nodes(self):
        """Sort internal node list from input to output. Sorting is not required for correct graph evaluation, but it can help debugging."""
        cur = set(x for x in self.nodes if len(x.inputs) == 0)
        remain_nodes = set(self.nodes) - cur
        l = list(cur)
        while len(remain_nodes) > 0:
            to_add = set(x for x in remain_nodes if all((y in cur) for y in x.inputs))
            cur = cur.union(to_add)
            l.extend(to_add)
            remain_nodes = remain_nodes - to_add
        assert len(l) == len(self.nodes)
        self.nodes = l

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

    def get_node_by_id(self, id: int) -> Node:
        for n in self.nodes:
            if n.id == id:
                return n
        raise KeyError(f"Node with id '{id}' not found.")

    def copy(self) -> NodeGraph:
        ng = NodeGraph()
        clones = {ng.input_node.id: ng.input_node}

        def clone_node_recursive(src: Node):
            if src == None:
                return None

            if src.id in clones:
                return clones[src.id]

            if type(src) is FunctionNode:
                fnn = FunctionNode(src.fn, src.id)
                fnn.input_nodes = [clone_node_recursive(src_in) for src_in in src.input_nodes]
                clones[fnn.id] = fnn
                return fnn

            raise RuntimeError(f"Cannot clone node of type {type(src)}")

        ng.nodes = [clone_node_recursive(n) for n in self.nodes]
        return ng

    def remove_unary_image(self, node: FunctionNode) -> Node:
        if (
            type(node) is not FunctionNode
            or len(node.input_nodes) != 1
            or node.fn.parameter_types[0] != ParameterType.Image
            or node.fn.return_type != ParameterType.Image
        ):
            raise ValueError("Invalid node")

        # remove unary function node by replacing it by its input
        replace_by = node.input_nodes[0]
        for m in self.nodes:
            if isinstance(m, FunctionNode):
                for i, inode in enumerate(m.input_nodes):
                    if inode == node:
                        m.input_nodes[i] = replace_by

        self.nodes.remove(node)
        return replace_by

    def serialize(self, debug: bool = False, simplify_ids: bool = True) -> dict:
        if simplify_ids:
            g = self.copy()
            for i, n in enumerate(g.nodes):
                n.id = i
        else:
            g = self

        vertices = [(n.id, n.name) for n in g.nodes]
        edges = []
        for n in g.nodes:
            if isinstance(n, FunctionNode):
                source_ids = [inode.id for inode in n.input_nodes]
            else:
                source_ids = []

            edges.append(source_ids)
        d = {"nodes": vertices, "inputs": edges}
        if debug:
            d["__debug"] = g.fmt()
        return d

    @classmethod
    def deserialize(cls, f: NodeFactory, d: dict):
        ng = cls()
        node_by_id = {ng.input_node.id: ng.input_node}
        vertices, edges = d["nodes"], d["inputs"]
        for id, fname in vertices:
            if fname == "input":
                assert id == 0
                continue
            n = f.create_node(fname)
            node_by_id[id] = n
            ng.add(n)

        for i, src_ids in enumerate(edges):
            n = ng.nodes[i]
            if isinstance(n, FunctionNode):
                for j, id in enumerate(src_ids):
                    n.connect_input(j, node_by_id[id])
        return ng


def register_functions(f: NodeFactory):
    # unary
    for i in range(10):
        f.register_unary(f"filter_color_{i}", partial(filter_color, id=i))
    for i in range(1, 10):
        f.register_unary(f"erase_color_{i}", partial(erase_color, col=i))
    for i in range(1, 10):
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
        f.register_unary(f"make_border2_{b}", partial(make_border2, usemaj=b))

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
    f.register_binary("filter_color_palette", filter_color_palette)
    f.register_binary("replace_colors", replace_colors)

    # split
    f.register("cut_image", cut_image, return_type=ParameterType.ImageList, parameter_types=[ParameterType.Image])
    f.register(
        "split_colors",
        partial(split_colors, include0=False),
        return_type=ParameterType.ImageList,
        parameter_types=[ParameterType.Image],
    )
    f.register("split_all", split_all, return_type=ParameterType.ImageList, parameter_types=[ParameterType.Image])
    f.register(
        "split_columns", split_columns, return_type=ParameterType.ImageList, parameter_types=[ParameterType.Image]
    )
    f.register("split_rows", split_rows, return_type=ParameterType.ImageList, parameter_types=[ParameterType.Image])
    f.register(
        "inside_marked", inside_marked, return_type=ParameterType.ImageList, parameter_types=[ParameterType.Image]
    )
    for d in range(4):
        f.register(
            f"gravity_{d}",
            partial(gravity, d=d),
            return_type=ParameterType.ImageList,
            parameter_types=[ParameterType.Image],
        )

    # join
    for id in range(14):
        f.register(
            f"pick_max_{id}",
            partial(pick_max, id=id),
            return_type=ParameterType.Image,
            parameter_types=[ParameterType.ImageList],
        )
    f.register(f"pick_unique", pick_unique, return_type=ParameterType.Image, parameter_types=[ParameterType.ImageList])
    f.register(
        "compose_growing", compose_growing, return_type=ParameterType.Image, parameter_types=[ParameterType.ImageList]
    )
    f.register("stack_line", stack_line, return_type=ParameterType.Image, parameter_types=[ParameterType.ImageList])
    for id in range(4):
        f.register(
            f"my_stack_list_{id}",
            partial(my_stack_list, orient=id),
            return_type=ParameterType.Image,
            parameter_types=[ParameterType.ImageList],
        )

    # vector
    for id in range(14):
        f.register(
            f"pick_maxes_{id}",
            partial(pick_maxes, id=id),
            return_type=ParameterType.ImageList,
            parameter_types=[ParameterType.ImageList],
        )
        f.register(
            f"pick_not_maxes_{id}",
            partial(pick_not_maxes, id=id),
            return_type=ParameterType.ImageList,
            parameter_types=[ParameterType.ImageList],
        )

    # for i in range(6):
    #     f.register(
    #         f"compose{i}",
    #         partial(compose, id=i),
    #         ParameterType.Image,
    #         [ParameterType.Image, ParameterType.Image],
    #     )
    #     f.register(f"compose_list{i}", partial(compose_list, id=i), ParameterType.Image, [ParameterType.ImageList])

    return f


def print_image(img):
    typer.echo(img.fmt(True))
    print(f"p: {img.p.x},{img.p.y}; sz: {img.sz.x}x{img.sz.y};")


class InputSampler:
    def __init__(
        self,
        riddle_ids: Sequence[int],
        include_outputs: bool = True,
        include_test: bool = True,
        color_permutation: bool = False,
        random_offsets: bool = False,
        add_noise_p: float = 0,
        noise_p: float = 0,
        add_parts_p=1.0,
        parts_min=0,
        parts_max=2,
    ):
        self.riddle_ids = riddle_ids
        self.riddles = [dataset.load_riddle_from_id(id) for id in self.riddle_ids]

        self.boards = [(r.riddle_id + f"_train{i}_in", t.input) for r in self.riddles for i, t in enumerate(r.train)]

        if include_outputs:
            self.boards = self.boards + [
                (r.riddle_id + f"_train{i}_out", t.output) for r in self.riddles for i, t in enumerate(r.train)
            ]

        if include_test:
            self.boards = self.boards + [
                (r.riddle_id + f"_test{i}_in", t.input) for r in self.riddles for i, t in enumerate(r.test)
            ]

        self.color_permutation = color_permutation
        self.random_offsets = random_offsets
        self.add_noise_p = add_noise_p
        self.noise_p = noise_p
        self.order = list(range(len(self.boards)))

        if add_parts_p > 0:
            # extract parts
            parts = []
            self.next_index = 0
            for i in range(len(riddle_ids)):
                img = self.next_image()
                for x in split_all(img):
                    if x.w > 2 and x.h > 2 and x.w < img.w // 2 and x.h < img.h // 2:
                        x.p = Point(0, 0)
                        parts.append(x)
            self.parts = parts

        self.add_parts_p = add_parts_p
        assert parts_min <= parts_max
        self.parts_min = parts_min
        self.parts_max = parts_max

        self.shuffle()

    def shuffle(self):
        random.shuffle(self.order)
        self.next_index = 0

    def next_board(self):
        if self.next_index >= len(self.order):
            self.shuffle()
        i = self.order[self.next_index]
        self.next_index += 1
        return self.boards[i][1]

    def next_image(self):
        board = self.next_board()
        return Image.from_board(board)

    def __randomize_offsets(self, images: List[Image]) -> None:
        max_w = max(i.w for i in images)
        max_h = max(i.h for i in images)

        for img in images:
            if img.w < max_w:
                img.x = random.randint(0, max_w - img.w)
            if img.h < max_h:
                img.y = random.randint(0, max_h - img.h)

    def next_augmented_image(self):
        image = self.next_image()
        image = rigid(image, random.randint(0, 8))
        if self.color_permutation:
            # use random mapping of colors 1-9
            color_map = list(range(1, 10))
            random.shuffle(color_map)
            color_map = [0] + color_map
            image.mask = [color_map[x] for x in image.mask]

        if self.add_noise_p > 0:
            if self.add_noise_p >= 1.0 or random.random() < self.add_noise_p:
                for i in range(len(image.mask)):
                    if self.noise_p >= 1.0 or random.random() < self.noise_p:
                        image.mask[i] = random.randint(0, 9)

        if self.add_parts_p > 0 and len(self.parts):
            if self.add_parts_p >= 1.0 or random.random() <= self.add_parts_p:
                composition = [image]
                for i in range(random.randint(self.parts_min, self.parts_max)):
                    part_to_add = random.choice(self.parts).copy()
                    composition.append(part_to_add)
                self.__randomize_offsets(composition)

        return image

    def next_composed_image(self, n=2):
        inputs = [self.next_augmented_image() for _ in range(n)]
        if self.random_offsets:
            self.__randomize_offsets(inputs)
        return compose_growing(inputs)


class SynthRiddleGen1:
    def __init__(
        self,
        node_factory: NodeFactory,
        input_sampler: InputSampler,
        sample_node_count: int = 10,
        min_depth: int = 2,
        max_depth: int = 5,
        max_input_sample_tries: int = 100,
        min_examles: int = 4,
        max_examles: int = 7,
        function_names: List[str] = None,
    ):
        assert min_depth > 0 and min_depth <= max_depth
        assert sample_node_count > min_depth
        assert min_examles <= max_examles

        self.node_factory = node_factory
        self.input_sampler = input_sampler
        self.sample_node_count = sample_node_count
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_input_sample_tries = max_input_sample_tries
        self.min_examples = min_examles
        self.max_examples = max_examles
        self.function_names = function_names

    @staticmethod
    def try_connect_inputs(n: FunctionNode, g: NodeGraph) -> bool:
        """Try to sample compatible source nodes in graph `g` for all input of the given node `n`."""
        input_types = set(n.fn.parameter_types)
        nodes_by_type = {t: g.filter_nodes_by_type(t) for t in input_types}
        for i, t in enumerate(n.fn.parameter_types):
            if len(nodes_by_type[t]) == 0:
                return False
            source_node = random.choice(nodes_by_type[t])
            n.connect_input(i, source_node)

        return True

    @staticmethod
    def generate_random_graph(f: NodeFactory, node_count: int = 5, function_names: List[str] = None) -> NodeGraph:
        """Genaret a random graph with a `node_count` number of random nodes.`"""
        g = NodeGraph()
        while len(g) < node_count:
            n = f.create_random_node(function_names)
            if SynthRiddleGen1.try_connect_inputs(n, g):
                g.add(n)
        return g

    @staticmethod
    def remove_nops(
        g: NodeGraph, output_node: Node, node_outputs: List[CacheDict], trainig_examples: List[Tuple[Image, Image]]
    ):
        g = g.copy()
        for n in g.nodes.copy():
            if type(n) is FunctionNode and n.is_unary_image:  # unary function
                # if all outputs of this node are identical for all training examples it can be removed from the graph
                v = [o[n.id] for o in node_outputs]
                if all(x == v[0] for x in v):
                    g.remove_unary_image(n)

        # for each unary function determine if it has an influence on the output of at least one examples, if not remove it
        for n in g.nodes.copy():
            if type(n) is FunctionNode and n.is_unary_image:  # unary function
                gc = g.copy()
                nc = gc.get_node_by_id(n.id)
                if n.id == output_node.id:
                    no = gc.remove_unary_image(nc)
                else:
                    gc.remove_unary_image(nc)
                    no = gc.get_node_by_id(output_node.id)

                all_equal = True
                for a, b in trainig_examples:
                    b_ = gc.evaluate(a)[no.id]
                    if b_.mask != b.mask:
                        all_equal = False
                        break

                if all_equal:
                    g = gc
                    output_node = no

        return g, output_node

    def __check_ouputs(self, node: Node, outputs: CacheDict, prev_outputs: List[CacheDict]):
        a = node.ancestors(include_self=True)

        input_image = outputs[0]
        output_image = outputs[node.id]

        # input must differ from output
        if input_image.mask == output_image.mask:
            return False

        if input_image.area < 4 or output_image.area < 1:
            return False

        if sum(output_image.mask) == 0:
            return False  # all zero

        # input and output must be unique
        for o in prev_outputs:
            other_input = o[0]
            other_output = o[node.id]
            if other_input == input_image or other_output == output_image:
                return False

        # all image must stay in limits
        for n in a:
            if isinstance(n, FunctionNode) and n.return_type == ParameterType.Image:
                img = outputs[n.id]
                if img is None or img.area <= 0 or img.w > 32 or img.h > 32:
                    return False

        return True

    def __find_pair(
        self, g: NodeGraph, node: Node, prev_outputs: List[CacheDict], max_tries: int = 100
    ) -> Tuple[Image, Image]:
        for trial in range(max_tries):
            input_image = self.input_sampler.next_composed_image(n=2)
            outputs = g.evaluate(input_image)
            if self.__check_ouputs(node, outputs, prev_outputs):
                prev_outputs.append(outputs)
                output_image = outputs[node.id]
                return (input_image, output_image)
            trial += 1

        raise RuntimeError("Max retries exceeded")

    def generate_riddle(self):
        assert self.min_depth > 0 and self.sample_node_count > self.min_depth

        g = SynthRiddleGen1.generate_random_graph(self.node_factory, self.sample_node_count, self.function_names)

        # single image output nodes with min_depth are candidates
        candidates = [
            n
            for n in g.nodes
            if n.depth >= self.min_depth and n.depth <= self.max_depth and n.return_type == ParameterType.Image
        ]

        for node in candidates:
            if self.max_examples > self.min_examples:
                num_examples = random.randint(self.min_examples, self.max_examples)
            else:
                num_examples = self.min_examples

            example_outputs = []
            try:
                g2 = NodeGraph.from_ancestors(node)
                trainig_examples = [
                    self.__find_pair(g2, node, example_outputs, max_tries=self.max_input_sample_tries)
                    for i in range(num_examples)
                ]
                # print("before NOP removal", g2.fmt())
                g2, node = SynthRiddleGen1.remove_nops(g2, node, example_outputs, trainig_examples)
                if node.depth < self.min_depth:
                    continue  # graph after pruning too shallow

                # print("after NOP remoal", g2.fmt())
                return trainig_examples, g2, node
            except RuntimeError:
                pass

        return None, None, None
