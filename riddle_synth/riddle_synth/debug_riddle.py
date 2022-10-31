import argparse
import json
from pathlib import Path
from arc.utils.dataset import load_riddle_from_file
from image import Image
from node_graph import FunctionNode, InputNode, NodeFactory, NodeGraph, ParameterType, print_image, register_functions


def print_steps(board_pair, g):
    input_image = Image.from_board(board_pair.input)
    output_image = Image.from_board(board_pair.output)
    outputs = g.evaluate(input_image)
    for n in g.nodes:
        print(f"x{n.id} = {n.fmt()}")
        x = outputs[n.id]
        if isinstance(n, InputNode):
            print_image(x)
        elif isinstance(n, FunctionNode):
            if n.fn.return_type == ParameterType.Image:
                print_image(x)
            elif n.fn.return_type == ParameterType.ImageList:
                for j, y in enumerate(x):
                    print(f"[{j}]:")
                    print_image(y)
        print()
    print("Expected output:")
    print_image(output_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--riddle", type=str, required=True, help="path to riddle.json file")
    parser.add_argument("--graph", type=str, default=None, help="path to riddle.graph.json file")
    parser.add_argument("--train", type=int, default=None, help="index of train example to analyse")
    parser.add_argument("--test", type=int, default=None, help="index of test example to analyse")
    parser.add_argument("--show_training", default=False, action="store_true", help="show training examples")
    parser.add_argument("--show_test", default=False, action="store_true", help="show test examples")
    opt = parser.parse_args()
    return opt


def show_board_pairs(board_pairs):
    for i, board_pair in enumerate(board_pairs):
        input_image = Image.from_board(board_pair.input)
        output_image = Image.from_board(board_pair.output)
        print("Input:")
        print_image(input_image)
        print("Output:")
        print_image(output_image)


def main():
    args = parse_args()

    riddle_path = Path(args.riddle)
    if not riddle_path.exists():
        print(f'Error: Riddle file "{riddle_path}" not found.')
        quit(code=-1)

    if args.graph is None:
        graph_path = riddle_path.parent / Path("graphs") / (riddle_path.stem + ".graph.json")
    else:
        graph_path = Path(args.graph)

    if not graph_path.exists():
        print(f'Error: Graph file "{graph_path}" not found.')
        quit(code=-1)

    nf = NodeFactory()
    register_functions(nf)

    # load graph
    with graph_path.open("r") as f:
        gd = json.load(f)

    g = NodeGraph.deserialize(nf, gd)
    print("Deserialized graph:")
    print(g.fmt())

    # load riddle
    print(f"Loading riddle: {riddle_path}")
    r = load_riddle_from_file(str(riddle_path))
    print(f"Training examples: {len(r.train)}")
    print(f"Test examples: {len(r.test)}")

    if args.show_training:
        show_board_pairs(r.train)

    if args.show_test:
        show_board_pairs(r.test)

    if args.train is not None:
        print_steps(r.train[args.train], g)

    if args.test is not None:
        print_steps(r.test[args.test], g)


if __name__ == "__main__":
    main()
