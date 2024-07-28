from typing import Dict, List, Tuple
from riddle_synth.image import *
from arc.utils import dataset
from riddle_synth.node_graph import NodeGraph, NodeFactory, register_functions
import json

def print_image_or_list(result):
    if isinstance(result, list):
        print(f"List of {len(result)} images:")
        for i, img in enumerate(result):
            print(f"Image {i + 1}:")
            print(img.fmt(True))
            print()
    else:
        print(result.fmt(True))

def display_transformations(graph, input_image, intermediate_results):
    print("Input Image:")
    print(input_image.fmt(True))
    print()

    for node in graph.nodes[1:-1]:  # Skip the input node
        print(f"Transformation: {node.name}")
        print(node.fmt())
        print("Result:")
        print_image_or_list(intermediate_results[node.id])
        print()

    print("Final Output:")
    print_image_or_list(intermediate_results[graph.nodes[-1].id])


# Load the graph data
graph_data = json.load(open("/private/tmp/outputs/graphs/9e3fae94.graph.json"))
f = NodeFactory()
register_functions(f)
ng = NodeGraph.deserialize(f, graph_data)

# Load the input image
riddle = dataset.load_riddle_from_file('/private/tmp/outputs/9e3fae94.json')
inp, out = riddle.train[0].input, riddle.train[0].output
input_image = Image.from_board(inp)

# Evaluate the graph and get intermediate results
intermediate_results = ng.evaluate(input_image)

# Display the transformations
display_transformations(ng, input_image, intermediate_results)
