from collections import OrderedDict
import random
import json
from typing import List, Tuple
from arc.interface import Board
from arc.utils import dataset
from node_graph import InputSampler, NodeFactory, SynthRiddleGen1, print_image, register_functions
from image import Image



def riddle_to_json(training_pairs: List[Tuple[Image, Image]]) -> str:
    assert len(training_pairs) > 1
    pairs = [OrderedDict(input=x[0].np.tolist(), output=x[1].np.tolist()) for x in training_pairs]
    riddle = OrderedDict(train=pairs[:-1], test=pairs[-1:])
    return json.dumps(riddle)


def main():
    random.seed(110)

    print("loading boards")
    eval_riddle_ids = dataset.get_riddle_ids(["training"])[:101]
    input_sampler = InputSampler(eval_riddle_ids, include_outputs=True, include_test=True)
    print(f"Total boards: {len(input_sampler.boards)}")

    f = NodeFactory()
    register_functions(f)
    print("Number of functions:", len(f.functions))

    function_names = [
        "rigid_1",
        "rigid_2",
        "rigid_3",
        "rigid_4",
        "rigid_5",
        "rigid_6",
        "rigid_7",
        "rigid_8",
        "half_0",
        "half_1",
        "half_2",
        "half_3",
    ]
    riddle_gen = SynthRiddleGen1(f, input_sampler, min_depth=1, max_depth=1, function_names=function_names)

    riddles = []
    while len(riddles) < 10:
        xs, g, node = riddle_gen.generate_riddle()
        if xs == None:
            print("fail")
            continue

        riddles.append(xs)
        print(f"RIDDLE {len(riddles)}")

        for i, x in enumerate(xs):
            print(f"example {i}")
            print("INPUT:")
            print_image(x[0])
            print("OUTPUT:")
            print_image(x[1])
            print()

            # print("begin")
            # blub = g.evaluate(x[0])
            # for n in g.nodes:
            #     if n.return_type == ParameterType.Image:
            #         print("node output:", type(n).__name__, n.id)
            #         print_image(blub[n.id])
            # print("end")

        print(g.fmt())
        print("node:", node.id)
        
        j = riddle_to_json(xs)
        print(j)
        quit()


if __name__ == "__main__":
    main()
