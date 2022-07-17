import random
from arc.utils import dataset
from node_graph import FunctionNode, InputSampler, NodeFactory, print_image, register_functions


def main():
    random.seed(72)

    f = NodeFactory()
    register_functions(f)

    eval_riddle_ids = dataset.get_riddle_ids(["training"])
    eval_riddle_ids = eval_riddle_ids[:100]
    input_sampler = InputSampler(eval_riddle_ids, include_outputs=True, include_test=True)
    print(f"Total boards: {len(input_sampler.boards)}")

    function_names = list(f.functions.keys())

    def find_image_unary(n: FunctionNode):
        for i in range(1000):
            num_inputs = len(n.input_nodes)

            input_images = [input_sampler.next_composed_image(n=2) for _ in range(num_inputs)]
            input_image = input_images[0]
            input_image.p.x = 2
            input_image.p.y = 2

            if input_image.area < 4:
                continue

            output_image = n.fn.evaluate(input_images)

            if (input_image == output_image or output_image.area < 1) and n.name not in ["to_origin"]:
                continue

            if sum(output_image.mask) == 0 and n.name not in ["get_size0", "hull0", "center"]:
                continue

            return input_images, output_image

        raise RuntimeError("no suitable input image found")


    for i,function_name in enumerate(function_names):
        n = f.create_node(function_name)
        print(f"Function #{i}: {function_name}")
        if n.is_unary_image or n.is_binary_image:

            input_images, output_image = find_image_unary(n)

            for j,input_image in enumerate(input_images):
                print(f"INPUT{j}:")
                print_image(input_image)
            print("OUTPUT:")
            print_image(output_image)
            print()

        else:
            print('skipped')


if __name__ == "__main__":
    main()
