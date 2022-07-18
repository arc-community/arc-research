from collections import OrderedDict
import dataclasses
from pathlib import Path
import random
import json
import argparse
import shutil
from tqdm import tqdm
from typing import List, Sequence, Tuple
from arc.utils import dataset
from node_graph import InputSampler, NodeFactory, SynthRiddleGen1, print_image, register_functions
from image import Image
from dataclasses import dataclass

PRODUCT_NAME = "riddle_synth"
PRODUCT_VERSION = (0, 1)


@dataclass
class InputSamplerConfiguration:
    subdirs: Sequence[str] = ("training",)
    first_n: int = None
    include_outputs: bool = True
    include_test: bool = True
    color_permutation: bool = False  # randomly map colors 1-9
    random_offsets: bool = True  # use random offsets for smaller images when composing
    add_noise_p: float = 0.0
    noise_p: float = 0.0
    add_parts_p: float = 0.0
    parts_min: int = 0
    parts_max: int = 2


@dataclass
class RiddleSynthConfiguration:
    seed: int = None
    num_riddles_to_generate: int = 1
    output_dir: str = "synth_riddles"
    write_graph: bool = True
    function_set: Sequence[str] = None
    exclude_functions: Sequence[str] = None
    min_depth: int = 1
    max_depth: int = 3
    min_examples: int = 4  # 3 examples + 1 test
    max_examples: int = 7
    sample_node_count: int = 10
    max_input_sample_tries: int = 100
    input_sampler: InputSamplerConfiguration = InputSamplerConfiguration()


def read_section(data: dict, config_type: type = RiddleSynthConfiguration) -> object:
    fields = {f.name: f for f in dataclasses.fields(config_type)}
    d = config_type()
    for k, v in data.items():
        f = fields.get(k)
        if f is None:
            raise RuntimeError(f'Configuration value "{k}" not supported.')
        if dataclasses.is_dataclass(f.type):
            v = read_section(v, f.type)
        d.__setattr__(k, v)
    return d


def load_configuration_file(file_path) -> RiddleSynthConfiguration:
    with open(file_path, "r") as f:
        data = json.load(f)
        return read_section(data)


def riddle_to_json(training_pairs: List[Tuple[Image, Image]]) -> str:
    assert len(training_pairs) > 1
    pairs = [OrderedDict(input=x[0].np.tolist(), output=x[1].np.tolist()) for x in training_pairs]
    riddle = OrderedDict(train=pairs[:-1], test=pairs[-1:])
    return json.dumps(riddle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configuration file to use")
    parser.add_argument("--show_functions", default=False, action="store_true", help="show used function names")
    parser.add_argument("--output_dir", default=None, help="override configuration output dir")
    parser.add_argument("--count", type=int, default=None, help="override num_riddles_to_generate configuration")
    parser.add_argument("--show_riddles", default=False, action="store_true")
    parser.add_argument("--dry", default=False, action="store_true", help="dry run: disable writing files")
    parser.add_argument("--overwrite", default=False, action="store_true", help="overwrite existing riddles")
    opt = parser.parse_args()
    return opt


def main():
    print(f"{PRODUCT_NAME} {PRODUCT_VERSION[0]}.{PRODUCT_VERSION[1]}")

    args = parse_args()
    cfg = load_configuration_file(args.config)

    if cfg.seed:
        random.seed(cfg.seed)

    # prepare node factory
    f = NodeFactory()
    register_functions(f)

    if cfg.function_set is not None:
        valid_function_names = set(f.functions.keys())
        function_names = []
        for n in cfg.function_set:
            if type(n) is str:
                if n in valid_function_names:
                    function_names.append(n)
                else:
                    print(f"WARNING: Ignoring unknown function name '{n}'.")
    else:
        function_names = list(f.functions.keys())

    if cfg.exclude_functions is not None:
        function_names = list(set(function_names) - set(cfg.exclude_functions))

    function_names.sort()
    print(f"Number of functions: {len(function_names)}")
    if args.show_functions:
        print(json.dumps(function_names, indent=2))

    # load boards from public training set for InputSampler
    print("Loading boards")
    eval_riddle_ids = dataset.get_riddle_ids(cfg.input_sampler.subdirs)
    if cfg.input_sampler.first_n is not None and cfg.input_sampler.first_n > 0:
        eval_riddle_ids = eval_riddle_ids[: cfg.input_sampler.first_n]
    input_sampler = InputSampler(
        eval_riddle_ids,
        include_outputs=cfg.input_sampler.include_outputs,
        include_test=cfg.input_sampler.include_test,
        color_permutation=cfg.input_sampler.color_permutation,
        random_offsets=cfg.input_sampler.random_offsets,
        add_noise_p=cfg.input_sampler.add_noise_p,
        noise_p=cfg.input_sampler.noise_p,
        add_parts_p=cfg.input_sampler.add_parts_p,
        parts_min=cfg.input_sampler.parts_min,
        parts_max=cfg.input_sampler.parts_max,
    )
    print(f"Total boards: {len(input_sampler.boards)}")

    # create output directory
    output_dir = args.output_dir if args.output_dir is not None else cfg.output_dir
    output_dir = Path(output_dir)
    if not args.dry:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_out_path = output_dir / "config"
        config_out_path.mkdir(exist_ok=True)

        graph_out_path = output_dir / "graphs"
        graph_out_path.mkdir(exist_ok=True)

        # copy config to output dir
        shutil.copy(args.config, str(config_out_path))

    riddle_gen = SynthRiddleGen1(
        f,
        input_sampler,
        sample_node_count=cfg.sample_node_count,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        max_input_sample_tries=cfg.max_input_sample_tries,
        function_names=function_names,
    )

    total_riddles = args.count if args.count is not None else cfg.num_riddles_to_generate
    i = 0

    with tqdm(total=total_riddles) as pbar:
        while i < total_riddles:
            xs, g, node = riddle_gen.generate_riddle()
            if xs == None:
                print("retrying...")
                continue

            # generate riddle hash
            h = hash(tuple(xs)) & 0xFFFFFFFF
            fn = f"{h:08x}.json"
            fn = output_dir / fn
            if not args.overwrite and fn.exists():
                print(f"File {fn} already exists.")
                continue

            j = riddle_to_json(xs)
            if not args.dry:
                fn.write_text(j)
            i += 1
            pbar.update(1)

            if args.show_riddles:
                print(f"{fn}:")
                for j, x in enumerate(xs):
                    print(f"Example {j}")
                    print("INPUT:")
                    print_image(x[0])
                    print("OUTPUT:")
                    print_image(x[1])
                    print()

                print(g.fmt())

                #     # print("begin")
                #     # blub = g.evaluate(x[0])
                #     # for n in g.nodes:
                #     #     if n.return_type == ParameterType.Image:
                #     #         print("node output:", type(n).__name__, n.id)
                #     #         print_image(blub[n.id])
                #     # print("end")

            # write graph to json file
            if not args.dry and cfg.write_graph:
                serialized_graph = g.serialize(debug=True)
                graph_fn = graph_out_path / f"{h:08x}.graph.json"
                graph_fn.write_text(json.dumps(serialized_graph))


if __name__ == "__main__":
    main()
