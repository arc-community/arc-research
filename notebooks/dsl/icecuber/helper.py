# prepare & load func_examples
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import display_markdown
import pickle
from pathlib import Path
from riddle_synth.node_graph import NodeFactory, register_functions
from riddle_synth.image import Image
import riddle_synth.image
import sys


def plot_image(ax, image, title=""):
    cmap = colors.ListedColormap(
        ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
    )
    norm = colors.Normalize(vmin=0, vmax=9)
    input_matrix = image.np
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)


def plot_imagelist(p, title):
    if isinstance(p, Image):
        fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True)
        fig.suptitle(title, fontsize=16)
        plot_image(ax, p, str(p))
    elif isinstance(p, list):
        if len(p) > 1:
            fig, axs = plt.subplots(ncols=len(p), nrows=1, figsize=(len(p), 2), constrained_layout=True)
            fig.suptitle(title, fontsize=16)
            for j, li in enumerate(p):
                plot_image(axs[j], li, str(li))
        elif len(p) == 1:
            plot_imagelist(p[0], title)


def plot_fn_example(fn_name, input_args, output_value):
    for i, p in enumerate(input_args):
        plot_imagelist(p, f"{fn_name} arg{i}")
    plot_imagelist(output_value, f"{fn_name} out")
    plt.show()


# visualize func_examples
def show_function_outputs(fn_name, v, f):
    display_markdown(f"# {fn_name}\n------", raw=True)
    for j, example in enumerate(v):

        fn_dsc = f.functions[fn_name]
        s = f"{fn_name} (" + ",".join([t.name for t in fn_dsc.parameter_types]) + ") -> " + fn_dsc.return_type.name
        print(f"Example: {j+1}/{len(v)}: {s}")

        input_args = example["input"]
        output_value = example["output"]
        plot_fn_example(fn_name, input_args, output_value)


def load_func_examples():
    f = NodeFactory()
    register_functions(f)
    with open("func_examples2.pkl", "rb") as file:
        data = pickle.load(file)
    
    func_examples = data["func_examples"]

    def show_func_examples(fn_name):
        show_function_outputs(fn_name, func_examples[fn_name], f)

    return show_func_examples
