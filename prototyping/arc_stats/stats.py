from pathlib import Path
import json
from xmlrpc.client import Boolean
import numpy as np


from dataclasses import dataclass



@dataclass
class ArcTrainingExample:
    input: np.ndarray
    output: np.ndarray


class ArcRiddle:
    def __init__(self, id):
        self.id = id
        self.train = []
        self.test = []


def read_arc_file(path):
    def parse_training_example(json_obj):
        input = np.array(json_obj['input'])
        output = np.array(json_obj['output'])
        return ArcTrainingExample(input=input, output=output)
        
    riddle = ArcRiddle(path.stem)
    with path.open() as f:
        data = json.load(f)
        for j in data['train']:
            riddle.train.append(parse_training_example(j))
        for j in data['test']:
            riddle.test.append(parse_training_example(j))
    return riddle


def count_shapes(shapes):
    d = {}
    for s in shapes:
        if s in d:
            d[s] += 1
        else:
            d[s] = 1
    return d


def print_bincounts(bincount: np.ndarray, skip_zeros:Boolean=False):
    total = bincount.sum()
    for i,c in enumerate(bincount):
        if skip_zeros and c == 0:
            continue 
        print(f'{i:2d}: {c:6d} ({float(c)/total:.2%})')
    print()


def print_dataset_stats(base_path):
    print('base_path: ', str(base_path))

    json_paths = base_path.glob('*.json')
    riddles = [read_arc_file(p) for p in json_paths]

    print(f'Number of training riddles: {len(riddles)}')
    print(f'Number of training examples: {sum(len(r.train) for r in riddles)}; (w/ test: {sum(len(r.train)+len(r.test) for r in riddles)})')
    print(f'Number of riddles with multiple tests: {sum(1 for r in riddles if len(r.test) > 1)}')
    print()

    # number of train examples
    train_example_count_hist = np.bincount([len(r.train) for r in riddles])
    print('Train examples per riddle (bincount):')
    print_bincounts(train_example_count_hist, skip_zeros=True)
    
    all_input_flat = np.concatenate([t.input.flatten() for r in riddles for t in r.train])
    train_input_color_distribution = np.bincount(all_input_flat)
    print('Input color distribution (bincount):')
    print_bincounts(train_input_color_distribution, skip_zeros=True)

    all_output_flat = np.concatenate([t.output.flatten() for r in riddles for t in r.train])
    train_outout_color_distribution = np.bincount(all_output_flat)
    print('Output color distribution (bincount):')
    print_bincounts(train_outout_color_distribution, skip_zeros=True)

    all_train = [t for r in riddles for t in r.train]
    all_train_input = [t.input for t in all_train]
    all_train_output = [t.output for t in all_train]

    unique_input_color_counts = [len(np.unique(i)) for i in all_train_input]
    print('Unique colors per input (bincount):')
    print_bincounts(np.bincount(unique_input_color_counts), skip_zeros=True)

    unique_output_color_counts = [len(np.unique(i)) for i in all_train_output]
    print('Unique colors per output (bincount):')
    print_bincounts(np.bincount(unique_output_color_counts), skip_zeros=True)

    riddles_with_equal_inout_size = [r for r in riddles if all(t.input.shape == t.output.shape for t in r.train )]
    print(f'Number of riddles with equal input and output shape: {len(riddles_with_equal_inout_size)}')

    riddles_with_strictly_smaller_output_size = [r for r in riddles if all(t.input.shape[0] > t.output.shape[0] and t.input.shape[1] > t.output.shape[1] for t in r.train )]
    print(f'Number of riddles with strictly smaller output shape: {len(riddles_with_strictly_smaller_output_size)}')

    riddles_with_strictly_greater_output_size = [r for r in riddles if all(t.input.shape[0] < t.output.shape[0] and t.input.shape[1] < t.output.shape[1] for t in r.train )]
    print(f'Number of riddles with strictly greater output shape: {len(riddles_with_strictly_greater_output_size)}')

    riddles_with_row_column_output = [r for r in riddles if all(t.output.shape[0] == 1 or t.output.shape[1] == 1 for t in r.train )]
    print(f'Number of riddles with only single row/column outputs: {len(riddles_with_row_column_output)}')

    input_shapes = [t.input.shape for r in riddles for t in r.train]
    output_shapes = [t.output.shape for r in riddles for t in r.train]

    input_areas = [s[0]*s[1] for s in input_shapes]
    output_areas = [s[0]*s[1] for s in output_shapes]

    print(f'min input area: {min(input_areas)}; max input area: {max(input_areas)};')
    print(f'min output area: {min(output_areas)}; max output area: {max(output_areas)};')

    print()
    x = count_shapes(input_shapes)
    sorted_input_shapes = list((v,k) for k,v in x.items())
    sorted_input_shapes.sort(key=lambda x: x[0])
    print(f'Top 10 input shapes (distinct shapes: {len(x)}):')
    for c,s in sorted_input_shapes[-10:][::-1]:
        print(f'{s}: {c}')
    print()

    x = count_shapes(output_shapes)
    sorted_input_shapes = list((v,k) for k,v in x.items())
    sorted_input_shapes.sort(key=lambda x: x[0])
    print(f'Top 10 output shapes (distinct shapes: {len(x)}):')
    for c,s in sorted_input_shapes[-10:][::-1]:
        print(f'{s}: {c}')
    print()


def main():
    arc_data_base_path = Path('/home/koepf/code/ARC/data/')
    print_dataset_stats(arc_data_base_path / 'training') 
    print_dataset_stats(arc_data_base_path / 'evaluation')


if __name__ == '__main__':
    main()
