from arc.utils import dataset
from arc.interface import Riddle, Board
from pathlib import Path
import csv


def parse_solution_data(line):
    rows = []
    for l in line:
        row = []
        for c in l:
            i = int(c)
            assert i >= 0 and i <= 9
            row.append(i)
        rows.append(row)
    assert all(len(rows[0]) == len(r) for r in rows)

    return Board(__root__=rows)


def check_riddles(path_iter):
    result = []
    num_total, num_solved = 0, 0
    for fn in path_iter:
        with fn.open(mode="r") as f:
            x = csv.reader(f, delimiter="|")
            riddle_id = next(x)[0]
            riddle_id = riddle_id.split("_")
            test_index = int(riddle_id[1])
            riddle_id = riddle_id[0]

            r = dataset.load_riddle_from_id(riddle_id)

            candidates = []
            while True:
                raw = next(x, None)
                if raw == None:
                    break
                ln = raw[1:-1]
                board = parse_solution_data(ln)
                candidates.append(board)

            test_pair = r.test[test_index]

            solved = any(c == test_pair.output for c in candidates)
            result.append((riddle_id, test_index, len(r.test), solved))
            #print(f"{riddle_id}, {test_index}, {solved}")

            if solved:
                num_solved += 1
            num_total += 1

    return result, num_total, num_solved



def update_wiki_files(arc_wiki_path, results):

    files = [
        'Training-Riddles-0-to-3.md',
        'Training Riddles 4 to 7.md',
        'Training Riddles 8 to b.md',
        'Training Riddles c to f.md',
        'Evaluation Riddles 0 to 3.md',
        'Evaluation Riddles 4 to 7.md',
        'Evaluation Riddles 8 to b.md',
        'Evaluation Riddles c to f.md',
    ]

    for fn in files:
        path = arc_wiki_path / fn
        text = path.read_text()
        in_lines = text.splitlines()
        out_lines = []
        for ln in in_lines:
            parts = ln.split('|')
            if parts[1] == ' Riddle ':
                if len(parts) < 6:
                    parts.insert(4, ' ice-dsl ')
            elif parts[1] == ' --- ':
                if len(parts) < 6:
                    parts.insert(4, ' --- ')
            else:
                riddle_id = parts[1].strip()
                if len(parts) < 6:
                    parts.insert(4, f' {results[riddle_id]} ')
                else:
                    parts[4] = f' {results[riddle_id]} '
                
            out_lines.append('|'.join(parts))
        text = '\n'.join(out_lines)
        path.write_text(text)


def main():
    # print for results for training and evaluation

    # get riddle ids of training folder
    trainig_ids = sorted(dataset.get_riddle_ids(['training']))
    evaluation_ids = sorted(dataset.get_riddle_ids(['evaluation']))

    all_results = {}

    # gather all results
    dir = "./output"
    run_args = [3, 23, 33, 4]
    for a in run_args:
        pattern = f"*_{a}.csv"
        result, num_total, num_solved = check_riddles(Path(dir).glob(pattern))

        for riddle_id, test_index, num_test, solved in result:
            if not riddle_id in all_results:
                test_results = [False] * num_test
                solved_by = [[]] * num_test
                all_results[riddle_id] = { 'id': riddle_id, 'test': test_results, 'solved_by': solved_by }

            if solved:
                all_results[riddle_id]['test'][test_index] = True
                all_results[riddle_id]['solved_by'][test_index].append(a)
        
        if num_total > 0:
            print(
                f"Arg: {a}; Total: {num_total}; Solved: {num_solved}; Accuracy: {num_solved/num_total:.1%}"
            )

    def check_ids(ids, results):
        for id in ids:
            if id in all_results:
                test_results = all_results[id]['test']
                solve_args = -1
                if all(test_results):
                    solved_by = all_results[id]['solved_by']
                    solve_args = max([s[0] for s in solved_by], key=lambda x: 40 if x==4 else x)     # max arg for first solution, treat 4 as 40
                results[id] = solve_args
                print(f'{id}, {all(test_results)}, {solve_args}')
            else:
                results[id] = -2
                print(f'{id}, False, -2')   # no solution generated
            
    results = {}
    print('Training')
    check_ids(trainig_ids, results)
    print()
    print('Evaluation')
    check_ids(evaluation_ids, results)

    arc_wiki_path = Path('../../../arc.wiki/')
    #update_wiki_files(arc_wiki_path, results) # uncomment to really update wiki files


if __name__ == "__main__":
    main()
