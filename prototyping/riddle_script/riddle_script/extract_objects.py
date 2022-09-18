from typing import Sequence
import pickle
from pathlib import Path
from arc.utils import dataset

from riddle_script import Image, Point, split_all, count_colors, compress 


def extract_objects(riddle_ids: Sequence[int]):
    # gather all training and test boards
    riddles = [dataset.load_riddle_from_id(id) for id in riddle_ids]
    boards = (
        [t.input for r in riddles for i, t in enumerate(r.train)]
        + [t.output for r in riddles for i, t in enumerate(r.train)]
        + [t.input for r in riddles for i, t in enumerate(r.test)]
        + [t.output for r in riddles for i, t in enumerate(r.test)]
    )

    parts = set()
    for board in boards:
        img = compress(Image.from_board(board))
        if img.w > 1 and img.w < 6 and img.h > 1 and img.h < 6 and count_colors(img) > 1:
            parts.add(img)  # treat small boards with more than one color as objects
        elif img.area > 8:
            for x in split_all(img):
                if x.w > 2 and x.h > 2 and x.w < img.w // 2 and x.h < img.h // 2:
                    x.p = Point(0, 0)
                    parts.add(x)

    return list(parts)


def count_shapes(shapes):
    d = {}
    for s in shapes:
        if s in d:
            d[s] += 1
        else:
            d[s] = 1
    return d


def write_all(fn):
    fn = Path(fn)
    subdirs = ["training", "evaluation"]
    riddle_ids = dataset.get_riddle_ids(subdirs)
    parts = extract_objects(riddle_ids)
    print(f"Unique parts: {len(parts)}")

    part_shapes = [p.sz.astuple() for p in parts]
    d = count_shapes(part_shapes)
    sorted_shapes = list((v, k) for k, v in d.items())
    sorted_shapes.sort(key=lambda x: x[0])
    print(f"Top 10 shapes (distinct shapes: {len(d)}):")
    for c, s in sorted_shapes[-10:][::-1]:
        print(f"{s}: {c}")

    print(f'writing: "{fn}"')
    with fn.open("wb") as f:
        pickle.dump(parts, f)


def main():
    write_all("arc1_obj_all.pkl")


if __name__ == "__main__":
    main()
