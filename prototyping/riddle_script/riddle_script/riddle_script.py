from __future__ import annotations
import random
from typing import Callable, Iterable, Tuple, List, overload
from pathlib import Path
import pickle
import numpy as np
from arc.interface import Board


class Point:
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, x: int, y: int) -> None:
        ...

    @overload
    def __init__(self, pos: Tuple[int]) -> None:
        ...

    def __init__(self, *args):
        if len(args) == 0:
            x, y = 0, 0
        elif len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
            x, y = args[0]
        elif len(args) == 1 and isinstance(args[0], Point):
            x, y = args[0].x, args[0].y
        elif len(args) == 2 and type(args[0]) is int and type(args[1]) is int:
            x, y = args
        else:
            raise ValueError("Invalid arguments for Point constructor")

        self.x = int(x)
        self.y = int(y)

    def copy(self) -> Point:
        return Point(self.x, self.y)

    def __add__(self, other) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other) -> Point:
        if isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other)
        return Point(self.x * other.x, self.y * other.y)

    def __div__(self, f: int) -> Point:
        assert self.x % f == 0 and self.y % f == 0
        return Point(self.x / f, self.y / f)

    def __floordiv__(self, f: int) -> Point:
        return Point(self.x // f, self.y // f)

    def __eq__(self, other: object) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: object) -> bool:
        return self.x != other.x or self.y != other.y

    def cross(self, other: Point) -> int:
        return self.x * other.y - self.y * other.x

    def __str__(self):
        return f"({self.x}, {self.y})"

    def astuple(self):
        return self.x, self.y

    def __iter__(self):
        return iter(self.astuple())

    def __getitem__(self, key):
        if key == 0 or key == "x":
            return self.x
        elif key == 1 or key == "y":
            return self.y

        raise IndexError("Index out of range.")


class Image:
    @staticmethod
    def from_board(board: Board) -> Image:
        return Image(Point(), Point(board.num_cols, board.num_rows), board.flat)

    @overload
    def __init__(self, p: Point, sz: Point, mask: Iterable[int]):
        ...

    @overload
    def __init__(self, p: Tuple[int], sz: Tuple[int], mask: Iterable[int]):
        ...

    def __init__(self, *args):
        if len(args) == 3:
            p = Point(args[0])
            sz = Point(args[1])
            mask = list(args[2])
        else:
            raise ValueError("Invalid number of arguments for Image constructor")

        self.p = p
        self.sz = sz
        self.mask = mask

    @property
    def x(self) -> int:
        return self.p.x

    @x.setter
    def x(self, value: int):
        self.p.x = value

    @property
    def y(self) -> int:
        return self.p.y

    @y.setter
    def y(self, value: int):
        self.p.y = value

    @property
    def w(self) -> int:
        return self.sz.x

    @w.setter
    def w(self, value: int):
        self.sz.x = value

    @property
    def h(self) -> int:
        return self.sz.y

    @h.setter
    def h(self, value: int):
        self.sz.y = value

    @property
    def area(self):
        return self.w * self.h

    def __getitem__(self, key) -> int:
        i, j = key
        assert i >= 0 and j >= 0 and i < self.h and j < self.w
        return self.mask[i * self.w + j]

    def __setitem__(self, key, value: int):
        i, j = key
        assert i >= 0 and j >= 0 and i < self.h and j < self.w
        self.mask[i * self.w + j] = value

    def safe(self, i: int, j: int) -> int:
        return 0 if i < 0 or j < 0 or i >= self.h or j >= self.w else self.mask[i * self.w + j]

    def __eq__(self, other: object) -> bool:
        return self.p == other.p and self.sz == other.sz and self.mask == other.mask

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __lt__(self, other: Image):
        if self.sz != other.sz:
            return (self.w, self.h) < (other.w, other.h)
        return self.mask < other.mask

    def __hash__(self):
        base = 137
        r = 1543
        r = r * base + self.w
        r = r * base + self.h
        r = r * base + self.x
        r = r * base + self.y
        for c in self.mask:
            r = r * base + c
        return r

    def __str__(self) -> str:
        return f"{self.x}:{self.y} {self.w}x{self.h}"

    def map(self, fn) -> Image:
        mapped = [fn(i, j, self[i, j]) for i in range(self.h) for j in range(self.w)]
        return Image(self.p, self.sz, mapped)

    def copy(self) -> Image:
        return Image(self.p, self.sz, self.mask.copy())

    @property
    def np(self) -> np.ndarray:
        return np.array(self.mask, dtype=np.int64).reshape(self.h, self.w)

    def to_board(self) -> Board:
        data = [[self[(i, j)] for j in range(self.w)] for i in range(self.h)]
        return Board.parse_obj(data)

    def fmt(self, colored=False) -> str:
        return self.to_board().fmt(colored=colored)


badImg = Image((0, 0), (0, 0), [])


@overload
def full(p: Point, sz: Point, filling: int = 0) -> Image:
    ...


@overload
def full(p: Tuple[int], sz: Tuple[int], filling: int = 0) -> Image:
    ...


@overload
def full(sz: Tuple[int], filling: int = 0) -> Image:
    ...


def full(*args):
    filling = 0
    if len(args) == 1:
        p = Point(0, 0)
        sz = Point(args[0])
    elif len(args) == 3:
        p = Point(args[0])
        sz = Point(args[1])
        filling = args[2]
    elif len(args) == 2:
        if type(args[1]) is int:
            p = Point(0, 0)
            sz = Point(args[0])
            filling = args[1]
        else:
            p = Point(args[0])
            sz = Point(args[1])
    else:
        raise ValueError("Invalid numer of arguments")
    return Image(p, sz, [filling] * (sz.x * sz.y))


@overload
def empty(p: Point, sz: Point) -> Image:
    ...


@overload
def empty(sz: Point) -> Image:
    ...


def empty(*args):
    if len(args) == 1:
        sz = Point(args[0])
        return full(sz, 0)
    elif len(args) == 2:
        p = Point(args[0])
        sz = Point(args[1])
        return full(p, sz, 0)


def Col(id: int) -> Image:
    assert id >= 0 and id < 10
    return full((0, 0), (1, 1), id)


def color_mask(img: Image) -> int:
    mask = 0
    for i in range(img.h):
        for j in range(img.w):
            mask |= 1 << img[i, j]
    return mask


def count_colors(img: Image) -> int:
    return len(set(img.mask))


def count_nonzero(img: Image) -> int:
    ans = 0
    for i in range(img.h):
        for j in range(img.w):
            ans += int(img[i, j] != 0)
    return ans


def color_shape_const(shape: Image, id: int) -> Image:
    """Set any non-zero pixel in image to the color specified by id."""
    assert id >= 0 and id < 10
    return Image(shape.p, shape.sz, [id if x != 0 else 0 for x in shape.mask])


def compress(img: Image, bg: Image = Col(0)):
    """Remove all border columns and rows which exclusively contain colors found in bg."""
    bg_mask = color_mask(bg)

    xmi, ymi = 1e9, 1e9
    xma, yma = 0, 0
    for i in range(img.h):
        for j in range(img.w):

            if (bg_mask >> img[i, j] & 1) == 0:
                xmi = min(xmi, j)
                xma = max(xma, j)
                ymi = min(ymi, i)
                yma = max(yma, i)

    if xmi == 1e9:
        return badImg

    ret = empty(img.p + Point(xmi, ymi), Point(xma - xmi + 1, yma - ymi + 1))
    for i in range(ymi, yma + 1):
        for j in range(xmi, xma + 1):
            ret[i - ymi, j - xmi] = img[i, j]

    return ret


def split_all(img: Image) -> List[Image]:
    ret = []
    done = empty(img.p, img.sz)

    for i in range(img.h):
        for j in range(img.w):
            if done[i, j] == 0:
                toadd = empty(img.p, img.sz)

                def dfs(r: int, c: int, col: int) -> None:
                    if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c] != 0:
                        return
                    toadd[r, c] = img[r, c] + 1
                    done[r, c] = 1
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        dfs(nr, nc, col)

                dfs(i, j, img[i, j])
                toadd = compress(toadd)

                for y in range(toadd.h):
                    for x in range(toadd.w):
                        toadd[y, x] = max(0, toadd[y, x] - 1)

                if count_nonzero(toadd) > 0:
                    ret.append(toadd)

    return ret


def compose_internal(a: Image, b: Image, f: Callable[[int, int], int], overlap_only: int) -> Image:
    ret = Image((0, 0), (0, 0), [])
    if overlap_only == 1:
        ret.p = Point(max(a.x, b.x), max(a.y, b.y))
        ra = a.p + a.sz
        rb = b.p + b.sz
        ret.sz = Point(min(ra.x, rb.x), min(ra.y, rb.y))
        ret.sz = ret.sz - ret.p
        if ret.w <= 0 or ret.h <= 0:
            return badImg
    elif overlap_only == 0:
        ret.p = Point(min(a.x, b.x), min(a.y, b.y))
        ra = a.p + a.sz
        rb = b.p + b.sz
        ret.sz = Point(max(ra.x, rb.x), max(ra.y, rb.y))
        ret.sz = ret.sz - ret.p
    elif overlap_only == 2:
        ret.p = a.p
        ret.sz = a.sz
    else:
        assert False

    # if ret.w > MAXSIDE or ret.h > MAXSIDE or ret.area > MAXAREA:
    #     return badImg

    ret.mask = [0] * ret.area

    da = ret.p - a.p
    db = ret.p - b.p
    for i in range(ret.h):
        for j in range(ret.w):
            ca = a.safe(i + da.y, j + da.x)
            cb = b.safe(i + db.y, j + db.x)
            ret[i, j] = f(ca, cb)

    return ret


def compose(a: Image, b: Image, id: int = 0) -> Image:
    if id == 0:
        return compose_internal(a, b, lambda a, b: b if b != 0 else a, 0)  # a then b, inside either
    elif id == 1:
        return compose_internal(a, b, lambda a, b: b if b != 0 else a, 1)  # a then b, inside both
    elif id == 2:
        return compose_internal(a, b, lambda a, b: a if b != 0 else 0, 1)  # a masked by b
    elif id == 3:
        return compose_internal(a, b, lambda a, b: b if b != 0 else a, 2)  # a then b, inside of a
    elif id == 4:
        return compose_internal(a, b, lambda a, b: 0 if b != 0 else a, 2)  # a masked by inverse of b, inside of a
    elif id == 5:
        return compose_internal(a, b, lambda a, b: max(a, b), 0)  # max(a, b), inside either
    else:
        assert id >= 0 and id < 5
    return badImg


def compose_list(imgs: List[Image], id: int=0) -> Image:
    if len(imgs) == 0:
        return badImg
    ret = imgs[0].copy()
    for i in range(1, len(imgs)):
        ret = compose(ret, imgs[i], id)
    return ret


def resolve_path(fn: Path, search_paths: List[Path]):
    for sp in search_paths:
        p = sp / fn
        if p.exists() and p.is_file():
            return p
    return None


def random_colors(n: int, lo: int = 1, hi: int = 9) -> List[int]:
    """Returns a list of n distinct random colors between lo and hi (inclusive)."""
    if n < 0:
        raise ValueError("Argument 'n' must be a positive integer.")
    l = list(range(lo, hi + 1))
    random.shuffle(l)
    return l[:n]


class PartSampler:
    def __init__(self, fn="arc1_obj_all.pkl"):
        fn = Path(fn)

        if not fn.is_absolute():
            search_paths = [Path.cwd(), Path(__file__).parent]
            p = resolve_path(fn, search_paths)
            if p is None:
                raise FileNotFoundError(f"File '{fn}' not found.")
        else:
            p = fn

        with p.open("rb") as f:
            self.parts = pickle.load(f)
        self.order = list(range(len(self.parts)))
        self.shuffle()

        parts_by_shape = {}
        for p in self.parts:
            shape = p.sz.astuple()
            if shape in parts_by_shape:
                parts_by_shape[shape].append(p)
            else:
                parts_by_shape[shape] = [p]

        self.index_by_shape = {}
        self.parts_by_shape = parts_by_shape

    def shuffle(self) -> None:
        random.shuffle(self.order)
        self.next_index = 0

    def next_part(self) -> Image:
        if self.next_index >= len(self.order):
            self.shuffle()
        i = self.order[self.next_index]
        self.next_index += 1
        return self.parts[i]

    def next_part_sized(self, sz: Tuple[int]) -> Image:
        parts = self.parts_by_shape[sz]
        if not sz in self.index_by_shape or self.index_by_shape[sz] >= len(parts):
            self.index_by_shape[sz] = 0
            random.shuffle(parts)
        i = self.index_by_shape[sz]
        self.index_by_shape[sz] += 1
        return parts[i]

    def distinct_symbols_sized(self, n: int, sz: Tuple[int], color: int = 1) -> List[Image]:
        symbols = set()

        while len(symbols) < n:
            p = self.next_part_sized(sz)
            p = color_shape_const(p, color)
            if p in symbols:
                continue
            symbols.add(p)

        return list(symbols)

    def distinct_symbols_size_range(
        self, n: int, min_w: int, min_h: int, max_w: int, max_h: int, color: int = 1, min_area: int = -1
    ) -> List[Image]:
        symbols = set()

        while len(symbols) < n:
            p = self.next_part()
            if p.w < min_w or p.h < min_h or p.w > max_w or p.h > max_h:
                continue
            p = color_shape_const(p, color)
            if p in symbols:
                continue
            symbols.add(p)

        return list(symbols)
