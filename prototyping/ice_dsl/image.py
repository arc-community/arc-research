from __future__ import annotations
from tkinter import W
from tokenize import Number
from typing import Iterable, Tuple, overload, List


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
        elif len(args) == 1 and type(args[0]) == tuple and len(args[0]) == 2:
            x, y = args[0]
        elif len(args) == 1 and type(args[0]) == Point:
            x, y = args[0].x, args[0].y
        elif len(args) == 2 and type(args[0]) == int and type(args[1]) == int:
            x, y = args
        else:
            raise ValueError("Invalid arguments for Point constructor")

        self.x = int(x)
        self.y = int(y)

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

    def __eq__(self, other: object) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: object) -> bool:
        return self.x != other.x or self.y != other.y

    def cross(self, other: Point) -> int:
        return self.x * other.y - self.y * other.x

    def __str__(self):
        return f"({self.x}, {self.y})"


class Image:
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

    def __getitem__(self, key) -> int:
        i, j = key
        assert i >= 0 and j >= 0 and i < self.h and j < self.w
        return self.mask[i * self.w + j]

    def __setitem__(self, key, value: int):
        i, j = key
        assert i >= 0 and j >= 0 and i < self.h and j < self.w
        self.mask[i * self.w + j] = value

    def safe(self, i: int, j: int) -> int:
        return (
            0
            if i < 0 or j < 0 or i >= self.h or j >= self.w
            else self.mask[i * self.w + j]
        )

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

    def clone(self) -> Image:
        return Image(self.p, self.sz, self.mask.copy())


badImg = Image((0, 0), (0, 0), [])
dummyImg = Image((0, 0), (1, 1), [0])


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
        if type(args[1]) == int:
            p = Point(0, 0)
            sz = Point(args[0])
            filling = args[1]
        else:
            p = Point(args[0])
            sz = Point(args[1])
    else:
        raise ValueError("Invalid numer of arguments")
    return Image(p, sz, (filling for _ in range(sz.x * sz.y)))


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


def colMask(img: Image) -> int:
    mask = 0
    for i in range(img.h):
        for j in range(img.w):
            mask |= 1 << img[(i, j)]
    return mask


def countCols(img: Image) -> int:
    return len(set(img.mask))


def count(img: Image) -> int:
    ans = 0
    for i in range(img.h):
        for j in range(img.w):
            ans += img[(i, j)] > 0
    return ans


def isRectangle(a: Image) -> bool:
    return count(a) == a.w * a.h


# def countComponents_dfs(Image&img, int r, int c) {
#     img(r,c) = 0;
#     for (int nr = r-1; nr <= r+1; nr++)
#       for (int nc = c-1; nc <= c+1; nc++)
# 	if (nr >= 0 && nr < img.h && nc >= 0 && nc < img.w && img(nr,nc))
# 	  countComponents_dfs(img,nr,nc);
#   }


def subImage(img: Image, p: Point, sz: Point) -> Image:
    assert (
        p.x >= 0
        and p.y >= 0
        and p.x + sz.x <= img.w
        and p.y + sz.y <= img.h
        and sz.x >= 0
        and sz.y >= 0
    )
    return Image(
        img.p + p,
        sz,
        [img[(i + p.y, j + p.x)] for i in range(sz.y) for j in range(sz.x)],
    )


def majorityCol(img: Image, include0: int = 0) -> int:
    cnt = [0 for _ in range(10)]

    for i in range(img.h):
        for j in range(img.w):
            c = img[(i, j)]
            if c >= 0 and c < 10:
                cnt[c] += 1

    if include0 == 0:
        cnt[0] = 0

    ret = 0
    ma = cnt[ret]
    for c in range(1, 10):
        if cnt[c] > ma:
            ma = cnt[c]
            ret = c

    return ret


def splitCols(img: img, include0=False) -> List[Image]:
    ret = []
    mask = colMask(img)
    for c in range(int(include0), 10):
        if mask >> c & 1:
            ret.append(Image(img.p, img.sz, [int(x == c) for x in img.mask]))
    return ret


def Col(id: int) -> Image:
    assert id >= 0 and id < 10
    return full((0, 0), (1, 1), id)


def Pos(dx: int, dy: int) -> Image:
    return full((dx, dy), (1, 1))


def Square(id: int) -> Image:
    assert id >= 1
    return full((0, 0), (id, id))


def Line(orient: int, id: int) -> Image:
    assert id >= 1
    if orient != 0:
        w, h = id, 1
    else:
        w, h = 1, id
    return full((0, 0), (w, h))


def getPos(img: Image) -> Image:
    return full(img.p, Point(1, 1), majorityCol(img))


def getSize(img: Image) -> Image:
    return full(Point(0, 0), img.sz, majorityCol(img))


def hull(img: Image) -> Image:
    return full(img.p, img.sz, majorityCol(img))


def toOrigin(img: Image) -> Image:
    return Image(Point(0, 0), img.sz, img.mask)


def getW(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(img.w, 1 if id == 0 else img.w), majorityCol(img))


def getH(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(1 if id == 0 else img.h, img.h), majorityCol(img))


def hull0(img: Image) -> Image:
    return full(img.p, img.sz, 0)


def getSize0(img: Image) -> Image:
    return full(Point(0, 0), img.sz, 0)


def Move(img: Image, p: Image) -> Image:
    return Image(img.p + p.p, img.sz, img.mask)


def invert(img: Image) -> Image:
    if img.w * img.h == 0:
        return img

    mask = colMask(img)
    col = 1
    while col < 10 and (mask >> col & 1) == 0:
        col += 1
    if col == 10:
        col = 1

    return img.map(lambda i, j, x: 0 if x != 0 else col)


@overload
def filterCol(img: Image, palette: Image) -> Image:
    ...


@overload
def filterCol(img: Image, id: int) -> Image:
    ...


def filterCol(*args) -> Image:
    if type(args[1]) == Image:
        img, palette = args
        col_mask = colMask(palette)
        filtered = [x if (col_mask >> x) & 1 else 0 for x in img.mask]
        return Image(img.p, img.sz, filtered)
    else:
        img, id = args
        assert id >= 0 and id < 10
        if id == 0:
            return invert(img)
        else:
            return filterCol(img, Col(id))
