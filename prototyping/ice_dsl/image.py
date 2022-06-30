from __future__ import annotations
from email.mime import image
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

    def astuple(self):
        return self.x, self.y

    def __iter__(self):
        return iter(self.astuple())

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y

        raise IndexError("Index out of range.")


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
    return count(a) == a.area


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


def splitCols(img: Image, include0: bool = False) -> List[Image]:
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


def getSize0(img: Image) -> Image:
    return full(Point(0, 0), img.sz, 0)


def hull(img: Image) -> Image:
    return full(img.p, img.sz, majorityCol(img))


def hull0(img: Image) -> Image:
    return full(img.p, img.sz, 0)


def toOrigin(img: Image) -> Image:
    return Image(Point(0, 0), img.sz, img.mask)


def getW(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(img.w, 1 if id == 0 else img.w), majorityCol(img))


def getH(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(1 if id == 0 else img.h, img.h), majorityCol(img))


def Move(img: Image, p: Image) -> Image:
    return Image(img.p + p.p, img.sz, img.mask)


def invert(img: Image) -> Image:
    if img.area == 0:
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


def broadcast(col: Image, shape: Image, include0: bool = True):
    """Return a resized version of col that matches the size of shape"""
    if shape.area == 0 or col.area == 0:
        return badImg

    if shape.w % col.w == 0 and shape.h % col.h == 0:
        ret = shape.clone()
        dh, dw = shape.h // col.h, shape.w // col.w

        for ii in range(col.h):
            for jj in range(col.w):
                c = col[ii, jj]
                for i in range(ii * dh, ii * dh + dh):
                    for j in range(jj * dw, jj * dw + dw):
                        ret[i, j] = c
        return ret

    # AKo: Should probably replaced by proper resampling code

    ret = shape.clone()
    fh, fw = col.h / shape.h, col.w / shape.w

    eps = 1e-9
    w0 = [0.0 for _ in range(10)]
    for c in col.mask:
        w0[c] += 1e-6

    tot = fh * fw

    for i in range(shape.h):
        for j in range(shape.w):
            weight = w0.copy()

            r0 = i * fh + eps
            r1 = (i + 1) * fh - eps
            c0 = j * fw + eps
            c1 = (j + 1) * fw - eps

            guess = int(not include0)
            y = int(r0)
            while y < r1:
                # wy = min(y + 1, r1) - max(y, r0)
                wy = r1 - r0

                x = int(c0)
                while x < c1:
                    # wx = min(x + 1, c1) - max(x, c0)
                    wx = c1 - c0
                    c = col[y, x]
                    weight[c] += wx * wy
                    guess = c

                    x += 1

                y += 1

            if weight[guess] * 2 > tot:
                ret[i, j] = guess
                continue

            maj = int(not include0)
            w = weight[maj]
            for c in range(1, 10):
                if weight[c] > w:
                    maj = c
                    w = weight[c]

            ret[i, j] = maj

    return ret


def colShape(col: Image, shape: Image) -> Image:
    """Return a resized copy of col with the size of shape with all pixels set to 0 that where 0 in shape"""
    if shape.area == 0 or col.area == 0:
        return badImg
    ret = broadcast(col, getSize(shape))
    ret.p = shape.p
    for i in range(ret.h):
        for j in range(ret.h):
            if shape[i, j] == 0:
                ret[i, j] = 0
    return ret


def colShape(shape: Image, id: int) -> Image:
    """Set any non-zero pixel in image to the color specified by id."""
    assert id >= 0 and id < 10
    return Image(shape.p, shape.sz, [id if x != 0 else 0 for x in shape.mask])


def compress(img: Image, bg: Image = Col(0)):
    """Remove all border columns and rows which contain only colors found in bg."""
    bg_mask = colMask(bg)

    xmi, xma, ymi, yma = 1e9, 0, 1e9, 0
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
    for i in range(ymi, yma):
        for j in range(xmi, xma):
            ret[i - ymi, j - xmi] = img[i, j]

    return ret


def center(img: Image) -> Image:
    sz = Point((img.w + 1) % 2 + 1, (img.h + 1) % 2 + 1)
    return full(img.p + (img.sz - sz) / 2, sz)
