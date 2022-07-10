from __future__ import annotations
from typing import Callable, Iterable, Tuple, List, overload
from functools import cmp_to_key
import numpy as np
from arc.interface import Board


MAXSIDE = 100
MAXAREA = 40 * 40
MAXPIXELS = 40 * 40 * 5


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
        return Image(Point(), Point(board.num_rows, board.num_cols), board.flat)

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


def isRectangle(a: Image) -> bool:
    return count_nonzero(a) == a.area


def clear_dfs(img: Image, r: int, c: int) -> None:
    """clear pixel at r,c and blob of non-zero adjacent pixels"""
    img[r, c] = 0
    for nr in (r - 1, r, r + 1):
        for nc in (c - 1, c, c + 1):
            if nr >= 0 and nr < img.h and nc >= 0 and nc < img.w and img[nr, nc] != 0:
                clear_dfs(img, nr, nc)


def count_components(img: Image) -> int:
    """Count number of"""
    img = img.copy()
    ans = 0
    for i in range(img.h):
        for j in range(img.w):

            if img[i, j] != 0:
                clear_dfs(img, i, j)
                ans += 1
    return ans


def sub_image(img: Image, p: Point, sz: Point) -> Image:
    assert p.x >= 0 and p.y >= 0 and p.x + sz.x <= img.w and p.y + sz.y <= img.h and sz.x >= 0 and sz.y >= 0
    return Image(
        img.p + p,
        sz,
        [img[(i + p.y, j + p.x)] for i in range(sz.y) for j in range(sz.x)],
    )


def majority_color(img: Image, include0: bool = False) -> int:
    cnt = [0] * 10

    for i in range(img.h):
        for j in range(img.w):
            c = img[(i, j)]
            if c >= 0 and c < 10:
                cnt[c] += 1

    if not include0:
        cnt[0] = 0

    ret = 0
    ma = cnt[ret]
    for c in range(1, 10):
        if cnt[c] > ma:
            ma = cnt[c]
            ret = c

    return ret


def split_colors(img: Image, include0: bool = False) -> List[Image]:
    ret = []
    mask = color_mask(img)
    for c in range(int(include0), 10):
        if (mask >> c) & 1:
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


def get_pos(img: Image) -> Image:
    return full(img.p, Point(1, 1), majority_color(img))


def get_size(img: Image) -> Image:
    return full(Point(0, 0), img.sz, majority_color(img))


def get_size0(img: Image) -> Image:
    return full(Point(0, 0), img.sz, 0)


def hull(img: Image) -> Image:
    return full(img.p, img.sz, majority_color(img))


def hull0(img: Image) -> Image:
    return full(img.p, img.sz, 0)


def to_origin(img: Image) -> Image:
    return Image(Point(0, 0), img.sz, img.mask)


def get_Width(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(img.w, 1 if id == 0 else img.w), majority_color(img))


def get_Height(img: Image, id: int) -> Image:
    return full(Point(0, 0), Point(1 if id == 0 else img.h, img.h), majority_color(img))


def move(img: Image, p: Image) -> Image:
    return Image(img.p + p.p, img.sz, img.mask)


def invert(img: Image) -> Image:
    if img.area == 0:
        return img

    mask = color_mask(img)
    col = 1
    while col < 10 and (mask >> col & 1) == 0:
        col += 1
    if col == 10:
        col = 1

    return img.map(lambda i, j, x: 0 if x != 0 else col)


def filter_color_palette(img: Image, palette: Image) -> Image:
    col_mask = color_mask(palette)
    filtered = [x if (col_mask >> x) & 1 else 0 for x in img.mask]
    return Image(img.p, img.sz, filtered)


def filter_color(img: Image, id: int) -> Image:
    assert id >= 0 and id < 10
    if id == 0:
        return invert(img)
    else:
        return filter_color_palette(img, Col(id))


def broadcast(col: Image, shape: Image, include0: bool = True) -> Image:
    """Return a resized version of col that matches the size of shape"""
    if shape.area == 0 or col.area == 0:
        return badImg

    if shape.w % col.w == 0 and shape.h % col.h == 0:
        ret = shape.copy()
        dh, dw = shape.h // col.h, shape.w // col.w

        for ii in range(col.h):
            for jj in range(col.w):
                c = col[ii, jj]
                for i in range(ii * dh, ii * dh + dh):
                    for j in range(jj * dw, jj * dw + dw):
                        ret[i, j] = c
        return ret

    # AKo: Should probably be replaced by proper resampling code

    ret = shape.copy()
    fh, fw = col.h / shape.h, col.w / shape.w

    eps = 1e-9
    w0 = [0.0] * 10
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


def color_shape(col: Image, shape: Image) -> Image:
    """Return a resized copy of col with the size of shape with all pixels set to 0 that are 0 in shape"""
    if shape.area == 0 or col.area == 0:
        return badImg
    ret = broadcast(col, get_size(shape))
    ret.p = shape.p
    for i in range(ret.h):
        for j in range(ret.h):
            if shape[i, j] == 0:
                ret[i, j] = 0
    return ret


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

    if ret.w > MAXSIDE or ret.h > MAXSIDE or ret.area > MAXAREA:
        return badImg

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


def compose_list(imgs: List[Image], id: int) -> Image:
    if len(imgs) == 0:
        return badImg
    ret = imgs[0].copy()
    for i in range(1, len(imgs)):
        ret = compose(ret, imgs[i], id)
    return ret


def fill(a: Image) -> Image:
    """Fill internal areas enclosed by non-zero borders with majority color."""
    ret = full(a.p, a.sz, majority_color(a, False))

    q = []
    for i in range(a.h):
        for j in range(a.w):
            if (i == 0 or j == 0 or i == a.h - 1 or j == a.w - 1) and a[i, j] == 0:
                q.append((i, j))
                ret[i, j] = 0

    while len(q) > 0:
        r, c = q.pop()
        for d in range(4):
            nr = r + int(d == 2) - int(d == 3)
            nc = c + int(d == 0) - int(d == 1)
            if nr >= 0 and nr < a.h and nc >= 0 and nc < a.w and a[nr, nc] == 0 and ret[nr, nc] != 0:
                q.append((nr, nc))
                ret[nr, nc] = 0

    return ret


def interior(a: Image) -> Image:
    return compose_internal(fill(a), a, lambda x, y: 0 if y != 0 else x, 0)


def border(a: Image) -> Image:
    ret = empty(a.p, a.sz)
    q = []

    for i in range(a.h):
        for j in range(a.w):
            if i == 0 or j == 0 or i == a.h - 1 or j == a.w - 1:
                if a[i, j] == 0:
                    q.append((i, j))
                ret[i, j] = 1

    def DO(nr, nc):
        if ret[nr, nc] == 0:
            ret[nr, nc] = 1
            if a[nr, nc] == 0:
                q.append((nr, nc))

    while len(q) > 0:
        r, c = q.pop()

        if r > 0:
            if c > 0:
                DO(r - 1, c - 1)
            DO(r - 1, c)
            if c + 1 < a.w:
                DO(r - 1, c + 1)

        if r + 1 < a.h:
            if c > 0:
                DO(r + 1, c - 1)
            DO(r + 1, c)
            if c + 1 < a.w:
                DO(r + 1, c + 1)

        if c > 0:
            DO(r, c - 1)
        if c + 1 < a.w:
            DO(r, c + 1)

    for i in range(len(a.mask)):
        ret.mask[i] = ret.mask[i] * a.mask[i]
    return ret


def alignx(a: Image, b: Image, id) -> Image:
    assert id >= 0 and id < 5
    ret = a.copy()
    if id == 0:
        ret.x = b.x - a.w
    elif id == 1:
        ret.x = b.x
    elif id == 2:
        ret.x = b.x + (b.w - a.w) / 2
    elif id == 3:
        ret.x = b.x + b.w - a.w
    elif id == 4:
        ret.x = b.x + b.w
    return ret


def aligny(a: Image, b: Image, id: int) -> Image:
    assert id >= 0 and id < 5
    ret = a.copy()
    if id == 0:
        ret.y = b.y - a.h
    elif id == 1:
        ret.y = b.y
    elif id == 2:
        ret.y = b.y + (b.h - a.h) / 2
    elif id == 3:
        ret.y = b.y + b.h - a.h
    elif id == 4:
        ret.y = b.y + b.h
    return ret


def align(a: Image, b: Image, idx: int, idy: int) -> Image:
    assert idx >= 0 and idx < 6
    assert idy >= 0 and idy < 6
    ret = a.copy()
    if idx == 0:
        ret.x = b.x - a.w
    elif idx == 1:
        ret.x = b.x
    elif idx == 2:
        ret.x = b.x + (b.w - a.w) / 2
    elif idx == 3:
        ret.x = b.x + b.w - a.w
    elif idx == 4:
        ret.x = b.x + b.w
    if idy == 0:
        ret.y = b.y - a.h
    elif idy == 1:
        ret.y = b.y
    elif idy == 2:
        ret.y = b.y + (b.h - a.h) / 2
    elif idy == 3:
        ret.y = b.y + b.h - a.h
    elif idy == 4:
        ret.y = b.y + b.h
    return ret


def align_color(a: Image, b: Image) -> Image:
    """Find most matching color and align a to b using it."""
    ret = a.copy()
    match_size = 0
    for c in range(1, 10):
        ca = compress(filter_color(a, c))
        cb = compress(filter_color(b, c))
        if ca.mask == cb.mask:
            cnt = count_nonzero(ca)
            if cnt > match_size:
                match_size = cnt
                ret.p = a.p + cb.p - ca.p
    if match_size == 0:
        return badImg
    return ret


def interior2(a: Image) -> Image:
    return compose(a, invert(border(a)), 2)


def embed(img: Image, shape: Image) -> Image:
    ret = empty(shape.p, shape.sz)
    d = shape.p - img.p
    sx = max(0, -d.x)
    sy = max(0, -d.y)
    ex = min(ret.w, img.w - d.x)
    ey = min(ret.h, img.h - d.y)

    retw = ret.w
    imgw = img.w
    off = d.y * img.w + d.x
    for i in range(sy, ey):
        for j in range(sx, ex):
            ret.mask[i * retw + j] = img.mask[i * imgw + j + off]

    return ret


def center(img: Image) -> Image:
    sz = Point((img.w + 1) % 2 + 1, (img.h + 1) % 2 + 1)
    return full(img.p + (img.sz - sz) // 2, sz)


def transform(img: Image, A00: int, A01: int, A10: int, A11: int) -> Image:
    if img.area == 0:
        return img

    c = center(img)
    off = Point(1 - c.w, 1 - c.h) + (img.p - c.p) * 2

    def t(p: Point) -> Point:
        p = p * 2 + off
        p = Point(A00 * p.x + A01 * p.y, A10 * p.x + A11 * p.y)
        p = p - off
        p = p // 2
        return p

    corners = [
        t(Point(0, 0)),
        t(Point(img.w - 1, 0)),
        t(Point(0, img.h - 1)),
        t(Point(img.w - 1, img.h - 1)),
    ]
    a = corners[0]
    b = corners[0]

    for c in corners:
        a = Point(min(a.x, c.x), min(a.y, c.y))
        b = Point(max(b.x, c.x), max(b.y, c.y))

    ret = empty(img.p, b - a + Point(1, 1))

    for i in range(img.h):
        for j in range(img.w):
            go = t(Point(j, i)) - a
            ret[go.y, go.x] = img[i, j]

    return ret


def mirror_heuristic(img: Image) -> int:
    # Meant to be used for mirroring, flip either x or y, depending on center of gravity
    cnt, sumx, sumy = 0, 0, 0
    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] != 0:
                cnt += 1
                sumx += j
                sumy += i

    return abs(sumx * 2 - (img.w - 1) * cnt) < abs(sumy * 2 - (img.h - 1) * cnt)


def rigid(img: Image, id: int) -> Image:
    if id == 0:
        return img
    elif id == 1:
        return transform(img, 0, 1, -1, 0)
        # CCW
    elif id == 2:
        return transform(img, -1, 0, 0, -1)
        # 180
    elif id == 3:
        return transform(img, 0, -1, 1, 0)
        # CW
    elif id == 4:
        return transform(img, -1, 0, 0, 1)
        # flip x
    elif id == 5:
        return transform(img, 1, 0, 0, -1)
        # flip y
    elif id == 6:
        return transform(img, 0, 1, 1, 0)
        # swap xy
    elif id == 7:
        return transform(img, 0, -1, -1, 0)
        # swap other diagonal
    elif id == 8:
        return rigid(img, 4 + mirror_heuristic(img))
    else:
        assert id >= 0 and id < 9
        return badImg


def get_regular_internal(col: List):
    colw = len(col)

    for w in range(1, colw):
        s = -1
        if colw % (w + 1) == w:  # No outer border
            s = w
        elif colw % (w + 1) == 1:  # Outer border
            s = 0

        if s != -1:
            ok = True
            for i in range(colw):
                if col[i] != (i % (w + 1) == s):
                    ok = False
                    break
            if ok:
                return

    for i in range(len(col)):
        col[i] = False


def get_regular(img: Image) -> Image:
    """Look for regular grid division in single color"""
    ret = img.copy()

    col = [True] * img.w
    row = [True] * img.h
    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] != img[i, 0]:
                row[i] = False
            if img[i, j] != img[0, j]:
                col[j] = False

    get_regular_internal(col)
    get_regular_internal(row)
    for i in range(img.h):
        for j in range(img.w):
            ret[i, j] = 1 if row[i] or col[j] else 0

    return ret


def count(img: Image, id: int, out_type: int) -> Image:
    assert id >= 0 and id < 7
    assert out_type >= 0 and out_type < 3

    if id == 0:
        num = count_nonzero(img)
    elif id == 1:
        num = count_colors(img)
    elif id == 2:
        num = count_components(img)
    elif id == 3:
        num = img.w
    elif id == 4:
        num = img.h
    elif id == 5:
        num = max(img.w, img.h)
    elif id == 6:
        num = min(img.w, img.h)
    else:
        assert False

    if out_type == 0:
        sz = Point(num, num)
    elif out_type == 1:
        sz = Point(num, 1)
    elif out_type == 2:
        sz = Point(1, num)
    else:
        raise ValueError("Unsupported output type")

    if max(sz.x, sz.y) > MAXSIDE or sz.x * sz.y > MAXAREA:
        return badImg
    return full(sz, majority_color(img))


def gravity(img: Image, d: int) -> List[Image]:
    pieces = split_all(img)
    out = hull0(img)

    # 0: (1,0), 1: (-1,0), 2: (0,1), 3: (0,-1)  (right, left, down, up)
    dx = int(d == 0) - int(d == 1)
    dy = int(d == 2) - int(d == 3)

    ret = []
    pieces.sort(key=cmp_to_key(lambda a, b: (a.x - b.x) * dx + (a.y - b.y) * dy), reverse=True)
    for p in pieces:

        def move():
            while True:
                p.x += dx
                p.y += dy
                for i in range(p.h):
                    for j in range(p.w):

                        if p[i, j] == 0:
                            continue

                        x = j + p.x - out.x
                        y = i + p.y - out.y

                        if x < 0 or y < 0 or x >= out.w or y >= out.h or out[y, x] != 0:
                            p.x -= dx
                            p.y -= dy
                            return

        move()
        ret.append(p)
        out = compose(out, p, 3)

    return ret


def my_stack(a: Image, b: Image, orient: int) -> Image:
    assert orient >= 0 and orient <= 3
    b = Image(a.p, b.sz, b.mask)
    if orient == 0:  # Horizontal
        b.x += a.w
    elif orient == 1:  # Vertical
        b.y += a.h
    elif orient == 2:  # Diagonal
        b.x += a.w
        b.y += a.h
    else:
        # Other diagonal, bottom-left / top-right
        c = a.copy()
        c.y += b.h
        b.x += a.w
        return compose(c, b)

    return compose(a, b)


def my_stack_list(lens: List[Image], orient: int) -> Image:
    """stack images sorted based on area"""
    n = len(lens)
    if n == 0:
        return badImg
    order = sorted([(lens[i].area, i) for i in range(n)])
    ret = lens[order[0][1]].copy()
    for i in range(1, n):
        ret = my_stack(ret, lens[order[i][1]], orient)
    return ret


def wrap(line: Image, area: Image) -> Image:
    if line.area == 0 or area.area == 0:
        return badImg

    ans = empty(area.sz)
    for i in range(line.h):
        for j in range(line.w):
            x, y = j, i

            x += y // area.h * line.w
            y %= area.h

            y += x // area.w * line.h
            x %= area.w

            if x >= 0 and y >= 0 and x < ans.w and y < ans.h:
                ans[y, x] = line[i, j]

    return ans


def smear(img: Image, id: int) -> Image:
    assert id >= 0 and id < 15

    R, L, D, U = (1, 0), (-1, 0), (0, 1), (0, -1)
    X, Y, Z, W = (1, 1), (-1, -1), (1, -1), (-1, 1)
    d = [
        [R],
        [L],
        [D],
        [U],
        [R, L],
        [D, U],
        [R, L, D, U],
        [X],
        [Y],
        [Z],
        [W],
        [X, Y],
        [Z, W],
        [X, Y, Z, W],
        [R, L, D, U, X, Y, Z, W],
    ]
    w = img.w
    ret = img.copy()

    for dx, dy in d[id]:
        di = dy * w + dx

        for i in range(ret.h):
            step = 1 if i == 0 or i == ret.h - 1 else max(ret.w - 1, 1)
            for j in range(0, ret.w, step):

                if i - dy < 0 or j - dx < 0 or i - dy >= img.h or j - dx >= img.w:
                    steps = MAXSIDE

                    if dx == -1:
                        steps = min(steps, j + 1)
                    if dx == 1:
                        steps = min(steps, img.w - j)
                    if dy == -1:
                        steps = min(steps, i + 1)
                    if dy == 1:
                        steps = min(steps, img.h - i)

                    ind = i * w + j
                    end_ind = ind + steps * di
                    c = 0
                    for ind in range(ind, end_ind, di):
                        if img.mask[ind]:
                            c = img.mask[ind]
                        if c != 0:
                            ret.mask[ind] = c

    return ret


"""
def smear(base: Image, room: Image, id: int) -> Image:
    assert id >= 0 and id < 7

    arr = [1, 2, 4, 8, 3, 12, 15]
    mask = arr[id]

    d = room.p - base.p

    ret = embed(base, hull(room))
    if mask & 1:
        for i in range(ret.h):
            c = 0
            for j in range(ret.w):
                if room[i, j] == 0:
                    c = 0
                elif base.safe(i + d.y, j + d.x) != 0:
                    c = base[i + d.y, j + d.x]
                if c != 0:
                    ret[i, j] = c

    if (mask >> 1) & 1:
        for i in range(ret.h):
            c = 0
            for j in range(ret.w - 1, -1, -1):
                if room[i, j] == 0:
                    c = 0
                elif base.safe(i + d.y, j + d.x) != 0:
                    c = base[i + d.y, j + d.x]
                if c != 0:
                    ret[i, j] = c

    if (mask >> 2) & 1:
        for j in range(ret.w):
            c = 0
            for i in range(ret.h):
                if room[i, j] == 0:
                    c = 0
                elif base.safe(i + d.y, j + d.x) != 0:
                    c = base[i + d.y, j + d.x]
                if c != 0:
                    ret[i, j] = c

    if (mask >> 3) & 1:
        for j in range(ret.w):
            c = 0
            for i in range(ret.h - 1, -1, -1):
                if room[i, j] == 0:
                    c = 0
                elif base.safe(i + d.y, j + d.x) != 0:
                    c = base[i + d.y, j + d.x]
                if c != 0:
                    ret[i, j] = c

    return ret
"""


def clamp(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x


def extend(img: Image, room: Image) -> Image:
    if img.area == 0:
        return badImg
    ret = room.copy()
    for i in range(ret.h):
        for j in range(ret.w):
            p = Point(j, i) + room.p - img.p
            p.x = clamp(p.x, 0, img.w - 1)
            p.y = clamp(p.y, 0, img.h - 1)
            ret[i, j] = img[p.y, p.x]

    return ret


def pick_max_internal(v: List[Image], f: Callable[[Image], int]) -> Image:
    if len(v) == 0:
        return badImg
    return max(v, key=f)


def max_criterion(img: Image, id: int) -> int:
    assert id >= 0 and id < 14

    if id == 0:
        return count_nonzero(img)
    elif id == 1:
        return -count_nonzero(img)
    elif id == 2:
        return img.w * img.h
    elif id == 3:
        return -img.w * img.h
    elif id == 4:
        return count_colors(img)
    elif id == 5:
        return -img.p.y
    elif id == 6:
        return img.p.y
    elif id == 7:
        return count_components(img)
    elif id == 8:
        comp = compress(img)
        return comp.w * comp.h - count_nonzero(comp)
    elif id == 9:
        comp = compress(img)
        return -(comp.w * comp.h - count_nonzero(comp))
    elif id == 10:
        return count_nonzero(interior(img))
    elif id == 11:
        return -count_nonzero(interior(img))
    elif id == 12:
        return -img.p.x
    elif id == 13:
        return img.p.x

    return -1


def pick_max(v: List[Image], id: int) -> Image:
    return pick_max_internal(v, lambda img: max_criterion(img, id))


def cut(img: Image, a: Image) -> List[Image]:
    ret = []
    done = empty(img.p, img.sz)
    d = img.p - a.p

    for i in range(img.h):
        for j in range(img.w):
            if done[i, j] == 0 and a.safe(i + d.y, j + d.x) == 0:
                toadd = empty(img.p, img.sz)

                def dfs(r: int, c: int) -> None:
                    if r < 0 or r >= img.h or c < 0 or c >= img.w or a.safe(r + d.y, c + d.x) != 0 or done[r, c] != 0:
                        return
                    toadd[r, c] = img[r, c] + 1
                    done[r, c] = 1
                    for nr in (r - 1, r, r + 1):
                        for nc in (c - 1, c, c + 1):
                            dfs(nr, nc)

                dfs(i, j)
                toadd = compress(toadd)
                for i in range(toadd.h):
                    for j in range(toadd.w):
                        toadd[i, j] = max(0, toadd[i, j] - 1)

                ret.append(toadd)

    return ret


def outer_product_is(a: Image, b: Image) -> Image:
    if a.w * b.w > MAXSIDE or a.h * b.h > MAXSIDE or a.w * b.w * a.h * b.h > MAXAREA:
        return badImg
    rpos = Point(a.p.x * b.w + b.p.x, a.p.y * b.h + b.p.y)
    ret = empty(rpos, Point(a.w * b.w, a.h * b.h))
    for i in range(a.h):
        for j in range(a.w):
            for k in range(b.h):
                for l in range(b.w):
                    ret[i * b.h + k, j * b.w + l] = a[i, j] * int(b[k, l] != 0)
    return ret


def outer_product_si(a: Image, b: Image) -> Image:
    if a.w * b.w > MAXSIDE or a.h * b.h > MAXSIDE or a.w * b.w * a.h * b.h > MAXAREA:
        return badImg
    rpos = Point(a.x * b.w + b.x, a.y * b.h + b.y)
    ret = empty(rpos, Point(a.w * b.w, a.h * b.h))
    for i in range(a.h):
        for j in range(a.w):
            for k in range(b.h):
                for l in range(b.w):
                    ret[i * b.h + k, j * b.w + l] = int(a[i, j] != 0) * b[k, l]
    return ret


def replace_colors(base: Image, cols: Image) -> Image:
    ret = base.copy()
    done = empty(base.p, base.sz)
    d = base.p - cols.p

    for i in range(base.h):
        for j in range(base.w):
            if done[i, j] == 0 and base[i, j] != 0:
                acol = base[i, j]
                cnt = [0] * 10
                path = []

                def dfs(r: int, c: int):
                    if r < 0 or r >= base.h or c < 0 or c >= base.w or base[r, c] != acol or done[r, c] != 0:
                        return
                    cnt[cols.safe(r + d.y, c + d.x)] += 1
                    path.append((r, c))
                    done[r, c] = 1
                    for nr in (r - 1, r, r + 1):
                        for nc in (c - 1, c, c + 1):
                            dfs(nr, nc)

                dfs(i, j)
                maj = (0, 0)
                for c in range(1, 10):
                    if cnt[c] > maj[0]:
                        maj = (cnt[c], -c)

                for r, c in path:
                    ret[r, c] = -maj[1]

    return ret


def repeat(a: Image, b: Image, pad: int = 0) -> Image:
    """Fill b with repeated copies of a"""
    if a.area <= 0 or b.area <= 0:
        return badImg

    ret = empty(b.p, b.sz)
    W = a.w + pad
    H = a.h + pad
    ai = ((b.y - a.y) % H + H) % H
    aj0 = ((b.x - a.x) % W + W) % W
    for i in range(ret.h):
        aj = aj0
        for j in range(ret.w):
            if ai < a.h and aj < a.w:
                ret[i, j] = a[ai, aj]
            aj += 1
            if aj == W:
                aj = 0

        ai += 1
        if ai == H:
            ai = 0

    return ret


def mirror(a: Image, b: Image, pad: int = 0) -> Image:
    """Like repeat but with every 2nd mirrored"""
    if a.area <= 0 or b.area <= 0:
        return badImg
    ret = empty(b.p, b.sz)

    W, H = a.w + pad, a.h + pad
    W2, H2 = W * 2, H * 2
    ai = ((b.y - a.y) % H2 + H2) % H2
    aj0 = ((b.x - a.x) % W2 + W2) % W2
    for i in range(ret.h):
        aj = aj0
        for j in range(ret.w):
            x, y = -1, -1
            if aj < a.w:
                x = aj
            elif aj >= W and aj < W + a.w:
                x = W + a.w - 1 - aj
            if ai < a.h:
                y = ai
            elif ai >= H and ai < H + a.h:
                y = H + a.h - 1 - ai
            if x != -1 and y != -1:
                ret[i, j] = a[y, x]
            aj += 1
            if aj == W2:
                aj = 0
        ai += 1
        if ai == H2:
            ai = 0
    return ret


def majority_color_image(img: Image):
    return Col(majority_color(img))


def heuristic_cut(img: Image) -> Image:
    """
    Return a single color
    Cut into at least 2 pieces
    No nested pieces
    Must touch at least 2 opposite sides
    Smallest piece should be as big as possible
    """
    ret = majority_color(img, 1)
    ret_score = -1

    mask = color_mask(img)

    for col in range(10):
        if (mask >> col) & 1 == 0:
            continue

        done = empty(img.p, img.sz)

        def edgy(r: int, c: int) -> None:
            if r < 0 or r >= img.h or c < 0 or c >= img.w or img[r, c] != col or done[r, c] != 0:
                return
            done[r, c] = 1
            for nr in (r - 1, r, r + 1):
                for nc in (c - 1, c, c + 1):
                    edgy(nr, nc)

        top, bot, left, right = False, False, False, False
        for i in range(img.h):
            for j in range(img.w):
                if img[i, j] == col:
                    if i == 0:
                        top = True
                    if j == 0:
                        left = True
                    if i == img.h - 1:
                        bot = True
                    if j == img.w - 1:
                        right = True
                if (i == 0 or j == 0 or i == img.h - 1 or j == img.w - 1) and img[i, j] == col and done[i, j] == 0:
                    edgy(i, j)

        if not (top and bot or left and right):
            continue

        score = 1e9
        components = 0
        nocontained = 1
        for i in range(img.h):
            for j in range(img.w):
                cnt, contained = 0, 1
                if done[i, j] == 0 and img[i, j] != col:

                    def dfs(r: int, c: int):
                        nonlocal contained
                        nonlocal cnt
                        if r < 0 or r >= img.h or c < 0 or c >= img.w:
                            return
                        if img[r, c] == col:
                            if done[r, c] != 0:
                                contained = 0
                            return

                        if done[r, c] != 0:
                            return
                        cnt += 1
                        done[r, c] = 1
                        for nr in (r - 1, r, r + 1):
                            for nc in (c - 1, c, c + 1):
                                dfs(nr, nc)

                    dfs(i, j)
                    components += 1
                    score = min(score, cnt)
                    if contained:
                        nocontained = 0

        if components >= 2 and nocontained and score > ret_score:
            ret_score = score
            ret = col
    return filter_color(img, ret)


def cut_image(img: Image) -> List[Image]:
    return cut(img, heuristic_cut(img))


def cut_pick_max(a: Image, b: Image, id: int) -> Image:
    return pick_max(cut(a, b), id)


def regular_cut_pick_max(a: Image, id: int) -> Image:
    b = get_regular(a)
    return pick_max(cut(a, b), id)


def split_pick_max(a: Image, id: int, include0: bool = False) -> Image:
    return pick_max(split_colors(a, include0), id)


def cut_compose(a: Image, b: Image, id: int) -> Image:
    v = cut(a, b)
    v = [to_origin(img) for img in v]
    return compose_list(v, id)


def regular_cut_compose(a: Image, id: int) -> Image:
    b = get_regular(a)
    v = cut(a, b)
    v = [to_origin(img) for img in v]
    return compose_list(v, id)


def split_compose(a: Image, id: int, include0: bool = False) -> Image:
    v = split_colors(a, include0)
    v = [to_origin(compress(img)) for img in v]
    return compose(v, id)


def cut_index(a: Image, b: Image, ind: int) -> Image:
    v = cut(a, b)
    if ind < 0 or ind >= len(v):
        return badImg
    return v[ind]


def pick_maxes_internal(v: List[Image], f, invert: bool = False) -> List[Image]:
    n = len(v)
    if n == 0:
        return []
    score = [f(img) for img in v]
    ma = max(score)

    ret_imgs = []
    for i in range(n):
        if (score[i] == ma) ^ invert:
            ret_imgs.append(v[i])
    return ret_imgs


def pick_maxes(v: List[Image], id: int) -> List[Image]:
    return pick_maxes_internal(v, lambda img: max_criterion(img, id), False)


def pick_not_maxes(v: List[Image], id: int) -> List[Image]:
    return pick_maxes_internal(v, lambda img: max_criterion(img, id), True)


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


def erase_color(img: Image, col: int) -> Image:
    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] == col:
                img[i, j] = 0
    return img


def split_columns(img: Image) -> List[Image]:
    ret = []
    if img.area > 0:
        for j in range(img.w):
            ret.append(Image(Point(j, 0), Point(1, img.h), [img.mask[i, j] for i in range(img.h)]))

    return ret


def split_rows(img: Image) -> List[Image]:
    ret = []
    if img.area > 0:
        for i in range(img.h):
            ret.append(Image(Point(0, i), Point(img.w, 1), [img.mask[i, j] for j in range(img.w)]))
    return ret


def half(img: Image, id: int) -> Image:
    assert id >= 0 and id < 4
    if id == 0:
        return sub_image(img, Point(0, 0), Point(img.w / 2, img.h))
    elif id == 1:
        return sub_image(img, Point(img.w - img.w / 2, 0), Point(img.w / 2, img.h))
    elif id == 2:
        return sub_image(img, Point(0, 0), Point(img.w, img.h / 2))
    elif id == 3:
        return sub_image(img, Point(0, img.h - img.h / 2), Point(img.w, img.h / 2))
    else:
        return badImg


def mirror2(a: Image, line: Image) -> Image:
    if line.w > line.h:
        ret = rigid(a, 5)
        ret.x = a.x
        ret.y = line.y * 2 + line.h - a.y - a.h
    else:
        ret = rigid(a, 4)
        ret.y = a.y
        ret.x = line.x * 2 + line.w - a.x - a.w

    return ret


def inside_marked(img: Image) -> List[Image]:
    """Looks for 4 corners"""
    ret = []
    for i in range(img.h - 1):
        for j in range(img.w - 1):
            for h in range(1, img.h - 1 - i, 1):
                for w in range(1, img.w - 1 - j, 1):
                    col = img[i, j]
                    if col == 0:
                        continue

                    def check():
                        test = True
                        for k in range(4):
                            x = j + (k % 2) * w
                            y = i + (k // 2) * h
                            for d in range(4):
                                if not (d == k) == (img[y + d // 2, x + d % 2] == col):
                                    return False
                        return True

                    if check():
                        inside = invert(full(Point(j + 1, i + 1), Point(w, h)))
                        ret.append(compose(inside, img, 3))

    return ret


def make_border(img: Image, bcol: int = 1) -> Image:
    ret = hull0(img)
    for i in range(ret.h):
        for j in range(ret.w):
            if img[i, j] == 0:
                ok = False
                for ni in (i - 1, i, i + 1):
                    for nj in (j - 1, j, j + 1):

                        if img.safe(ni, nj) != 0:
                            ok = True
                            break
                if ok:
                    ret[i, j] = bcol

    return ret


def make_border2(img: Image, usemaj: bool = True) -> Image:
    bcol = 1
    if usemaj:
        bcol = majority_color(img)

    rsz = img.sz + Point(2, 2)
    if max(rsz.x, rsz.y) > MAXSIDE or rsz.x * rsz.y > MAXAREA:
        return badImg

    ret = full(img.p - Point(1, 1), rsz, bcol)
    for i in range(img.h):
        for j in range(img.w):
            ret[i + 1, j + 1] = img[i, j]
    return ret


def compress2(img: Image) -> Image:
    """Delete black rows / cols"""
    row = [0] * img.h
    col = [0] * img.w
    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] != 0:
                row[i] = col[j] = 1
    rows = [i for i in range(img.h) if row[i] != 0]
    cols = [j for j in range(img.w) if col[j] != 0]

    ret = empty(Point(len(cols), len(rows)))
    for i in range(ret.h):
        for j in range(ret.w):
            ret[i, j] = img[rows[i], cols[j]]
    return ret


def compress3(img: Image) -> Image:
    """Group single color rectangles"""
    if img.area <= 0:
        return badImg
    row = [0] * img.h
    col = [0] * img.w

    row[0] = col[0] = 1

    for i in range(img.h):
        for j in range(img.w):
            if i > 0 and img[i, j] != img[i - 1, j]:
                row[i] = 1
            if j > 0 and img[i, j] != img[i, j - 1]:
                col[j] = 1

    rows = [i for i in range(img.h) if row[i] != 0]
    cols = [j for j in range(img.w) if col[j] != 0]
    ret = empty(Point(len(cols), len(rows)))
    for i in range(ret.h):
        for j in range(ret.w):
            ret[i, j] = img[rows[i], cols[j]]
    return ret


def connect(img: Image, id: int) -> Image:
    """Zero pixels between two pixels of same color are filled. In the returned image only the infill pixels are set."""
    assert id >= 0 and id < 3
    ret = empty(img.p, img.sz)

    # horizontal
    if id == 0 or id == 2:
        for i in range(img.h):
            last, lastc = -1, -1
            for j in range(img.w):
                if img[i, j] != 0:
                    if img[i, j] == lastc:
                        for k in range(last + 1, j, 1):
                            ret[i, k] = lastc

                    lastc = img[i, j]
                    last = j

    # vertical
    if id == 1 or id == 2:
        for j in range(img.w):
            last, lastc = -1, -1
            for i in range(img.h):
                if img[i, j] != 0:
                    if img[i, j] == lastc:
                        for k in range(last + 1, i, 1):
                            ret[k, j] = lastc
                    lastc = img[i, j]
                    last = i

    return ret


def spread_colors(img: Image, skipmaj: bool = False) -> Image:
    skipcol = -1
    if skipmaj:
        skipcol = majority_color(img)

    done = hull0(img)
    q = []

    for i in range(img.h):
        for j in range(img.w):
            if img[i, j] != 0:
                if img[i, j] != skipcol:
                    q.append((j, i, img[i, j]))
                    done[i, j] = 1

    while len(q) > 0:
        j, i, c = q.pop(0)
        for d in range(4):
            ni = i + (d == 0) - (d == 1)
            nj = j + (d == 2) - (d == 3)
            if ni >= 0 and nj >= 0 and ni < img.h and nj < img.w and done[ni, nj] == 0:
                img[ni, nj] = c
                done[ni, nj] = 1
                q.append((nj, ni, c))

    return img


def stack_line(shapes: List[Image]) -> Image:
    n = len(shapes)
    if n == 0:
        return badImg
    elif n == 1:
        return shapes[0]

    xs = sorted([shapes[i].x for i in range(n)])
    ys = sorted([shapes[i].y for i in range(n)])

    xmin, ymin = 1e9, 1e9
    for i in range(1, n):
        xmin = min(xmin, xs[i] - xs[i - 1])
        ymin = min(ymin, ys[i] - ys[i - 1])

    dx, dy = 1, 0
    if xmin < ymin:
        dx, dy = 0, 1

    order = sorted([(shapes[i].x * dx + shapes[i].y * dy, i) for i in range(n)])
    out = shapes[order[0][1]].copy()
    for i in range(1, n):
        out = my_stack(out, shapes[order[i][1]], dy)
    return out


def compose_growing(imgs: List[Image]) -> Image:
    n = len(imgs)
    if n == 0:
        return badImg

    order = sorted([(count_nonzero(imgs[i]), i) for i in range(n)])
    ret = imgs[order[0][1]].copy()
    for i in range(1, n):
        ret = compose(ret, imgs[order[i][1]], 0)
    return ret


def pick_unique(imgs: List[Image]) -> Image:
    """Pick the one with the unique color"""
    n = len(imgs)
    if n == 0:
        return badImg

    mask = [color_mask(imgs[i]) for i in range(n)]
    cnt = [0] * 10
    for i in range(n):
        for c in range(10):
            if (mask[i] >> c) & 1 != 0:
                cnt[c] += 1

    reti = -1
    for i in range(n):
        for c in range(10):
            if (mask[i] >> c) & 1 != 0:
                if cnt[c] == 1:
                    if reti == -1:
                        reti = i
                    else:
                        return badImg

    if reti == -1:
        return badImg
    return imgs[reti]
