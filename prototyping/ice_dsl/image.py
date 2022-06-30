from __future__ import annotations
from typing import Callable, Iterable, Tuple, overload, List

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
    assert p.x >= 0 and p.y >= 0 and p.x + sz.x <= img.w and p.y + sz.y <= img.h and sz.x >= 0 and sz.y >= 0
    return Image(
        img.p + p,
        sz,
        [img[(i + p.y, j + p.x)] for i in range(sz.y) for j in range(sz.x)],
    )


def majorityCol(img: Image, include0: bool = False) -> int:
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
    """Remove all border columns and rows which exclusively contain colors found in bg."""
    bg_mask = colMask(bg)

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
    for i in range(ymi, yma):
        for j in range(xmi, xma):
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
    ret = full(a.p, a.sz, majorityCol(a, False))

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


def mirrorHeuristic(img: Image) -> int:
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
        return rigid(img, 4 + mirrorHeuristic(img))
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


def getRegular(img: Image) -> Image:
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


def clamp(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x


def myStack(a: Image, b: Image, orient: int) -> Image:
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


def outerProductIS(a: Image, b: Image) -> Image:
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


def outerProductSI(a: Image, b: Image) -> Image:
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


def align(a: Image, b: Image) -> Image:
    """Find most matching color and align a to b using it."""
    ret = a.copy()
    match_size = 0
    for c in range(1, 10):
        ca = compress(filterCol(a, c))
        cb = compress(filterCol(b, c))
        if ca.mask == cb.mask:
            cnt = count(ca)
            if cnt > match_size:
                match_size = cnt
                ret.p = a.p + cb.p - ca.p
    if match_size == 0:
        return badImg
    return ret


def replaceCols(base: Image, cols: Image) -> Image:
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


# def mirror(a: Image, b: Image, pad: int=0)-> Image:
#     if a.area <= 0 or b.area <= 0:
#         return badImg
#     ret = empty(b.p, b.sz)

#   const int W = a.w+pad, H = a.h+pad;
#   const int W2 = W*2, H2 = H*2;
#   int ai  = ((b.y-a.y)%H2+H2)%H2;
#   int aj0 = ((b.x-a.x)%W2+W2)%W2;
#   for (int i = 0; i < ret.h; i++) {
#     int aj = aj0;
#     for (int j = 0; j < ret.w; j++) {
#       int x = -1, y = -1;
#       if (aj < a.w) x = aj;
#       else if (aj >= W && aj < W+a.w) x = W+a.w-1-aj;
#       if (ai < a.h) y = ai;
#       else if (ai >= H && ai < H+a.h) y = H+a.h-1-ai;
#       if (x != -1 && y != -1)
# 	ret(i,j) = a(y,x);
#       if (++aj == W2) aj = 0;
#     }
#     if (++ai == H2) ai = 0;
#   }
#     return ret


def majCol(img: Image):
    return Col(majorityCol(img))
