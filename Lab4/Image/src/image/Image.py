# Image.py
from typing import Optional

from PIL import Image as PILImage

from image.RGBA import RGBA


class Image:
    def __init__(self, width: int = 0, height: int = 0, fill: Optional[RGBA] = None):
        self._width = width
        self._height = height
        default = fill if fill is not None else RGBA()
        self._pixels = [RGBA() for _ in range(width * height)]
        if fill:
            self.clear(fill)

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def _index(self, x: int, y: int) -> int:
        return y * self._width + x

    def get_pixel(self, x: int, y: int) -> RGBA:
        return self._pixels[self._index(x, y)]

    def set_pixel(
        self,
        x: int,
        y: int,
        r: Optional[int] = None,
        g: Optional[int] = None,
        b: Optional[int] = None,
        a: Optional[int] = None,
        color: Optional[RGBA] = None,
    ):
        if 0 <= x < self._width and 0 <= y < self._height:
            if color is not None:
                self._pixels[self._index(x, y)] = color
            else:
                self._pixels[self._index(x, y)] = RGBA(r or 0, g or 0, b or 0, a or 0)

    def clear(self, fill: RGBA):
        self._pixels = [RGBA(*fill.get_rgba()) for _ in range(self._width * self._height)]

    def save(self, fname: str) -> bool:
        try:
            img = PILImage.new("RGBA", (self._width, self._height))
            img.putdata([p.get_rgba() for p in self._pixels])
            img.save(fname)
            return True
        except Exception:
            return False

    def line(self, sx: int, sy: int, ex: int, ey: int, color: RGBA):
        dx, dy = abs(ex - sx), abs(ey - sy)
        x, y = sx, sy
        sx_sign = 1 if ex > sx else -1
        sy_sign = 1 if ey > sy else -1
        if dx > dy:
            err = dx / 2
            while x != ex:
                self.set_pixel(x, y, color=color)
                err -= dy
                if err < 0:
                    y += sy_sign
                    err += dx
                x += sx_sign
        else:
            err = dy / 2
            while y != ey:
                self.set_pixel(x, y, color=color)
                err -= dx
                if err < 0:
                    x += sx_sign
                    err += dy
                y += sy_sign
        self.set_pixel(ex, ey, color=color)

    def rectangle(self, tx: int, ty: int, bx: int, by: int, color: RGBA):
        x0, x1 = sorted((tx, bx))
        y0, y1 = sorted((ty, by))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y, color=color)
