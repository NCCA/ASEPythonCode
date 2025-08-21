# RGBA.py
from typing import Tuple


class RGBA:
    red_mask = 0xFF000000
    green_mask = 0x00FF0000
    blue_mask = 0x0000FF00
    alpha_mask = 0x000000FF

    def __init__(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255):
        self.pixel = (r & 0xFF) << 24 | (g & 0xFF) << 16 | (b & 0xFF) << 8 | (a & 0xFF)

    def get_rgba(self) -> Tuple[int, int, int, int]:
        r = (self.pixel & self.red_mask) >> 24
        g = (self.pixel & self.green_mask) >> 16
        b = (self.pixel & self.blue_mask) >> 8
        a = self.pixel & self.alpha_mask
        return r, g, b, a

    def __eq__(self, other):
        if not isinstance(other, RGBA):
            return False
        return self.pixel == other.pixel

    def __repr__(self):
        r, g, b, a = self.get_rgba()
        return f"RGBA(r={r}, g={g}, b={b}, a={a})"
