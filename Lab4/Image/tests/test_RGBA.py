# tests/test_RGBA.py
from image.Image import RGBA


def test_ctor_defaults():
    p = RGBA()
    assert p.get_rgba() == (0, 0, 0, 255)


def test_ctor_values():
    p = RGBA(10, 20, 30, 40)
    assert p.get_rgba() == (10, 20, 30, 40)
