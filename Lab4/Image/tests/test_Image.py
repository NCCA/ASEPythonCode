# tests/test_Image.py
from image.Image import RGBA, Image


def test_image_ctor_and_clear():
    img = Image(2, 3)
    assert img.width() == 2
    assert img.height() == 3
    default = RGBA()
    for x in range(2):
        for y in range(3):
            assert img.get_pixel(x, y) == default

    red = RGBA(255, 0, 0, 255)
    img.clear(red)
    for x in range(2):
        for y in range(3):
            assert img.get_pixel(x, y) == red


def test_set_get_pixel():
    img = Image(3, 3)
    green = RGBA(0, 255, 0, 255)
    img.set_pixel(1, 1, color=green)
    assert img.get_pixel(1, 1) == green


def test_line_horizontal():
    img = Image(5, 5)
    blue = RGBA(0, 0, 255, 255)
    img.line(0, 2, 4, 2, blue)
    for x in range(5):
        assert img.get_pixel(x, 2) == blue


def test_rectangle():
    img = Image(4, 4)
    yellow = RGBA(255, 255, 0, 255)
    img.rectangle(1, 1, 3, 2, yellow)
    for x in range(4):
        for y in range(4):
            if 1 <= x <= 3 and 1 <= y <= 2:
                assert img.get_pixel(x, y) == yellow
            else:
                assert img.get_pixel(x, y) == RGBA()


def test_save(tmp_path):
    img = Image(10, 10)
    cyan = RGBA(0, 255, 255, 255)
    img.clear(cyan)
    fname = tmp_path / "test.png"
    assert img.save(str(fname))
