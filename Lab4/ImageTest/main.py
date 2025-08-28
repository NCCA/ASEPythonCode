#!/usr/bin/env -S uv run --script
import random

from image import RGBA, Image


def main():
    width = 400
    height = 400
    num_images = 5

    for i in range(num_images):
        img = Image(width, height, fill=RGBA(0, 0, 0))
        num_shapes = random.randint(5, 500)

        for _ in range(num_shapes):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = RGBA(r, g, b)

            shape_type = random.choice(["line", "rectangle"])

            if shape_type == "line":
                sx = random.randint(0, width - 1)
                sy = random.randint(0, height - 1)
                ex = random.randint(0, width - 1)
                ey = random.randint(0, height - 1)
                img.line(sx, sy, ex, ey, color)
            elif shape_type == "rectangle":
                tx = random.randint(0, width - 1)
                ty = random.randint(0, height - 1)
                bx = random.randint(0, width - 1)
                by = random.randint(0, height - 1)
                img.rectangle(tx, ty, bx, by, color)

        filename = f"random_image_{i}.png"
        if img.save(filename):
            print(f"Saved {filename}")
        else:
            print(f"Failed to save {filename}")


if __name__ == "__main__":
    main()
