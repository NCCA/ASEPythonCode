#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 4: images and colour

    So far, we have used arrays with shape `(N, 3)` to store positions: one axis selects a point, and the other selects its XYZ components. The [Lab 4 Image class](../Lab4/Image/src/image/Image.py) uses shape `(height, width, 4)` to store an image. Here the axes select a row, a column and an RGBA component.

    The same NumPy operations work on both arrays, but we need to understand what each axis represents. In this lesson, we will use indexing, slicing and broadcasting to work with pixels, then connect that storage to the texture data used in the Lab 5 blitting demos.

    ## Rows before columns

    We usually describe a pixel's position as `(x, y)`, with X measuring across the image and Y measuring down it. NumPy indexes the image by row first, then column, so we access that pixel using:

    ```python
    pixel = pixels[y, x]
    ```

    For an RGBA image, this selects four values in the order red, green, blue and alpha. In the Lab 4 image storage, each component is an unsigned byte from 0 to 255. Alpha describes opacity: zero is fully transparent and 255 is fully opaque.

    I use a deliberately small, non-square image so that we can see the difference between its axes:

    ```python
    pixels = np.zeros((3, 5, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    ```

    This image has three rows and five columns. It starts as opaque black: the RGB components are zero, and the alpha component is 255. The expression `pixels[..., 3]` selects alpha for every pixel.

    We can set the pixel at `x=4`, `y=1` to opaque orange:

    ```python
    pixels[1, 4] = (255, 128, 0, 255)
    ```

    The reversed expression, `pixels[4, 1]`, would try to access row four and raise an `IndexError`, as our image only has rows zero to two. With a square image, swapping X and Y can still produce a valid index, making this mistake harder to spot.

    We can also select one component directly:

    ```python
    red = pixels[1, 4, 0]
    alpha = pixels[1, 4, 3]
    ```

    In the display below, row zero appears at the top and the row index increases downwards. This is the display convention we are using; the NumPy array itself only stores values and their indices.

    Hover over a pixel to inspect its RGBA values and compare its displayed position with its row and column indices.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    print("NumPy", np.__version__)
    return go, make_subplots, mo, np


@app.cell
def _(go, mo, np):
    pixels = np.full((4, 6, 4), [30, 40, 70, 255], dtype=np.uint8)
    pixels[1, 4] = [255, 100, 0, 255]
    pixels[2:, :2] = [0, 220, 120, 255]
    print("shape", pixels.shape, "pixel x=4, y=1", pixels[1, 4])
    print("red channel shape", pixels[..., 0].shape)
    pixel_figure = go.Figure(
        go.Image(
            z=pixels,
            colormodel="rgba256",
            hovertemplate="x=%{x}<br>y=%{y}<br>RGBA=%{z}<extra></extra>",
        )
    )
    pixel_figure.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="x / column",
        yaxis_title="y / row",
    )
    pixel_figure.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    mo.ui.plotly(pixel_figure, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Broadcasting a gradient and a mask

    [`indices`](https://numpy.org/doc/stable/reference/generated/numpy.indices.html) creates coordinate arrays for each axis of a given shape. For an image, we can use it to find the row and column of every pixel:

    ```python
    height, width = 100, 200
    rows, columns = np.indices((height, width))
    ```

    Both arrays have shape `(height, width)`. `rows` contains the row number at each location, whilst `columns` contains the column number. These coordinates let us calculate patterns across the image without writing a loop over individual pixels.

    For a horizontal gradient, we convert the column coordinates to floating-point values between zero and one:

    ```python
    gradient = columns.astype(np.float32) / (width - 1)
    ```

    The first column contains `0.0`, the last contains `1.0`, and the columns between them increase evenly. This calculation assumes the image is at least two pixels wide.

    We can use broadcasting to turn each gradient value into an RGB colour:

    ```python
    colour = np.array([255.0, 128.0, 0.0], dtype=np.float32)
    rgb = gradient[..., np.newaxis] * colour
    ```

    `gradient` has shape `(height, width)`. Adding a final axis changes this to `(height, width, 1)`, allowing each value to multiply the three components of `colour`. The result has shape `(height, width, 3)` and runs from black on the left to orange on the right.

    We keep the calculation in floating point until we are ready to store the image:

    ```python
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., :3] = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    pixels[..., 3] = 255
    ```

    Clipping keeps the values within the byte range, and `rint` rounds them before conversion. Alpha is set separately so every pixel is opaque.

    The coordinate arrays also let us describe a region. For a circle, we compare each pixel's squared distance from the centre with the squared radius:

    ```python
    centre_x, centre_y = width // 2, height // 2
    radius = 30

    mask = (
        (columns - centre_x) ** 2
        + (rows - centre_y) ** 2
    ) <= radius ** 2
    ```

    `mask` is a Boolean array with shape `(height, width)`. Entries inside or on the circle are `True`; entries outside are `False`. Comparing squared distances avoids calculating a square root for every pixel.

    We can use this mask to assign a colour to the selected pixels:

    ```python
    pixels[mask] = (255, 255, 255, 255)
    ```

    The mask matches the first two axes of `pixels`, so it selects complete RGBA pixels. The channel axis stays intact: if the mask selects `K` pixels, `pixels[mask]` has shape `(K, 4)`. The four-component colour broadcasts across those selected pixels.

    I use the same approach for brush regions and particle splats: calculate a condition from the coordinates, then use the resulting mask to select the affected pixels.

    Keeping intermediate calculations out of `uint8` matters because byte arithmetic can wrap around:

    ```python
    np.array([250], dtype=np.uint8) + np.array([10], dtype=np.uint8)
    # array([4], dtype=uint8)
    ```

    The mathematical result is 260, but an unsigned byte can only store values from 0 to 255. NumPy's byte-array arithmetic wraps here rather than clamping. Calculate in floating point, then clip and convert once the result is ready.
    """)
    return


@app.cell
def _(np):
    _rows, _cols = np.indices((96, 160))
    image = np.ones((96, 160, 4), dtype=np.float32)
    image[..., 0] = _cols / 159
    image[..., 1] = _rows / 95
    image[..., 2] = 0.2
    circle_mask = ((_cols - 80) ** 2 + (_rows - 48) ** 2) < 28**2
    image[circle_mask, :3] = [1, 0.2, 0.1]
    print(
        "mask", circle_mask.shape, "selected pixels", np.count_nonzero(circle_mask)
    )
    print(
        "byte overflow",
        np.array([250], dtype=np.uint8) + np.array([10], dtype=np.uint8),
    )
    return circle_mask, image


@app.cell
def _(go, make_subplots, np):
    _height, _width = 200, 400
    _rows, _columns = np.indices((_height, _width))

    _gradient = _columns.astype(np.float32) / (_width - 1)
    _colour = np.array([255.0, 128.0, 0.0], dtype=np.float32)
    _rgb = _gradient[..., np.newaxis] * _colour

    _pixels = np.zeros((_height, _width, 4), dtype=np.uint8)
    _pixels[..., :3] = np.rint(np.clip(_rgb, 0.0, 255.0)).astype(np.uint8)
    _pixels[..., 3] = 255

    _centre_x = (_width - 1) / 2
    _centre_y = (_height - 1) / 2
    _radius = 70

    _mask = ((_columns - _centre_x) ** 2 + (_rows - _centre_y) ** 2) <= _radius**2

    _masked_pixels = _pixels.copy()
    _masked_pixels[_mask] = (255, 255, 255, 255)

    _figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Horizontal gradient",
            "Gradient with circular mask",
        ),
    )

    for _column, _image in enumerate((_pixels, _masked_pixels), start=1):
        _figure.add_trace(
            go.Image(
                z=_image,
                colormodel="rgba256",
                hovertemplate=("x=%{x}, y=%{y}<br>RGBA=%{z}<extra></extra>"),
            ),
            row=1,
            col=_column,
        )

    _figure.update_xaxes(title_text="Column (x)")
    _figure.update_yaxes(title_text="Row (y)", autorange="reversed")
    _figure.update_layout(height=380, margin=dict(l=40, r=20, t=60, b=40))
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting colours with `np.where`

    The [introductory threshold example](../Lecture5/IntroductionToNumpy.py) uses [`where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) to choose between scalar values. The same operation lets us choose between colours for each pixel:

    ```python
    result = np.where(condition, value_if_true, value_if_false)
    ```

    For our circle, the mask has shape `(height, width)`. Each entry tells us whether that pixel is inside the circle. To select RGB colours, we add a final axis:

    ```python
    mask[..., None]
    ```

    This changes the shape to `(height, width, 1)`. The final dimension broadcasts across the three colour channels, so one Boolean selects all three components of a pixel.

    Here we choose green inside the circle and blue outside:

    ```python
    green = np.array([0, 255, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)

    selected = np.where(mask[..., None], green, blue)
    ```

    The colour arrays each have shape `(3,)`. Together with the expanded mask, they broadcast to produce an RGB image with shape `(height, width, 3)`. Where the mask is `True`, the result contains green; where it is `False`, it contains blue.

    `where` returns a new array. We can produce the same values by allocating an image and assigning colours through the mask:

    ```python
    assigned = np.empty((*mask.shape, 3), dtype=np.uint8)
    assigned[:] = blue
    assigned[mask] = green
    ```

    The first assignment fills every pixel with blue, so all components are initialised before we read them. The second replaces the pixels inside the circle with green.

    Notice that masked assignment uses `mask` directly. It selects whole pixels across the first two axes, leaving the channel axis intact. In `where`, we need `mask[..., None]` because the condition participates in broadcasting with the colour arrays.

    Both versions create a separate image and leave `image` from the earlier cell unchanged. I use `where` when expressing a choice between two values, and masked assignment when updating selected pixels in an array I have allocated.
    """)
    return


@app.cell
def _(circle_mask, go, image, mo, np):
    _foreground = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    _background = np.array([0.1, 0.2, 0.8], dtype=np.float32)
    selected_image = np.where(circle_mask[..., None], _foreground, _background)
    assigned_image = np.broadcast_to(_background, selected_image.shape).copy()
    assigned_image[circle_mask] = _foreground
    print("mask", circle_mask.shape, "RGB result", selected_image.shape)
    print("same values", np.array_equal(selected_image, assigned_image))
    print("shares with the earlier image", np.shares_memory(selected_image, image))
    assert np.array_equal(selected_image, assigned_image)
    _bytes = np.rint(selected_image * 255).astype(np.uint8)
    selection_figure = go.Figure(go.Image(z=_bytes, colormodel="rgb"))
    selection_figure.update_layout(
        template="plotly_white",
        height=360,
        xaxis_title="column",
        yaxis_title="row",
    )
    mo.ui.plotly(selection_figure)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alpha over an opaque background

    Alpha describes how much a foreground pixel contributes when we place it over a background. An alpha of `1.0` makes the foreground fully opaque; an alpha of `0.0` lets the background show through completely.

    For straight alpha, the foreground RGB values have not already been multiplied by alpha. When the background is opaque, we calculate the resulting RGB colour using:

    $$
    C = \alpha C_f + (1-\alpha) C_b
    $$

    Here $C_f$ is the foreground colour and $C_b$ is the background colour. With $\alpha=0.25$, the result contains 25% foreground and 75% background. The output remains opaque because the background is opaque.

    Our RGBA image stores components as bytes, so we first convert them to floating point and normalise them to the range zero to one:

    ```python
    foreground = image[..., :3].astype(np.float32) / 255.0
    alpha = image[..., 3:4].astype(np.float32) / 255.0

    background = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    blended = alpha * foreground + (1.0 - alpha) * background
    ```

    `foreground` has shape `(height, width, 3)`, whilst `alpha` has shape `(height, width, 1)`. Broadcasting applies each pixel's alpha to all three of its colour components. The background has shape `(3,)`, so the same background colour is used at every pixel.

    The slice used to extract alpha matters:

    ```python
    image[..., 3]    # (height, width)
    image[..., 3:4]  # (height, width, 1)
    ```

    Selecting a single channel with `3` removes that axis. Using `3:4` keeps a one-component channel axis, giving us the shape needed for broadcasting.

    After blending, we can convert the RGB result back to bytes:

    ```python
    blended_bytes = np.rint(
        np.clip(blended, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    ```

    This produces an RGB image. If we need RGBA storage, we can add an alpha channel filled with 255 because the composited result is fully opaque.

    For this calculation, I treat our generated colour values as linear RGB, where the weighted sum represents a mixture of light. Loaded sRGB photographs need their RGB values decoded to linear light before blending, then encoded back to sRGB for display or storage. Alpha represents coverage or opacity and does not receive that sRGB conversion.

    This example handles a straight-alpha foreground over an opaque background. Blending two transparent layers also requires calculating the output alpha, whilst premultiplied-alpha colours require a different form of the RGB calculation.
    """)
    return


@app.cell
def _(mo):
    opacity = mo.ui.slider(
        0, 1, step=0.05, value=0.5, label="foreground alpha", show_value=True
    )
    opacity
    return (opacity,)


@app.cell
def _(go, image, make_subplots, mo, np, opacity):
    _foreground = image.copy()
    _foreground[..., 3] = opacity.value
    _background = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    _alpha = _foreground[..., 3:4]
    blended = _alpha * _foreground[..., :3] + (1 - _alpha) * _background
    encoded_bytes = np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8)
    print(
        "RGB", blended.shape, "alpha", _alpha.shape, "bytes", encoded_bytes.dtype
    )
    blend_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["generated foreground", "over blue background"],
    )
    _source_bytes = np.rint(np.clip(image[..., :3], 0, 1) * 255).astype(np.uint8)
    for _column, _pixels in enumerate((_source_bytes, encoded_bytes), start=1):
        blend_figure.add_trace(
            go.Image(
                z=_pixels,
                colormodel="rgb",
                hovertemplate="column=%{x}<br>row=%{y}<br>RGB=%{z}<extra></extra>",
            ),
            row=1,
            col=_column,
        )
        blend_figure.update_xaxes(title_text="column", row=1, col=_column)
        blend_figure.update_yaxes(
            title_text="row",
            autorange="reversed",
            scaleratio=1,
            scaleanchor="x" if _column == 1 else "x2",
            row=1,
            col=_column,
        )
    blend_figure.update_layout(
        template="plotly_white", height=430, uirevision="pixel-view"
    )
    mo.ui.plotly(blend_figure, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packed pixels and byte order

    [`ImageUint32`](../Lab4/Image/src/image/ImageUint32.py) stores each pixel as one big-endian 32-bit unsigned integer. The four colour components occupy eight bits each, giving us the packed value:

    ```text
    0xRRGGBBAA
    ```

    For example, the RGBA colour `(255, 128, 0, 255)` becomes `0xFF8000FF`. Big-endian storage places the most significant byte first, so the bytes appear in memory in the order red, green, blue and alpha.

    We can describe this explicitly with NumPy's `">u4"` dtype. The `>` means big-endian, `u` means unsigned integer, and `4` specifies four bytes per value:

    ```python
    packed = np.array([[0xFF8000FF, 0x0080FFFF]], dtype=">u4")
    ```

    This array has shape `(1, 2)`: one row containing two packed pixels. To access the individual colour bytes, we use [`view`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html):

    ```python
    rgba = packed.view(np.uint8).reshape(*packed.shape, 4)
    ```

    The result has shape `(1, 2, 4)`:

    ```text
    [[[255 128   0 255]
      [  0 128 255 255]]]
    ```

    `view` reinterprets the existing bytes using another dtype. It does not calculate new colour values, and this byte view shares storage with `packed`. Changing a component through `rgba` therefore changes the corresponding packed pixel.

    [`astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) performs a numeric conversion instead. Converting a 32-bit integer to `uint8` does not unpack it into four channels:

    ```python
    packed.astype(np.uint8)
    ```

    This still gives one value per pixel. For these unsigned integers, only the lowest eight bits remain, which are the alpha component in our packing scheme. We need `view` when inspecting the bytes and `astype` when changing the numeric representation.

    Naming the byte order matters because native integer storage depends on the machine. On a little-endian machine, a native `uint32` containing `0xFF8000FF` stores its bytes in the reverse order. Using an explicit big-endian dtype makes the RGBA byte order predictable for this layout.

    Byte order is separate from row padding. A texture readback buffer may contain extra bytes after each row to meet its transfer alignment requirements. Those bytes are not pixels. In the optional WebGPU lesson, we will describe the padded rows, slice out each row's pixel bytes, then reshape them into an image.
    """)
    return


@app.cell
def _(np):
    _packed = np.array([0xFF8000FF], dtype=">u4")
    _channels = _packed.view(np.uint8).reshape(1, 1, 4)
    print("RGBA bytes", _channels[0, 0])
    assert np.array_equal(_channels[0, 0], [255, 128, 0, 255])
    _original = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    _restored = np.frombuffer(_original.tobytes(), dtype=np.uint8).reshape(2, 3, 4)
    assert np.array_equal(_original, _restored)
    print("byte round trip", _restored.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Paint a horizontal red line without looping over pixels.
    2. Replace the circle mask with a checkerboard using integer division and remainder.
    3. Set alpha to 0 and 1. Predict the output before moving the slider.
    4. Reverse the rows using `image[::-1]`. What changes in its strides?
    5. Use `where` to give a checkerboard two RGB colours. Make the same image with masked assignment and check that the results agree.

    [Part 5](NumPyForCGPart5Particles.py) builds random points and updates a particle batch.
    """)
    return


if __name__ == "__main__":
    app.run()
