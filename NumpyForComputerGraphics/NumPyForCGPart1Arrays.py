#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 1: arrays and vertex data

    I use NumPy throughout the graphics demos to build the data we send to the GPU.
    In this series we will take those operations apart, starting with a triangle and
    ending with a deformed mesh. You should read [IntroductionToNumpy](../Lecture5/IntroductionToNumpy.py) first as well as look at the lecture slides. Most of what is covered here will occur at some stage in the examples in the [PyNGLDemos](https://github.com/NCCA/PyNGLDemos) or lecture / lab code and slides.

    The six main parts cover arrays, shapes and grids, transformations, images,
    random points and particles, then a lit mesh. Part 7 adds an optional WebGPU render.
    We will generate all the data here, so there are no models or images to download.
    The figures use [Plotly](https://plotly.com/python/): hover to read values, drag
    to zoom or rotate a 3D view, and double-click a 2D plot to reset its axes.

    ## [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)

    A vertex has three coordinates. A triangle has three vertices. I store one vertex
    per row, giving an array with shape `(3, 3)`. This is the position data used in
    [FirstTriangles](../GraphicsDemos/FirstTriangles/WebGPU/WebGPUTriangle.py), without the window code.

    ```python
    np.array(object, dtype=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `object` | required | values to store |
    | `dtype` | inferred | type of each component |

    The dtype is part of the buffer contract. Three `float32` coordinates occupy
    12 bytes; the same three values as `float64` occupy 24. Most graphics api's want to use float so we must ensure we specify this, numpy defaults to double and if we get errors most of the time it will be because we forgot this (ask me how I know :-) )
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    print("NumPy", np.__version__)
    return go, mo, np


@app.cell
def _(np):
    triangle = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    print("positions:\n", triangle)
    print("shape", triangle.shape, "dtype", triangle.dtype)
    print("bytes per component", triangle.itemsize, "total bytes", triangle.nbytes)
    assert triangle.nbytes == 3 * 3 * 4
    return (triangle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allocating and sampling

    Before we calculate our points or assign colours, we need somewhere to store the data. NumPy lets us create an array with a particular shape and choose how its values are initialised. For example, an array with shape `(100, 3)` contains 100 rows of three components. We could use this to store 100 positions as `(x, y, z)` values, or 100 colours as `(r, g, b)` values.

    [`zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) creates an array where every component starts at zero. This is useful when zero is a sensible starting value, or when we will only update some of the components. Here I allocate space for 100 positions using 32-bit floating-point values:

    ```python
    positions = np.zeros((100, 3), dtype=np.float32)
    ```

    Each row initially contains `[0.0, 0.0, 0.0]`. If we later assign only the `x` and `y` components, the `z` component will remain zero, keeping our points in the XY plane.

    [`full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html) creates an array filled with a value we provide. We can also pass an RGB colour and let NumPy broadcast it across the rows:

    ```python
    colours = np.full((100, 3), (1.0, 0.5, 0.0), dtype=np.float32)
    ```

    The colour has shape `(3,)`, which matches the three components in each row of our destination array. Broadcasting repeats this colour across all 100 rows, giving every point the same orange colour without us having to write a Python loop. Here I am using floating-point colour components in the range `0.0` to `1.0`.

    [`empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html) allocates storage without initialising its numeric values:

    ```python
    positions = np.empty((100, 3), dtype=np.float32)
    ```

    The values already in that memory are not meaningful data, even if they happen to look like zeros. I only use `empty` when the next calculation will assign every component before anything reads the array. For example, assigning `x` and `y` whilst leaving `z` untouched would give us undefined heights for our points. In that case, `zeros` is the better choice.

    Once we have storage, we need to decide where to sample our curve. I use [`linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) when I want a fixed number of evenly spaced parameter values. For a circle, we can sample angles over one complete turn:

    ```python
    angles = np.linspace(0.0, 2.0 * np.pi, num=100, endpoint=False)
    ```

    This gives us 100 angles, starting at zero with a spacing of `2π / 100` radians. We can then calculate the circle's coordinates using `np.cos(angles)` and `np.sin(angles)`.

    By default, `linspace` includes both ends of the interval. For a circle, the angles `0` and `2π` describe the same position, so including both would store the starting point twice. Using `endpoint=False` leaves out the final angle and gives us 100 distinct sample positions around the circle.

    When drawing the closing line, we connect the last sample back to the first. If the drawing method expects a continuous sequence of points, we append the first point to that sequence for drawing. This keeps the stored curve samples free of the duplicated endpoint.
    """)
    return


@app.cell
def _(np):
    colours = np.full((3, 3), [1.0, 0.4, 0.1], dtype=np.float32)
    velocity = np.zeros_like(colours)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False, dtype=np.float32)
    circle = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    print("colours:\n", colours)
    print("velocity", velocity.shape, velocity.dtype)
    print("circle", circle.shape, "first point", circle[0])
    assert np.allclose(circle[:, :2] ** 2 @ np.ones(2), 1)
    return (circle,)


@app.cell
def _(circle, go, mo, np, triangle):
    geometry_figure = go.Figure()
    for _points, _label in [(triangle, "triangle"), (circle, "12 circle samples")]:
        _closed = np.concatenate((_points, _points[:1]))
        geometry_figure.add_trace(
            go.Scatter(
                x=_closed[:, 0],
                y=_closed[:, 1],
                mode="lines+markers",
                name=_label,
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    geometry_figure.update_layout(
        template="plotly_white", height=480, xaxis_title="x", yaxis_title="y"
    )
    geometry_figure.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.ui.plotly(
        geometry_figure, config={"displaylogo": False, "scrollZoom": True}
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps, counts and whole-array functions

    The [introductory lecture](../Lecture5/IntroductionToNumpy.py) uses [`arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) to generate integer sequences. We give it a start, a stop and a step:

    ```python
    indices = np.arange(0, 10, 2)
    ```

    This produces `[0, 2, 4, 6, 8]`. The start is included, whilst the stop is excluded. The step determines the spacing between entries, and the number of entries follows from the interval.

    [`linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) works the other way around: we specify how many samples we want, and NumPy calculates their spacing.

    ```python
    angles = np.linspace(0.0, 2.0 * np.pi, num=100, endpoint=False)
    ```

    Here we ask for exactly 100 angles around a circle. For a GPU buffer, I usually know the required vertex count, so `linspace` makes this explicit. With `endpoint=False`, the interval includes zero but excludes `2π`, avoiding a second sample at the starting position.

    We can also use `arange` with a floating-point step, but many decimal fractions cannot be represented exactly in binary floating point. Rounding can affect the generated values and whether a value near the stop is included. I use `arange` for integer steps and `linspace` when a calculation needs a particular number of samples.

    Once we have our angles, NumPy's trigonometric functions calculate coordinates for the whole array:

    ```python
    x = np.cos(angles)
    y = np.sin(angles)
    ```

    Each entry in `x` is the cosine of the corresponding entry in `angles`, and each entry in `y` is its sine. NumPy's trigonometric functions expect angles in radians. These coordinates describe a unit circle centred at the origin; multiplying them by a radius changes its size.

    Functions such as `np.sin`, `np.cos` and `np.exp` are [universal functions, or ufuncs](https://numpy.org/doc/stable/user/basics.ufuncs.html). They apply an operation element by element across an array and return the results without us writing a Python loop. For these calculations, the output has the same shape as the input.

    An exponential falloff uses the same pattern. For example, we can model the fraction of light transmitted through a uniform fog at different distances:

    ```python
    distances = np.linspace(0.0, 100.0, num=101)
    density = 0.05
    transmission = np.exp(-density * distances)
    ```

    Multiplying `distances` by `-density` produces an array of exponents. `np.exp` then calculates the exponential of every entry, giving us a transmission value for each distance. At distance zero, transmission is `1.0`; as distance increases, it falls towards zero. Here distance and density must use compatible units so that `density * distances` is dimensionless.
    """)
    return


@app.cell
def _(np):
    print("indices from arange", np.arange(0, 8, 2))
    print("four samples including the stop", np.linspace(0, 1, 4))
    print("four samples excluding the stop", np.linspace(0, 1, 4, endpoint=False))
    _distance = np.linspace(0, 3, 5, dtype=np.float32)
    _transmission = np.exp(-0.75 * _distance)
    print("distance", _distance)
    print("exponential transmission", _transmission)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A buffer with named fields

    The [WebGPU triangle grid](../GraphicsDemos/TriangleGrid/WebGPU/WebGPUTriangleGrid.py) uses a [structured dtype](https://numpy.org/doc/stable/user/basics.rec.html) to describe the data stored for each vertex. Instead of treating a vertex as a row of numbered columns, we give each attribute a name, a component type and a shape:

    ```python
    vertex_dtype = np.dtype([
        ("position", np.float32, (3,)),
        ("normal", np.float32, (3,)),
        ("uv", np.float32, (2,)),
    ])

    vertices = np.zeros(vertex_count, dtype=vertex_dtype)
    ```

    Each entry in `vertices` is one record containing a position, a normal and a UV coordinate. The array itself has shape `(vertex_count,)`, whilst its fields provide views of the individual attributes:

    ```python
    vertices["position"].shape
    # (vertex_count, 3)

    vertices["normal"].shape
    # (vertex_count, 3)

    vertices["uv"].shape
    # (vertex_count, 2)
    ```

    These field views let us assign values using the attribute names. For a grid in the XZ plane, for example, we can give every vertex an upward-facing normal:

    ```python
    vertices["normal"] = (0.0, 1.0, 0.0)
    ```

    NumPy broadcasts this normal across all the records. We can also change one position or one component:

    ```python
    vertices["position"][0] = (-1.0, 0.0, -1.0)
    vertices["position"][:, 1] = 0.0
    ```

    The second assignment sets the Y coordinate of every position to zero. I find the field names easier to follow than remembering which numbered columns contain positions, normals and texture coordinates.

    The attributes are stored together for each vertex, giving us an interleaved buffer. Each `float32` occupies four bytes, so the layout is:

    | Field | Components | Byte offset | Size |
    | --- | --- | --- | --- |
    | `position` | 3 floats | 0 | 12 bytes |
    | `normal` | 3 floats | 12 | 12 bytes |
    | `uv` | 2 floats | 24 | 8 bytes |

    With this dtype, each record occupies 32 bytes. We can inspect that size using `vertex_dtype.itemsize`. The WebGPU vertex buffer description must match these offsets, formats and the 32-byte stride so that the shader receives the correct attributes.

    This layout describes vertex attributes. Uniform buffers have separate WGSL alignment and padding rules, so a tightly packed NumPy record is not automatically suitable for a uniform struct. When preparing uniform data, we need to check the offsets, alignment and total size against the WGSL declaration.
    """)
    return


@app.cell
def _(np, triangle):
    vertex_dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("uv", np.float32, 2),
        ]
    )
    vertices = np.zeros(3, dtype=vertex_dtype)
    vertices["position"] = triangle
    vertices["normal"] = [0, 0, 1]
    vertices["uv"] = [[0, 0], [1, 0], [0.5, 1]]
    for _name in vertex_dtype.names:
        print(_name, "byte offset", vertex_dtype.fields[_name][1])
    print("stride", vertex_dtype.itemsize, "upload bytes", len(vertices.tobytes()))
    assert vertex_dtype.itemsize == 32
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Converting colour values

    [`astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) converts array values to a different data type. It does not automatically scale them to a colour range or clamp values which fall outside that range.

    The Lab 4 image classes store colour components as unsigned 8-bit integers (`np.uint8`), with values from 0 to 255. Calculations can produce values outside this range, so we need to clamp the results before converting them:

    ```python
    colour = np.array([280.0, 127.5, -20.0])
    colour_bytes = np.clip(colour, 0.0, 255.0).astype(np.uint8)
    ```

    [`clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) limits each component to the supplied range. Values below zero become zero, values above 255 become 255, and values within the range remain unchanged. The conversion then produces `[255, 127, 0]`. Converting to an integer discards the fractional part; if we want to round to the nearest integer first, we can use `np.rint`.

    The point demos use floating-point RGB values, where `0.0` represents no contribution from a component and `1.0` represents its full contribution. To convert these colours to bytes, we first clamp to this range, then multiply by 255:

    ```python
    colour = np.array([1.0, 0.5, 0.0])
    colour_bytes = np.rint(
        np.clip(colour, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    ```

    This produces `[255, 128, 0]`. Calling `astype(np.uint8)` directly on the original array would produce `[1, 0, 0]`, as the conversion does not know that our floating-point values represent normalised colours.
    """)
    return


@app.cell
def _(np):
    _values = np.array([-20.0, 127.9, 300.0])
    print("cast directly", _values.astype(np.uint8))
    print("clamp and round", np.rint(np.clip(_values, 0, 255)).astype(np.uint8))
    try:
        _values.astype(np.uint8, casting="safe")
    except TypeError as _error:
        print("safe casting rejected the conversion:", _error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Add a fourth position and draw a closed square. Predict its buffer size first.
    2. Change the circle to 32 samples. What changes if you include the endpoint?
    3. Add an RGB field to the vertex dtype. Print its offset and the new stride.
    4. Create a `(480, 640, 4)` byte array. How many bytes does one frame occupy?
    5. Make eight samples in `[0, 1)` using integer `arange` values and using `linspace`. Compare their counts and final values.

    [Part 2](NumPyForCGPart2Shapes.py) uses shapes, views and indexing to build a grid.
    """)
    return


if __name__ == "__main__":
    app.run()
