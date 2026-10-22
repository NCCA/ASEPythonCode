#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 2: shapes, views and grids

    In [Part 1](NumPyForCGPart1Arrays.py) we stored one point per row. Now we will make
    a rectangular grid and turn its corners into triangles, following the
    [TriangleGrid demo](../GraphicsDemos/TriangleGrid/WebGPU/WebGPUTriangleGrid.py).
    I use a non-square grid here because a square can hide an accidental transpose.

    ## [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)

    `reshape` changes the shape of an array whilst preserving its values and total number of elements. For example, we can arrange twelve values into four rows of three:

    ```python
    values = np.arange(12)
    points = values.reshape(4, 3)
    ```

    The resulting array is:

    ```text
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    ```

    This is useful when a flat sequence contains groups of components, such as three coordinates per point. The new dimensions must multiply to the original number of elements: twelve values fit into shape `(4, 3)`, but attempting shape `(5, 3)` raises a `ValueError`.

    We can use `-1` for one dimension and let NumPy calculate its size:

    ```python
    points = values.reshape(-1, 3)
    ```

    Here we are asking for three columns and however many rows are needed. NumPy calculates four rows from the twelve available values. Only one dimension can be inferred this way.

    A reshape does not necessarily allocate new storage. If the existing memory layout supports the requested shape, NumPy can return a view of the same data. Changes through that view also affect the original array:

    ```python
    points[0, 0] = 99
    print(values[0])
    # 99
    ```

    Other layouts may require NumPy to copy the data. For example, transposing an array changes how its indices map to memory, which can prevent a later reshape from sharing the original storage.

    I use [`shares_memory`](https://numpy.org/doc/stable/reference/generated/numpy.shares_memory.html) to check this directly:

    ```python
    np.shares_memory(values, points)
    # True
    ```

    A result of `True` means the arrays overlap in memory, so changing shared elements through one array can affect the other. If we need an independent array, we can request a copy explicitly:

    ```python
    independent_points = values.reshape(-1, 3).copy()
    ```
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
    _flat = np.arange(12, dtype=np.float32)
    _points = _flat.reshape(4, 3)
    _points[0, 0] = 99
    print("reshaped points:\n", _points)
    print(
        "shared storage", np.shares_memory(_flat, _points), "first value", _flat[0]
    )
    _copy = _points.copy()
    _copy[0] = 0
    print("copy leaves original alone", _points[0])
    try:
        _flat.reshape(5, 3)
    except ValueError as _error:
        print("wrong component count:", _error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Combining arrays: new axes or existing axes?

    When working with geometry, we often calculate coordinates separately and then combine them into one array. The function we choose determines how those values are organised: we can add more rows, preserve separate batches, or place attributes beside each other.

    Consider two arrays, each containing two XYZ positions:

    ```python
    a = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    b = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    ```

    Both have shape `(2, 3)`. Axis `0` selects the rows, whilst axis `1` selects the components within each row.

    [`concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) joins arrays along an existing axis:

    ```python
    vertices = np.concatenate((a, b), axis=0)
    ```

    This produces shape `(4, 3)`, placing the rows from `b` after those from `a`. We have combined two batches into one sequence of four vertices. All dimensions except the joining axis must match.

    [`stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html) creates a new axis:

    ```python
    batches = np.stack((a, b), axis=0)
    ```

    The result has shape `(2, 2, 3)`. The new first axis selects the batch, so `batches[0]` contains the values from `a` and `batches[1]` contains those from `b`. The remaining axes select the vertex and its XYZ component. Unlike `concatenate`, `stack` requires every input to have the same shape.

    [`column_stack`](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html) places columns beside each other:

    ```python
    attributes = np.column_stack((a, b))
    ```

    For these two-dimensional inputs, the result has shape `(2, 6)`. Each row contains three values from `a` followed by three from `b`. We might use this arrangement for positions and normals, where corresponding rows describe attributes of the same vertex.

    | Operation on two `(2, 3)` arrays | Result shape | What it represents here |
    | --- | --- | --- |
    | `np.concatenate((a, b), axis=0)` | `(4, 3)` | Append another batch of XYZ rows |
    | `np.stack((a, b), axis=0)` | `(2, 2, 3)` | Keep the batches on a new axis |
    | `np.column_stack((a, b))` | `(2, 6)` | Put two attributes beside each other |

    For separate one-dimensional coordinate arrays, `column_stack` treats each input as a column:

    ```python
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([1.0, 1.0, 1.0])

    positions = np.column_stack((x, y, z))
    ```

    Each input has shape `(3,)`, and the result has shape `(3, 3)`:

    ```text
    [[0. 0. 1.]
     [1. 0. 1.]
     [2. 0. 1.]]
    ```

    More generally, three coordinate arrays of length `N` produce an `(N, 3)` array. We used this form for the circle in Part 1.

    When working with coordinate grids, I use `stack` with `axis=-1`:

    ```python
    positions = np.stack((x_grid, y_grid, z_grid), axis=-1)
    ```

    Here `-1` places the new axis at the end. If each input has shape `(rows, columns)`, the result has shape `(rows, columns, 3)`. We can access a position with `positions[row, column]`, which gives us its three XYZ components.

    This preserves the grid structure whilst we calculate the geometry. When we need one row per vertex for a buffer, we can flatten the two grid dimensions:

    ```python
    vertices = positions.reshape(-1, 3)
    ```

    The resulting shape is `(rows * columns, 3)`, with the XYZ components still grouped together for each vertex.
    """)
    return


@app.cell
def _(np):
    _first = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    _second = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    combined_vertices = np.concatenate((_first, _second), axis=0)
    stacked_vertices = np.stack((_first, _second), axis=0)
    packed_attributes = np.column_stack((_first, _second))
    for _label, _array in [
        ("concatenate", combined_vertices),
        ("stack", stacked_vertices),
        ("column_stack", packed_attributes),
    ]:
        print(_label, _array.shape, "\n", _array)
    print("flattened xyz entries", combined_vertices.ravel())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)

    `meshgrid` takes one-dimensional coordinate arrays and builds arrays containing every combination of their values. This is useful for creating a regular grid: we choose the samples along each axis, then generate the coordinates at their intersections.

    For a grid in the XZ plane, we start with four X samples and three Z samples:

    ```python
    x = np.linspace(-1.0, 1.0, num=4)
    z = np.linspace(-1.0, 1.0, num=3)

    x_grid, z_grid = np.meshgrid(x, z, indexing="xy")
    ```

    Here `"xy"` describes NumPy's indexing convention; it does not require us to use the X and Y axes in our geometry. With two inputs, this convention puts the first coordinate along the columns and the second along the rows.

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | coordinate arrays | required | One-dimensional samples along each axis |
    | `indexing` | `"xy"` | For two inputs, gives output shape `(len(second), len(first))` |

    Both output arrays have shape `(3, 4)`. The X values repeat across each row, whilst each row has a different Z value. Rounded for display, they look like this:

    ```text
    x_grid:
    [[-1.    -0.333  0.333  1.   ]
     [-1.    -0.333  0.333  1.   ]
     [-1.    -0.333  0.333  1.   ]]

    z_grid:
    [[-1. -1. -1. -1.]
     [ 0.  0.  0.  0.]
     [ 1.  1.  1.  1.]]
    ```

    Entries at the same row and column belong to the same grid point. For example, `x_grid[0, 0]` and `z_grid[0, 0]` give the coordinates of the first point: `x=-1.0` and `z=-1.0`.

    To create XYZ positions, we also need a Y coordinate. I use [`zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html) to create an array of zeros with the same shape and dtype as `x_grid`:

    ```python
    y_grid = np.zeros_like(x_grid)
    positions = np.stack((x_grid, y_grid, z_grid), axis=-1)
    ```

    Stacking along the last axis groups the three components at each grid location. The result has shape `(3, 4, 3)`: three rows, four columns and three components per position. We can now access a complete XYZ position using `positions[row, column]`.

    When preparing a vertex buffer, we usually want one row per vertex:

    ```python
    vertices = positions.reshape(-1, 3)
    ```

    The final dimension stays at three, keeping XYZ together. NumPy infers the other dimension from the number of values, so the result has shape `(12, 3)`.

    With the default reshape order, the vertices run across the first grid row before moving to the next. This gives us the grid's positions; we still need to define which vertices form each triangle when building the mesh.
    """)
    return


@app.cell
def _(np):
    x_samples = np.linspace(-2, 2, 4, dtype=np.float32)
    z_samples = np.linspace(-1, 1, 3, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(x_samples, z_samples, indexing="xy")
    grid_positions = np.stack((grid_x, np.zeros_like(grid_x), grid_z), axis=-1)
    points = grid_positions.reshape(-1, 3)
    print("coordinate grids", grid_x.shape, grid_z.shape)
    print("positions", grid_positions.shape, "flattened", points.shape)
    print("first row of grid:\n", points[:4])
    return grid_positions, points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Broadcasting a translation

    Broadcasting lets NumPy apply an operation to arrays with different shapes, provided their dimensions are compatible. For a translation, we can add one XYZ offset to every position in an array:

    ```python
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    offset = np.array([2.0, 3.0, 4.0])
    translated = positions + offset
    ```

    `positions` has shape `(3, 3)` and `offset` has shape `(3,)`. NumPy applies the offset to each row, producing:

    ```text
    [[2. 3. 4.]
     [3. 3. 4.]
     [2. 4. 4.]]
    ```

    Each point moves two units along X, three along Y and four along Z. The expression creates a new result array. If we want to update the existing positions, we can use `positions += offset`, provided its dtype supports the result.

    NumPy checks broadcasting compatibility by comparing dimensions from the right. Each pair must either match or contain a dimension of size `1`. Missing dimensions on the left are treated as size `1`, so our shapes behave like this:

    ```text
    positions: (N, 3)
    offset:    (1, 3)
    result:    (N, 3)
    ```

    We do not need to construct an `(N, 3)` array containing repeated offsets ourselves.

    If we want a different XYZ translation for every point, we supply offsets with shape `(N, 3)`. A different scalar amount per point needs shape `(N, 1)` if that amount should be added to all three components:

    ```python
    amounts = np.array([0.1, 0.2, 0.3])
    shifted = positions + amounts[:, np.newaxis]
    ```

    [`np.newaxis`](https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools) adds a dimension, changing `amounts` from shape `(N,)` to `(N, 1)`. Each row's scalar then broadcasts across its X, Y and Z components.

    This distinction matters: an array with shape `(N,)` does not automatically mean “one value per point”. NumPy matches it against the final axis. If `N` happens to be three, the operation succeeds but applies those values as one shared XYZ offset.

    ## Views and gathered copies

    How we select positions also affects whether later changes reach the original array. Basic slicing returns a view:

    ```python
    first_two = positions[:2]
    first_two[:, 1] = 5.0
    ```

    Here `first_two` shares storage with `positions`, so the assignment also changes the Y coordinates of the first two original points.

    Integer array indexing gathers values into a new array:

    ```python
    indices = np.array([0, 2, 1])
    triangle = positions[indices]
    triangle[:, 1] = 5.0
    ```

    `triangle` contains a copy of the selected positions. Changing it does not update `positions`. This matters when we gather grid vertices into triangle order: the resulting triangle array has its own storage.

    If we intend to change the original vertices, we can assign directly through the indices:

    ```python
    positions[indices, 1] = 5.0
    ```

    I use [`np.shares_memory`](https://numpy.org/doc/stable/reference/generated/numpy.shares_memory.html) when I want to check whether two arrays share storage.
    """)
    return


@app.cell
def _(np, points):
    print("translated first two points:\n", (points + [0, 2, 0])[:2])
    try:
        points + np.arange(len(points))
    except ValueError as _error:
        print("one value per point needs another axis:", _error)
    print("per-point offsets", (points + np.arange(len(points))[:, None]).shape)
    _source = points.copy()
    _view = _source[:2]
    _gathered = _source[[0, 1]]
    _view[0, 1] = 5
    _gathered[1, 1] = 9
    print("slice shared", np.shares_memory(_source, _view))
    print("gather shared", np.shares_memory(_source, _gathered))
    print("source y values", _source[:2, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two triangles per quad

    Our grid gives us rows and columns of points. Four neighbouring points define a quad, but WebGPU does not provide a quad primitive, so we split each grid cell into two triangles for drawing.

    Triangles are useful because three non-collinear points always define a plane. A quad's four corners can lie on different planes, particularly when we change the height of individual grid points. Splitting the quad into triangles gives each part a well-defined surface. The GPU can then determine which pixels each triangle covers and interpolate attributes such as colour and UV coordinates across it.

    We number the grid points row by row. For each cell, we identify its four corners and split it along the diagonal between bottom-left and top-right:

    ```text
    top-left -------- top-right
        |           / |
        |         /   |
        |       /     |
        |     /       |
        |   /         |
    bottom-left ----- bottom-right
    ```

    The two triangles use these corners:

    ```text
    triangle 1: top-left, bottom-left, top-right
    triangle 2: top-right, bottom-left, bottom-right
    ```

    The order of the vertices is called the winding order. For our XZ grid, where X increases across columns and Z increases down rows, these orders give both triangles a geometric normal pointing along positive Y. Reversing the order reverses the normal. Consistent winding also matters when back-face culling is enabled, as it determines which side of each triangle is considered the front. We will check the normal direction in Part 6.

    To find the corner indices, we first arrange the vertex numbers into the same shape as the grid:

    ```python
    grid_indices = np.arange(rows * columns).reshape(rows, columns)

    top_left = grid_indices[:-1, :-1]
    bottom_left = grid_indices[1:, :-1]
    top_right = grid_indices[:-1, 1:]
    bottom_right = grid_indices[1:, 1:]
    ```

    Each slice has shape `(rows - 1, columns - 1)`, with one entry per quad. Entries at the same location in these four arrays identify the corners of the same cell.

    We can then put the indices into triangle order:

    ```python
    triangle_indices = np.stack(
        (
            top_left, bottom_left, top_right,
            top_right, bottom_left, bottom_right,
        ),
        axis=-1,
    ).reshape(-1, 3)
    ```

    Each row of `triangle_indices` now describes one triangle. A grid with `rows` by `columns` points contains `(rows - 1) * (columns - 1)` quads, giving us twice that many triangles.

    The demo expands each quad into six vertex records. Slices such as `0::6` and `1::6` select the first and second vertex slots in every group of six. Here I use an index array first so we can see the connectivity before gathering the positions:

    ```python
    triangle_positions = positions.reshape(-1, 3)[triangle_indices]
    ```

    This produces shape `(triangle_count, 3, 3)`: one entry per triangle, three vertices per triangle and three coordinates per vertex. Reshaping to `(-1, 3)` gives the flat sequence of positions needed for an expanded vertex buffer.

    Gathering with an integer index array creates a copy. Shared grid corners therefore appear more than once in the expanded data. Alternatively, an indexed draw can use the original vertex buffer together with an index buffer, allowing neighbouring triangles to reference the same vertex records.
    """)
    return


@app.cell
def _(grid_positions, np, points):
    _ids = np.arange(len(points)).reshape(grid_positions.shape[:2])
    _tl, _tr = _ids[:-1, :-1].ravel(), _ids[:-1, 1:].ravel()
    _bl, _br = _ids[1:, :-1].ravel(), _ids[1:, 1:].ravel()
    faces = np.stack((_tl, _bl, _tr, _tr, _bl, _br), axis=1).reshape(-1, 3)
    triangles = points[faces]
    print("first quad indices:\n", faces[:2])
    print("faces", faces.shape, "triangle positions", triangles.shape)
    assert faces.shape == (12, 3)
    assert triangles.shape == (12, 3, 3)
    return faces, triangles


@app.cell
def _(faces, go, mo, np, points):
    _edges = points[faces[:, [0, 1, 2, 0]]]
    _separated = np.full((len(faces), 5, 3), np.nan)
    _separated[:, :4] = _edges
    _lines = _separated.reshape(-1, 3)
    grid_figure = go.Figure(
        go.Scatter(
            x=_lines[:, 0],
            y=_lines[:, 2],
            mode="lines",
            line={"color": "grey"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    grid_figure.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 2],
            mode="markers+text",
            text=np.arange(len(points)),
            textposition="top center",
            name="grid vertices",
            customdata=np.arange(len(points)),
            hovertemplate="vertex %{customdata}<br>x=%{x:.3f}<br>z=%{y:.3f}<extra></extra>",
        )
    )
    grid_figure.update_layout(
        template="plotly_white",
        height=480,
        title="12 points, 6 quads, 12 triangles",
        xaxis_title="x",
        yaxis_title="z",
        showlegend=False,
    )
    grid_figure.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.ui.plotly(grid_figure, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packing a strided view

    An array's shape tells us how many entries it has along each axis. Its strides tell us how many bytes NumPy moves through memory when advancing one entry along an axis.

    For a C-contiguous array of XYZ positions stored as `float32`, each component occupies four bytes and each row occupies twelve bytes:

    ```python
    positions = np.zeros((6, 3), dtype=np.float32)

    print(positions.strides)
    # (12, 4)
    ```

    Moving to the next vertex advances twelve bytes; moving to the next component advances four. The rows follow one another in memory without gaps.

    A slice can change this arrangement without copying the data:

    ```python
    selected = positions[::2]

    print(selected.shape)
    # (3, 3)

    print(selected.strides)
    # (24, 4)
    ```

    `selected` contains every other vertex. Its components are still adjacent, but moving between rows now skips twenty-four bytes because an unselected vertex sits between them. NumPy can work with this view because it knows the strides.

    A transpose can also change the shape and strides without moving the underlying values. This is useful within NumPy, but a buffer consumer may expect consecutive rows in C order and may not interpret NumPy's stride information.

    I use [`ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html) to make that memory requirement explicit:

    ```python
    packed = np.ascontiguousarray(selected)

    print(packed.strides)
    # (12, 4)

    print(packed.flags.c_contiguous)
    # True
    ```

    Here NumPy copies the selected values into consecutive rows. If the input already has the required contiguous layout and dtype, `ascontiguousarray` can return it without copying. It therefore guarantees a layout, rather than independent storage.

    We can also request the component type when preparing data for a GPU buffer:

    ```python
    packed = np.ascontiguousarray(selected, dtype=np.float32)
    ```

    This preserves the array's shape whilst ensuring C-contiguous storage and 32-bit floating-point components. We still need to match that shape and dtype to the buffer layout expected by the drawing code.

    If we also need a one-dimensional array, [`ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) and [`flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) differ in how they handle storage:

    ```python
    flat_values = packed.ravel()
    independent_values = packed.flatten()
    ```

    With their default ordering, both arrange the values row by row. `ravel` returns a view where possible and copies when necessary. `flatten` always creates a copy, so changing `independent_values` will not affect `packed`.

    I use `ascontiguousarray` when the requirement is packed storage, `ravel` when I need a flat sequence, and `flatten` when I specifically need an independent flat copy.
    """)
    return


@app.cell
def _(np, triangles):
    _expanded = triangles.reshape(-1, 3)
    _every_other = _expanded[::2]
    _packed = np.ascontiguousarray(_every_other, dtype=np.float32)
    print("strided", _every_other.strides, _every_other.flags.c_contiguous)
    print("packed ", _packed.strides, _packed.flags.c_contiguous)
    print("flatten shares", np.shares_memory(_expanded, _expanded.flatten()))
    assert np.array_equal(_packed, _every_other)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Change the grid to five x samples and three z samples. Predict the triangle count.
    2. Reverse each face with `faces[:, ::-1]`. Does the wireframe reveal the change?
    3. Translate only the y component using a column slice.
    4. Gather two vertices, modify the result, then check the source. Repeat using a slice.
    5. Combine two `(2, 3)` vertex arrays with `concatenate`, `stack` and `column_stack`. Predict each output shape and explain which one adds more vertices.

    [Part 3](NumPyForCGPart3Transforms.py) applies matrices to our points.
    """)
    return


if __name__ == "__main__":
    app.run()
