#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics: exercise solutions

    These are worked answers to all 36 exercises in [Parts 1–7](README.md). I have kept
    the numbering from each lesson so we can compare the answer with the original
    question. Try the exercises first; there is usually more than one way to write the code.

    This notebook generates its own data and runs without opening the other lessons.
    The plots use Plotly, as in the series. The WebGPU answers also show their array
    calculations on the CPU; enable the checkbox in this notebook's Part 7 section to render the images.

    - [Part 1: arrays](#part-1-arrays-and-vertex-data)
    - [Part 2: grids](#part-2-shapes-views-and-grids)
    - [Part 3: transformations](#part-3-transformations)
    - [Part 4: images](#part-4-images-and-colour)
    - [Part 5: particles](#part-5-random-points-and-particles)
    - [Part 6: meshes](#part-6-a-lit-mesh)
    - [Part 7: WebGPU](#part-7-webgpu)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: arrays and vertex data

    [Back to Part 1](NumPyForCGPart1Arrays.py).

    ### 1.1 Four positions and a closed square

    I have moved the triangle's top point to `(1, 1)` and added `(-1, 1)` to make a
    square. Four xyz positions stored as `float32` occupy `4 * 3 * 4 = 48` bytes.
    The repeated first point closes the plotted line; it is separate from the four-corner array.
    """)
    return


@app.cell
def _(go, mo, np):
    square_positions = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32
    )
    _closed = np.concatenate((square_positions, square_positions[:1]))
    print(
        "four positions",
        square_positions.nbytes,
        "bytes; closed polyline",
        _closed.nbytes,
        "bytes",
    )
    assert square_positions.nbytes == 48
    _fig = go.Figure(go.Scatter(x=_closed[:, 0], y=_closed[:, 1], mode="lines+markers"))
    _fig.update_layout(
        template="plotly_white", height=360, xaxis_title="x", yaxis_title="y"
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.ui.plotly(_fig)
    return (square_positions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 A circle with 32 samples

    `endpoint=False` gives 32 different positions around the circle. With `endpoint=True`
    the last angle is $2\pi$, which returns us to the first position (allowing for float
    error), so there are only 31 distinct locations. We still have 32 rows in both arrays.
    I use an absolute tolerance of `1e-6` when comparing these `float32` coordinates near zero.
    """)
    return


@app.cell
def _(go, mo, np):
    _fig = go.Figure()
    for _endpoint in (False, True):
        _angles = np.linspace(0, 2 * np.pi, 32, endpoint=_endpoint, dtype=np.float32)
        _points = np.column_stack((np.cos(_angles), np.sin(_angles)))
        print(
            "endpoint",
            _endpoint,
            "shape",
            _points.shape,
            "first equals last",
            np.allclose(_points[0], _points[-1], atol=1e-6),
        )
        assert bool(np.allclose(_points[0], _points[-1], atol=1e-6)) == _endpoint
        _fig.add_trace(
            go.Scatter(
                x=_points[:, 0],
                y=_points[:, 1],
                mode="markers",
                name=f"endpoint={_endpoint}",
            )
        )
    _fig.update_layout(
        template="plotly_white", height=380, xaxis_title="x", yaxis_title="y"
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Add an RGB attribute

    The original position, normal and UV fields take 32 bytes. Three more `float32`
    components add 12 bytes. The colour therefore starts at byte 32 and the stride is
    44 bytes. A renderer using this buffer needs the same stride and colour offset.
    """)
    return


@app.cell
def _(np):
    extended_vertex_dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("uv", np.float32, 2),
            ("colour", np.float32, 3),
        ]
    )
    for _field in extended_vertex_dtype.names:
        print(_field, "offset", extended_vertex_dtype.fields[_field][1])
    print("stride", extended_vertex_dtype.itemsize)
    assert extended_vertex_dtype.fields["colour"][1] == 32
    assert extended_vertex_dtype.itemsize == 44
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 How large is an RGBA frame?

    There are 480 rows, 640 pixels per row and four one-byte channels per pixel.
    That is 1,228,800 bytes: 1.2288 MB using decimal units, or 1.171875 MiB.
    This is the array's pixel storage, without Python object overhead or GPU row padding.
    """)
    return


@app.cell
def _(np):
    _frame = np.zeros((480, 640, 4), dtype=np.uint8)
    frame_bytes = _frame.nbytes
    print("frame", _frame.shape, "bytes", frame_bytes, "MiB", frame_bytes / 1024**2)
    assert frame_bytes == 480 * 640 * 4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.5 Eight samples in a half-open interval

    Use integer indices for `arange`, then divide by eight. The equivalent `linspace`
    call asks for eight samples and excludes the endpoint. Both end at 7/8, not 1.
    """)
    return


@app.cell
def _(np):
    _indices = np.arange(8) / 8
    _counted = np.linspace(0, 1, 8, endpoint=False)
    print("arange version", _indices, "linspace version", _counted)
    assert len(_indices) == len(_counted) == 8
    assert np.array_equal(_indices, _counted)
    assert _indices[-1] == 7 / 8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: shapes, views and grids

    [Back to Part 2](NumPyForCGPart2Shapes.py).

    ### 2.1 Five x samples and three z samples

    There are `(5-1) * (3-1) = 8` quads and two triangles per quad, giving 16 triangles.
    There are 15 shared positions. I have put the grid calculation in a small function
    so we can reuse it for the mesh answers later; it is the same indexing as the lesson.
    """)
    return


@app.cell
def _(np):
    def make_grid(x_count: int, z_count: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build an xz grid and its upward-facing triangle indices.

        Parameters
        ----------
        x_count : int
            number of samples along x
        z_count : int
            number of samples along z
        """
        x = np.linspace(-3, 3, x_count, dtype=np.float32)
        z = np.linspace(-2, 2, z_count, dtype=np.float32)
        xx, zz = np.meshgrid(x, z, indexing="xy")
        positions = np.stack((xx, np.zeros_like(xx), zz), axis=-1).reshape(-1, 3)
        ids = np.arange(len(positions)).reshape(zz.shape)
        tl, tr = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
        bl, br = ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
        faces = np.stack((tl, bl, tr, tr, bl, br), axis=1).reshape(-1, 3)
        return positions, faces

    grid_points, grid_faces = make_grid(5, 3)
    print("points", grid_points.shape, "triangles", grid_faces.shape)
    assert grid_faces.shape == (16, 3)
    return grid_faces, grid_points, make_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Reverse the winding

    The wireframe looks the same because each triangle still connects the same three
    positions. Its orientation changes: the cross product points along -y instead of
    +y. A wireframe alone cannot tell us the winding. Lighting or back-face culling can.
    """)
    return


@app.cell
def _(go, grid_faces, grid_points, make_subplots, mo, np):
    reverse_faces = grid_faces[:, ::-1]
    _fig = make_subplots(
        rows=1, cols=2, subplot_titles=["original winding", "reversed winding"]
    )
    for _column, _faces in enumerate((grid_faces, reverse_faces), start=1):
        _closed = grid_points[_faces[:, [0, 1, 2, 0]]]
        _lines = np.full((len(_faces), 5, 3), np.nan)
        _lines[:, :4] = _closed
        _lines = _lines.reshape(-1, 3)
        _fig.add_trace(
            go.Scatter(x=_lines[:, 0], y=_lines[:, 2], mode="lines", showlegend=False),
            row=1,
            col=_column,
        )
        _a, _b, _c = grid_points[_faces[0]]
        print("first face", _faces[0], "cross product", np.cross(_b - _a, _c - _a))
        _fig.update_xaxes(title_text="x", row=1, col=_column)
        _fig.update_yaxes(
            title_text="z",
            scaleanchor="x" if _column == 1 else "x2",
            scaleratio=1,
            row=1,
            col=_column,
        )
    _fig.update_layout(template="plotly_white", height=350)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Translate only y

    The slice `[:, 1:2]` keeps the coordinate axis, giving shape `(15, 1)`.
    Adding 2 to that slice moves all y values and leaves x and z alone.
    I copy the positions first so this solution does not change the previous cells.
    """)
    return


@app.cell
def _(grid_points, np):
    raised_points = grid_points.copy()
    raised_points[:, 1:2] += 2
    print("before", grid_points[:3], "after", raised_points[:3], sep="\n")
    np.testing.assert_array_equal(raised_points[:, [0, 2]], grid_points[:, [0, 2]])
    assert np.all(raised_points[:, 1] == 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 A gathered copy and a sliced view

    Integer array indexing gathers a copy, so changing `gathered` leaves the source
    alone. The basic slice shares storage, so changing `view` changes the source.
    Both examples start from the same values and make the same assignment.
    """)
    return


@app.cell
def _(np):
    _source = np.arange(12, dtype=np.float32).reshape(4, 3)
    _gathered = _source[[0, 1]]
    _gathered[:] = -1
    print("after changing gathered rows:\n", _source)
    assert np.all(_source >= 0)
    _view = _source[:2]
    _view[:] = -1
    print("after changing sliced rows:\n", _source)
    assert np.all(_source[:2] == -1)
    print(
        "gather shares",
        np.shares_memory(_source, _gathered),
        "slice shares",
        np.shares_memory(_source, _view),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.5 Combine vertex arrays in three ways

    Concatenating along axis zero appends rows, so that is the operation for adding
    vertices. Stack makes a new batch axis. Column stacking adds more components to
    each row, which is useful for attributes but does not add vertices.
    """)
    return


@app.cell
def _(np):
    _a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    _b = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    for _name, _result in [
        ("concatenate", np.concatenate((_a, _b))),
        ("stack", np.stack((_a, _b))),
        ("column_stack", np.column_stack((_a, _b))),
    ]:
        print(_name, _result.shape, "\n", _result)
    assert np.concatenate((_a, _b)).shape == (4, 3)
    assert np.stack((_a, _b)).shape == (2, 2, 3)
    assert np.column_stack((_a, _b)).shape == (2, 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: transformations

    [Back to Part 3](NumPyForCGPart3Transforms.py).

    ### 3.1 Change the order to R @ T @ S

    For `T @ R @ S`, the centre moves to `(2, 1)`. For `R @ T @ S`, the translation
    is rotated too: the centre becomes approximately `(1.232, 1.866)`. The final
    rotation is about the origin, rather than about the translated centre.
    """)
    return


@app.cell
def _(go, mo, np, square_positions):
    _theta = np.deg2rad(30)
    _c, _s = np.cos(_theta), np.sin(_theta)
    _rotation = np.array(
        [[_c, -_s, 0, 0], [_s, _c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    _translation = np.eye(4, dtype=np.float32)
    _translation[:3, 3] = [2, 1, 0]
    _scale = np.diag([1.5, 1, 1, 1]).astype(np.float32)
    trs_matrix = _translation @ _rotation @ _scale
    rts_matrix = _rotation @ _translation @ _scale
    original_homogeneous = np.column_stack(
        (square_positions, np.ones(4, dtype=np.float32))
    )
    transformed_trs = original_homogeneous @ trs_matrix.T
    _transformed_rts = original_homogeneous @ rts_matrix.T
    _fig = go.Figure()
    for _points, _label in [
        (transformed_trs, "T @ R @ S"),
        (_transformed_rts, "R @ T @ S"),
    ]:
        _closed = np.concatenate((_points, _points[:1]))
        _fig.add_trace(
            go.Scatter(
                x=_closed[:, 0], y=_closed[:, 1], mode="lines+markers", name=_label
            )
        )
        print(_label, "centre", _points.mean(axis=0)[:2])
    _fig.update_layout(
        template="plotly_white", height=420, xaxis_title="x", yaxis_title="y"
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.ui.plotly(_fig)
    return original_homogeneous, transformed_trs, trs_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 A direction and a point

    The point has $w=1$, the direction $w=0$. Both are scaled and rotated, but only
    the point receives the translation. Their transformed xyz values differ by `(2, 1, 0)`.
    The homogeneous w component remains 1 for the point and 0 for the direction.
    """)
    return


@app.cell
def _(np, trs_matrix):
    _point = trs_matrix @ np.array([1, 0, 0, 1], dtype=np.float32)
    _direction = trs_matrix @ np.array([1, 0, 0, 0], dtype=np.float32)
    print("point", _point, "direction", _direction)
    print("xyz difference", (_point - _direction)[:3])
    assert np.allclose((_point - _direction)[:3], [2, 1, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Undo the transformation

    The inverse must undo the whole product. With points stored in rows we multiply
    by `inverse.T`, just as we used `model.T` for the forward operation.
    The non-zero scale makes this matrix invertible; a zero scale would lose information.
    """)
    return


@app.cell
def _(np, original_homogeneous, transformed_trs, trs_matrix):
    recovered_points = transformed_trs @ np.linalg.inv(trs_matrix).T
    print("recovered:\n", recovered_points)
    print("maximum error", np.abs(recovered_points - original_homogeneous).max())
    assert np.allclose(recovered_points, original_homogeneous, atol=1e-6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Multiplying components instead of matrices

    Both results have shape `(2, 2)`, so checking the shape alone misses this mistake.
    For the top-left matrix product we calculate `1*2 + 2*1 = 4`.
    The corresponding component product is just `1*2 = 2`.
    """)
    return


@app.cell
def _(np):
    _a = np.array([[1, 2], [3, 4]])
    _b = np.array([[2, 0], [1, 2]])
    print("A * B:\n", _a * _b, "shape", (_a * _b).shape)
    print("A @ B:\n", _a @ _b, "shape", (_a @ _b).shape)
    assert not np.array_equal(_a * _b, _a @ _b)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.5 Comparing a float32 result with zero

    Relative tolerance is multiplied by the reference magnitude, which is zero here.
    The float32 sine result has an absolute error of about $9\times10^{-8}$, so an
    absolute tolerance of `1e-6` accepts it. This tolerance is a choice for this
    calculation, not a rule that every small scene coordinate should become zero.
    """)
    return


@app.cell
def _(np):
    _value = np.sin(np.float32(np.pi))
    print("sin(pi)", _value)
    print("relative only", np.isclose(_value, 0, rtol=1e-5, atol=0))
    print("absolute tolerance", np.isclose(_value, 0, rtol=0, atol=1e-6))
    assert not np.isclose(_value, 0, rtol=1e-5, atol=0)
    assert np.isclose(_value, 0, rtol=0, atol=1e-6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4: images and colour

    [Back to Part 4](NumPyForCGPart4Images.py).

    ### 4.1 Paint a horizontal red line

    Selecting one row and all columns gives the whole line. Assigning RGBA broadcasts
    one colour over that row. There is no loop over pixels.
    """)
    return


@app.cell
def _(go, mo, np):
    _line_pixels = np.full((8, 12, 4), [20, 30, 50, 255], dtype=np.uint8)
    _line_pixels[3, :] = [255, 0, 0, 255]
    assert np.all(_line_pixels[3] == [255, 0, 0, 255])
    _fig = go.Figure(go.Image(z=_line_pixels, colormodel="rgba256"))
    _fig.update_layout(
        template="plotly_white", height=300, xaxis_title="column", yaxis_title="row"
    )
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Replace the circle with a checkerboard

    Integer division finds the tile number. Adding the row and column tile numbers,
    then taking remainder 2, alternates between zero and one. The extra final axis on
    the mask lets `where` select all three RGB components for each pixel.
    """)
    return


@app.cell
def _(go, mo, np):
    _rows, _cols = np.indices((96, 160))
    _mask = ((_rows // 16 + _cols // 16) % 2) == 0
    checker_pixels = np.where(_mask[..., None], [240, 240, 240], [30, 30, 30]).astype(
        np.uint8
    )
    print("checkerboard", checker_pixels.shape, "white pixels", np.count_nonzero(_mask))
    _fig = go.Figure(go.Image(z=checker_pixels, colormodel="rgb"))
    _fig.update_layout(
        template="plotly_white", height=360, xaxis_title="column", yaxis_title="row"
    )
    mo.ui.plotly(_fig)
    return (checker_pixels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Check alpha zero and one

    At alpha zero only the blue background remains. At alpha one only the foreground
    remains. These are useful endpoint checks before trying fractions. The calculation
    here uses the same linear RGB assumption and opaque background as the lesson.
    """)
    return


@app.cell
def _(checker_pixels, go, make_subplots, mo, np):
    _foreground = checker_pixels.astype(np.float32) / 255
    _background = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    alpha_zero = 0 * _foreground + 1 * _background
    alpha_one = 1 * _foreground + 0 * _background
    assert np.allclose(alpha_zero, _background)
    assert np.allclose(alpha_one, _foreground)
    _fig = make_subplots(rows=1, cols=2, subplot_titles=["alpha = 0", "alpha = 1"])
    for _column, _image in enumerate((alpha_zero, alpha_one), start=1):
        _bytes = np.rint(_image * 255).astype(np.uint8)
        _fig.add_trace(go.Image(z=_bytes, colormodel="rgb"), row=1, col=_column)
        _fig.update_xaxes(title_text="column", row=1, col=_column)
        _fig.update_yaxes(title_text="row", row=1, col=_column)
    _fig.update_layout(template="plotly_white", height=360)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Reverse the rows

    The row stride changes from +480 to -480 bytes. Each move to the next displayed
    row now moves backwards through the source array. The column and channel strides
    stay 3 and 1 bytes. It is a view, so no pixel data has been copied.
    """)
    return


@app.cell
def _(checker_pixels, np):
    flipped_pixels = checker_pixels[::-1]
    print("original strides", checker_pixels.strides)
    print("reversed strides", flipped_pixels.strides)
    print("shares storage", np.shares_memory(checker_pixels, flipped_pixels))
    assert flipped_pixels.strides[0] == -checker_pixels.strides[0]
    assert np.array_equal(flipped_pixels[0], checker_pixels[-1])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.5 Two-colour selection and masked assignment

    Keep a length-one channel axis on the mask so `where` can broadcast two RGB
    colours. The assignment version starts from a blue image and paints red into
    the same selected pixels. Both allocate new storage, leaving the checkerboard alone.
    """)
    return


@app.cell
def _(checker_pixels, np):
    _mask = checker_pixels[..., 0] > 128
    _selected = np.where(_mask[..., None], [255, 0, 0], [0, 0, 255]).astype(np.uint8)
    _assigned = np.full(checker_pixels.shape, [0, 0, 255], dtype=np.uint8)
    _assigned[_mask] = [255, 0, 0]
    print("mask shape", _mask.shape, "RGB result", _selected.shape)
    assert np.array_equal(_selected, _assigned)
    assert not np.shares_memory(_selected, checker_pixels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5: random points and particles

    [Back to Part 5](NumPyForCGPart5Particles.py).

    ### 5.1 Halve the time step

    Both runs cover one second. For constant acceleration, semi-implicit Euler adds
    an extra $\frac12 a\,t\,\Delta t$ to the analytic position. Halving the time step
    halves that error. I compare trajectories before applying lifetime or ground masks,
    so we are checking the integrator on the same particles.
    """)
    return


@app.cell
def _(np):
    def integrate_batch(
        initial_velocity: np.ndarray, dt: float, steps: int, wind: float = 0.0
    ) -> np.ndarray:
        """
        Advance a particle batch using semi-implicit Euler.

        Parameters
        ----------
        initial_velocity : np.ndarray
            one xyz velocity per particle
        dt : float
            time step in seconds
        steps : int
            number of updates
        wind : float
            constant x acceleration
        """
        positions = np.zeros_like(initial_velocity)
        velocity = initial_velocity.copy()
        acceleration = np.array([wind, -9.81, 0], dtype=np.float32)
        for _ in range(steps):
            velocity += acceleration * dt
            positions += velocity * dt
        return positions

    _rng = np.random.default_rng(7)
    launch_velocities = _rng.uniform([-2, 4, -2], [2, 8, 2], (200, 3)).astype(
        np.float32
    )
    coarse_positions = integrate_batch(launch_velocities, 1 / 60, 60)
    _fine_positions = integrate_batch(launch_velocities, 1 / 120, 120)
    _exact = launch_velocities + np.array([0, -9.81 / 2, 0], dtype=np.float32)
    coarse_error = float(np.max(np.abs(coarse_positions - _exact)))
    fine_error = float(np.max(np.abs(_fine_positions - _exact)))
    print(
        "60 steps error",
        coarse_error,
        "120 steps error",
        fine_error,
        "ratio",
        fine_error / coarse_error,
    )
    assert fine_error < coarse_error
    return coarse_positions, integrate_batch, launch_velocities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Add wind along x

    I treat wind here as a constant acceleration of 2 units/s² along +x. After one
    second the cloud shifts by about +1.017 units with this integrator. Its x extent
    does not widen because every particle receives the same acceleration; y and z are unchanged.
    """)
    return


@app.cell
def _(coarse_positions, integrate_batch, launch_velocities, np):
    _windy = integrate_batch(launch_velocities, 1 / 60, 60, wind=2)
    wind_shift = (_windy - coarse_positions).mean(axis=0)
    print(
        "no wind x bounds", coarse_positions[:, 0].min(), coarse_positions[:, 0].max()
    )
    print("wind x bounds", _windy[:, 0].min(), _windy[:, 0].max())
    print("mean shift", wind_shift)
    assert np.allclose(wind_shift, [1 + 1 / 60, 0, 0], atol=1e-5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Respawn a complete particle

    The same mask must reset position, velocity and age. Resetting only position
    leaves the old falling velocity or expired lifetime, which can kill the particle
    again on the next update. I keep the assigned lifetime for each slot here; it
    could also be sampled again when the particle is born.
    """)
    return


@app.cell
def _(go, mo, np):
    _rng = np.random.default_rng(12)
    respawn_positions = np.zeros((200, 3), dtype=np.float32)
    respawn_velocities = _rng.uniform([-2, 4, -2], [2, 8, 2], (200, 3)).astype(
        np.float32
    )
    respawn_ages = np.zeros(200, dtype=np.float32)
    _lifetimes = _rng.uniform(0.3, 0.7, 200).astype(np.float32)
    respawn_events = 0
    _dt = 1 / 60
    for _ in range(60):
        respawn_velocities += np.array([0, -9.81, 0], dtype=np.float32) * _dt
        respawn_positions += respawn_velocities * _dt
        respawn_ages += _dt
        _dead = (respawn_ages >= _lifetimes) | (respawn_positions[:, 1] < 0)
        _count = int(np.count_nonzero(_dead))
        respawn_events += _count
        respawn_positions[_dead] = [0, 0, 0]
        respawn_velocities[_dead] = _rng.uniform(
            [-2, 4, -2], [2, 8, 2], (_count, 3)
        ).astype(np.float32)
        respawn_ages[_dead] = 0
        assert np.all(respawn_positions[_dead] == 0)
        assert np.all(respawn_velocities[_dead, 1] >= 4)
        assert np.all(respawn_ages[_dead] == 0)
    print("births after the initial batch", respawn_events)
    assert np.all(respawn_ages < _lifetimes)
    _fig = go.Figure(
        go.Scatter3d(
            x=respawn_positions[:, 0],
            y=respawn_positions[:, 1],
            z=respawn_positions[:, 2],
            mode="markers",
            marker={
                "size": 3,
                "color": respawn_ages,
                "colorscale": "Plasma",
                "colorbar": {"title": "age / s"},
            },
        )
    )
    _fig.update_layout(
        template="plotly_white",
        height=430,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "height / y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "camera": {"up": {"x": 0, "y": 1, "z": 0}},
        },
    )
    mo.ui.plotly(_fig)
    return (respawn_positions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 Points inside a sphere

    A sphere's enclosed volume grows as $r^3$. If $u$ is uniform on 0–1, choosing
    $r=u^{1/3}$ makes enclosed volume uniform. The mean radius should approach 0.75,
    and about one eighth of the points should lie within radius 0.5.

    Sampling the radius uniformly instead gives mean radius 0.5 and puts half the
    points in that small central region. The histogram compares these radius distributions;
    more points near the outside is what uniform volume requires.
    """)
    return


@app.cell
def _(go, mo, np):
    _rng = np.random.default_rng(21)
    _directions = _rng.normal(size=(6000, 3))
    _directions /= np.linalg.norm(_directions, axis=1, keepdims=True)
    _samples = _rng.random(6000)
    volume_radii = _samples ** (1 / 3)
    uniform_radii = _samples
    _inside = _directions * volume_radii[:, None]
    assert np.allclose(np.linalg.norm(_inside, axis=1), volume_radii)
    _fig = go.Figure()
    for _radii, _name in [
        (volume_radii, "uniform volume"),
        (uniform_radii, "uniform radius"),
    ]:
        print(
            _name,
            "mean radius",
            _radii.mean(),
            "fraction inside 0.5",
            np.mean(_radii < 0.5),
        )
        _fig.add_trace(
            go.Histogram(
                x=_radii,
                xbins={"start": 0, "end": 1, "size": 0.05},
                histnorm="probability",
                opacity=0.6,
                name=_name,
            )
        )
    _fig.update_layout(
        template="plotly_white",
        height=380,
        barmode="overlay",
        xaxis_title="radius",
        yaxis_title="fraction per bin",
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.5 Reducing a cloud along its axes

    Use the respawned cloud from 5.3. Axis zero combines particles, leaving xyz;
    axis one combines components within a particle. The kept column mean has shape
    `(1, 3)` and the kept squared-radius array has shape `(200, 1)`.
    """)
    return


@app.cell
def _(np, respawn_positions):
    _centroid = respawn_positions.mean(axis=0)
    _kept_centroid = respawn_positions.mean(axis=0, keepdims=True)
    _radius_squared = np.sum(
        respawn_positions * respawn_positions, axis=1, keepdims=True
    )
    print("centroid", _centroid, _centroid.shape, "kept", _kept_centroid.shape)
    print("bounds", respawn_positions.min(axis=0), respawn_positions.max(axis=0))
    print("radius squared", _radius_squared.shape)
    assert _kept_centroid.shape == (1, 3)
    assert _radius_squared.shape == (200, 1)
    assert np.allclose((respawn_positions - _kept_centroid).mean(axis=0), 0, atol=1e-6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6: a lit mesh

    [Back to Part 6](NumPyForCGPart6Mesh.py).

    ### 6.1 Zero amplitude

    At zero amplitude all triangles lie in the xz plane. With the lesson's winding,
    their unit normals are `(0, 1, 0)`. For light `[sin(45°), 1, cos(45°)]`, normalising
    it gives a y component of $1/\sqrt{2}$, so every diffuse value is about 0.7071.
    """)
    return


@app.cell
def _(make_grid, np):
    def normalise_faces(triangles: np.ndarray) -> np.ndarray:
        """
        Calculate face normals, leaving zero-area faces at zero.

        Parameters
        ----------
        triangles : np.ndarray
            xyz positions with shape (face count, 3, 3)
        """
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(cross, axis=1, keepdims=True)
        normals = np.zeros_like(cross)
        np.divide(cross, lengths, out=normals, where=lengths > 1e-8)
        return normals

    flat_points, dense_faces = make_grid(25, 17)
    flat_normals = normalise_faces(flat_points[dense_faces])
    _theta = np.deg2rad(45)
    light_vector = np.array([np.sin(_theta), 1, np.cos(_theta)], dtype=np.float32)
    light_vector /= np.linalg.norm(light_vector)
    flat_diffuse = np.clip(flat_normals @ light_vector, 0, 1)
    print(
        "first normal",
        flat_normals[0],
        "diffuse range",
        flat_diffuse.min(),
        flat_diffuse.max(),
    )
    assert np.allclose(flat_normals, [0, 1, 0])
    assert np.allclose(flat_diffuse, 1 / np.sqrt(2))
    return dense_faces, flat_points, light_vector, normalise_faces


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Reverse the winding and clip the light

    Reversing a flat face changes its normal to `(0, -1, 0)`. The dot product with
    our upward light becomes negative, so `clip(..., 0, 1)` makes the diffuse term zero.
    The lesson's ambient term still contributes 0.15 of the base colour; the surface
    becomes dark rather than completely black. Culling, if enabled, is a separate effect.
    """)
    return


@app.cell
def _(dense_faces, flat_points, light_vector, normalise_faces, np):
    _reversed = normalise_faces(flat_points[dense_faces[:, ::-1]])
    _dot = _reversed @ light_vector
    _diffuse = np.clip(_dot, 0, 1)
    print("reversed normal", _reversed[0], "dot", _dot[0], "clipped", _diffuse[0])
    print("remaining colour factor", 0.15 + 0.85 * _diffuse[0])
    assert np.allclose(_diffuse, 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 A radial ripple

    I use `cos(3*r)` where $r=\sqrt{x^2+z^2}$. Recalculate the face normals after
    changing height. Cosine gives a smooth centre at the origin; `sin(3*r)` would
    have a cusp there. Both use radius to make concentric ripples.
    """)
    return


@app.cell
def _(dense_faces, flat_points, go, light_vector, mo, normalise_faces, np):
    ripple_points = flat_points.copy()
    _radius = np.sqrt(ripple_points[:, 0] ** 2 + ripple_points[:, 2] ** 2)
    ripple_points[:, 1] = 0.5 * np.cos(3 * _radius)
    ripple_normals = normalise_faces(ripple_points[dense_faces])
    _diffuse = np.clip(ripple_normals @ light_vector, 0, 1)
    _rgb = np.rint(
        (0.15 + 0.85 * _diffuse[:, None]) * np.array([0.15, 0.65, 0.9]) * 255
    ).astype(np.uint8)
    _fig = go.Figure(
        go.Mesh3d(
            x=ripple_points[:, 0],
            y=ripple_points[:, 1],
            z=ripple_points[:, 2],
            i=dense_faces[:, 0],
            j=dense_faces[:, 1],
            k=dense_faces[:, 2],
            facecolor=[f"rgb({r},{g},{b})" for r, g, b in _rgb],
            flatshading=True,
            lighting={"ambient": 1, "diffuse": 0, "specular": 0, "fresnel": 0},
        )
    )
    _fig.update_layout(
        template="plotly_white",
        height=480,
        scene={
            "xaxis_title": "x",
            "yaxis_title": "height / y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "eye": {"x": 1.5, "y": 1, "z": 1.5},
            },
        },
    )
    assert np.allclose(np.linalg.norm(ripple_normals, axis=1), 1)
    mo.ui.plotly(_fig)
    return ripple_normals, ripple_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 A degenerate triangle

    Here all three vertices lie on a line. The cross product is zero and cannot be
    normalised. The masked division leaves the initial zero normal alone, without
    producing NaNs. It contributes no diffuse light; a mesh-cleaning step could remove it.
    """)
    return


@app.cell
def _(dense_faces, normalise_faces, np, ripple_points):
    _good = ripple_points[dense_faces[:1]]
    _bad = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
    _normals = normalise_faces(np.concatenate((_good, _bad)))
    degenerate_normal = _normals[-1]
    print("valid normal", _normals[0], "degenerate normal", degenerate_normal)
    assert np.array_equal(degenerate_normal, [0, 0, 0])
    assert np.isfinite(_normals).all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 Repeat each normal, or tile the whole array?

    `repeat` produces `A A A B B B C C C`, keeping a face's normal with its three
    corners. `tile` produces `A B C A B C A B C`, repeating the whole batch.
    Using tile here would assign other faces' normals to the corners.
    """)
    return


@app.cell
def _(np):
    _labels = np.array(["A", "B", "C"])
    repeat_labels = np.repeat(_labels, 3)
    tile_labels = np.tile(_labels, 3)
    print("repeat", repeat_labels)
    print("tile  ", tile_labels)
    _normals = np.eye(3, dtype=np.float32)
    print(
        "normals for the first triangle with repeat:\n",
        np.repeat(_normals, 3, axis=0)[:3],
    )
    print("normals for the first triangle with tile:\n", np.tile(_normals, (3, 1))[:3])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.6 Inverse-transpose normals

    The tangent becomes `(2, 1, 0)`. Direct scaling gives normal `(-2, 1, 0)`, whose
    dot product with that tangent is -3 before normalisation. The inverse transpose
    gives `(-0.5, 1, 0)` and a dot product of zero. Normalising changes lengths, not
    that perpendicularity. This uses column vectors; normal rows would multiply by
    the transpose of the normal matrix.
    """)
    return


@app.cell
def _(np):
    _scale = np.diag([2.0, 1.0, 1.0])
    _tangent = _scale @ np.array([1.0, 1.0, 0.0])
    _direct = _scale @ np.array([-1.0, 1.0, 0.0])
    _correct = np.linalg.inv(_scale).T @ np.array([-1.0, 1.0, 0.0])
    print("tangent", _tangent, "direct normal", _direct, "correct normal", _correct)
    print("dot products", _tangent @ _direct, _tangent @ _correct)
    assert np.isclose(_tangent @ _correct, 0, atol=1e-12)
    assert not np.isclose(_tangent @ _direct, 0, atol=1e-12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7: WebGPU

    [Back to Part 7](NumPyForCGPart7WebGPU.py).

    The following cells use the lesson's shader with an offscreen render helper
    that can also swap attribute offsets. This keeps the solutions notebook
    independent. The GPU checkbox is off initially; the CPU checks still run.
    """)
    return


@app.cell
def _(mo):
    gpu_solutions = mo.ui.checkbox(value=False, label="run the WebGPU solution renders")
    gpu_solutions
    return (gpu_solutions,)


@app.cell
def _():
    solution_shader = """
    struct VertexOut {
        @builtin(position) position: vec4<f32>,
        @location(0) colour: vec3<f32>,
    };
    @vertex
    fn vertex_main(@location(0) position: vec3<f32>,
                   @location(1) colour: vec3<f32>) -> VertexOut {
        var output: VertexOut;
        output.position = vec4<f32>(position, 1.0);
        output.colour = colour;
        return output;
    }
    @fragment
    fn fragment_main(input: VertexOut) -> @location(0) vec4<f32> {
        return vec4<f32>(input.colour, 1.0);
    }
    """
    return (solution_shader,)


@app.cell
def _(np):
    def render_exercise(
        data: np.ndarray,
        source: str,
        width: int = 512,
        height: int = 384,
        swap_attributes: bool = False,
    ) -> np.ndarray:
        """
        Render packed position and colour rows into an RGBA array.

        Parameters
        ----------
        data : np.ndarray
            float32 rows containing xyz followed by RGB
        source : str
            WGSL vertex and fragment shader source
        width : int
            output width in pixels
        height : int
            output height in pixels
        swap_attributes : bool
            read colour as position and position as colour for the mistake demonstration
        """
        import wgpu
        from wgpu.utils import get_default_device

        position_offset, colour_offset = (12, 0) if swap_attributes else (0, 12)
        device = get_default_device()
        shader = device.create_shader_module(code=source)
        pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 24,
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": "float32x3",
                                "offset": position_offset,
                                "shader_location": 0,
                            },
                            {
                                "format": "float32x3",
                                "offset": colour_offset,
                                "shader_location": 1,
                            },
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [{"format": "rgba8unorm"}],
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
        )
        vertex_buffer = device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        colour_texture = device.create_texture(
            size=(width, height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        depth_texture = device.create_texture(
            size=(width, height, 1),
            format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        try:
            encoder = device.create_command_encoder()
            render_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": colour_texture.create_view(),
                        "resolve_target": None,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (0.08, 0.08, 0.1, 1),
                    }
                ],
                depth_stencil_attachment={
                    "view": depth_texture.create_view(),
                    "depth_clear_value": 1,
                    "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
            )
            render_pass.set_pipeline(pipeline)
            render_pass.set_vertex_buffer(0, vertex_buffer)
            render_pass.draw(len(data))
            render_pass.end()
            device.queue.submit([encoder.finish()])
            row_bytes = ((width * 4 + 255) // 256) * 256
            raw = device.queue.read_texture(
                {"texture": colour_texture, "mip_level": 0, "origin": (0, 0, 0)},
                {"offset": 0, "bytes_per_row": row_bytes, "rows_per_image": height},
                (width, height, 1),
            )
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, row_bytes)
            return rows[:, : width * 4].reshape(height, width, 4).copy()
        finally:
            vertex_buffer.destroy()
            colour_texture.destroy()
            depth_texture.destroy()

    return (render_exercise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.1 A width of 512 pixels

    An RGBA row is `512*4 = 2048` bytes. This is already divisible by 256, so the
    padded row size is also 2048 and there are no padding bytes. At width 510 there
    are still 8 padding bytes. Our optional offscreen renderer below uses this alignment.
    """)
    return


@app.cell
def _():
    for _width in (510, 512):
        _raw = _width * 4
        _aligned = ((_raw + 255) // 256) * 256
        print("width", _width, "row bytes", _raw, "padding", _aligned - _raw)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.2 Red vertices and an interior pixel

    I project the ripple mesh with the fixed view angle of 30 degrees from Part 7,
    then set the final three columns to red. The fragment shader should return
    `[255, 0, 0, 255]` for covered pixels. We find an interior pixel from the red mask
    rather than assuming a particular screen coordinate lies inside the mesh.
    """)
    return


@app.cell
def _(dense_faces, np, ripple_points):
    _theta = np.deg2rad(30)
    _rotation = np.array(
        [
            [np.cos(_theta), 0, np.sin(_theta)],
            [0, 1, 0],
            [-np.sin(_theta), 0, np.cos(_theta)],
        ],
        dtype=np.float32,
    )
    _rotated = ripple_points[dense_faces].reshape(-1, 3) @ _rotation.T
    clip_positions = np.column_stack(
        (
            _rotated[:, 0] / 4.5,
            (_rotated[:, 1] - 0.5 * _rotated[:, 2]) / 3,
            0.5 + _rotated[:, 2] / 12,
        )
    ).astype(np.float32)
    red_upload = np.zeros((len(clip_positions), 6), dtype=np.float32)
    red_upload[:, :3] = clip_positions
    red_upload[:, 3:] = [1, 0, 0]
    print("red upload", red_upload.shape, "stride", red_upload.strides[0])
    assert np.all(red_upload[:, 3:] == [1, 0, 0])
    return clip_positions, red_upload


@app.cell
def _(gpu_solutions, mo, np, red_upload, render_exercise, solution_shader):
    if gpu_solutions.value:
        _red_image = render_exercise(red_upload, solution_shader)
        _mask = np.all(_red_image == [255, 0, 0, 255], axis=-1)
        assert np.any(_mask)
        _row, _column = np.argwhere(_mask)[len(np.argwhere(_mask)) // 2]
        print("interior pixel x, y", _column, _row, "RGBA", _red_image[_row, _column])
        _display = mo.image(_red_image, alt="Red mesh from the exercise vertex buffer")
    else:
        _display = mo.md(
            "Enable the GPU checkbox to render the red mesh and check an interior pixel."
        )
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.3 Swap the position and colour offsets

    The formats are both `float32x3` with a 24-byte stride, so these offsets are still
    valid to WebGPU. The data is wrong rather than the layout being invalid.

    With our constant red colour, every position read by the vertex shader becomes
    `(1, 0, 0)`. All triangles collapse to that point and no fragments are drawn.
    The result is just the clear colour. With varying colours the mistaken positions
    would depend on those colours, so a different buffer could produce different geometry.
    """)
    return


@app.cell
def _(gpu_solutions, mo, np, red_upload, render_exercise, solution_shader):
    _wrong_positions = red_upload[:, 3:].reshape(-1, 3, 3)
    _wrong_cross = np.cross(
        _wrong_positions[:, 1] - _wrong_positions[:, 0],
        _wrong_positions[:, 2] - _wrong_positions[:, 0],
    )
    assert np.all(_wrong_cross == 0)
    print("zero-area triangles after swapping offsets", len(_wrong_cross))
    if gpu_solutions.value:
        _wrong_image = render_exercise(
            red_upload, solution_shader, swap_attributes=True
        )
        assert np.all(_wrong_image == _wrong_image[0, 0])
        print("clear RGBA", _wrong_image[0, 0])
        _display = mo.image(
            _wrong_image,
            alt="Clear background after swapping position and colour offsets",
        )
    else:
        _display = mo.md("Enable the GPU checkbox to render the swapped-offset result.")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.4 Move the light, keep the view fixed

    The slider changes the light vector and the face colours. `clip_positions` remains
    unchanged, so this is a lighting change rather than a camera rotation. The Plotly
    preview shows the same per-face colours without requiring a local wgpu adapter;
    the optional image below comes from the GPU.
    """)
    return


@app.cell
def _(mo):
    light_heading = mo.ui.slider(
        -180, 180, value=45, label="light direction in degrees", show_value=True
    )
    light_heading
    return (light_heading,)


@app.cell
def _(
    clip_positions,
    dense_faces,
    go,
    light_heading,
    mo,
    np,
    ripple_normals,
    ripple_points,
):
    _theta = np.deg2rad(light_heading.value)
    _light = np.array([np.sin(_theta), 1, np.cos(_theta)], dtype=np.float32)
    _light /= np.linalg.norm(_light)
    _diffuse = np.clip(ripple_normals @ _light, 0, 1)
    _colours = (0.15 + 0.85 * _diffuse[:, None]) * np.array(
        [0.2, 0.7, 1], dtype=np.float32
    )
    lit_upload = np.ascontiguousarray(
        np.column_stack((clip_positions, np.repeat(_colours, 3, axis=0))),
        dtype=np.float32,
    )
    assert np.array_equal(lit_upload[:, :3], clip_positions)
    _rgb = np.rint(_colours * 255).astype(np.uint8)
    _fig = go.Figure(
        go.Mesh3d(
            x=ripple_points[:, 0],
            y=ripple_points[:, 1],
            z=ripple_points[:, 2],
            i=dense_faces[:, 0],
            j=dense_faces[:, 1],
            k=dense_faces[:, 2],
            facecolor=[f"rgb({r},{g},{b})" for r, g, b in _rgb],
            flatshading=True,
            lighting={"ambient": 1, "diffuse": 0, "specular": 0, "fresnel": 0},
        )
    )
    _fig.update_layout(
        template="plotly_white",
        height=460,
        uirevision="solution-light-view",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "height / y",
            "zaxis_title": "z",
            "aspectmode": "data",
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "eye": {"x": 1.5, "y": 1, "z": 1.5},
            },
        },
    )
    mo.ui.plotly(_fig)
    return (lit_upload,)


@app.cell
def _(gpu_solutions, lit_upload, mo, render_exercise, solution_shader):
    if gpu_solutions.value:
        _display = mo.image(
            render_exercise(lit_upload, solution_shader),
            alt="Ripple mesh with the slider-controlled light and fixed projection",
        )
    else:
        _display = mo.md(
            "The Plotly preview updates with the light; enable the GPU checkbox for the WebGPU image."
        )
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.5 What should change per frame?

    The animated MarimoDemos version updates its rotation and matrices each frame.
    Its random point positions and colours can stay in GPU buffers until the point
    cloud changes. Creating the pipeline, textures and readback buffer once avoids
    repeating those allocations on every frame.

    | Data or resource | When it changes | What I would keep |
    | --- | --- | --- |
    | Point positions and colours | when regenerating the cloud | the arrays and vertex buffers |
    | Rotation, view and projection matrices | rotation per frame; camera or resize as needed | one uniform buffer, updated with `write_buffer` |
    | Pipeline and shader module | when shader or layout changes | the compiled pipeline and shader |
    | Colour, depth and multisample textures | when output size or format changes | the textures and their views |
    | Readback buffer | when output size changes | a buffer sized for the padded rows |
    | Command encoder and render pass | each frame | create and finish them for that frame |
    | Returned RGBA bytes | after each rendered frame | the widget; update its pixel data |

    For the solutions here, changing the light only changes the colour columns in
    `lit_upload`. A persistent renderer could update that vertex buffer while reusing
    its pipeline and textures. A deformation would require new positions and normals
    too. In Part 7's CPU-projected version the clip positions change with the view;
    in the animated PyNGL demo that transformation is carried by the uniform matrices.

    The short helper deliberately creates and releases its resources on each call so
    we can inspect one render without managing a long-lived scene. The optional checkbox
    controls GPU work; the array answers and Plotly figures remain available without it.
    """)
    return


if __name__ == "__main__":
    app.run()
