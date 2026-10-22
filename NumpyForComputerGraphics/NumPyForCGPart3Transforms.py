#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 3: transformations

    [Part 2](NumPyForCGPart2Shapes.py) built a grid by combining coordinate arrays. Now we will use matrices to move, rotate and scale a shape. The same approach appears in the model, view and projection calculations in the graphics demos.

    A model matrix places an object in the scene. A view matrix expresses the scene relative to the camera, and a projection matrix maps view coordinates into clip coordinates before the perspective divide. We will start with a two-dimensional rotation so we can check the calculation by hand.

    ## [Matrix multiplication](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

    I use column-vector maths, where a transformation matrix multiplies a point on its left:

    $$
    p' = M p
    $$

    For a two-dimensional point, this means:

    $$
    \begin{bmatrix}
    x' \\
    y'
    \end{bmatrix}
    =
    \begin{bmatrix}
    m_{00} & m_{01} \\
    m_{10} & m_{11}
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    $$

    Each output component is a row–column product:

    $$
    x' = m_{00}x + m_{01}y
    $$

    $$
    y' = m_{10}x + m_{11}y
    $$

    In NumPy, `@` performs matrix multiplication. For example, this matrix rotates a point 90 degrees anticlockwise in an XY plane where positive X points right and positive Y points up:

    ```python
    rotation = np.array([
        [0.0, -1.0],
        [1.0,  0.0],
    ])

    point = np.array([1.0, 0.0])
    rotated_point = rotation @ point
    ```

    The result is `[0.0, 1.0]`. We can check it directly: the first component is `0 × 1 + -1 × 0 = 0`, and the second is `1 × 1 + 0 × 0 = 1`.

    Here `point` has shape `(2,)`. NumPy's one-dimensional arrays do not have a separate row or column orientation; their role depends on the operation. In `rotation @ point`, NumPy treats the point as the vector on the right of the matrix.

    Our geometry arrays store one point per row, giving a batch shape of `(N, 2)`. To apply the same column-vector transformation to every point, we write:

    ```python
    points = np.array([
        [ 1.0, 0.0],
        [ 0.0, 1.0],
        [-1.0, 0.0],
    ])

    rotated_points = points @ rotation.T
    ```

    The result is:

    ```text
    [[ 0.  1.]
     [-1.  0.]
     [ 0. -1.]]
    ```

    The transpose follows from rewriting the original equation for a point stored as a row:

    $$
    p'^T = p^T M^T
    $$

    We are applying the same transformation, with the multiplication arranged to suit our array layout. For a matrix with shape `(D, D)` and points with shape `(N, D)`, the batch expression is therefore `points @ matrix.T`. Transposing the matrix does not rearrange our stored points or change the intended transformation.

    The `*` operator performs element-by-element multiplication instead:

    ```python
    rotation * rotation
    ```

    This squares each component independently. It does not calculate the row–column sums needed to compose transformations. I use `*` for operations such as scaling individual coordinate components, and `@` when applying or combining matrices.
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
    _rotation90 = np.array([[0, -1], [1, 0]], dtype=np.float32)
    _point = np.array([1, 0], dtype=np.float32)
    print("column vector", _rotation90 @ _point)
    print("row batch", _point[None, :] @ _rotation90.T)
    print("component multiplication:\n", _rotation90 * _point)
    assert np.allclose(_rotation90 @ _point, [0, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Homogeneous coordinates and order

    A point has $w=1$, so the last column of a 4×4 matrix can translate it.
    A direction has $w=0$, so translation does not affect it.

    With column vectors, `T @ R @ S` scales first, then rotates, then translates.
    Try swapping the order below: rotating a translated point moves it around the origin.
    This is the same ordering idea as `projection @ view @ model` in the demos.
    """)
    return


@app.cell
def _(np):
    _translation = np.eye(4, dtype=np.float32)
    _translation[:3, 3] = [2, 3, 0]
    print("point    ", _translation @ [1, 0, 0, 1])
    print("direction", _translation @ [1, 0, 0, 0])
    _rotation = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    print("rotate then translate", _translation @ _rotation @ [1, 0, 0, 1])
    print("translate then rotate", _rotation @ _translation @ [1, 0, 0, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change the angle and x scale. Watch the square's corners and the location of its
    centre. The shape is rebuilt from its original positions on each run, so moving
    a slider does not accumulate transformations.
    """)
    return


@app.cell
def _(mo):
    angle = mo.ui.slider(
        -180, 180, value=30, label="angle in degrees", show_value=True
    )
    x_scale = mo.ui.slider(
        0.25, 3, step=0.25, value=1.5, label="x scale", show_value=True
    )
    mo.hstack([angle, x_scale])
    return angle, x_scale


@app.cell
def _(angle, go, mo, np, x_scale):
    _shape = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32
    )
    _homogeneous = np.column_stack((_shape, np.ones(4, dtype=np.float32)))
    _theta = np.deg2rad(angle.value)
    _c, _s = np.cos(_theta), np.sin(_theta)
    _rotate = np.array(
        [[_c, -_s, 0, 0], [_s, _c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    _scale = np.diag([x_scale.value, 1, 1, 1]).astype(np.float32)
    _translate = np.eye(4, dtype=np.float32)
    _translate[:3, 3] = [2, 1, 0]
    model = _translate @ _rotate @ _scale
    transformed = _homogeneous @ model.T
    transform_figure = go.Figure()
    for _data, _label in [(_homogeneous, "original"), (transformed, "T @ R @ S")]:
        _closed = np.concatenate((_data, _data[:1]))
        transform_figure.add_trace(
            go.Scatter(
                x=_closed[:, 0],
                y=_closed[:, 1],
                mode="lines+markers",
                name=_label,
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    transform_figure.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title="x",
        yaxis_title="y",
        uirevision="transform-view",
    )
    transform_figure.update_xaxes(range=[-4, 6])
    transform_figure.update_yaxes(range=[-4, 6], scaleanchor="x", scaleratio=1)
    mo.ui.plotly(
        transform_figure, config={"displaylogo": False, "scrollZoom": True}
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Floating-point checks need a tolerance

    The [introduction](../Lecture5/IntroductionToNumpy.py) explains why `0.1 + 0.2` may not equal `0.3` exactly. Transformations and trigonometric functions also introduce rounding errors. A rotation that should produce zero may instead give us a small value close to zero, so an exact equality check can fail even when the result is accurate enough for our calculation.

    [`isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) compares corresponding values and returns a Boolean mask. [`allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) returns one Boolean indicating whether every corresponding pair is close enough:

    ```python
    expected = np.array([0.0, 1.0])

    np.isclose(rotated_point, expected, atol=1e-6, rtol=0.0)
    np.allclose(rotated_point, expected, atol=1e-6, rtol=0.0)
    ```

    For each entry, NumPy checks whether:

    ```text
    abs(actual - reference) <= atol + rtol * abs(reference)
    ```

    The absolute tolerance, `atol`, allows a fixed difference in the units of our values. The relative tolerance, `rtol`, allows a difference proportional to the magnitude of the reference value.

    When the reference is zero, the relative term is also zero. We therefore need an absolute tolerance to accept small rounding errors around zero. In this example, `rtol=0.0` makes the check depend entirely on the absolute tolerance.

    I use `1e-6` here for the expected rounding error in this float32 rotation example. It is not a universal threshold: the tolerance should reflect the precision, scale and units of the calculation, as well as how much error we can accept.
    """)
    return


@app.cell
def _(np):
    _near_zero = np.sin(np.float32(np.pi))
    print("sin(pi), float32", _near_zero)
    print("relative tolerance only", np.isclose(_near_zero, 0, rtol=1e-5, atol=0))
    near_zero_match = np.isclose(_near_zero, 0, rtol=0, atol=1e-6)
    print("absolute tolerance", near_zero_match)
    assert near_zero_match
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perspective division and matrix bytes

    A projection matrix produces homogeneous clip coordinates with four components: `(x, y, z, w)`. To obtain normalised device coordinates (NDC), we divide the first three components by `w`:

    $$
    (x_{\mathrm{ndc}}, y_{\mathrm{ndc}}, z_{\mathrm{ndc}})
    =
    \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)
    $$

    The `w` component is part of the projection calculation. For a conventional perspective projection, it depends on the point's depth, so dividing by it makes objects appear smaller as they move away from the camera.

    We can check the operation using invented clip coordinates:

    ```python
    clip = np.array([
        [2.0, 4.0, 1.0, 2.0],
        [3.0, 6.0, 2.0, 3.0],
    ])

    ndc = clip[:, :3] / clip[:, 3:4]
    ```

    `clip[:, :3]` selects XYZ and has shape `(2, 3)`. The slice `clip[:, 3:4]` selects W whilst retaining shape `(2, 1)`. Broadcasting then divides all three components in each row by that row's W value:

    ```text
    [[1.         2.         0.5       ]
     [1.         2.         0.66666667]]
    ```

    Keeping the second dimension with `3:4` matters. Using `clip[:, 3]` would produce shape `(2,)`, which does not broadcast against `(2, 3)` in the way we need.

    A point with `w=0` cannot undergo this division. Values very close to zero can also produce very large results. In a CPU example, we need to identify these cases before dividing.

    Perspective division alone does not clip geometry or establish visibility. In the graphics pipeline, clipping operates on homogeneous clip coordinates before the divide. A triangle crossing a clipping plane may need new vertices at its intersections with that plane; checking or dividing its original vertices does not perform that work.

    ### Packing a matrix for WGSL

    The mathematical convention and the order of bytes in a buffer are separate questions. We use column-vector maths, `p' = M p`, but store batches of points as NumPy rows and calculate `points @ M.T` on the CPU.

    Uploading a matrix introduces another requirement: its bytes must match the shader's expected layout. A WGSL `mat4x4<f32>` is stored as four columns, each containing four 32-bit floats. For a conventional NumPy matrix whose entries are addressed as `model[row, column]`, we can pack those columns like this:

    ```python
    model_bytes = model.T.copy(order="C").astype(np.float32).tobytes()
    ```

    The transpose makes each original column a row, and the C-order copy stores those rows consecutively. The resulting byte sequence therefore contains the original matrix's columns in order.

    This prepares the storage expected by WGSL; it does not change the transformation we intend the shader to apply.

    I only do this when I know the source matrix follows that NumPy convention. A graphics library may already provide matrix data in the required order. Before adding a transpose, check the library's upload contract and test a transformation with an obvious result, such as translating the origin to `(2, 3, 4)`.
    """)
    return


@app.cell
def _(model, np):
    _clip = np.array([[2, 4, 1, 2], [1, 1, 0, 0]], dtype=np.float32)
    _valid = ~np.isclose(_clip[:, 3], 0)
    _ndc = np.full((2, 3), np.nan, dtype=np.float32)
    np.divide(_clip[:, :3], _clip[:, 3:4], out=_ndc, where=_valid[:, None])
    print("projectable", _valid, "NDC:\n", _ndc)
    _shader_bytes = model.T.copy().tobytes()
    _round_trip = np.frombuffer(_shader_bytes, dtype=np.float32).reshape(4, 4).T
    print("matrix bytes", len(_shader_bytes))
    assert np.allclose(_ndc[0], [1, 2, 0.5])
    assert np.allclose(_round_trip, model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Change the model to `R @ T @ S`. Explain where the centre moves.
    2. Transform a direction and a point with the same matrix. Which components differ?
    3. Apply `np.linalg.inv(model)` to the transformed points and compare with the originals.
    4. Replace `@` with `*` in a small example. Check both the shape and the numbers.
    5. Compare float32 `sin(pi)` with zero using relative tolerance alone, then choose an absolute tolerance. Explain the difference.

    [Part 4](NumPyForCGPart4Images.py) treats an image as another shaped buffer.
    """)
    return


if __name__ == "__main__":
    app.run()
