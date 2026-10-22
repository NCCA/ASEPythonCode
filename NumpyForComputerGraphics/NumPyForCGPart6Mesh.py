#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 6: putting it together — a lit mesh

    We will now combine the operations from the earlier lessons to build a mesh. We start with a regular grid, displace its Y coordinates to create a surface, then split each grid cell into two triangles. From those triangles, we calculate face normals, use them for a diffuse lighting calculation and pack the attributes into vertex records.

    This brings together ideas from the [TriangleGrid](../GraphicsDemos/TriangleGrid/WebGPU/WebGPUTriangleGrid.py), [UpdateMesh](../GraphicsDemos/UpdateMesh/WebGPU/WebGPUTriangleGridWireFrame.py) and [FlatShaded](../GraphicsDemos/FlatShaded/WebGPU/PhongShaderWithWireFrame.wgsl) examples.

    I use a directional diffuse light here so we can see how the normals affect the surface colour. A directional light has the same direction at every point, so we do not need to calculate a separate direction or distance to a light position. The full shader also includes ambient and specular terms.

    Our plot uses one calculated colour per face, giving us flat shading. It lets us inspect the geometry and lighting before passing the vertex buffer to a renderer. Adjacent triangles can have different colours because their normals point in different directions, even when they share an edge.

    Drag the [Plotly mesh](https://plotly.com/python/3d-mesh/) to rotate it and hover over it to inspect the coordinates. I set Plotly's diffuse and specular lighting contributions to zero and use its ambient term to display the supplied face colours. The shading we inspect therefore comes from our NumPy calculation.

    ## Check the cross product first

    Before calculating normals for the whole mesh, I check one triangle whose normal direction we already know. This helps catch a reversed winding order before it affects every face.

    For triangle vertices $a$, $b$ and $c$, we construct two edges starting at $a$:

    $$
    e_1 = b-a,\qquad e_2 = c-a
    $$

    Their cross product gives a vector perpendicular to the triangle:

    $$
    n = e_1 \times e_2
    $$

    The order matters. Swapping the edges reverses the result, so reversing the triangle's winding also reverses its normal.

    For example, this triangle lies in the XZ plane:

    ```python
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    c = np.array([1.0, 0.0, 0.0])

    normal = np.cross(b - a, c - a)
    ```

    The result is `[0.0, 1.0, 0.0]`, pointing along positive Y as intended.

    The cross product's length is twice the triangle's area. Its direction is suitable for a face normal, but its length depends on the geometry. For lighting, we divide by that length to obtain a unit normal:

    $$
    \hat{n} = \frac{n}{\lVert n\rVert}
    $$

    A zero-area triangle produces a zero vector and has no defined face normal. We need to detect these degenerate triangles before dividing. Very small triangles can also require care, with the threshold chosen to suit the scale and precision of the geometry.

    [`cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) and [`norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) can perform these calculations across a batch. If `triangles` has shape `(triangle_count, 3, 3)`, its axes select the triangle, the vertex and the XYZ component:

    ```python
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]

    face_normals = np.cross(edge_1, edge_2)
    normal_lengths = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    ```

    `face_normals` has shape `(triangle_count, 3)`. Using `axis=-1` calculates a length across each normal's XYZ components, whilst `keepdims=True` retains shape `(triangle_count, 1)`. Once we have handled degenerate faces, those lengths broadcast across the three components so we can normalise each face independently.
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
    _test_triangle = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    _normal = np.cross(
        _test_triangle[1] - _test_triangle[0],
        _test_triangle[2] - _test_triangle[0],
    )
    print("forward winding", _normal, "reverse", -_normal)
    assert np.allclose(_normal, [0, 1, 0])
    _degenerate = np.cross([1, 0, 0], [2, 0, 0])
    print("collinear edges", _degenerate)
    assert np.linalg.norm(_degenerate) == 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Change the amplitude and light angle separately. With zero amplitude all normals
    should point upwards. Moving the light changes the colours without moving a vertex.
    The phase shifts the wave; we rebuild from the original xz grid each time.
    """)
    return


@app.cell
def _(mo):
    amplitude = mo.ui.slider(
        0, 1.5, step=0.1, value=0.6, label="wave amplitude", show_value=True
    )
    phase = mo.ui.slider(
        0, 6.28, step=0.1, value=0, label="wave phase", show_value=True
    )
    light_angle = mo.ui.slider(
        -180, 180, value=45, label="light angle", show_value=True
    )
    mo.hstack([amplitude, phase, light_angle])
    return amplitude, light_angle, phase


@app.cell
def _(amplitude, np, phase):
    _x = np.linspace(-3, 3, 25, dtype=np.float32)
    _z = np.linspace(-2, 2, 17, dtype=np.float32)
    _xx, _zz = np.meshgrid(_x, _z, indexing="xy")
    _yy = amplitude.value * np.sin(_xx + phase.value) * np.cos(_zz)
    mesh_points = (
        np.stack((_xx, _yy, _zz), axis=-1).reshape(-1, 3).astype(np.float32)
    )
    _ids = np.arange(len(mesh_points)).reshape(_xx.shape)
    _tl, _tr = _ids[:-1, :-1].ravel(), _ids[:-1, 1:].ravel()
    _bl, _br = _ids[1:, :-1].ravel(), _ids[1:, 1:].ravel()
    mesh_faces = np.stack((_tl, _bl, _tr, _tr, _bl, _br), axis=1).reshape(-1, 3)
    mesh_triangles = mesh_points[mesh_faces]
    print("points", mesh_points.shape, "faces", mesh_faces.shape)
    print("bounds", mesh_points.min(axis=0), mesh_points.max(axis=0))
    assert mesh_faces.shape == (24 * 16 * 2, 3)
    return mesh_faces, mesh_points, mesh_triangles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normalise along the component axis

    Our face normals have shape `(face_count, 3)`: one row per face, with three XYZ components. To normalise them, we need a separate length for each row:

    ```python
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    ```

    `axis=1` combines the XYZ components within each face normal. `keepdims=True` retains that axis with size one, giving `lengths` shape `(face_count, 1)`. These lengths can then broadcast across the three components of their corresponding normals.

    Omitting the axis would calculate one length from every component in the array. Dividing by that scalar would run, but it would scale the whole collection together rather than give each face a unit normal. Our diffuse lighting calculation would then depend on triangle area as well as orientation.

    We also need to handle degenerate faces. A zero-area triangle produces a zero-length normal, which we cannot normalise by division. [`divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html) lets us supply both an output array and a condition controlling where division happens:

    ```python
    unit_normals = np.zeros_like(face_normals)

    np.divide(
        face_normals,
        lengths,
        out=unit_normals,
        where=lengths > 0.0,
    )
    ```

    The condition has shape `(face_count, 1)` and broadcasts across each row. Non-zero normals are divided by their lengths, whilst zero-length normals are skipped.

    Initialising `unit_normals` matters because entries where `where` is `False` retain their existing output values. Starting with zeros gives skipped faces a predictable result instead of leaving uninitialised data.

    A zero vector is a placeholder for an undefined normal, not a valid unit normal. It contributes zero to our diffuse dot product, but we may instead choose to remove or report degenerate faces. For nearly degenerate triangles, we can use a positive threshold rather than an exact zero check, chosen to suit the scale and precision of the geometry.
    """)
    return


@app.cell
def _(mesh_triangles, np):
    _edges_a = mesh_triangles[:, 1] - mesh_triangles[:, 0]
    _edges_b = mesh_triangles[:, 2] - mesh_triangles[:, 0]
    _cross = np.cross(_edges_a, _edges_b)
    _lengths = np.linalg.norm(_cross, axis=1, keepdims=True)
    face_normals = np.zeros_like(_cross)
    np.divide(_cross, _lengths, out=face_normals, where=_lengths > 1e-8)
    print("normal shape", face_normals.shape, "length shape", _lengths.shape)
    print("wrong global length", np.linalg.norm(_cross))
    assert np.allclose(np.linalg.norm(face_normals, axis=1), 1)
    assert np.all(face_normals[:, 1] > 0)
    return (face_normals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A dot product per face

    The dot product between a unit normal and a unit direction towards the light tells us how directly the face points towards that light:

    $$
    \hat{n} \cdot \hat{l} = \cos(\theta)
    $$

    Here $\theta$ is the angle between the two directions. A face pointing directly towards the light gives `1.0`. A face perpendicular to the light direction gives zero, whilst a face pointing away gives a negative value.

    Both vectors must have unit length for this interpretation. We have already normalised the face normals, so we also normalise the light direction:

    ```python
    light = np.array([1.0, 2.0, 1.0])
    light /= np.linalg.norm(light)
    ```

    Here `light` points from the surface towards the light source. If a renderer instead supplies the direction in which the light travels, we need to reverse that direction for this calculation.

    With normals of shape `(face_count, 3)` and a light direction of shape `(3,)`, matrix multiplication calculates one dot product per face:

    ```python
    diffuse = unit_normals @ light
    ```

    The result has shape `(face_count,)`. Each value combines the three components of one face normal with the corresponding components of the shared light direction. We can express the same calculation using broadcasting and a reduction:

    ```python
    diffuse = np.sum(unit_normals * light, axis=1)
    ```

    Multiplication applies the light's XYZ components to every normal, then the sum reduces those three products to one value per face.

    A light behind the surface should add no diffuse contribution, so we clamp negative values to zero:

    ```python
    diffuse = np.clip(diffuse, 0.0, 1.0)
    ```

    The upper limit also handles small floating-point overshoots above one. This gives us a diffuse factor which we can apply to a base RGB colour:

    ```python
    base_colour = np.array([0.1, 0.6, 0.9])
    face_colours = diffuse[:, None] * base_colour
    ```

    Adding the final axis changes `diffuse` to shape `(face_count, 1)`, allowing each face's factor to multiply all three colour components. The result has shape `(face_count, 3)`: one RGB colour per triangle. These colours represent the diffuse term alone, so faces receiving no direct light are black unless we add another contribution, such as ambient light.
    """)
    return


@app.cell
def _(face_normals, light_angle, np):
    _theta = np.deg2rad(light_angle.value)
    _light = np.array([np.sin(_theta), 1, np.cos(_theta)], dtype=np.float32)
    _light /= np.linalg.norm(_light)
    diffuse = np.clip(face_normals @ _light, 0, 1)
    assert np.allclose(
        face_normals @ _light, np.sum(face_normals * _light, axis=1)
    )
    face_colours = (0.15 + 0.85 * diffuse[:, None]) * np.array(
        [0.15, 0.65, 0.9], dtype=np.float32
    )
    print("diffuse range", diffuse.min(), diffuse.max())
    assert np.all((face_colours >= 0) & (face_colours <= 1))
    return (face_colours,)


@app.cell
def _(face_colours, go, mesh_faces, mesh_points, mesh_triangles, mo, np):
    _rgb = np.rint(face_colours * 255).astype(np.uint8)
    _colours = [f"rgb({r},{g},{b})" for r, g, b in _rgb]
    mesh_figure = go.Figure(
        go.Mesh3d(
            x=mesh_points[:, 0],
            y=mesh_points[:, 1],
            z=mesh_points[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            facecolor=_colours,
            flatshading=True,
            lighting={
                "ambient": 1,
                "diffuse": 0,
                "specular": 0,
                "fresnel": 0,
                "roughness": 1,
            },
            hovertemplate="x=%{x:.3f}<br>height=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            name="NumPy face colours",
        )
    )
    _edges = np.full((len(mesh_triangles), 5, 3), np.nan)
    _edges[:, :4] = mesh_triangles[:, [0, 1, 2, 0]]
    _lines = _edges.reshape(-1, 3)
    mesh_figure.add_trace(
        go.Scatter3d(
            x=_lines[:, 0],
            y=_lines[:, 1],
            z=_lines[:, 2],
            mode="lines",
            line={"color": "rgba(40,40,40,0.25)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mesh_figure.update_layout(
        template="plotly_white",
        height=600,
        uirevision="mesh-view",
        scene={
            "xaxis": {"title": "x", "range": [-3, 3]},
            "yaxis": {"title": "height / y", "range": [-1.6, 1.6]},
            "zaxis": {"title": "z", "range": [-2, 2]},
            "aspectmode": "data",
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "eye": {"x": 1.5, "y": 1, "z": 1.5},
            },
        },
    )
    mo.ui.plotly(mesh_figure, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pack the upload buffer

    Each face has its own normal, so we duplicate it for all three corners using
    [`repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html).
    That gives flat shading. Smooth shading would need shared vertex normals averaged
    from neighbouring faces (and care at hard edges).

    The UVs map x to 0–1 and z to 1–0, following the flipped v coordinate in the demo.
    Our 768 triangles become 2304 vertex records with a 32-byte stride. Changing the
    wave requires updating normals as well as positions.

    ## Why the surface looks faceted

    The surface is made from flat triangles. We calculate one normal per triangle and use it to calculate one colour for the whole face. This is called flat shading.

    Neighbouring triangles generally point in slightly different directions after we displace the grid's Y coordinates. Their normals therefore make different angles with the light. In our diffuse lighting calculation, the brightness depends on the dot product between the unit normal and the unit direction towards the light:

    $$
    I = \max(0, \hat{n} \cdot \hat{l})
    $$

    A face pointing towards the light receives a larger contribution, whilst a face pointing away receives none. Because each triangle has one normal and one colour, brightness changes abruptly at its edges. These changes reveal the individual faces and give the mesh its faceted appearance.

    Increasing the grid resolution creates smaller triangles, which follow the curved surface more closely. The facets become less obvious, but each triangle still has a constant colour.

    For smooth shading, we instead provide normals at the vertices and interpolate them across each triangle. Shared vertices need consistent normals, often calculated by averaging the normals of neighbouring faces. A shader then normalises the interpolated normal and calculates lighting for each fragment.

    Smooth shading makes the lighting vary gradually across the surface, but it does not change the geometry. The silhouette still follows the triangle edges. Here I keep flat shading so we can inspect each face's orientation and check our normal calculation.
    """)
    return


@app.cell
def _(face_normals, mesh_triangles, np):
    _dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("uv", np.float32, 2),
        ]
    )
    mesh_buffer = np.zeros(mesh_triangles.shape[0] * 3, dtype=_dtype)
    mesh_buffer["position"] = mesh_triangles.reshape(-1, 3)
    mesh_buffer["normal"] = np.repeat(face_normals, 3, axis=0)
    mesh_buffer["uv"][:, 0] = (mesh_buffer["position"][:, 0] + 3) / 6
    mesh_buffer["uv"][:, 1] = 1 - (mesh_buffer["position"][:, 2] + 2) / 4
    print(
        "vertices",
        len(mesh_buffer),
        "stride",
        _dtype.itemsize,
        "bytes",
        mesh_buffer.nbytes,
    )
    print("first vertex", mesh_buffer[0])
    assert mesh_buffer.nbytes == 2304 * 32
    assert np.isfinite(mesh_buffer["normal"]).all()
    assert np.all((mesh_buffer["uv"] >= 0) & (mesh_buffer["uv"] <= 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transforming normals under non-uniform scale

    [Part 3](NumPyForCGPart3Transforms.py) uses a model matrix to transform points and directions. A surface normal has an additional requirement: it must remain perpendicular to the surface after the transformation.

    A tangent points along the surface, so we transform it using the model matrix's linear 3×3 part, $A$. Applying that same matrix to a normal works for rotations and, after normalisation, uniform positive scaling. Non-uniform scaling can change the angle between the two vectors, leaving the transformed normal tilted away from its correct direction.

    For an invertible transform, we instead use the inverse transpose:

    $$
    N = (A^{-1})^T
    $$

    This is the normal matrix. If the original normal $n$ and tangent $t$ satisfy $n \cdot t = 0$, then:

    $$
    (Nn) \cdot (At) = n \cdot t = 0
    $$

    The inverse transpose therefore preserves the perpendicular relationship. We normalise the transformed normal afterwards because its length may have changed.

    In NumPy, we construct the matrix from the linear part of the model transform:

    ```python
    linear = model[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    ```

    Translation does not appear here. Moving a surface changes its position, but does not change the directions of its tangents or normals.

    Our arrays store vectors as rows, so the batch calculations are:

    ```python
    transformed_tangents = tangents @ linear.T
    transformed_normals = normals @ normal_matrix.T

    transformed_normals /= np.linalg.norm(
        transformed_normals, axis=1, keepdims=True
    )
    ```

    To see why the distinction matters, consider these perpendicular directions:

    ```text
    tangent = (1, 1, 0)
    normal  = (1, -1, 0)
    ```

    Stretching X by two transforms the tangent to `(2, 1, 0)`. Applying the same scale directly to the normal gives `(2, -1, 0)`. Their dot product is `3`, so they are no longer perpendicular. Normalising them changes their lengths but does not repair the angle.

    The inverse transpose instead gives the normal `(0.5, -1, 0)`. Its dot product with the transformed tangent is zero, as required.

    A zero scale makes the matrix singular, so its inverse does not exist. The slider deliberately stays above zero to keep this example invertible.

    Change the X scale and compare the two dot products. After normalising the transformed tangent and each candidate normal, the correct pair should give a value close to zero. At scale one, both methods agree, which is why this mistake can go unnoticed until a model is stretched. Some axis-aligned normals also remain correct under non-uniform scale, so I use a sloping tangent and normal to make the difference visible.
    """)
    return


@app.cell
def _(mo):
    normal_x_scale = mo.ui.slider(
        0.25,
        3,
        step=0.25,
        value=2,
        label="normal example: x scale",
        show_value=True,
    )
    normal_x_scale
    return (normal_x_scale,)


@app.cell
def _(go, mo, normal_x_scale, np):
    _tangent = np.array([1.0, 1.0, 0.0])
    _normal = np.array([-1.0, 1.0, 0.0])
    _linear = np.diag([normal_x_scale.value, 1.0, 1.0])
    _normal_matrix = np.linalg.inv(_linear).T
    scaled_tangent = _linear @ _tangent
    scaled_tangent /= np.linalg.norm(scaled_tangent)
    wrong_normal = _linear @ _normal
    wrong_normal /= np.linalg.norm(wrong_normal)
    correct_normal = _normal_matrix @ _normal
    correct_normal /= np.linalg.norm(correct_normal)
    wrong_normal_dot = float(scaled_tangent @ wrong_normal)
    normal_dot = float(scaled_tangent @ correct_normal)
    print("directly scaled normal dot tangent", wrong_normal_dot)
    print("inverse-transpose normal dot tangent", normal_dot)
    assert np.isclose(normal_dot, 0, atol=1e-12)
    normal_figure = go.Figure()
    for _vector, _name in [
        (scaled_tangent, "transformed tangent"),
        (wrong_normal, "directly scaled normal"),
        (correct_normal, "inverse-transpose normal"),
    ]:
        normal_figure.add_trace(
            go.Scatter(
                x=[0, _vector[0]],
                y=[0, _vector[1]],
                mode="lines+markers",
                name=_name,
            )
        )
    normal_figure.update_layout(
        template="plotly_white",
        height=440,
        xaxis_title="x",
        yaxis_title="y",
        uirevision="normal-example",
    )
    normal_figure.update_xaxes(range=[-1.2, 1.2])
    normal_figure.update_yaxes(range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)
    mo.vstack(
        [
            mo.md(
                f"Dot products with the tangent: direct scale `{wrong_normal_dot:.4f}`; inverse transpose `{normal_dot:.4f}`."
            ),
            mo.ui.plotly(normal_figure),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Set amplitude to zero and check the normals and diffuse values.
    2. Reverse the winding. Explain why clipping the dot product makes the surface dark.
    3. Replace the wave with a radial ripple using `np.sqrt(x*x + z*z)`.
    4. Add a deliberately degenerate face and confirm its normal remains zero.
    5. Compare `repeat(face_normals, 3, axis=0)` with `tile(face_normals, (3, 1))`.
       Which one keeps each face's three corners together?
    6. Scale the tangent `(1, 1, 0)` and normal `(-1, 1, 0)` by `(2, 1, 1)`. Compare a directly transformed normal with the inverse-transpose result using dot products.

    [Part 7](NumPyForCGPart7WebGPU.py) renders a NumPy triangle buffer with WebGPU and
    reads the result back as an RGBA image, following the MarimoDemos WebGPU examples.
    """)
    return


if __name__ == "__main__":
    app.run()
