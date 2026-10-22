#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 5: random points and particles

    The [Points demo](../GraphicsDemos/Points/OpenGL/OpenGLPoints.py) generates positions and colours as NumPy arrays. The [Lab 6 emitter](../Lab6/Particle/Emitter.py) instead manages individual particle objects, each with its own state.

    Here we will connect these approaches by storing particle properties in arrays and updating many particles in one calculation. Positions and velocities can each use shape `(N, 3)`, whilst properties such as age or lifetime can use shape `(N,)`. Matching entries across the arrays describe the same particle.

    This is a NumPy variation on the lab, rather than a replacement for its classes. It gives us another way to organise the data and compare how the updates are expressed.

    ## [default_rng](https://numpy.org/doc/stable/reference/random/generator.html)

    I use `np.random.default_rng` to create a random number generator. Passing a fixed seed gives us a repeatable starting point, which is useful when debugging or comparing calculations:

    ```python
    rng = np.random.default_rng(42)
    positions = rng.uniform(-1.0, 1.0, size=(1000, 3))
    ```

    This creates 1,000 XYZ positions inside a cube. Each component is sampled independently from the interval between `-1.0` and `1.0`, with the upper bound normally excluded.

    The generator keeps its own state. Each call advances that state, so generating another array gives us another set of samples. Creating a new generator with the same seed repeats the sequence for the same sequence of calls in the same environment.

    In this notebook, I create the generator inside the data cell. Rerunning that cell therefore starts from the same seed and rebuilds the same cloud. This makes it easier to see the effect of changing the calculation without also changing the random input.

    The point demos use the older `np.random.uniform` interface, which draws from shared global state. A separate generator keeps this notebook's random state local, so unrelated calls to the older interface do not change our cloud.

    ### Sampling directions on a sphere

    Random points inside a cube are not the same as random directions on a sphere. To generate directions with no preferred orientation, we can start with three independent standard normal samples per point:

    ```python
    vectors = rng.standard_normal(size=(1000, 3))
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / lengths
    ```

    The Gaussian distribution is rotationally symmetric: rotating the distribution does not change it. Normalising these vectors removes their lengths whilst preserving that symmetry, giving uniformly distributed directions on the unit sphere.

    `axis=1` calculates one length per vector. `keepdims=True` retains shape `(1000, 1)`, allowing each length to broadcast across its vector's three components. A zero-length vector cannot be normalised; general-purpose code should detect and resample it before dividing.

    Normalising uniformly sampled cube points does not give the same result. A cube extends farther from its centre towards its corners than towards its faces. More samples therefore map towards the corner directions when projected onto a sphere, producing a directional bias.

    These normalised Gaussian samples lie on the sphere's surface. Sampling uniformly throughout its volume also requires choosing an appropriate radius; multiplying by a uniformly sampled radius would concentrate points towards the centre.

    Drag each cloud to inspect its shape, then hover over a point to read its coordinates. Compare the cube's volume with the sphere's surface, and look for any preferred directions in the distribution.
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
def _(np):
    _rng = np.random.default_rng(42)
    cube_points = _rng.uniform(-1, 1, (500, 3)).astype(np.float32)
    _directions = _rng.normal(size=(500, 3)).astype(np.float32)
    _lengths = np.linalg.norm(_directions, axis=1, keepdims=True)
    sphere_points = np.zeros_like(_directions)
    np.divide(_directions, _lengths, out=sphere_points, where=_lengths > 0)
    print("cube bounds", cube_points.min(axis=0), cube_points.max(axis=0))
    print(
        "sphere radius range",
        np.linalg.norm(sphere_points, axis=1).min(),
        np.linalg.norm(sphere_points, axis=1).max(),
    )
    assert np.allclose(np.linalg.norm(sphere_points, axis=1), 1)
    return cube_points, sphere_points


@app.cell
def _(cube_points, go, make_subplots, mo, sphere_points):
    cloud_figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["uniform cube", "sphere surface"],
    )
    for _column, _points, _title in [
        (1, cube_points, "uniform cube"),
        (2, sphere_points, "sphere surface"),
    ]:
        cloud_figure.add_trace(
            go.Scatter3d(
                x=_points[:, 0],
                y=_points[:, 1],
                z=_points[:, 2],
                mode="markers",
                name=_title,
                marker={"size": 3, "opacity": 0.65},
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=_column,
        )
    cloud_figure.update_scenes(
        xaxis={"title": "x", "range": [-1.1, 1.1]},
        yaxis={"title": "y", "range": [-1.1, 1.1]},
        zaxis={"title": "z", "range": [-1.1, 1.1]},
        aspectmode="cube",
    )
    cloud_figure.update_layout(
        template="plotly_white", height=500, showlegend=False
    )
    mo.ui.plotly(cloud_figure, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reductions: the centroid, bounds and radius of a cloud

    The [introduction](../Lecture5/IntroductionToNumpy.py) shows reductions on a small matrix. A reduction combines values along an axis, using an operation such as a sum, mean, minimum or maximum.

    For positions with shape `(N, 3)`, axis `0` runs through the points and axis `1` runs through the XYZ components of each point. Choosing the axis determines which values we combine.

    ### Finding the centroid and bounds

    Taking the mean along axis `0` combines the point rows whilst keeping the three coordinate components separate:

    ```python
    centroid = points.mean(axis=0)
    ```

    The result has shape `(3,)` and contains the mean X, mean Y and mean Z. This is the centroid of the sampled points, with every point contributing equally.

    Minimum and maximum reductions along the same axis give us the bounds:

    ```python
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    ```

    These two XYZ positions describe opposite corners of an axis-aligned bounding box. Each component is calculated independently, so a box corner need not coincide with an actual point in the cloud.

    We can calculate the box's size and centre from these bounds:

    ```python
    box_size = maximum - minimum
    box_centre = (minimum + maximum) / 2.0
    ```

    The box centre and the centroid need not be the same. The box centre depends only on the extreme coordinates, whilst the centroid depends on every point.

    ### Measuring distance from a centre

    Reducing along axis `1` combines the components within each point. For example, summing squared components gives the squared distance from the origin:

    ```python
    squared_distances = np.sum(points * points, axis=1)
    ```

    The result has shape `(N,)`, with one squared distance per point. Taking the square root gives the distances themselves.

    To measure distances from the cloud's centroid, we subtract it first:

    ```python
    centroid = points.mean(axis=0, keepdims=True)
    centred_points = points - centroid

    squared_distances = np.sum(centred_points * centred_points, axis=1)
    radius = np.sqrt(squared_distances.max())
    ```

    The centroid has shape `(1, 3)` and broadcasts across all point rows. The final maximum selects the largest squared distance, giving us a radius for a sphere centred at the centroid that contains every sampled point. This is not necessarily the smallest possible enclosing sphere.

    | Expression | Result shape | Meaning |
    | --- | --- | --- |
    | `points.mean(axis=None)` | `()` | One mean across every coordinate |
    | `points.mean(axis=0)` | `(3,)` | Centroid XYZ |
    | `points.mean(axis=0, keepdims=True)` | `(1, 3)` | Centroid with its row axis kept |
    | `points.min(axis=0)` | `(3,)` | Minimum X, Y and Z |
    | `points.max(axis=0)` | `(3,)` | Maximum X, Y and Z |
    | `np.sum(points * points, axis=1, keepdims=True)` | `(N, 1)` | Squared distance from the origin per point |

    Without an axis, the mean combines all coordinate values into one scalar, losing the distinction between X, Y and Z. Usually that is not the quantity we want for geometry.

    `keepdims=True` retains the reduced axis with size one. A centroid with shape `(3,)` already broadcasts against `(N, 3)`, but keeping shape `(1, 3)` makes its role as one shared position explicit.

    For per-point lengths, retaining the component axis is particularly useful. In Part 6, lengths with shape `(N, 1)` will let us divide each normal by its own length, broadcasting that divisor across its three components.
    """)
    return


@app.cell
def _(cube_points, np):
    point_centroid = cube_points.mean(axis=0)
    column_centroid = cube_points.mean(axis=0, keepdims=True)
    squared_lengths = np.sum(cube_points * cube_points, axis=1, keepdims=True)
    print("global mean", cube_points.mean(axis=None))
    print("centroid", point_centroid, "kept shape", column_centroid.shape)
    print("bounds", cube_points.min(axis=0), cube_points.max(axis=0))
    print("squared radii", squared_lengths.shape, squared_lengths[:3].ravel())
    print("centred mean", (cube_points - column_centroid).mean(axis=0))
    assert np.allclose(
        squared_lengths[:, 0], np.linalg.norm(cube_points, axis=1) ** 2, atol=1e-6
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One particle, then a batch

    We will use semi-implicit Euler: first add `gravity * dt` to velocity, then add
    `velocity * dt` to position. The Lab 6 emitter uses a different gravity factor
    and frame-based lifetimes; here lifetime is measured in seconds.

    The integration has error, so halving the time step should change the result.
    For constant acceleration we can compare against the analytic position. The particle
    plot can be rotated to inspect its depth; hover shows the particle index and age.
    """)
    return


@app.cell
def _(np):
    _dt = 0.1
    _initial_velocity = np.array([1, 5, 0], dtype=np.float32)
    _gravity = np.array([0, -9.81, 0], dtype=np.float32)
    _next_velocity = _initial_velocity + _gravity * _dt
    _next_position = _next_velocity * _dt
    _exact_position = _initial_velocity * _dt + 0.5 * _gravity * _dt**2
    print("Euler position", _next_position)
    print("analytic      ", _exact_position)
    print("difference    ", _next_position - _exact_position)
    return


@app.cell
def _(mo):
    steps = mo.ui.slider(
        0, 120, value=30, label="simulation steps (1/60 s)", show_value=True
    )
    steps
    return (steps,)


@app.cell
def _(go, mo, np, steps):
    _rng = np.random.default_rng(7)
    _initial = np.zeros((200, 3), dtype=np.float32)
    _velocity = _rng.uniform([-2, 4, -2], [2, 8, 2], (200, 3)).astype(np.float32)
    _lifetimes = _rng.uniform(0.8, 2, 200).astype(np.float32)
    positions = _initial.copy()
    _age = np.zeros(200, dtype=np.float32)
    _gravity = np.array([0, -9.81, 0], dtype=np.float32)
    _dt = 1 / 60
    for _ in range(steps.value):
        _velocity += _gravity * _dt
        positions += _velocity * _dt
        _age += _dt
    alive = (_age < _lifetimes) & (positions[:, 1] >= 0)
    print("alive", np.count_nonzero(alive), "of", len(alive))
    print("positions", positions.shape, "lifetime", _lifetimes.shape)
    particle_figure = go.Figure(
        go.Scatter3d(
            x=positions[alive, 0],
            y=positions[alive, 1],
            z=positions[alive, 2],
            mode="markers",
            customdata=np.column_stack((np.flatnonzero(alive), _age[alive])),
            marker={
                "size": 4,
                "color": _age[alive],
                "cmin": 0,
                "cmax": 2,
                "colorscale": "Plasma",
                "colorbar": {"title": "age / s"},
            },
            hovertemplate="particle %{customdata[0]:.0f}<br>x=%{x:.3f}<br>height=%{y:.3f}<br>z=%{z:.3f}<br>age=%{customdata[1]:.2f} s<extra></extra>",
        )
    )
    particle_figure.update_layout(
        template="plotly_white",
        height=540,
        title="particle batch, replayed from its seed",
        uirevision="particle-view",
        scene={
            "xaxis": {"title": "x", "range": [-4, 4]},
            "yaxis": {"title": "height / y", "range": [0, 4]},
            "zaxis": {"title": "z", "range": [-4, 4]},
            "aspectmode": "data",
            "camera": {
                "up": {"x": 0, "y": 1, "z": 0},
                "eye": {"x": 1.5, "y": 1, "z": 1.5},
            },
        },
    )
    mo.ui.plotly(
        particle_figure, config={"displaylogo": False, "scrollZoom": True}
    )
    return alive, positions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Masks and bounds

    A Boolean mask lets us identify particles that satisfy a condition. For example, we can check which particles have moved outside a box defined by minimum and maximum XYZ coordinates:

    ```python
    outside = (positions < minimum) | (positions > maximum)
    dead = np.any(outside, axis=1)
    ```

    The comparisons broadcast the bounds, each with shape `(3,)`, across the `(N, 3)` positions. `outside` therefore has shape `(N, 3)`, with one Boolean per coordinate. Reducing with `any(axis=1)` gives shape `(N,)`: a particle is outside the box if any of its components exceeds the bounds. Points exactly on a boundary remain inside.

    We use `&` for element-wise AND and `|` for element-wise OR. For example, checking whether an X coordinate lies within an interval requires both comparisons to be true:

    ```python
    inside_x = (
        (positions[:, 0] >= minimum[0])
        & (positions[:, 0] <= maximum[0])
    )
    ```

    Keep parentheses around each comparison so Python groups the operations correctly. Python's `and` and `or` ask for a single truth value from each operand. A multi-element Boolean array has no unambiguous single truth value, so these operators raise an error. Use `np.any` or `np.all` when we deliberately want to reduce several Boolean values to one result.

    Reading through a Boolean mask gathers a copy:

    ```python
    escaped_positions = positions[dead]
    ```

    Changing `escaped_positions` will not update `positions`. Direct masked assignment behaves differently:

    ```python
    positions[dead] = 0.0
    ```

    This writes into the original array, setting all three components of each selected position to zero. The scalar broadcasts across the selected rows.

    In this notebook, I make a fresh copy before resetting particles so the operation does not change another cell's input:

    ```python
    reset_positions = positions.copy()
    reset_positions[dead] = 0.0
    ```

    This moves escaped particles back to the origin. A complete emitter would also reset their velocity and age, and any other state needed to start a new particle lifetime. The same mask can select corresponding entries in each particle array.
    """)
    return


@app.cell
def _(alive, np, positions):
    try:
        _bad_mask = (positions[:, 1] > 0) and (positions[:, 1] < 2)
    except ValueError as _error:
        print("array truth value:", _error)
    _reset = positions.copy()
    _reset[~alive] = [0, 0, 0]
    if np.any(alive):
        _visible = positions[alive]
        print("visible bounds", _visible.min(axis=0), _visible.max(axis=0))
    else:
        print("no visible particles; there are no bounds to reduce")
    print("reset positions", np.count_nonzero(~alive))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Repeat the simulation with half the time step and twice as many steps.
    2. Add wind along x. Predict how the cloud's bounds will change.
    3. Respawn dead particles, resetting position, velocity and age together.
    4. Generate points inside a unit sphere: multiply each surface direction by
       `rng.random(N) ** (1/3)`. Compare with a radius sampled uniformly.
    5. Compute the cloud centroid, bounding-box corners and squared radius per point. Print the shapes with and without `keepdims`.

    [Part 6](NumPyForCGPart6Mesh.py) brings shapes, vector maths and buffer layout together.
    """)
    return


if __name__ == "__main__":
    app.run()
