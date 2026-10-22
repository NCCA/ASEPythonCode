#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Computer Graphics, Part 7: an interactive WebGPU mesh

    We will build a wave mesh using the operations from [Part 6](NumPyForCGPart6Mesh.py)
    and draw it in a browser WebGPU canvas. Drag to tumble the mesh, scroll to zoom
    and double-click to reset the view.

    NumPy builds the vertex positions and colours. A small Python helper packages
    them into an HTML document, and JavaScript handles the camera and GPU drawing.
    We keep the vertex layout visible so we can follow the data from the array to
    the shader. There is no Qt window or animation thread.

    The interactive view needs a browser with an available WebGPU adapter. The NumPy
    cells still run without one, and the viewer reports any browser GPU error beneath
    its canvas. No extra Python widget package is required.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    print("NumPy", np.__version__)
    return mo, np


@app.cell
def _(mo):
    wave_height = mo.ui.slider(
        0, 1, step=0.1, value=0.5, label="wave height", show_value=True
    )
    view_angle = mo.ui.slider(-180, 180, value=30, label="view angle", show_value=True)
    mo.hstack([wave_height, view_angle])
    return view_angle, wave_height


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From world positions to a vertex buffer

    We use the same winding, cross products and normalisation as Part 6. The colour is
    calculated once per face on the CPU, then repeated for its three corners. The GPU
    will interpolate those identical colours, so each triangle stays one colour.

    We upload world positions to the viewer. JavaScript calculates an orthographic
    view for each camera update and maps depth into WebGPU's 0–1 range. The shader
    uses `w=1`, so no perspective division is needed. We explain that transform below.
    """)
    return


@app.cell
def _(np, wave_height):
    _x = np.linspace(-3, 3, 25, dtype=np.float32)
    _z = np.linspace(-2, 2, 17, dtype=np.float32)
    _xx, _zz = np.meshgrid(_x, _z, indexing="xy")
    _yy = wave_height.value * np.sin(_xx) * np.cos(_zz)
    _points = np.stack((_xx, _yy, _zz), axis=-1).reshape(-1, 3)
    _ids = np.arange(len(_points)).reshape(_xx.shape)
    _tl, _tr = _ids[:-1, :-1].ravel(), _ids[:-1, 1:].ravel()
    _bl, _br = _ids[1:, :-1].ravel(), _ids[1:, 1:].ravel()
    _faces = np.stack((_tl, _bl, _tr, _tr, _bl, _br), axis=1).reshape(-1, 3)
    _triangles = _points[_faces]
    _normals = np.cross(
        _triangles[:, 1] - _triangles[:, 0], _triangles[:, 2] - _triangles[:, 0]
    )
    _normals /= np.linalg.norm(_normals, axis=1, keepdims=True)
    _light = np.array([1, 2, 1], dtype=np.float32)
    _light /= np.linalg.norm(_light)
    _brightness = 0.15 + 0.85 * np.clip(_normals @ _light, 0, 1)
    _colours = _brightness[:, None] * np.array([0.2, 0.7, 1], dtype=np.float32)
    world_upload = np.ascontiguousarray(
        np.column_stack((_triangles.reshape(-1, 3), np.repeat(_colours, 3, axis=0))),
        dtype=np.float32,
    )
    print("upload", world_upload.shape, world_upload.dtype, "stride", world_upload.strides[0])
    assert world_upload.shape == (2304, 6)
    assert world_upload.strides == (24, 4)
    return (world_upload,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The shader and its layout

    Each row contains `position.xyz, colour.rgb`: six `float32` values and a 24-byte
    stride. Attribute 0 starts at byte 0, attribute 1 at byte 12. These offsets must
    agree with the shader's `@location` declarations.

    The vertex shader returns clip coordinates; the fragment shader returns an opaque
    colour. We use a depth texture so the closest triangle wins regardless of draw order.
    """)
    return


@app.cell
def _():
    shader_source = """
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
    return (shader_source,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tumble the mesh

    Drag to rotate the camera, scroll to zoom and double-click to reset. The view-angle
    slider sets the starting heading. Changing the wave height rebuilds the mesh.
    This viewer uses the browser's WebGPU adapter and starts when the cell runs.

    I split the viewer into three files so we can follow where each part runs:

    | File | Runs in | What it does |
    | :--- | :--- | :--- |
    | `NumPyForCGPart7WebGPU.py` | Python kernel | Builds the mesh, calculates colours and supplies the WGSL shader |
    | `webgpu_widget.py` | Python kernel | Packages the arrays and JavaScript into an HTML document |
    | `webgpu_viewer.mjs` | Browser | Creates the GPU resources, handles mouse input and draws the mesh |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is an `.mjs` file?

    `.mjs` is a filename extension for a JavaScript module, often called an ECMAScript
    module or ES module. It contains ordinary JavaScript. Modules let us export names
    from one file and import them into another, much like Python modules. Here we
    export the `Camera` class and the `mount` function. Node.js can import `Camera`
    directly for our tests without starting a browser or requesting a GPU.

    In the browser, `type="module"` on a script element tells it to treat the code as
    a module. A `.js` file can also contain module code; I use `.mjs` here to make its
    role explicit. See the [JavaScript modules guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules).

    ### How Python passes data to JavaScript

    The notebook calls:

    ```python
    webgpu_html(world_upload, shader_source, view_angle.value)
    ```

    `world_upload` has shape `(2304, 6)`. Each row holds world-space `x, y, z` followed
    by `r, g, b`. We keep world positions so the viewer can calculate a fresh
    projection whenever the camera moves.

    Inside `webgpu_html`, we check the array's shape and finite values, then flatten
    it with `rows.ravel().tolist()`. `json.dumps` serialises that list, the WGSL source
    string and the starting angle into JSON text. This copies values across the
    Python/browser boundary; the browser does not share the NumPy array's memory.

    The helper finds `webgpu_viewer.mjs` beside its own Python file using
    `Path(__file__).with_name(...)` and reads it as text. It embeds that text inside
    an HTML `<script type="module">` element, followed by the JSON payload and this
    call:

    ```javascript
    const [data, source, angle] = /* embedded JSON payload */;
    mount(document.querySelector('canvas'), status, data, source, angle);
    ```

    This shows the structure of the generated code; the real document contains the
    JSON value on the right-hand side. Python is reading the `.mjs` file, not executing
    or importing it as Python. The browser receives its contents inside the HTML,
    so it does not fetch the `.mjs` file separately. Keep both helper files together
    when copying the notebook.

    Finally, `mo.iframe(...)` displays the HTML document in a small page embedded in
    the notebook. Its `<canvas>` is the drawing surface and its status paragraph
    shows the controls or an error message.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What happens when the viewer starts?

    `mount` is an asynchronous JavaScript function. It uses `await` whilst requesting
    the browser's WebGPU adapter and device. It then configures the canvas, compiles
    our WGSL source and creates a render pipeline with the 24-byte vertex layout
    described above. A depth texture lets nearer triangles hide farther ones.

    `new Float32Array(data)` rebuilds the packed numeric data in JavaScript. We keep
    an `original` array of world positions and colours, a `projected` working copy,
    and one GPU vertex buffer. The pipeline, buffer and depth texture are reused
    during interaction. The WGSL string travels through Python and JavaScript as
    text; the WebGPU implementation compiles it for execution on the GPU.

    ### From a mouse drag to a new frame

    1. A left-button `pointerdown` captures the pointer, so the drag continues when
       it leaves the canvas. Releasing or cancelling the pointer ends the drag.
    2. `pointermove` passes horizontal and vertical movement to `Camera.drag`.
       Horizontal movement changes yaw; vertical movement changes pitch. We clamp
       pitch just short of the poles to stop the view flipping over.
    3. `redraw` requests a frame using
       [`requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame).
       The `pending` flag groups several input events into one scheduled draw.
    4. `draw` rotates the original positions around the origin using yaw and pitch,
       applies orthographic scaling and maps depth into WebGPU's 0–1 range.
       This calculation runs in JavaScript on the CPU. The shader receives the
       resulting clip positions; it does not calculate the camera transform.
    5. `device.queue.writeBuffer` uploads the working array. A render pass clears
       colour and depth, draws the triangles and submits the commands to the GPU.

    Wheel input changes the orthographic scale, with limits to keep it positive.
    Double-click restores the starting yaw, a 30-degree pitch and scale one. We draw
    on demand, so there is no continuous animation loop whilst the view is idle.
    Colours stay as calculated in NumPy; tumbling changes the view without
    recalculating the lighting.

    Mouse input stays inside the browser. It does not rerun a Python cell or update
    the view-angle slider. Changing either notebook slider reruns dependent cells
    and creates a new viewer, resetting its camera. There is no two-way widget
    synchronisation in this example.

    The browser viewer renders directly to its canvas, using the GPU on the machine
    running the browser. On a remote notebook server, the Python kernel may be on a
    different machine. We send mesh data to the browser without reading rendered
    pixels back into NumPy. The solutions notebook includes an optional offscreen
    readback example for exploring pixel storage and row padding.
    When the viewer receives `pagehide`, it releases its buffer, depth texture and
    device. A device-loss notification also stops drawing and reports the problem.
    """)
    return


@app.cell
def _(mo, shader_source, view_angle, world_upload):
    from webgpu_widget import webgpu_html

    browser_view = mo.iframe(
        webgpu_html(world_upload, shader_source, view_angle.value), height="660px"
    )
    mo.output.replace(browser_view)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
