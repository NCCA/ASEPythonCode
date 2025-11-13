#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from NumpyBufferWidget import NumpyBufferWidget
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device


class WebGPUScene(NumpyBufferWidget):
    """
    A concrete implementation of NumpyBufferWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Lines")
        self.device = None
        self.pipeline = None
        self.vertex_buffer = None
        self.ratio = self.devicePixelRatio()
        self.texture_size = (int(self.width() * self.ratio), int(self.height() * self.ratio))
        self.msaa_sample_count = 4
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True

        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._init_buffers()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _create_lines(self, width: float, depth: float, rows: int, cols: int) -> np.ndarray:
        """Creates vertex data for a grid of lines using numpy."""
        half_w, half_d = width / 2, depth / 2

        # Horizontal lines
        # np.linspace(-half_d, half_d, rows + 1)` creates an array `z`
        # containing evenly spaced Z-coordinates for the horizontal lines.
        z = np.linspace(-half_d, half_d, rows + 1)
        # pre-allocates a 3D NumPy array called `h_lines` to hold the start and end points for each horizontal line.
        # The shape means `(number of lines, 2 points per line, 3 coordinates per point)`.
        h_lines = np.zeros((len(z), 2, 3), dtype=np.float32)
        h_lines[:, 0, 0] = -half_w  # start x
        h_lines[:, 1, 0] = half_w  # end x
        # The Z-coordinates for both start and end points of each line are set from the `z` array.
        # `z[:, np.newaxis]` is used to make the 1D `z`
        # array compatible for assignment into the 3D `h_lines` array.
        h_lines[:, :, 2] = z[:, np.newaxis]  # z for start and end

        # Vertical lines mirrors above
        x = np.linspace(-half_w, half_w, cols + 1)
        v_lines = np.zeros((len(x), 2, 3), dtype=np.float32)
        v_lines[:, :, 0] = x[:, np.newaxis]  # x for start and end
        v_lines[:, 0, 2] = -half_d  # start z
        v_lines[:, 1, 2] = half_d  # end z

        # Combine and flatten
        # .reshape(-1, 3)` transforms this array into a 2D list of vertices,
        # where each row is a single [x, y, z] point.
        # The `-1` automatically calculates the correct number of rows.
        points = np.concatenate([h_lines, v_lines]).reshape(-1, 3)
        self.line_vertex_count = len(points)
        return points

    def _create_render_buffer(self):
        # This is the texture that the multisampled texture will be resolved to
        colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture = colour_buffer_texture
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()

        # This is the multisampled texture that will be rendered to
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.msaa_sample_count,
        )
        self.depth_buffer_view = depth_texture.create_view()

        # Calculate aligned buffer size for texture copy
        buffer_size = self._calculate_aligned_buffer_size()
        self.readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def _init_buffers(self):
        points = self._create_lines(10, 10, 30, 30)
        vertex_data = np.array(points, dtype=np.float32)

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("LineShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="particle_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 12,  # 3 floats x 4 bytes per float
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
            },
        )

        # Create a uniform buffer
        self.uniform_data = np.zeros((), dtype=[("MVP", "float32", (16))])

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def paint(self) -> None:
        """
        Paint the WebGPU content.
        """
        try:
            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.multisample_texture_view,
                        "resolve_target": self.colour_buffer_texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.4, 0.4, 0.4, 1.0),
                    }
                ],
                depth_stencil_attachment={
                    "view": self.depth_buffer_view,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            )
            self.update_uniform_buffers()
            render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.line_vertex_count)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def resizeEvent(self, event) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            event: The resize event object.
        """
        # Update the stored width and height, considering high-DPI displays
        width = int(event.size().width() * self.ratio)
        height = int(event.size().height() * self.ratio)

        # Update texture size to match window dimensions
        self.texture_size = (width, height)

        # Update projection matrix
        self.project = perspective(45.0, width / height if height > 0 else 1, 0.1, 100.0)

        # Recreate render buffers for the new window size
        self._create_render_buffer()

        # Resize the numpy buffer to match new window dimensions
        if self.frame_buffer is not None:
            self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

        self.update()

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        rotation = Mat4.rotate_y(self.rotation)
        mvp_matrix = (self.project @ self.view @ rotation).to_numpy().astype(np.float32)
        self.uniform_data["MVP"] = mvp_matrix.flatten()
        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )

    def _calculate_aligned_row_size(self) -> int:
        """
        Calculate the aligned row size for texture copy operations.
        Many GPUs require row alignment to 256 or 512 bytes.
        """
        bytes_per_pixel = 4  # RGBA8 = 4 bytes per pixel
        raw_row_size = self.texture_size[0] * bytes_per_pixel

        # Align to 256 bytes (common GPU requirement)
        alignment = 256
        aligned_row_size = ((raw_row_size + alignment - 1) // alignment) * alignment

        return aligned_row_size

    def _calculate_aligned_buffer_size(self) -> int:
        """
        Calculate the aligned buffer size needed for texture copy operations.
        Many GPUs require row alignment to 256 or 512 bytes.
        """
        aligned_row_size = self._calculate_aligned_row_size()
        return aligned_row_size * self.texture_size[1]

    def _update_colour_buffer(self) -> None:
        """
        Update the colour buffer with the rendered texture data.
        """
        # Use the aligned row size calculation
        bytes_per_row = self._calculate_aligned_row_size()

        try:
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                {"texture": self.colour_buffer_texture},
                {
                    "buffer": self.readback_buffer,
                    "bytes_per_row": bytes_per_row,  # Aligned row stride
                    "rows_per_image": self.texture_size[1],  # Number of rows in the texture
                },
                (
                    self.texture_size[0],
                    self.texture_size[1],
                    1,
                ),  # Copy size: width, height, depth
            )
            self.device.queue.submit([command_encoder.finish()])

            # Map the buffer for reading
            self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)

            # Access the mapped memory
            raw_data = self.readback_buffer.read_mapped()
            width, height = self.texture_size

            # Create a strided view of the raw data and then copy it to a contiguous array.
            # This is necessary because the raw data from the buffer includes padding bytes
            # to meet row alignment requirements, so we can't just reshape it.
            strided_view = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw_data, dtype=np.uint8),
                shape=(height, width, 4),
                strides=(bytes_per_row, 4, 1),
            )
            self.frame_buffer = np.copy(strided_view)

            # Unmap the buffer when done
            self.readback_buffer.unmap()
        except Exception as e:
            print(f"Failed to update colour buffer: {e}")
            # Fallback: create a simple gray buffer if texture copy fails
            if self.frame_buffer is not None:
                self.frame_buffer.fill(128)

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_Space:
            self.animate = not self.animate

        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        if self.animate:
            self.rotation += 0.5
        self.update()


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU Line demo")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
