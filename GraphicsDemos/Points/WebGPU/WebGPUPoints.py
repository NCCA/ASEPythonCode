#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, Vec3, look_at, perspective
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

    def __init__(self, num_points=10000):
        super().__init__()
        self.setWindowTitle("WebGPU Points")
        self.device = None
        self.pipeline = None
        self.vertex_buffer = None
        self.num_points = num_points
        self.window_width = 1024
        self.window_height = 1024
        self.texture_size = (1024, 1024)
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        gl_to_web = Mat4.from_list([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.project = gl_to_web @ perspective(45.0, self.window_width / self.window_height, 0.1, 100.0)
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
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _init_buffers(self):
        vertex_data = np.empty((self.num_points, 6), dtype=np.float32)

        # Populate the first 3 columns with position data
        vertex_data[:, 0:3] = np.random.uniform(-4.0, 4.0, size=(self.num_points, 3))

        # Populate the next 3 columns with colour data
        vertex_data[:, 3:6] = np.random.uniform(0.0, 1.0, size=(self.num_points, 3))
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("PointShader.wgsl", "r") as f:
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
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.point_list},
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

        This method renders the WebGPU content for the scene.
        """
        self.render_text(
            10,
            20,
            f"WebGPU Static Points :- {self.num_points}",
            size=20,
            colour=Qt.yellow,
        )
        try:
            texture = self.device.create_texture(
                size=(self.window_width, self.window_height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            )
            texture_view = texture.create_view()
            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": texture_view,
                        "resolve_target": None,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.3, 0.3, 0.3, 1.0),
                    }
                ]
            )
            self.update_uniform_buffers()
            render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.num_points)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer(texture)
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

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

    def _update_colour_buffer(self, texture) -> None:
        """
        Update the colour buffer with the rendered texture data.
        """
        buffer_size = (
            self.window_width * self.window_height * 4
        )  # Width * Height * Bytes per pixel (RGBA8 is 4 bytes per pixel)
        try:
            readback_buffer = self.device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                {"texture": texture},
                {
                    "buffer": readback_buffer,
                    "bytes_per_row": self.window_width * 4,  # Row stride (width * bytes per pixel)
                    "rows_per_image": self.window_height,  # Number of rows in the texture
                },
                (
                    self.window_width,
                    self.window_height,
                    1,
                ),  # Copy size: width, height, depth
            )
            self.device.queue.submit([command_encoder.finish()])

            # Map the buffer for reading
            readback_buffer.map_sync(mode=wgpu.MapMode.READ)

            # Access the mapped memory
            raw_data = readback_buffer.read_mapped()
            self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((
                self.window_width,
                self.window_height,
                4,
            ))  # Height, Width, Channels

            # Unmap the buffer when done
            readback_buffer.unmap()
        except Exception as e:
            print(f"Failed to update colour buffer: {e}")

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        self.buffer = np.zeros([self.window_height, self.window_width, 4], dtype=np.uint8)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        self.rotation += 0.5
        self.update()


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU points demo")
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=10000,
        help="The number of points to generate.",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = WebGPUScene(num_points=args.points)
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
