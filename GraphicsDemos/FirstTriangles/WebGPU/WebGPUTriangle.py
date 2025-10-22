#!/usr/bin/env -S uv run --active --script
import sys

import numpy as np
import wgpu
import wgpu.utils

# from WebGPUWidget import WebGPUWidget
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
        self.setWindowTitle("WebGPU Triangle")
        self.device = None
        self.pipeline = None
        self.vertex_buffer = None
        self.angle = 0.0
        # fmt: off
        self.vertices = np.array([
            -0.75, -0.75,0.0,1.0, 0.0, 0.0,  # Bottom-left vertex (red)
            0.0,  0.75,0.0, 0.0, 1.0,  0.0,  # Top vertex (green)
            0.75,  -0.75,0.0, 0.0,  0.0, 1.0,  # Bottom-right vertex (blue)
            ],dtype=np.float32)
        # fmt: on
        self.width = 1024
        self.height = 1024
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
            self.startTimer(100)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _init_buffers(self):
        self.vertex_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Create a copy buffer to update the vertex buffer
        self.vertex_buffer.copy_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        vertex_shader_code = """
        struct VertexIn {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) fragColor: vec3<f32>,
        };

        @vertex
        fn main(input: VertexIn) -> VertexOut {
            var output: VertexOut;
            output.position = vec4<f32>(input.position, 1.0);
            output.fragColor = input.color;
            return output;
        }
        """

        fragment_shader_code = """
        @fragment
        fn main(@location(0) fragColor: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(fragColor, 1.0); // Simple color output
        }
        """

        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[])
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.device.create_shader_module(code=vertex_shader_code),
                "entry_point": "main",
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
                "module": self.device.create_shader_module(code=fragment_shader_code),
                "entry_point": "main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def paint(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(10, 20, "First Triangle WebGPU", size=20, colour=Qt.black)
        try:
            texture = self.device.create_texture(
                size=(self.width, self.height, 1),
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
                        "clear_value": (0.4, 0.4, 0.4, 1.0),
                    }
                ]
            )
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(3)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer(texture)
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def _update_colour_buffer(self, texture) -> None:
        """
        Update the color buffer with the rendered texture data.
        """
        buffer_size = self.width * self.height * 4  # Width * Height * Bytes per pixel (RGBA8 is 4 bytes per pixel)
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
                    "bytes_per_row": self.width * 4,  # Row stride (width * bytes per pixel)
                    "rows_per_image": self.height,  # Number of rows in the texture
                },
                (self.width, self.height, 1),  # Copy size: width, height, depth
            )
            self.device.queue.submit([command_encoder.finish()])

            # Map the buffer for reading
            readback_buffer.map_sync(mode=wgpu.MapMode.READ)

            # Access the mapped memory
            raw_data = readback_buffer.read_mapped()
            self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((
                self.width,
                self.height,
                4,
            ))  # Height, Width, Channels

            # Unmap the buffer when done
            readback_buffer.unmap()
        except Exception as e:
            print(f"Failed to update color buffer: {e}")

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        self.buffer = np.zeros([self.height, self.width, 4], dtype=np.uint8)

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        tmp_buffer = self.vertex_buffer.copy_buffer
        tmp_buffer.map_sync(wgpu.MapMode.WRITE)
        tmp_buffer.write_mapped(self.vertices.tobytes())
        tmp_buffer.unmap()
        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(tmp_buffer, 0, self.vertex_buffer, 0, self.vertices.nbytes)
        self.device.queue.submit([command_encoder.finish()])

        self.update()


app = QApplication(sys.argv)
win = WebGPUScene()
win.resize(800, 600)
win.show()
sys.exit(app.exec())
