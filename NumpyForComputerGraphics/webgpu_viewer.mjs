export class Camera {
  constructor(angle = 30) {
    this.initialYaw = angle * Math.PI / 180;
    this.reset();
  }
  reset() {
    this.yaw = this.initialYaw;
    this.pitch = Math.PI / 6;
    this.scale = 1;
  }
  drag(dx, dy) {
    this.yaw += dx * 0.008;
    this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.008));
  }
  zoom(delta) {
    this.scale = Math.max(0.3, Math.min(4, this.scale * Math.exp(delta * 0.001)));
  }
}

export async function mount(canvas, status, data, source, angle) {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser. Open the notebook in a browser with WebGPU support.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No browser WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({device, format, alphaMode: 'opaque'});
  const shader = device.createShaderModule({code: source});
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {module: shader, entryPoint: 'vertex_main', buffers: [{arrayStride: 24,
      attributes: [{format: 'float32x3', offset: 0, shaderLocation: 0},
        {format: 'float32x3', offset: 12, shaderLocation: 1}]}]},
    fragment: {module: shader, entryPoint: 'fragment_main', targets: [{format}]},
    primitive: {topology: 'triangle-list', cullMode: 'none'},
    depthStencil: {format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less'},
  });
  const original = new Float32Array(data);
  const projected = new Float32Array(original);
  const buffer = device.createBuffer({size: original.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST});
  const depth = device.createTexture({size: [canvas.width, canvas.height],
    format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT});
  const camera = new Camera(angle);
  let pending = 0;
  let disposed = false;
  function draw() {
    pending = 0;
    if (disposed) return;
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const aspect = canvas.width / canvas.height;
    for (let i = 0; i < original.length; i += 6) {
      const x = original[i], y = original[i + 1], z = original[i + 2];
      const rx = cy * x + sy * z, rz = -sy * x + cy * z;
      projected[i] = rx / (3.6 * aspect * camera.scale);
      projected[i + 1] = (cp * y - sp * rz) / (3.6 * camera.scale);
      projected[i + 2] = 0.5 + (sp * y + cp * rz) / 16;
    }
    device.queue.writeBuffer(buffer, 0, projected);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({colorAttachments: [{
      view: context.getCurrentTexture().createView(), clearValue: [0.08, 0.08, 0.1, 1],
      loadOp: 'clear', storeOp: 'store'}], depthStencilAttachment: {
      view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store'}});
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, buffer);
    pass.draw(original.length / 6);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  function redraw() {
    if (!pending && !disposed) pending = requestAnimationFrame(draw);
  }
  let pointer = null;
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || pointer !== null) return;
    pointer = event.pointerId;
    canvas.setPointerCapture(pointer);
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', event => {
    if (event.pointerId !== pointer) return;
    camera.drag(event.movementX, event.movementY);
    redraw();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(name, event => {
      if (event.pointerId !== pointer) return;
      pointer = null;
      canvas.style.cursor = 'grab';
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    });
  }
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    camera.zoom(event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.height : 1));
    redraw();
  }, {passive: false});
  canvas.addEventListener('dblclick', () => { camera.reset(); redraw(); });
  function dispose() {
    disposed = true;
    cancelAnimationFrame(pending);
    buffer.destroy();
    depth.destroy();
    context.unconfigure();
    device.destroy();
  }
  window.addEventListener('pagehide', dispose, {once: true});
  device.lost.then(info => {
    if (!disposed) { status.textContent = `WebGPU device lost: ${info.message}`; dispose(); }
  });
  status.textContent = 'Drag to tumble · scroll to zoom · double-click to reset';
  draw();
}
