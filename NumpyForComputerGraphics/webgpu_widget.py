import json
from pathlib import Path

import numpy as np


def webgpu_html(data: np.ndarray, source: str, angle: float = 30) -> str:
    """Build a browser WebGPU viewer from world-space xyz and RGB rows."""
    rows = np.asarray(data, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError('expected finite (vertex_count, 6) position and colour rows')
    payload = json.dumps([rows.ravel().tolist(), source, float(angle)]).replace('<', '\\u003c')
    script = Path(__file__).with_name('webgpu_viewer.mjs').read_text()
    return f'''<!doctype html><html><head><style>
      body {{ margin: 0; font: 14px sans-serif; color: #666; }}
      canvas {{ width: 100%; height: auto; touch-action: none; cursor: grab; }}
      p {{ margin: 8px 0; }}
      </style></head><body>
      <canvas width="900" height="600" aria-label="Interactive WebGPU wave mesh"></canvas>
      <p id="status">Starting WebGPU…</p>
      <script type="module">{script}
      const [data, source, angle] = {payload};
      const status = document.querySelector('#status');
      mount(document.querySelector('canvas'), status, data, source, angle)
        .catch(error => {{ status.textContent = error.message; }});
      </script></body></html>'''
