"""Standalone SVG diagnostics; no plotting dependency or browser service required."""
from html import escape
from pathlib import Path
import numpy as np


def lines_svg(path, panels, title):
    """Render labeled line panels with axes and units; each series is (label,x,y)."""
    width, panel_h = 1100, 220
    height = 45 + len(panels) * panel_h
    colors = ["#1565c0", "#e64a19", "#2e7d32", "#6a1b9a", "#d19100", "#00796b"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
             f'<text x="20" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>']
    for i, (name, series) in enumerate(panels):
        left, top, w, h = 85, 70 + i * panel_h, 980, 150
        arrays = [(label, np.asarray(x, float), np.asarray(y, float)) for label, x, y in series if len(x)]
        finite_y = np.concatenate([y[np.isfinite(y)] for _, _, y in arrays]) if arrays else np.array([0, 1])
        if not len(finite_y):
            finite_y = np.array([0, 1])
        xmin = min((float(np.min(x)) for _, x, _ in arrays), default=0)
        xmax = max((float(np.max(x)) for _, x, _ in arrays), default=1)
        ymin, ymax = float(finite_y.min()), float(finite_y.max())
        xmax = max(xmax, xmin + 1e-6)
        ymax = max(ymax, ymin + 1e-6)
        parts.append(f'<text x="15" y="{top-13}" font-family="sans-serif" font-size="14">{escape(name)}</text>')
        for k in range(5):
            x = left + w * k / 4
            y = top + h * k / 4
            parts.extend([f'<path d="M {x} {top} V {top+h}" stroke="#ddd"/>', f'<path d="M {left} {y} H {left+w}" stroke="#ddd"/>',
                          f'<text x="{x}" y="{top+h+17}" font-size="11">{xmin+(xmax-xmin)*k/4:.2f}</text>',
                          f'<text x="8" y="{y+4}" font-size="11">{ymax-(ymax-ymin)*k/4:.3g}</text>'])
        for j, (label, x, y) in enumerate(arrays):
            take = np.linspace(0, len(x)-1, min(len(x), 2000)).astype(int)
            points = " ".join(f'{left+(x[q]-xmin)/(xmax-xmin)*w:.1f},{top+h-(y[q]-ymin)/(ymax-ymin)*h:.1f}' for q in take if np.isfinite(y[q]))
            color = colors[j % len(colors)]
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1"/>')
            parts.append(f'<text x="{left+j*150}" y="{top+h+35}" font-size="11" fill="{color}">{escape(label)}</text>')
    parts.append('</svg>')
    Path(path).write_text("".join(parts), encoding="utf-8")
