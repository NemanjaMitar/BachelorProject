"""Occupancy-grid builder for the vision observation. Pure numpy, no pygame,
so it can be unit-tested in isolation. Coordinates are the game's 480x360
screen pixels (pygame: x right, y DOWN, y=0 is the top / the climb direction)."""
import numpy as np

GRID_CELL = 8   # pixels per cell -> 360/8=45 rows, 480/8=60 cols

def build_occupancy_grid(platforms, king_x, king_y, screen_w, screen_h, cell=GRID_CELL):
    """Return (3, H, W) float32: channel 0 solid, 1 king, 2 hazard (ice/snow).
    `platforms` is an iterable of objects with .x,.y,.width,.height and an
    optional .type in {'Land','Ice','Snow'} (Platform exposes exactly this)."""
    H, W = screen_h // cell, screen_w // cell
    g = np.zeros((3, H, W), dtype=np.float32)
    for p in platforms:
        x0 = max(0, int(p.x) // cell)
        y0 = max(0, int(p.y) // cell)
        x1 = min(W, (int(p.x) + int(p.width)  + cell - 1) // cell)   # ceil so thin ledges register
        y1 = min(H, (int(p.y) + int(p.height) + cell - 1) // cell)
        if x1 <= x0 or y1 <= y0:
            continue
        g[0, y0:y1, x0:x1] = 1.0
        if getattr(p, "type", "Land") in ("Ice", "Snow"):
            g[2, y0:y1, x0:x1] = 1.0
    kx = min(W - 1, max(0, int(king_x) // cell))
    ky = min(H - 1, max(0, int(king_y) // cell))
    g[1, ky, kx] = 1.0
    return g