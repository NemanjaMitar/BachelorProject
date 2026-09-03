"""Occupancy-grid builder for the vision observation. Pure numpy, no pygame,
so it can be unit-tested in isolation. Coordinates are the game's 480x360
screen pixels (pygame: x right, y DOWN, y=0 is the top / the climb direction).

The channel SET is configurable so a level family can be given an observation
that carries only what that family needs -- e.g. the fully-icy levels 36-38,
where every platform is ice and the hazard channel is therefore an exact copy
of the solid channel. The default is the historical (solid, king, hazard)
triple, so every model trained before this was configurable keeps working."""
import numpy as np

GRID_CELL = 8 # pixels per cell -> 360/8=45 rows, 480/8=60 cols

# The channel set every pre-existing checkpoint was trained on. Loaders use it
# as the default when a checkpoint's cfg predates the grid_channels field.
DEFAULT_CHANNELS = ("solid", "king", "hazard")
SUPPORTED_CHANNELS = ("solid", "king", "hazard", "slope")


def resolve_channels(channels):
    """Normalise a channel spec (None / str / iterable) to a validated tuple."""
    if channels is None:
        return DEFAULT_CHANNELS
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",") if c.strip()]
    channels = tuple(channels)
    if not channels:
        raise ValueError("grid channels must not be empty")
    bad = [c for c in channels if c not in SUPPORTED_CHANNELS]
    if bad:
        raise ValueError(f"unknown grid channel(s) {bad}; "
                         f"supported: {list(SUPPORTED_CHANNELS)}")
    if len(set(channels)) != len(channels):
        raise ValueError(f"duplicate grid channels in {list(channels)}")
    return channels


def build_occupancy_grid(platforms, king_x, king_y, screen_w, screen_h,
                         cell=GRID_CELL, channels=DEFAULT_CHANNELS):
    """Return (C, H, W) float32, one plane per name in `channels`:
        solid   1 where any platform is
        king    1 in the single cell holding the king
        hazard  1 where the platform type is Ice or Snow
        slope   signed steepness of a ramp, 0 on flat ground

    The slope plane carries Platform.slope[0] clipped to [-1, 1] -- that is
    exactly the value King copies into king.slope on contact, and its SIGN is
    which way gravity drags the king along the ramp (King._check_collisions).
    It matters most on the fully-icy levels 36-38, where a third of the
    platforms are ramps and slip changes the along-ramp deceleration from 0.35
    to 0.10 per frame: without this plane the ramp that generates the momentum
    is invisible to the policy. NOT encoded: slope[1] (which diagonal of the
    rect is solid), and the solid plane still fills the whole rect rather than
    the true diagonal surface.
    `platforms` is an iterable of objects with .x,.y,.width,.height and an
    optional .type in {'Land','Ice','Snow'} (Platform exposes exactly this).

    `channels` is assumed already validated -- this runs on every observation,
    so the checking lives in resolve_channels(), called once at env construction
    and once per checkpoint load. An unrecognised name here yields no plane at
    all rather than an error, which is why callers must go through it."""
    H, W = screen_h // cell, screen_w // cell
    order = {name: i for i, name in enumerate(channels)}
    i_solid = order.get("solid")
    i_king = order.get("king")
    i_haz = order.get("hazard")
    i_slope = order.get("slope")
    g = np.zeros((len(channels), H, W), dtype=np.float32)
    if i_solid is not None or i_haz is not None or i_slope is not None:
        for p in platforms:
            x0 = max(0, int(p.x) // cell)
            y0 = max(0, int(p.y) // cell)
            x1 = min(W, (int(p.x) + int(p.width)  + cell - 1) // cell)   # ceil so thin ledges register
            y1 = min(H, (int(p.y) + int(p.height) + cell - 1) // cell)
            if x1 <= x0 or y1 <= y0:
                continue
            if i_solid is not None:
                g[i_solid, y0:y1, x0:x1] = 1.0
            if i_haz is not None and getattr(p, "type", "Land") in ("Ice", "Snow"):
                g[i_haz, y0:y1, x0:x1] = 1.0
            if i_slope is not None:
                sl = getattr(p, "slope", 0)      # 0 on flat, (steepness, side) on a ramp
                if sl:
                    g[i_slope, y0:y1, x0:x1] = min(1.0, max(-1.0, float(sl[0])))
    if i_king is not None:
        kx = min(W - 1, max(0, int(king_x) // cell))
        ky = min(H - 1, max(0, int(king_y) // cell))
        g[i_king, ky, kx] = 1.0
    return g
