#!/usr/bin/env python
"""
Headless, training-oriented wrapper around the Jump King game.

Design notes (read these before changing anything):

* HEADLESS: we set SDL's dummy video/audio drivers, so pygame loads all the
  real sprite/sound *files* (they must exist on disk in the game directory)
  but never opens a window or audio device. We never blit to a display and we
  never cap the framerate, so physics runs as fast as the CPU allows.

* DECISION POINTS: the King only reads input while grounded (King.update skips
  _check_events when isFalling). So one *agent action* == one full
  charge -> release -> fly -> land cycle. We fast-forward the physics through
  the whole jump inside step(), and only return control to the agent once the
  King is grounded and settled (move_available() is True). At those moments the
  velocity is ~0, so (level, x, y) is a sufficient Markov state.

* ACTIONS (macro): a jump is "hold SPACE for C frames, then release toward
  {left, up, right}". jumpCount (== frames SPACE held) sets the jump strength:
      speed = 1.5 + (jumpCount/5)**1.13   (+0.9 for a directional jump)
  We drive the King by faking pygame.key.get_pressed() instead of the 4-action
  get_action_dict() in King.py, because that dict can't charge an up-jump and
  can't control charge length. We ALSO expose a couple of pure-walk actions for
  fine horizontal positioning.

* CURRICULUM: if you pass start_states="starts/start_states.json" (a list of captured
  checkpoints produced by capture.py), reset() will, with probability
  (1 - p_bottom), drop the King at a random captured checkpoint instead of the
  bottom. This is what feeds the agent useful near-the-exit starts. With
  probability p_bottom (or if the pool is empty) it resets to the bottom as
  usual. Explicit reset(level=..., rect_x=..., rect_y=...) calls bypass the
  curriculum entirely, so old callers keep working.

* FALL RULE: with terminate_on_fall_below_start=True, an episode ends as a
  FAILURE the moment the King drops below the level it started this attempt on.
  Combined with level-N start states, this makes a fall re-anchor the next
  attempt on level N instead of forcing a full re-climb from the bottom.

* This file does NOT modify King.py / Level.py etc. It only orchestrates them.

IMPORTANT: place this file in the game root (same folder as King.py) so the
imports resolve and the relative asset paths (images\\..., Audio\\...) work.
"""

import os
import json
import math

# --- must be set BEFORE pygame is imported/initialised -----------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import numpy as np
from Occupancy import build_occupancy_grid, resolve_channels, GRID_CELL


# Wind force = sin(wind_var) * WIND_AMPLITUDE (see Wind.calculate_wind).
WIND_AMPLITUDE = 2.5 ** 2                       # 6.25
# 5 phase buckets by signed force: strong-left, weak-left, calm, weak-right,
# strong-right. The "wait until bucket B" macro uses these so the agent (and
# the search) can DETERMINISTICALLY reach any wind state before jumping.
WIND_BUCKET_EDGES = (-3.0, -0.5, 0.5, 3.0)      # 4 edges -> 5 buckets (0..4)
N_WIND_BUCKETS = len(WIND_BUCKET_EDGES) + 1
WIND_JUMP_CHARGES = (8, 12, 16, 20, 26, 32)     # charges for timed jump combos


def wind_force(wind_var):
    return math.sin(wind_var) * WIND_AMPLITUDE


def wind_bucket(w):
    """Classify a wind force into 0..4 (strong-left .. strong-right)."""
    for i, e in enumerate(WIND_BUCKET_EDGES):
        if w < e:
            return i
    return N_WIND_BUCKETS - 1


# Each bucket's "ready" moment is a SINGLE consistent phase (a narrow sin band
# on the strengthening edge), so the wait macros launch into the exact same
# wind every time regardless of the episode's starting phase. Strong buckets
# target near-peak wind (what crossings need); weak buckets a mid value.
WIND_BUCKET_TARGET_SIN = (-0.9, -0.35, 0.0, 0.35, 0.9)
WIND_READY_TOL = 0.08


def wind_ready(wind_var, target_bucket):
    """True at the ONE consistent phase per cycle for `target_bucket`: sin near
    the bucket's target on the strengthening edge (approaching its peak). This
    makes 'wait for bucket B' deterministic across start phases -- the jump
    always launches into the identical wind."""
    s = math.sin(wind_var)
    tgt = WIND_BUCKET_TARGET_SIN[target_bucket]
    if abs(s - tgt) > WIND_READY_TOL:
        return False
    c = math.cos(wind_var)
    if target_bucket < 2:
        return c < 0.0          # strengthening left (sin heading to -1)
    if target_bucket > 2:
        return c > 0.0          # strengthening right (sin heading to +1)
    return c > 0.0              # calm: pick the rising-through-zero crossing


def build_action_table(charges=(4, 8, 12, 16, 20, 26, 32), walk_frames=10,
                       fine_walk_frames=0, extra_charges=(), wait_frames=(),
                       wait_wind=(), wind_jump=(), approach_jump=(), wind_combo=(),
                       settle_action=False):
    """THE canonical action-table layout. Any tool that needs to know what a
    model's action index means (e.g. Play's per-model tables) must build the
    table through this function with that model's OWN action config --
    index meaning depends on the exact extra_charges list, so two models
    trained with different lists are NOT index-compatible."""
    actions = []
    for d in ("left", "up", "right"):
        for c in charges:
            actions.append(("jump", d, c))
    actions.append(("walk", "left", int(walk_frames)))
    actions.append(("walk", "right", int(walk_frames)))
    if fine_walk_frames:
        actions.append(("walk", "left", int(fine_walk_frames)))
        actions.append(("walk", "right", int(fine_walk_frames)))
    # Extra charges go last so the base indices keep their meaning.
    for d in ("left", "up", "right"):
        for c in (extra_charges or ()):
            actions.append(("jump", d, c))
    # Fixed-length wait actions (stand still N frames to let the wind advance).
    for f in (wait_frames or ()):
        actions.append(("wait", "none", int(f)))
    # "Wait until the wind reaches bucket B" macros -- the timing tool. Each
    # stands still until the wind force enters bucket B (or a one-cycle cap),
    # then returns control so the NEXT action launches into that wind.
    for b in (wait_wind or ()):
        actions.append(("wait_wind", int(b), 1100))
    # ATOMIC "wait for wind bucket B, then jump" combos. The wait and jump
    # happen in ONE macro-action, so the search (which resets phase per node)
    # can still discover wind-assisted crossings. All three directions: UP
    # (the wind bucket carries it sideways) AND left/right (the near-exit
    # crossing is a directional max-charge jump ridden by the strong wind).
    # Encoded as ("jump_wind", (bucket, direction), charge).
    for b in (wind_jump or ()):
        for d in ("up", "left", "right"):
            for c in WIND_JUMP_CHARGES:
                actions.append(("jump_wind", (int(b), d), c))
    # ATOMIC "walk to a target x, then wind-timed jump" macros. For crossings
    # whose launch WINDOW the net can't perceive (a few px on a uniform shelf,
    # collapsed by the conv grid): the macro walks deterministically to target_x
    # on the (windless) snow, holds for the wind bucket, then jumps -- so ONE
    # action succeeds from anywhere on the shelf and the policy needs no fine
    # position sense. Each entry is (target_x, bucket, direction, charge).
    # Encoded as ("approach_jump", (target_x, bucket, direction), charge).
    for (tx, b, d, c) in (approach_jump or ()):
        actions.append(("approach_jump", (int(tx), int(b), d), int(c)))
    # ATOMIC multi-step wind route: a proven sequence of wind-jumps executed as
    # ONE action (settling between steps). This makes
    # the entire level a single decision at the entry, so no intermediate state
    # -- which the 8px conv grid often can't tell apart from its neighbour -- is
    # ever a decision point. Each combo is a tuple of (bucket, direction, charge)
    # steps. Encoded as ("wind_combo", combo_index, steps_tuple).
    # STANDALONE SETTLE: let the policy choose to kill its own momentum before
    # committing to a jump. On ice a grounded king keeps sliding, and a jump
    # launched mid-slide lands somewhere else -- measured: success drops from
    # 0.51 to 0.11 purely from arrival momentum, and is fully recovered by
    # settling first. A multi-step route opens with a settle for the same
    # reason; this exposes the primitive as an action the agent can learn.
    if settle_action:
        actions.append(("settle", "none", 0))
    for i, steps in enumerate(wind_combo or ()):
        enc = []
        for s in steps:
            if s[0] == "walk":                 # ("walk", target_x): reposition
                enc.append(("walk", int(s[1])))
            elif s[0] == "settle":             # ("settle",): kill ice momentum
                enc.append(("settle", 0))
            elif s[0] == "jump":               # ("jump", dir, charge): plain jump
                enc.append(("jump", s[1], int(s[2])))   # (windless/ice levels)
            else:                              # (bucket, dir, charge): wind-jump
                b, d, c = s
                enc.append((int(b), d, int(c)))
        actions.append(("wind_combo", int(i), tuple(enc)))
    return actions


# Distance over which the dense route term decays to zero, in pixels. A leg of
# the demonstrated route is rarely longer than this, so the term is informative
# along the whole leg instead of saturating.
ROUTE_DENSE_SCALE = 220.0

STARTS_DIR = "starts"          # every start-state pool lives in this folder


def resolve_start_states(path):
    """Accept both 'starts/starts_L12.json' and the older bare 'starts_L12.json'.

    The pools moved into starts/ after the 0->42 run; older commands (and the
    ones quoted in the logs) still name them without the folder."""
    if not path or os.path.exists(path):
        return path
    cand = os.path.join(STARTS_DIR, os.path.basename(path))
    return cand if os.path.exists(cand) else path


def obs_selector(env, channels=None, wind_obs=False, vel_obs=False, who="model",
                 vel_encoding=None):
    """Return f(obs) narrowing the env's observation to what ONE model expects.

    A relay drives many models with a single env, so the env is built as the
    SUPERSET -- the widest grid channel set and the widest scalar vector -- and
    each model takes only its own planes and its own scalar columns. This is the
    one place that knows the packing order: grid (C*H*W) first, then the scalars
    [x, y, level, altitude] (+ wind sin/cos if the env has wind) (+ vx, vy if
    the env has velocity).

    A model whose channel list equals the env's takes the identity path -- no
    reshape, no copy -- which is what every checkpoint from the 0->42 run does.

    The returned function carries .obs_dim, .n_scalars and .grid_shape so a
    loader can build the matching network without recomputing the packing."""
    channels = resolve_channels(channels)
    missing = [c for c in channels if c not in env.grid_channels]
    if missing:
        raise ValueError(f"{who}: needs grid channel(s) {missing}; this env "
                         f"produces {list(env.grid_channels)}")
    if wind_obs and not env.wind_obs:
        raise ValueError(f"{who}: trained with wind scalars; this env has none")
    if vel_obs and not env.vel_obs:
        raise ValueError(f"{who}: trained with velocity scalars; this env has none")
    venc = vel_encoding or "xy"
    if vel_obs and env.vel_encoding != "both" and venc != env.vel_encoding:
        raise ValueError(f"{who}: trained with vel_encoding={venc!r} but this env "
                         f"reports {env.vel_encoding!r}; build the env with "
                         f"vel_encoding='both' to serve both families at once")

    _, H, W = env.grid_shape
    gf = env._grid_flat
    idx = [env.grid_channels.index(c) for c in channels]
    same = tuple(channels) == tuple(env.grid_channels)
    src_c = len(env.grid_channels)

    cols = [0, 1, 2, 3]
    if wind_obs:
        cols += [4, 5]
    if vel_obs:                       # velocity sits AFTER wind when both exist
        base = 6 if env.wind_obs else 4
        if env.vel_encoding == "both":
            # env emits [vx, vy, |v|, dir_x, dir_y]; take this model's slice
            cols += (list(range(base, base + 2)) if venc == "xy"
                     else list(range(base + 2, base + 5)))
        else:
            cols += list(range(base, base + env._n_vel))

    def select(obs):
        grid = obs[:gf] if same else obs[:gf].reshape(src_c, H, W)[idx].ravel()
        return np.concatenate([grid, obs[gf:][cols]]).astype(np.float32)

    select.grid_shape = (len(channels), H, W)
    select.n_scalars = len(cols)
    select.obs_dim = len(channels) * H * W + len(cols)
    return select


class JumpKingEnv:
    """A single headless Jump King environment. Gymnasium-style API.

    NOTE: pygame uses a lot of *process-global* state (the display, the mixer,
    os.environ, and our key monkeypatch). Do NOT instantiate more than one of
    these in the same process. For parallelism use the subprocess vec env
    (vec_env.py), which gives each environment its own process.
    """

    def __init__(self,
                 charges=(4, 8, 12, 16, 20, 26, 32),
                 walk_frames=10,
                 fine_walk_frames=0,   # if > 0, add a second walk pair of this
                                       # many frames (~1.4 px/frame) for micro-
                                       # positioning into narrow launch windows.
                                       # CHANGES num_actions: needs a fresh model.
                 extra_charges=(),     # extra jump charges APPENDED to the end
                                       # of the action table (e.g. (22,24,28,30))
                                       # so existing models' indices stay valid.
                                       # CHANGES num_actions: growth-compatible
                                       # with old checkpoints, see Play.ModelBank.
                 wait_frames=(),       # wind levels: "stand still N frames"
                                       # actions appended after the extras,
                                       # e.g. (30, 60) -- lets the agent TIME
                                       # its jumps to the wind phase.
                 wait_wind=(),         # wind levels: "wait until wind bucket B"
                                       # macro actions, e.g. (0,1,2,3,4). The
                                       # timing tool -- deterministically reach
                                       # a wind state, then jump into it.
                 wind_jump=(),         # wind levels: ATOMIC "wait for bucket B
                                       # then jump" combos, e.g. (0,4). Makes
                                       # wind crossings SEARCH-provable (the
                                       # wait+jump is one action, immune to the
                                       # search's per-node phase reset).
                 approach_jump=(),     # wind levels: ATOMIC "walk to target_x,
                                       # then wait-for-bucket-then-jump" combos,
                                       # each (target_x, bucket, direction,
                                       # charge). For crossings whose launch
                                       # window is too small for the conv grid
                                       # to perceive -- one action crosses from
                                       # anywhere on the shelf.
                 wind_combo=(),        # wind levels: ATOMIC multi-step wind
                                       # routes, each a tuple of (bucket,dir,
                                       # charge) steps -- encodes a whole short
                                       # windy level's proven sequence as one
                                       # action so intermediate (perceptually
                                       # ambiguous) states are never decisions.
                 settle_action=False,  # add a standalone "settle" action that
                                       # damps the king's momentum to rest. On
                                       # ice this is what lets a policy trained
                                       # on at-rest states cope with sliding
                                       # arrivals. Appended LAST, so old
                                       # checkpoints keep their indices.
                 wind_obs=False,       # append sin/cos of the wind phase to the
                                       # observation scalars (4 -> 6) AND
                                       # randomize the phase at every teleport
                                       # so training covers all phases.
                 vel_obs=False,        # append the king's (vx, vy) velocity to
                                       # the scalars. On ICE the king keeps
                                       # sliding at a grounded decision point, so
                                       # position alone aliases momentum states;
                                       # velocity makes them distinguishable ->
                                       # lets PPO genuinely learn icy levels.
                 vel_encoding="xy",    # how vel_obs reports motion.
                                       # "xy": (vx, vy) -- the historical pair,
                                       # what every existing checkpoint expects.
                                       # "polar": (|v|, dir_x, dir_y) -- HOW MUCH
                                       # momentum, separately from WHICH WAY.
                                       # Measured on the demo decision points:
                                       # the king is almost never sliding along
                                       # x, he is sliding DOWN A RAMP (|vy|/|v|
                                       # is 0.6-1.0 at most of them, and 1.00 at
                                       # L37 k=9), so direction needs both
                                       # components -- but magnitude is what the
                                       # charge-time drift scales with, and a
                                       # product of the two is a poor thing to
                                       # ask a small MLP to disentangle.
                 fail_penalty=0.0,     # UNIFORM cost for ending an episode
                                       # without reaching the goal, however it
                                       # ended. Without it the ways of failing
                                       # are priced wildly differently -- a fall
                                       # costs the level penalty PLUS the whole
                                       # accumulated route potential (measured
                                       # -40 on L38) while standing around until
                                       # the time limit costs 0 -- so the risk-
                                       # adjusted best move is to do nothing,
                                       # and a from-scratch policy correctly
                                       # learns exactly that (measured: it put
                                       # p=1e-4 on the only two actions that
                                       # cross, and 0.46 on a tiny safe walk).
                                       # Charging every failure the same F makes
                                       # exploring beat failing safely whenever
                                       # F < the crossing reward. Setting it also
                                       # stops the route potential resetting on a
                                       # fall, so the cost is not counted twice.
                 route_dense=0.0,      # DENSE route shaping. The default
                                       # potential only moves when the king
                                       # lands INSIDE the next waypoint box, so
                                       # a policy with no prior earns nothing at
                                       # all and has no gradient to follow. With
                                       # this > 0 the potential also credits how
                                       # far along the leg he is, scaled by this
                                       # weight -- still potential-based, so no
                                       # cycle can be farmed.
                 stuck_limit=0,        # end the episode after this many
                                       # CONSECUTIVE actions that leave the
                                       # king in the same place. On the icy
                                       # levels he can wedge (measured on L36:
                                       # x pinned at 294.9, y at 80, where every
                                       # jump, walk and settle moves him zero
                                       # pixels) -- move_available() stays True
                                       # so the physics-trap guard never fires,
                                       # and the episode burns its whole budget
                                       # earning exactly nothing. 0 = off, which
                                       # is what every existing model was
                                       # trained and is replayed with.
                 grid_channels=None,   # which occupancy planes the observation
                                       # carries. None = the historical
                                       # ("solid","king","hazard") triple that
                                       # every existing checkpoint expects. A
                                       # level family whose planes are redundant
                                       # (L36-38 are 100% ice, so hazard is an
                                       # exact copy of solid) can be trained on
                                       # a subset -- the checkpoint records it,
                                       # so relays stay index-compatible.
                 no_wind=False,        # DISABLE the wind force entirely (levels
                                       # 25-31 become deterministic). One switch,
                                       # no geometry changes. Use with wind_obs
                                       # OFF (the phase is meaningless with no
                                       # force). Keep this consistent between
                                       # training and play for a level.
                 goal_x_min=None,      # optional DELIVERY REGION: reaching the
                 goal_x_max=None,      # goal level only counts as success when
                                       # the king's x lies in [min, max] -- use
                                       # to steer a model toward the arrival
                                       # spot the NEXT level can actually work
                                       # with. Outside the region the episode
                                       # continues (he can still walk/jump in).
                 start_vel_jitter=0.0, # randomise the momentum a curriculum
                                       # start begins with (in observation
                                       # units, |v| <= this). A teleport places
                                       # the king at REST, but in a real climb
                                       # he ARRIVES sliding -- decisive on ice.
                                       # Jitter makes the policy robust to the
                                       # arrival momentum instead of over-
                                       # fitting the at-rest state. Ignored for
                                       # states that carry their own vx/vy.
                 goal_boxes=(),        # ROUTE waypoints: ordered boxes
                                       # (y, x_lo, x_hi) taken from a VERIFIED
                                       # route, each sized to its measured
                                       # basin -- the x window from which the
                                       # rest of the route still works. On ice
                                       # that window can be ~13px, so a height-
                                       # only waypoint is useless: the same
                                       # ledge reached 20px off is a dead end.
                                       # Progress is counted IN ORDER, so a step
                                       # that gains no height (walking out from
                                       # under an overhang) still pays.
                 goal_ys=(),           # WAYPOINT rewards: platform heights
                                       # (king rect_y). The first time in an
                                       # episode the king lands grounded at or
                                       # above each height he is paid once.
                                       # The episode is NOT cut short there --
                                       # a route on these screens goes up AND
                                       # down, so height is not route order, and
                                       # ending at a height would teach nothing
                                       # about whether that landing can be
                                       # continued from. Starts stay real (no
                                       # teleport), so momentum is whatever the
                                       # agent itself produced.
                 goal_y_reward=5.0,    # paid per newly reached waypoint
                 quarantine_after=150, # consecutive failures (with easier peers
                                       # mastered) before a start state's sample
                                       # weight collapses. Raise it (e.g. 600)
                                       # when a solvable state needs a rare
                                       # multi-action discovery so it is not
                                       # benched before the agent finds it.
                 max_steps=600,
                 max_settle_frames=2000,
                 goal_level=None,
                 transition_fail_y=None,      # if set, reaching the goal level
                                              # via a level transition with
                                              # king.rect_y >= this value ends
                                              # the episode as a failure
                 level_reward=10.0,            # reward per level climbed (the objective)
                 level_penalty=10.0,           # penalty per level fallen (magnitude)
                 level_penalty_cap=None,       # charge at most this many levels
                                               # per fall. Without it the cost of
                                               # failing grows with how far you
                                               # tumble, so standing HIGHER is
                                               # strictly more dangerous and the
                                               # agent learns to fail early
                                               # instead of climbing. Measured on
                                               # level 36: -20.6 average from the
                                               # entry vs -29.4 one jump higher.
                                               # None = original behaviour.
                 step_cost=0.0,                # charged for EVERY action. Without
                                               # it, idling is free: once falling
                                               # is the only thing that costs
                                               # anything, the safest policy is to
                                               # shuffle in place until the step
                                               # limit -- measured: 76 of 80
                                               # episodes ran to truncation with
                                               # zero route progress.
                 altitude_breadcrumb=0.01,     # weak within-level gradient toward the exit
                 start_states=None,            # path to starts/start_states.json (curriculum), or None
                 p_bottom=0.0,                 # prob of resetting to the bottom anyway
                 curriculum=False,             # enable the reverse (easy->hard) curriculum
                 cur_window=30,                # episodes per success-rate evaluation
                 cur_advance_rate=0.6,         # success rate that unlocks the next stage
                 cur_target_p_bottom=0.6,      # bottom-start fraction once fully unlocked
                 terminate_on_fall_below_start=True,   # end the attempt if the king drops
                                                       # below the level it started on
                                                       # (per-level models: default ON)
                 auto_frontier=False,        # bank grounded high-level states as new starts
                 frontier_min_level=1,       # only bank states on this level or above
                 frontier_per_level_cap=20,  # max banked spots kept per level
                 frontier_grid=12,           # dedup granularity in pixels
                 seed=None):

        self.charges = tuple(charges)
        self.walk_frames = int(walk_frames)
        self.fine_walk_frames = int(fine_walk_frames or 0)
        self.extra_charges = tuple(extra_charges or ())
        self.wait_frames = tuple(wait_frames or ())
        self.wait_wind = tuple(wait_wind or ())
        self.wind_jump = tuple(wind_jump or ())
        self.approach_jump = tuple(tuple(a) for a in (approach_jump or ()))
        self.wind_combo = tuple(tuple(tuple(s) for s in r)
                                for r in (wind_combo or ()))
        self.settle_action = bool(settle_action)
        self.wind_obs = bool(wind_obs)
        self.vel_obs = bool(vel_obs)
        # validated ONCE here; the hot path (build_occupancy_grid) trusts it
        self.grid_channels = resolve_channels(grid_channels)
        if vel_encoding not in ("xy", "polar", "both"):
            raise ValueError("vel_encoding must be 'xy', 'polar' or 'both', "
                             f"got {vel_encoding!r}")
        self.vel_encoding = vel_encoding
        self.route_dense = float(route_dense)
        self.fail_penalty = float(fail_penalty)
        self.stuck_limit = int(stuck_limit)
        self._stuck_at = None
        self._stuck_n = 0
        self.no_wind = bool(no_wind)
        self.goal_x_min = goal_x_min
        self.goal_x_max = goal_x_max
        self.max_steps = int(max_steps)
        self.max_settle_frames = int(max_settle_frames)
        self.goal_level = goal_level          # if set: terminate+reward on reaching it
        self.transition_fail_y = (None if transition_fail_y is None
                                  else int(transition_fail_y))
        self.level_reward = float(level_reward)
        self.level_penalty = float(level_penalty)
        self.level_penalty_cap = (None if level_penalty_cap is None
                                  else int(level_penalty_cap))
        self.step_cost = float(step_cost)
        self.altitude_breadcrumb = float(altitude_breadcrumb)
        self.p_bottom = float(p_bottom)
        self.terminate_on_fall_below_start = bool(terminate_on_fall_below_start)
        self.start_vel_jitter = float(start_vel_jitter)
        self.goal_boxes = tuple((int(y), int(xl), int(xh))
                                for (y, xl, xh) in (goal_boxes or ()))
        self._route_i = -1                   # last route waypoint passed
        # waypoints, highest y (lowest on screen) first
        self.goal_ys = tuple(sorted((int(y) for y in (goal_ys or ())), reverse=True))
        self.goal_y_reward = float(goal_y_reward)
        self._gy_phi = -1                    # current waypoint index (potential)
        self.rng = np.random.default_rng(seed)

        # Curriculum checkpoint pool (list of dicts). Empty list => bottom-only.
        self._start_pool = self._load_start_states(start_states)
        # ---- automatic frontier capture (augments the manual pool) -----------
        self.auto_frontier = bool(auto_frontier)
        self.frontier_min_level = int(frontier_min_level)
        self.frontier_per_level_cap = int(frontier_per_level_cap)
        self.frontier_grid = int(frontier_grid)
        self._best_seen_level = 0
        # Seed dedup bookkeeping from the existing pool so we never re-bank a known spot.
        self._frontier_keys, self._frontier_count = set(), {}
        for _s in self._start_pool:
            _lv = int(self._state_field(_s, "level", "current_level", default=0))
            _x = int(self._state_field(_s, "x", "rect_x", default=0))
            _y = int(self._state_field(_s, "y", "rect_y", default=0))
            self._frontier_keys.add((_lv, _x // self.frontier_grid, _y // self.frontier_grid))
            self._frontier_count[_lv] = self._frontier_count.get(_lv, 0) + 1

        # ---- reverse curriculum state ---------------------------------------
        # Rank checkpoints EASIEST-first: closest to the goal = highest level,
        # and within a level the smallest y (highest on screen). We unlock from
        # the easy end and add harder/lower starts as the agent succeeds, so it
        # masters the final hop first, then the climb leading up to it, and only
        # then the full level from the bottom. This is what stops it from
        # over-fitting the trivial near-exit hop and never learning the rest.
        self.curriculum = bool(curriculum)
        self.cur_window = int(cur_window)
        self.cur_advance_rate = float(cur_advance_rate)
        self.cur_target_p_bottom = float(cur_target_p_bottom)
        # Ranking key, easiest first: level, then VALIDATOR SCORE (a state the
        # BFS proved is d actions from the exit beats any unproven one -- pure
        # height misleads on levels where the route goes down before up), then
        # height as the tie-breaker for unscored states.
        self._cp_ranked = sorted(
            self._start_pool,
            key=lambda s: (self._state_field(s, "level", "current_level", default=0) * 100000
                           + float(s.get("score", 0.0) if isinstance(s, dict) else 0.0) * 1000
                           - self._state_field(s, "y", "rect_y", default=0)),
            reverse=True)                      # easiest (closest to goal) first
        self._unlocked = 1 if self._cp_ranked else 0   # how many easy checkpoints are live
        self._cur_p_bottom = 0.0 if self.curriculum else self.p_bottom
        self._succ = []                        # recent episode successes (1/0)
        self._episode_is_curric = False        # did THIS episode start from a checkpoint?
        # Per-checkpoint failure-rate EMA for prioritized sampling. Starts at
        # 1.0 (assume failing) so new/unlocked states immediately get focus.
        self._cp_fail = [1.0] * len(self._cp_ranked)
        # Consecutive failed attempts per checkpoint. A state that never
        # succeeds within the quarantine threshold is likely UNSOLVABLE
        # (disconnected from the exit); its sampling weight collapses so it
        # cannot eat an unattended run. Success clears the counter.
        self._cp_attempts = [0] * len(self._cp_ranked)
        self.quarantine_after = int(quarantine_after)
        self._episode_cp_idx = None            # which checkpoint THIS episode used

        self._build_action_table()

        # ---- boot pygame headlessly -----------------------------------------
        pygame.init()
        try:
            pygame.mixer.init()
        except Exception:
            pass

        # Imported lazily so SDL dummy drivers are already in place.
        from environment import Environment
        from Level import Levels
        from King import King
        from Babe import Babe

        self.environment = Environment()      # sets os.environ defaults, mixer channels

        w = int(os.environ.get("screen_width"))
        h = int(os.environ.get("screen_height"))
        # A real display surface is needed so convert_alpha() works, even headless.
        pygame.display.set_mode((w, h))
        self.game_screen = pygame.Surface((w, h))
        self.screen_w, self.screen_h = w, h

        self.levels = Levels(self.game_screen)
        self.king = King(self.game_screen, self.levels)
        self.babe = Babe(self.game_screen, self.levels)
        self.max_level = self.levels.max_level

        # ---- fake keyboard --------------------------------------------------
        # King._check_events() calls pygame.key.get_pressed() when agentCommand
        # is None. We replace it with a dict we control. A defaultdict(int)
        # returns 0 for any key we didn't press.
        from collections import defaultdict
        self._keys = defaultdict(int)
        # NOTE: this is a GLOBAL. Constructing a second env in the same process
        # would rebind it to that env's dict and leave this one permanently
        # unable to press a key -- the king would drift but never charge or
        # jump. _bind_keys() re-claims it every physics frame, which is what
        # makes several envs in one process behave independently.
        self._bind_keys()

        # ---- vision observation: occupancy grid (flattened) + scalars ------
        # We PACK grid+scalars into one flat vector so VecEnv/RolloutBuffer/the
        # train loop keep seeing a plain float vector; the network unpacks it.
        self.grid_cell = GRID_CELL
        self.grid_h = self.screen_h // self.grid_cell
        self.grid_w = self.screen_w // self.grid_cell
        self.grid_shape = (len(self.grid_channels), self.grid_h, self.grid_w)
        self._n_vel = (0 if not self.vel_obs else
                       {"xy": 2, "polar": 3, "both": 5}[self.vel_encoding])
        self.n_scalars = 4 + (2 if self.wind_obs else 0) + self._n_vel
        self._grid_flat = len(self.grid_channels) * self.grid_h * self.grid_w
        self.obs_dim = self._grid_flat + self.n_scalars
        self.num_actions = len(self.actions)

        self._steps = 0
        self._prev_level = 0
        self._episode_start_level = 0
        self._best_alt = 0.0
        self._episode_return = 0.0
        self._route_frac = 0.0

    # ------------------------------------------------------------------ setup
    def _build_action_table(self):
        """actions[i] = (kind, direction, magnitude)."""
        self.actions = build_action_table(self.charges, self.walk_frames,
                                          self.fine_walk_frames,
                                          self.extra_charges,
                                          self.wait_frames,
                                          self.wait_wind,
                                          self.wind_jump,
                                          self.approach_jump,
                                          self.wind_combo,
                                          self.settle_action)

    # ------------------------------------------------------------- curriculum
    def _load_start_states(self, path):
        """Load captured checkpoints into a flat list of {level, x, y} dicts.

        Understands capture.py's native format -- a dict keyed by level string
        with a list of [x, y] points per level:
            {"0": [[193, 16], [49, 160]], "1": [[...]]}
        Also tolerates a {'states': ...} wrapper and a plain list of dicts.
        Returns [] on any problem so we silently fall back to bottom-only
        starts instead of crashing a training run."""
        if not path:
            return []
        path = resolve_start_states(path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

        if isinstance(data, dict) and "states" in data:
            data = data["states"]

        pool = []
        if isinstance(data, dict):
            # capture.py format: {level_str: [[x, y], ...], ...}
            for lvl, points in data.items():
                try:
                    lvl_i = int(lvl)
                except (ValueError, TypeError):
                    continue
                if not isinstance(points, (list, tuple)):
                    continue
                for pt in points:
                    if isinstance(pt, dict):
                        pool.append(pt)
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        pool.append({"level": lvl_i, "x": int(pt[0]), "y": int(pt[1])})
        elif isinstance(data, list):
            # alt format: already a list of {level, x, y} dicts
            pool = [p for p in data if isinstance(p, dict)]
        return pool

    def _sample_start(self):
        """Pick a checkpoint to start at, or None to start at the bottom.

        Plain mode: uniform over the whole pool, bottom with prob p_bottom.
        Curriculum mode: PRIORITIZED over the currently-unlocked checkpoints --
        each state is weighted by its recent failure rate (EMA), so a start the
        policy keeps failing gets sampled hard until it is solved, and a solved
        one keeps a small floor weight as rehearsal against forgetting.
        (Same idea as Prioritized Level Replay, Jiang et al. 2021.)"""
        self._episode_cp_idx = None
        if not self._start_pool:
            return None
        if self.curriculum:
            if self.rng.random() < self._cur_p_bottom:
                return None
            k = max(1, self._unlocked)
            w = np.asarray(self._cp_fail[:k], dtype=np.float64) + 0.1
            for i in range(k):                   # quarantined: near-zero weight
                if self._cp_attempts[i] >= self.quarantine_after:
                    w[i] = 0.02
            idx = int(self.rng.choice(k, p=w / w.sum()))
            self._episode_cp_idx = idx
            return self._cp_ranked[idx]
        # plain mode
        if self.rng.random() < self.p_bottom:
            return None
        # If states carry a 'score' (reach-exit depth from
        # Validate_Start_States.py), favour checkpoints that actually lead
        # onward over dead ends. Falls back to uniform when no scores present.
        scores = np.array(
            [float(s.get("score", 1.0)) if isinstance(s, dict) else 1.0
             for s in self._start_pool], dtype=np.float64)
        if scores.sum() <= 0:
            idx = int(self.rng.integers(len(self._start_pool)))
        else:
            idx = int(self.rng.choice(len(self._start_pool), p=scores / scores.sum()))
        return self._start_pool[idx]

    def _maybe_bank_frontier(self):
        """Bank the king's current spot as a NEW start state if it is grounded
        and settled on a level >= frontier_min_level. The move_available() gate is
        the SAME grounded-and-settled check capture.py uses -- it is what keeps an
        un-grounded state (the instant-death spawn from Test B) out of the pool.
        Augments the manual pool; never removes anything. Returns the banked dict
        (for persistence by the trainer) or None."""
        if not self.auto_frontier or not self.move_available():
            return None
        level = self.levels.current_level
        if level < self.frontier_min_level:
            return None
        x, y = int(self.king.rect_x), int(self.king.rect_y)
        key = (level, x // self.frontier_grid, y // self.frontier_grid)
        if key in self._frontier_keys:
            return None                              # already have a near-identical spot
        if self._frontier_count.get(level, 0) >= self.frontier_per_level_cap:
            return None                              # enough diversity banked on this level
        state = {"level": level, "x": x, "y": y}
        self._start_pool.append(state)               # plain-mode sampler reads this
        self._frontier_keys.add(key)
        self._frontier_count[level] = self._frontier_count.get(level, 0) + 1
        self._best_seen_level = max(self._best_seen_level, level)
        return state

    def _record_episode(self, success):
        """Called once per finished episode. Drives the reverse curriculum:
        when recent success on the current stage is high enough, unlock the
        next-harder checkpoint; once all are unlocked, ramp up bottom starts."""
        if not self.curriculum or not self._cp_ranked:
            return
        # Per-checkpoint failure EMA drives the prioritized sampler: slow decay
        # on success (0.9) keeps a solved state rehearsed a while; a failure
        # pulls its weight back up fast.
        if self._episode_cp_idx is not None:
            i = self._episode_cp_idx
            self._cp_fail[i] = 0.9 * self._cp_fail[i] + (0.0 if success else 0.1)
            if success:
                self._cp_attempts[i] = 0
            else:
                self._cp_attempts[i] += 1
                if self._cp_attempts[i] == self.quarantine_after:
                    # Quarantine is a LAST resort: bench the state only when
                    # every EASIER-ranKED state (its ladder of prerequisites,
                    # by curriculum order) is mastered. While any rung above
                    # it still fails, this state SHOULD be failing -- parole
                    # it and keep the pressure on the rungs instead.
                    ladder = self._cp_fail[:i]
                    if ladder and float(max(ladder)) > 0.4:
                        self._cp_attempts[i] = self.quarantine_after // 2
                    else:
                        s = self._cp_ranked[i]
                        print(f"JK_Env: QUARANTINED start state "
                              f"({self._state_field(s,'x','rect_x')},"
                              f"{self._state_field(s,'y','rect_y')}) after "
                              f"{self.quarantine_after} straight failures "
                              f"with peers mastered -- likely unsolvable, "
                              f"sampling weight dropped")
        self._succ.append(1.0 if success else 0.0)
        if len(self._succ) < self.cur_window:
            return
        rate = sum(self._succ) / len(self._succ)
        if rate >= self.cur_advance_rate:
            if self._unlocked < len(self._cp_ranked):
                self._unlocked += 1                       # add the next harder start
            elif self._cur_p_bottom < self.cur_target_p_bottom:
                self._cur_p_bottom = min(self.cur_target_p_bottom,
                                         self._cur_p_bottom + 0.1)
            self._succ = []                               # fresh window for the new stage

    def goal_y_status(self):
        if not self.goal_ys:
            return None
        return {"at": self._gy_phi + 1, "of": len(self.goal_ys),
                "y": (self.goal_ys[self._gy_phi] if self._gy_phi >= 0 else None)}

    def curriculum_status(self):
        rate = (sum(self._succ) / len(self._succ)) if self._succ else float("nan")
        status = {"unlocked": self._unlocked,
                  "total": len(self._cp_ranked),
                  "p_bottom": round(self._cur_p_bottom, 2),
                  "succ_rate": round(rate, 2)}
        if self._cp_ranked and self._unlocked:
            # the start state the prioritized sampler is currently grinding
            k = max(1, self._unlocked)
            i = int(np.argmax(self._cp_fail[:k]))
            s = self._cp_ranked[i]
            status["focus"] = (int(self._state_field(s, "x", "rect_x", default=-1)),
                               int(self._state_field(s, "y", "rect_y", default=-1)),
                               round(self._cp_fail[i], 2))
        return status

    @staticmethod
    def _state_field(state, *names, default=None):
        """Read the first present key from a checkpoint dict, tolerating the
        different names capture.py might have used (level/current_level, x/rect_x...)."""
        for n in names:
            if n in state:
                return state[n]
        return default

    # ------------------------------------------------------------- key helper
    def _set_keys(self, space=False, left=False, right=False):
        self._keys[pygame.K_SPACE] = 1 if space else 0
        self._keys[pygame.K_LEFT] = 1 if left else 0
        self._keys[pygame.K_RIGHT] = 1 if right else 0

    # --------------------------------------------------------- physics driver
    def _bind_keys(self):
        """Point pygame.key.get_pressed() at THIS env's fake keyboard.

        pygame.key.get_pressed is process-global, so with N envs alive only the
        one that patched it last could actually press a key; every other env
        silently read an empty keyboard. Measured on L38: the same snapshot and
        the same action crossed the level in the last-built env (+21.12) and did
        nothing at all in the others, which is why in-process vectorised
        training learned from mostly dead rollouts."""
        pygame.key.get_pressed = lambda: self._keys

    def _physics_frame(self):
        """Advance the game one logic frame. Skips all cosmetic systems
        (audio/npc/flyer/readable/name/hiddenwall) for speed, but keeps wind,
        because wind perturbs the King's physics on windy levels."""
        self._bind_keys()          # several envs may share this process
        if not self.no_wind:
            self.levels.update_wind(self.king)         # physics-affecting
        prev_level = self.levels.current_level
        self.king.update(agentCommand=None)            # full physics + level change
        # Wind starts FRESH when the king first enters the windy biome (level
        # 25). This makes the 24->25 handoff deterministic (the landing no
        # longer depends on how long the climb took) -- the wind then builds
        # up from calm as the agent climbs the biome, exactly as in play.
        if (not self.no_wind and prev_level < 25 <= self.levels.current_level):
            self.levels.wind.wind_var = 0.0
        if self.levels.current_level == self.babe.level:
            # Only relevant at the very top; this is what flips levels.ending.
            self.babe.update(self.king)

    def move_available(self):
        k = self.king
        return (not k.isFalling
                and not self.levels.ending
                and (not k.isSplat or k.splatCount > k.splatDuration))

    # ----------------------------------------------------------------- action
    def _apply_action(self, action_idx, render_cb=None):
        kind, direction, magnitude = self.actions[action_idx]

        def frame():
            # One physics frame, plus an optional render so callers (play.py)
            # can WATCH the jump arc instead of only seeing the settled frame.
            self._physics_frame()
            if render_cb is not None:
                render_cb()

        def _charge_and_release(jdir, mag):
            # King._check_events fires a jump in TWO ways, and maxJumpCount=30:
            #   (a) while SPACE is held, once jumpCount > maxJumpCount it
            #       auto-fires using whatever arrow is held THAT frame, or "up"
            #       if none;
            #   (b) when SPACE is released while crouched, it fires the arrow
            #       held on the release frame, or "up" if none.
            # Input is only read while grounded (update() skips _check_events
            # when isFalling), so once a jump fires the king is airborne and
            # further keypresses are ignored.
            #
            # Therefore we hold the DIRECTION key during the whole charge (not
            # just at release). For charges <= maxJumpCount that is inert and
            # the release frame fires the direction; for charges > maxJumpCount
            # the auto-fire reads the held direction instead of defaulting to
            # "up" -- which is the bug that made every long directional jump go
            # straight up. For "up" we hold no arrow, so it fires up either way.
            hold_left = (jdir == "left")
            hold_right = (jdir == "right")
            for _ in range(mag):
                self._set_keys(space=True, left=hold_left, right=hold_right)
                frame()
                if self.levels.ending or self.king.isJump or self.king.isFalling:
                    break
            self._set_keys(left=hold_left, right=hold_right)
            frame()

        def _walk_to_x(target_x, cap):
            # Walk (either direction) until within a couple px of target_x, on
            # the windless snow shelf this is fully deterministic. Lets one
            # macro reach the crossing window from anywhere on the shelf.
            for _ in range(cap):
                dx = self.king.rect_x - target_x
                if dx < -2.0:
                    self._set_keys(right=True)
                elif dx > 2.0:
                    self._set_keys(left=True)
                else:
                    break
                frame()
                if self.levels.ending or self.king.isFalling:
                    break
            self._set_keys()

        def _kill_momentum(cap=400, eps=0.001, min_frames=20):
            # ICE: after a horizontal jump the king keeps sliding (slip>0) and
            # move_available() returns True WHILE still moving -- so a plain-jump
            # combo would launch mid-slide, and that residual speed makes the next
            # jump land somewhere else (breaks the handoff). Just let the slide
            # finish: with no keys held, friction (speed*=slip each frame) damps
            # it to a deterministic natural stop. Active counter-walk is worse on
            # ice (each direction switch re-accelerates). On non-ice slip==0 so
            # speed is already 0 and this returns immediately.
            for k in range(cap):
                # settle to a canonical AT-REST state: run at least min_frames
                # (so a just-crossed king whose move_available fired early -- non-
                # resting angle/flags -- fully stabilizes) then until speed ~= 0.
                if k >= min_frames and self.king.speed <= eps:
                    break
                self._set_keys()
                frame()
                if self.levels.ending or self.king.isFalling:
                    break
            self._set_keys()
            # canonicalize the AT-REST state to match a fresh teleport: a king
            # that just crossed levels can be grounded+stopped but keep a stale
            # facing angle (e.g. 3.8 vs the resting pi/2), which changes the very
            # next jump. Reset it so a multi-step route reproduces its own path.
            if not self.king.isFalling and not self.king.isJump:
                self.king.speed = 0.0
                self.king.angle = math.pi / 2

        def _hold_until_wind(target_bucket, cap):
            # Wait until wind_ready(target_bucket), HOLDING x against wind drift
            # by counter-walking toward the spot we started on (the agent's
            # "move opposite the wind" technique). On deep snow the wind exerts
            # no force so this is inert; elsewhere it keeps the launch spot
            # fixed so the crossing is identical from any starting phase.
            hold_x = self.king.rect_x
            for _ in range(cap):
                if self.king.rect_x > hold_x + 1.0:
                    self._set_keys(left=True)
                elif self.king.rect_x < hold_x - 1.0:
                    self._set_keys(right=True)
                else:
                    self._set_keys()
                frame()
                if self.levels.ending:
                    break
                if wind_ready(self.levels.wind.wind_var, target_bucket):
                    break
            self._set_keys()

        if kind == "jump":
            _charge_and_release(direction, magnitude)
        elif kind == "jump_wind":
            # ATOMIC wind-timed jump: direction == (target_bucket, jdir).
            # Hold position until the wind reaches the target, THEN launch.
            target_b, jdir = direction
            _hold_until_wind(target_b, 1100)
            _charge_and_release(jdir, magnitude)
        elif kind == "approach_jump":
            # ATOMIC "walk to target_x, then wind-timed jump".
            # direction == (target_x, target_bucket, jdir).
            target_x, target_b, jdir = direction
            _walk_to_x(target_x, 600)
            _hold_until_wind(target_b, 1100)
            _charge_and_release(jdir, magnitude)
        elif kind == "wind_combo":
            # ATOMIC multi-step wind route: magnitude is a tuple of
            # (bucket, jdir, charge) steps. Execute each wind-timed jump in
            # sequence, settling between so the next hold-until-wind starts
            # grounded. Bail out early if the king leaves the start level.
            for step in magnitude:
                if step[0] == "walk":
                    # deterministic reposition to a fixed launch x (snow ledge is
                    # windless) -> makes the combo robust to entry-x variance
                    _walk_to_x(int(step[1]), 600)
                    continue
                if step[0] == "settle":
                    # ICE: counter-walk off the arrival momentum before launching
                    _kill_momentum()
                    continue
                if step[0] == "jump":
                    # plain jump, no wind wait (windless / ice levels)
                    _charge_and_release(step[1], int(step[2]))
                else:
                    b, jd, c = step
                    _hold_until_wind(b, 1100)
                    _charge_and_release(jd, c)
                self._set_keys()
                f = 0
                while (not self.move_available()
                       and f < self.max_settle_frames
                       and not self.levels.ending):
                    frame()
                    f += 1
                if self.levels.ending:
                    break
        elif kind == "settle":
            # stand still until friction has damped the slide to a canonical
            # at-rest state (the same routine the ice combos use internally)
            _kill_momentum()
        elif kind == "wait":
            # Stand still and let the world (the wind phase) move on.
            self._set_keys()
            for _ in range(magnitude):
                frame()
                if self.levels.ending:
                    break
        elif kind == "wait_wind":
            # Hold position until the wind reaches the target bucket, so the
            # NEXT action launches into it. (direction = bucket index.)
            _hold_until_wind(direction, magnitude)
        else:  # walk
            for _ in range(magnitude):
                self._set_keys(left=(direction == "left"),
                               right=(direction == "right"))
                frame()

        # Let go and simulate until grounded & settled (or capped).
        self._set_keys()
        frames = 0
        while (not self.move_available()
               and frames < self.max_settle_frames
               and not self.levels.ending):
            frame()
            frames += 1
        # False => physics trap: the king never came to rest (e.g. bouncing in
        # a slope pocket). step() ends the episode on it.
        return self.move_available() or self.levels.ending

    # ------------------------------------------------------------------ state
    def _route_frac_now(self, level, route_i):
        """How far along the current route leg the king is, in [0, 1].

        Potential-based shaping only works if the potential is evaluated at the
        START state too. Initialising it to 0 instead handed every episode a
        free jump in potential on its first step -- measured as a flat +5 that
        the agent collected no matter what it did."""
        if not self.goal_boxes or self.route_dense <= 0.0:
            return 0.0
        if level > self._episode_start_level:
            return 1.0
        if level < self._episode_start_level:
            # with a uniform failure penalty the shaping must not punish a fall
            # as well, or the two costs stack and exploring goes negative-value
            return self._route_frac if self.fail_penalty > 0.0 else 0.0
        j = route_i + 1
        if j >= len(self.goal_boxes):
            return 1.0
        gy, xl, xh = self.goal_boxes[j]
        gx = 0.5 * (xl + xh)
        d = math.hypot(self.king.rect_x - gx, self.king.rect_y - gy)
        return max(0.0, 1.0 - d / ROUTE_DENSE_SCALE)

    def _altitude(self):
        """Monotonic 'how high have we climbed', in pixels. Higher = better.
        Each level up is +screen_h; within a screen, smaller rect_y = higher."""
        return (self.levels.current_level * self.screen_h
                + (self.screen_h - self.king.rect_y))

    def _obs(self):
        # Occupancy grid of the CURRENT screen (solid/king/hazard), flattened,
        # followed by the same 4 scalars as before (sub-cell position still
        # matters for jump precision, and the grid is only 8px-resolution).
        try:
            platforms = self.levels.levels[self.levels.current_level].platforms or []
        except Exception:
            platforms = []
        grid = build_occupancy_grid(platforms,
                                    self.king.rect_x, self.king.rect_y,
                                    self.screen_w, self.screen_h, self.grid_cell,
                                    self.grid_channels)
        vals = [
            self.king.rect_x / self.screen_w,
            self.king.rect_y / self.screen_h,
            self.levels.current_level / self.max_level,
            self._altitude() / ((self.max_level + 1) * self.screen_h),
        ]
        if self.wind_obs:
            # Complete wind state: force right now (sin) and where the cycle
            # is heading (cos). Together they make windy levels Markov again.
            wv = float(self.levels.wind.wind_var)
            vals += [math.sin(wv), math.cos(wv)]
        if self.vel_obs:
            # (vx, vy) = the actual per-frame displacement (King.py: rect_x +=
            # sin(a)*speed, rect_y -= cos(a)*speed), normalized by maxSpeed=11.
            # ~0 at rest / on non-ice; nonzero mid-slide on ice -> momentum state.
            s, a = float(self.king.speed), float(self.king.angle)
            vx, vy = math.sin(a) * s / 11.0, -math.cos(a) * s / 11.0
            if self.vel_encoding in ("xy", "both"):
                vals += [vx, vy]
            if self.vel_encoding in ("polar", "both"):
                mag = math.hypot(vx, vy)
                if mag > 1e-9:
                    vals += [mag, vx / mag, vy / mag]
                else:
                    vals += [0.0, 0.0, 0.0]      # at rest: no direction to report
        scalars = np.array(vals, dtype=np.float32)
        return np.concatenate([grid.ravel(), scalars]).astype(np.float32)

    # ---------------------------------------------------------------- teleport
    def teleport(self, level, x=None, y=None, vx=None, vy=None):
        """THE canonical way to place the King anywhere. Everything that moves
        him (reset, the validator, the prober, future explorers) must go
        through here, so grounding behaves identically everywhere.

        Fully resets the King's physics state (flags, momentum, charge),
        positions him, then marks him airborne and runs physics until he is
        grounded and settled -- so a point captured slightly above a platform
        lands on it instead of spawning half-clipped with stale flags.

        Returns the settled (level, x, y), which may differ from the request
        (e.g. a bad point that slides off a ledge)."""
        self.king.reset()
        self.levels.reset()

        os.environ["start"] = "1"
        os.environ["gaming"] = "1"
        os.environ["pause"] = ""
        os.environ["active"] = "1"
        os.environ["mode"] = "normal"

        self.levels.current_level = int(level)
        if x is not None:
            self.king.rect_x = float(x)
        if y is not None:
            self.king.rect_y = float(y)

        self._set_keys()
        # Airborne with zero momentum: gravity grounds him onto whatever is
        # below, and _check_events stays skipped until he has truly landed.
        # Settle at wind_var=0 (levels.reset zeroed it) so the SETTLED POSITION
        # is deterministic -- randomizing the phase before this would make the
        # king land in a different spot every episode on windy levels.
        self.king.isFalling = True
        frames = 0
        while (not self.move_available()
               and frames < self.max_settle_frames
               and not self.levels.ending):
            self._physics_frame()
            frames += 1

        # NOW set the episode's wind phase (after the king is grounded). Without
        # this every episode would train against the exact same wind timeline
        # and overfit phase 0; in the relay the king climbs into arbitrary
        # phases, so training must cover them all.
        if self.wind_obs:
            self.levels.wind.wind_var = float(self.rng.uniform(0.0, 2.0 * math.pi))

        # Optional ARRIVAL MOMENTUM. A reverse-curriculum start normally drops
        # the king at REST, but in a real climb he arrives sliding -- and on ice
        # a grounded king keeps moving, so "at rest" is a state the relay almost
        # never hands the policy. Applying the velocity AFTER the settle keeps
        # the landing spot deterministic while making the STATE realistic.
        if vx is not None or vy is not None:
            self.set_velocity(float(vx or 0.0), float(vy or 0.0))
        return (self.levels.current_level,
                int(self.king.rect_x), int(self.king.rect_y))

    def set_velocity(self, vx, vy):
        """Give the grounded king momentum, in the SAME normalised units the
        observation reports (fractions of King.maxSpeed = 11).

        The engine stores motion as (speed, angle) with
            rect_x += sin(angle) * speed ;  rect_y -= cos(angle) * speed
        so vx = sin(a)*s/11 and vy = -cos(a)*s/11 invert to the below."""
        speed = math.hypot(float(vx), float(vy)) * 11.0
        self.king.speed = speed
        if speed > 0.0:
            self.king.angle = math.atan2(float(vx), -float(vy))

    def velocity(self):
        """The king's current (vx, vy) in observation units."""
        s, a = float(self.king.speed), float(self.king.angle)
        return math.sin(a) * s / 11.0, -math.cos(a) * s / 11.0

    # ------------------------------------------------------------------- gym
    # ---------------------------------------------------------------- snapshot
    # Attributes of King that are NOT physics state: surfaces, sprite tables,
    # back-references. Everything else that is a plain number/bool/string IS
    # state and gets captured automatically, so a field added to King later is
    # picked up without touching this list.
    _KING_SKIP = ("screen", "sprites", "levels", "timer", "current_image",
                  "walkAngles", "jumpAngles", "lastCollision")

    def snapshot(self):
        """Capture the COMPLETE physics state as a JSON-able dict.

        teleport() cannot express every state: it resets the king, places him,
        settles him to rest and only then injects a velocity, so the flags a
        real arrival produced (isJump, collideBottom, slope, slip, the mid-slide
        phase) are lost. On non-ice that does not matter -- slip==0 makes every
        grounded state converge to the same rest point -- but on ice the state
        IS (position, velocity, phase), so a reverse curriculum built from
        teleport samples trains on states the game never actually produces.

        snapshot/restore is the way out: capture what really happened, replay it
        exactly. This is the mechanism Go-Explore calls returning to a cell by
        restoring the simulator state rather than reconstructing the position."""
        k = self.king
        st = {a: v for a, v in vars(k).items()
              if a not in self._KING_SKIP
              and isinstance(v, (int, float, bool, str, type(None)))}
        # lastCollision is a Platform object; store WHICH platform of the
        # current level it is, so restore can rebind it to the live object.
        lc = getattr(k, "lastCollision", None)
        st["lastCollision_i"] = -1
        if lc is not None:
            try:
                plats = self.levels.levels[self.levels.current_level].platforms or []
                st["lastCollision_i"] = list(plats).index(lc)
            except (ValueError, KeyError, AttributeError, IndexError):
                st["lastCollision_i"] = -1
        return {"v": 1,
                "king": st,
                "level": int(self.levels.current_level),
                "ending": bool(self.levels.ending),
                "wind_var": float(self.levels.wind.wind_var)}

    def restore(self, snap, route_i=-1):
        """Put the king back into exactly the state snapshot() captured.

        Returns the observation, like reset(). Episode bookkeeping is anchored
        on the restored level, so the fall rule and the altitude breadcrumb
        behave as if the episode had started here.

        `route_i` is how far along goal_boxes this state already is; the route
        shaping then looks for waypoint route_i+1 next. It matters when a
        backward curriculum restores a state from the MIDDLE of a route: with
        the default -1 the env would wait for waypoint 0, which the king already
        passed, so every step of a mid-route episode would earn nothing."""
        if not isinstance(snap, dict) or "king" not in snap:
            raise ValueError("restore() expects a dict from snapshot()")
        k = self.king
        lc_i = snap["king"].get("lastCollision_i", -1)
        for a, v in snap["king"].items():
            if a == "lastCollision_i":
                continue
            setattr(k, a, v)
        self.levels.current_level = int(snap["level"])
        self.levels.ending = bool(snap.get("ending", False))
        self.levels.wind.wind_var = float(snap.get("wind_var", 0.0))
        k.lastCollision = None
        if lc_i >= 0:
            try:
                plats = self.levels.levels[self.levels.current_level].platforms or []
                k.lastCollision = list(plats)[lc_i]
            except (IndexError, KeyError, AttributeError):
                k.lastCollision = None
        self._set_keys()                       # no key may be left held down

        self._steps = 0
        self._episode_return = 0.0
        self._prev_level = self.levels.current_level
        self._episode_start_level = self.levels.current_level
        self._best_alt = self._altitude()
        self._gy_phi = -1
        self._route_i = int(route_i)
        self._route_frac = self._route_frac_now(self.levels.current_level,
                                                self._route_i)
        self._episode_is_curric = False
        self._stuck_at, self._stuck_n = None, 0
        return self._obs(), {}

    def reset(self, level=None, rect_x=None, rect_y=None, vx=None, vy=None):
        """Reset to a curriculum checkpoint (default) or to an explicit start.

        Calling reset() with no args enables the curriculum: with probability
        (1 - p_bottom) we drop the King at a random captured checkpoint from
        starts/start_states.json; otherwise (or if the pool is empty) we reset to the
        bottom. Passing any of level/rect_x/rect_y explicitly bypasses the
        curriculum and forces that exact start (back-compatible with old code).

        Placement goes through teleport(), which settles the King onto the
        ground -- so a captured point a few pixels above a platform is fine."""
        explicit = (level is not None or rect_x is not None or rect_y is not None)
        if not explicit:
            s = self._sample_start()
            self._episode_is_curric = s is not None
            if s is not None:
                level = self._state_field(s, "level", "current_level", default=0)
                rect_x = self._state_field(s, "x", "rect_x")
                rect_y = self._state_field(s, "y", "rect_y")
                # start states may carry the momentum the king really arrives
                # with; absent -> at rest, exactly as before
                if vx is None:
                    vx = self._state_field(s, "vx")
                if vy is None:
                    vy = self._state_field(s, "vy")
                if vx is None and vy is None and self.start_vel_jitter > 0.0:
                    j = self.start_vel_jitter
                    vx = float(self.rng.uniform(-j, j))
                    vy = float(self.rng.uniform(-j, j))
        else:
            self._episode_is_curric = False

        if level is None:
            level = 0

        settled_level, _, _ = self.teleport(level, rect_x, rect_y, vx, vy)

        self._steps = 0
        self._episode_return = 0.0
        # Anchor the episode on the level he actually SETTLED on, not the one
        # requested: a checkpoint that slips a level down should not count the
        # very first step as "fell below start".
        self._prev_level = settled_level
        self._episode_start_level = settled_level    # anchor for the fall rule
        self._best_alt = self._altitude()            # best height reached this episode
        self._gy_phi = -1                            # below every waypoint
        self._route_i = -1                           # no route waypoint passed
        self._route_frac = self._route_frac_now(settled_level, -1)
        self._stuck_at, self._stuck_n = None, 0
        return self._obs(), {}

    def step(self, action_idx, render_cb=None):
        settled = self._apply_action(int(action_idx), render_cb=render_cb)
        self._steps += 1

        level = self.levels.current_level
        alt = self._altitude()

        # ---- reward: LEVEL TRANSITIONS dominate ------------------------------
        # Passing a level is the objective; falling back down a level is the
        # core Jump King punishment and must be felt. A weak new-max-altitude
        # breadcrumb keeps a gradient alive *within* a level so the agent has
        # some signal pointing toward the exit. Re-treading height and falling
        # both earn ZERO from the breadcrumb -> no hop-in-place optimum.
        reward = -self.step_cost          # time is not free

        d_level = level - self._prev_level
        if d_level > 0:
            reward += self.level_reward * d_level          # climbed one or more screens
        elif d_level < 0:
            n = -d_level
            if self.level_penalty_cap is not None:
                n = min(n, self.level_penalty_cap)
            reward -= self.level_penalty * n               # fell back down
        self._prev_level = level

        if alt > self._best_alt:
            reward += (alt - self._best_alt) * self.altitude_breadcrumb
            self._best_alt = alt

        reached_goal = False
        terminated = False
        transition_failed = False
        if self.levels.ending:                       # reached the babe at the top
            terminated = True
            reached_goal = True
            reward += 100.0
        elif self.goal_level is not None and level >= self.goal_level:
            transition_failed = (
                self.transition_fail_y is not None
                and d_level > 0
                and self.king.rect_y >= self.transition_fail_y
            )
            if transition_failed:
                terminated = True
                reward -= self.level_reward
            else:
                in_region = ((self.goal_x_min is None
                              or self.king.rect_x >= self.goal_x_min)
                             and (self.goal_x_max is None
                                  or self.king.rect_x <= self.goal_x_max))
                if in_region:
                    terminated = True
                    reached_goal = True
                    reward += self.level_reward            # same scale as a normal pass
                # outside the delivery region: no success yet -- the episode
                # continues so the king can still move into the region.

        # WAYPOINT SHAPING, potential-based (Ng et al. 1999): pay the CHANGE in
        # how many platform heights the king is above, not a one-off bounty for
        # each. Paying once per new height makes FAILING profitable -- collect
        # +5 four times, fall one level for -10, and a doomed episode still nets
        # +10, which is exactly what the agent learned to farm. With a
        # difference, climbing pays and dropping back charges the same amount
        # again, so no cycle and no failure can be milked; only real progress
        # that is KEPT survives to the end of the episode.
        # ROUTE PROGRESS, potential-based and IN ORDER. Only the next waypoint
        # counts, so the agent cannot collect a later box by luck, and a step
        # that gains no height still pays if it is the one the route needs.
        if self.goal_boxes:
            if level < self._episode_start_level and self.fail_penalty <= 0.0:
                phi_r = -1                       # fell off the route: charge it back
            elif level > self._episode_start_level:
                # LEFT THE LEVEL UPWARD -- the route is what leads here, so it is
                # complete, not abandoned. Resetting the potential to -1 here
                # made the potential difference pay back every waypoint on the
                # very step that succeeds: measured -8*goal_y_reward against a
                # +level_reward, i.e. the agent was PUNISHED for crossing.
                phi_r = len(self.goal_boxes) - 1
            else:
                phi_r = self._route_i
                if phi_r + 1 < len(self.goal_boxes) and self.move_available():
                    gy, xl, xh = self.goal_boxes[phi_r + 1]
                    if (abs(self.king.rect_y - gy) <= 6
                            and xl <= self.king.rect_x <= xh):
                        phi_r += 1
            # DENSE within-leg term. Without it the potential only moves when
            # the king lands inside the next waypoint box, which a policy with
            # no prior essentially never does -- so it sees a flat reward and
            # has nothing to follow. frac is how far along the current leg he
            # is, so the full potential is goal_y_reward*route_i + w*frac and a
            # step that merely gets CLOSER already pays.
            frac = self._route_frac_now(level, phi_r)
            reward += self.goal_y_reward * (phi_r - self._route_i)
            reward += self.route_dense * (frac - self._route_frac)
            self._route_i = phi_r
            self._route_frac = frac

        if self.goal_ys:
            phi = -1
            if level == self._episode_start_level and self.move_available():
                for i, gy in enumerate(self.goal_ys):
                    if self.king.rect_y <= gy:
                        phi = i
            reward += self.goal_y_reward * (phi - self._gy_phi)
            self._gy_phi = phi

        # Fall rule: dropped below the level this attempt started on -> the
        # attempt is over. End it as a FAILURE (reached_goal stays False) so the
        # next reset re-anchors on the same level instead of forcing a full
        # re-climb. The negative level_penalty applied above is the cost of it.
        if (self.terminate_on_fall_below_start
                and not terminated
                and level < self._episode_start_level):
            terminated = True

        # Physics-trap guard: the settle loop capped out, the king is still
        # bouncing (slope pocket etc.). Nothing meaningful can be learned or
        # rendered past this point -- end the attempt as a failure.
        if not settled and not terminated:
            terminated = True

        # WEDGE guard: grounded, settled, and yet completely immobile. Distinct
        # from the trap above -- move_available() is True here, the king simply
        # cannot be moved by any action from where he is.
        if self.stuck_limit > 0 and not terminated:
            here = (level, round(float(self.king.rect_x), 1),
                    round(float(self.king.rect_y), 1))
            if here == self._stuck_at:
                self._stuck_n += 1
                if self._stuck_n >= self.stuck_limit:
                    terminated = True
                    # Charge it like a fall. A zero-cost terminal is worse than
                    # useless here: wedging would DOMINATE falling, and a policy
                    # with no prior learns to wedge on purpose -- measured, the
                    # from-scratch arm sat at exactly 0.00 return forever.
                    # (fail_penalty, when set, already prices every failure the
                    # same, so this would double-charge.)
                    if self.fail_penalty <= 0.0:
                        reward -= self.level_penalty
            else:
                self._stuck_at, self._stuck_n = here, 0

        truncated = self._steps >= self.max_steps

        if (terminated or truncated) and not reached_goal:
            reward -= self.fail_penalty

        if terminated or truncated:
            # success = reached the GOAL, not merely 'episode ended'. A fall
            # below the start level terminates the episode but is NOT a success.
            self._record_episode(success=reached_goal)

        banked = self._maybe_bank_frontier()   # grounded-gated; None unless a new spot

        self._episode_return += reward
        info = {"level": self.levels.current_level,
                "altitude": alt,
                "x": self.king.rect_x,
                "y": self.king.rect_y,
                "success": reached_goal,
                "from_checkpoint": self._episode_is_curric,
                "frontier": banked,
                "curriculum": self.curriculum_status()}
        if terminated or truncated:
            # Episode stats for single-env consumers (smoke test, --render
            # mode); the subprocess vec-env computes its own identical dict.
            info["episode"] = {"r": self._episode_return,
                               "l": self._steps,
                               "level": self.levels.current_level}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


def make_env(**kwargs):
    return JumpKingEnv(**kwargs)


# ---------------------------------------------------------------- smoke test
if __name__ == "__main__":
    # Run from the game root:  python JK_Env.py
    env = JumpKingEnv(max_steps=20)
    obs, _ = env.reset()
    print("obs_dim:", env.obs_dim, "num_actions:", env.num_actions)
    print("start obs:", obs)
    for t in range(20):
        a = int(env.rng.integers(env.num_actions))
        obs, r, term, trunc, info = env.step(a)
        print(f"t={t:2d} act={a:2d} {env.actions[a]!s:24s} "
              f"r={r:+.3f} lvl={info['level']} y={info['y']} alt={info['altitude']:.0f}")
        if term or trunc:
            print("episode end (term=%s trunc=%s)" % (term, trunc))
            obs, _ = env.reset()
    env.close()