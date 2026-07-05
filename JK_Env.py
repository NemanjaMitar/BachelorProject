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

* CURRICULUM: if you pass start_states="start_states.json" (a list of captured
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
from Occupancy import build_occupancy_grid, GRID_CELL


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
                       wait_wind=(), wind_jump=()):
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
    return actions


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
                 wind_obs=False,       # append sin/cos of the wind phase to the
                                       # observation scalars (4 -> 6) AND
                                       # randomize the phase at every teleport
                                       # so training covers all phases.
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
                 altitude_breadcrumb=0.01,     # weak within-level gradient toward the exit
                 start_states=None,            # path to start_states.json (curriculum), or None
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
        self.wind_obs = bool(wind_obs)
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
        self.altitude_breadcrumb = float(altitude_breadcrumb)
        self.p_bottom = float(p_bottom)
        self.terminate_on_fall_below_start = bool(terminate_on_fall_below_start)
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
        pygame.key.get_pressed = lambda: self._keys

        # ---- vision observation: occupancy grid (flattened) + scalars ------
        # We PACK grid+scalars into one flat vector so VecEnv/RolloutBuffer/the
        # train loop keep seeing a plain float vector; the network unpacks it.
        self.grid_cell = GRID_CELL
        self.grid_h = self.screen_h // self.grid_cell
        self.grid_w = self.screen_w // self.grid_cell
        self.grid_shape = (3, self.grid_h, self.grid_w)
        self.n_scalars = 6 if self.wind_obs else 4
        self._grid_flat = 3 * self.grid_h * self.grid_w
        self.obs_dim = self._grid_flat + self.n_scalars
        self.num_actions = len(self.actions)

        self._steps = 0
        self._prev_level = 0
        self._episode_start_level = 0
        self._best_alt = 0.0
        self._episode_return = 0.0

    # ------------------------------------------------------------------ setup
    def _build_action_table(self):
        """actions[i] = (kind, direction, magnitude)."""
        self.actions = build_action_table(self.charges, self.walk_frames,
                                          self.fine_walk_frames,
                                          self.extra_charges,
                                          self.wait_frames,
                                          self.wait_wind,
                                          self.wind_jump)

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
    def _physics_frame(self):
        """Advance the game one logic frame. Skips all cosmetic systems
        (audio/npc/flyer/readable/name/hiddenwall) for speed, but keeps wind,
        because wind perturbs the King's physics on windy levels."""
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
                                    self.screen_w, self.screen_h, self.grid_cell)
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
        scalars = np.array(vals, dtype=np.float32)
        return np.concatenate([grid.ravel(), scalars]).astype(np.float32)

    # ---------------------------------------------------------------- teleport
    def teleport(self, level, x=None, y=None):
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
        return (self.levels.current_level,
                int(self.king.rect_x), int(self.king.rect_y))

    # ------------------------------------------------------------------- gym
    def reset(self, level=None, rect_x=None, rect_y=None):
        """Reset to a curriculum checkpoint (default) or to an explicit start.

        Calling reset() with no args enables the curriculum: with probability
        (1 - p_bottom) we drop the King at a random captured checkpoint from
        start_states.json; otherwise (or if the pool is empty) we reset to the
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
        else:
            self._episode_is_curric = False

        if level is None:
            level = 0

        settled_level, _, _ = self.teleport(level, rect_x, rect_y)

        self._steps = 0
        self._episode_return = 0.0
        # Anchor the episode on the level he actually SETTLED on, not the one
        # requested: a checkpoint that slips a level down should not count the
        # very first step as "fell below start".
        self._prev_level = settled_level
        self._episode_start_level = settled_level    # anchor for the fall rule
        self._best_alt = self._altitude()            # best height reached this episode
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
        reward = 0.0

        d_level = level - self._prev_level
        if d_level > 0:
            reward += self.level_reward * d_level          # climbed one or more screens
        elif d_level < 0:
            reward += self.level_penalty * d_level         # fell back down (negative)
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

        truncated = self._steps >= self.max_steps

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