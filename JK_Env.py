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
                 max_steps=600,
                 max_settle_frames=2000,
                 goal_level=None,
                 level_reward=10.0,            # reward per level climbed (the objective)
                 level_penalty=10.0,           # penalty per level fallen (magnitude)
                 altitude_breadcrumb=0.01,     # weak within-level gradient toward the exit
                 start_states=None,            # path to start_states.json (curriculum), or None
                 p_bottom=0.0,                 # prob of resetting to the bottom anyway
                 curriculum=False,             # enable the reverse (easy->hard) curriculum
                 cur_window=30,                # episodes per success-rate evaluation
                 cur_advance_rate=0.6,         # success rate that unlocks the next stage
                 cur_target_p_bottom=0.6,      # bottom-start fraction once fully unlocked
                 terminate_on_fall_below_start=False,  # end the attempt if the king drops
                                                       # below the level it started on
                 auto_frontier=False,        # bank grounded high-level states as new starts
                 frontier_min_level=1,       # only bank states on this level or above
                 frontier_per_level_cap=20,  # max banked spots kept per level
                 frontier_grid=12,           # dedup granularity in pixels
                 seed=None):

        self.charges = tuple(charges)
        self.walk_frames = int(walk_frames)
        self.max_steps = int(max_steps)
        self.max_settle_frames = int(max_settle_frames)
        self.goal_level = goal_level          # if set: terminate+reward on reaching it
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
        self._cp_ranked = sorted(
            self._start_pool,
            key=lambda s: (self._state_field(s, "level", "current_level", default=0) * 100000
                           - self._state_field(s, "y", "rect_y", default=0)),
            reverse=True)                      # easiest (closest to goal) first
        self._unlocked = 1 if self._cp_ranked else 0   # how many easy checkpoints are live
        self._cur_p_bottom = 0.0 if self.curriculum else self.p_bottom
        self._succ = []                        # recent episode successes (1/0)
        self._episode_is_curric = False        # did THIS episode start from a checkpoint?

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
        self.n_scalars = 4
        self._grid_flat = 3 * self.grid_h * self.grid_w
        self.obs_dim = self._grid_flat + self.n_scalars
        self.num_actions = len(self.actions)

        self._steps = 0
        self._prev_level = 0
        self._episode_start_level = 0
        self._best_alt = 0.0

    # ------------------------------------------------------------------ setup
    def _build_action_table(self):
        """actions[i] = (kind, direction, magnitude)."""
        self.actions = []
        for d in ("left", "up", "right"):
            for c in self.charges:
                self.actions.append(("jump", d, c))
        self.actions.append(("walk", "left", self.walk_frames))
        self.actions.append(("walk", "right", self.walk_frames))

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
        Curriculum mode: uniform over only the currently-unlocked EASIEST
        checkpoints, bottom with the (ramping) curriculum bottom fraction."""
        if not self._start_pool:
            return None
        if self.curriculum:
            if self.rng.random() < self._cur_p_bottom:
                return None
            k = max(1, self._unlocked)
            idx = int(self.rng.integers(k))
            return self._cp_ranked[idx]
        # plain mode
        if self.rng.random() < self.p_bottom:
            return None
        idx = int(self.rng.integers(len(self._start_pool)))
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
        return {"unlocked": self._unlocked,
                "total": len(self._cp_ranked),
                "p_bottom": round(self._cur_p_bottom, 2),
                "succ_rate": round(rate, 2)}

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
        self.levels.update_wind(self.king)             # physics-affecting
        self.king.update(agentCommand=None)            # full physics + level change
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

        if kind == "jump":
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
            hold_left = (direction == "left")
            hold_right = (direction == "right")
            for _ in range(magnitude):
                self._set_keys(space=True, left=hold_left, right=hold_right)
                frame()
                # Stop charging the instant the king has launched (auto-fire)
                # or the level is ending; extra charge frames would re-crouch.
                if self.levels.ending or self.king.isJump or self.king.isFalling:
                    break
            # Release: drop SPACE, keep the direction held one frame so a
            # short-charge crouch fires _jump(direction). (No-op if already
            # airborne from an auto-fire.)
            self._set_keys(left=hold_left, right=hold_right)
            frame()
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
        scalars = np.array([
            self.king.rect_x / self.screen_w,
            self.king.rect_y / self.screen_h,
            self.levels.current_level / self.max_level,
            self._altitude() / ((self.max_level + 1) * self.screen_h),
        ], dtype=np.float32)
        return np.concatenate([grid.ravel(), scalars]).astype(np.float32)

    # ------------------------------------------------------------------- gym
    def reset(self, level=None, rect_x=None, rect_y=None):
        """Reset to a curriculum checkpoint (default) or to an explicit start.

        Calling reset() with no args enables the curriculum: with probability
        (1 - p_bottom) we drop the King at a random captured checkpoint from
        start_states.json; otherwise (or if the pool is empty) we reset to the
        bottom. Passing any of level/rect_x/rect_y explicitly bypasses the
        curriculum and forces that exact start (back-compatible with old code).

        For level==0 the King's own reset() gives a valid grounded start.
        For level>0 you MUST have a known-grounded (rect_x, rect_y) -- that's
        exactly what the captured checkpoints provide."""
        self.king.reset()
        self.levels.reset()

        os.environ["start"] = "1"
        os.environ["gaming"] = "1"
        os.environ["pause"] = ""
        os.environ["active"] = "1"
        os.environ["mode"] = "normal"

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

        self.levels.current_level = int(level)
        if rect_x is not None:
            self.king.rect_x = int(rect_x)
        if rect_y is not None:
            self.king.rect_y = int(rect_y)

        self._set_keys()
        self._steps = 0
        self._prev_level = self.levels.current_level
        self._episode_start_level = self.levels.current_level   # anchor for the fall rule
        self._best_alt = self._altitude()            # best height reached this episode
        return self._obs(), {}

    def step(self, action_idx, render_cb=None):
        self._apply_action(int(action_idx), render_cb=render_cb)
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
        if self.levels.ending:                       # reached the babe at the top
            terminated = True
            reached_goal = True
            reward += 100.0
        elif self.goal_level is not None and level >= self.goal_level:
            terminated = True
            reached_goal = True
            reward += self.level_reward                # same scale as a normal pass

        # Fall rule: dropped below the level this attempt started on -> the
        # attempt is over. End it as a FAILURE (reached_goal stays False) so the
        # next reset re-anchors on the same level instead of forcing a full
        # re-climb. The negative level_penalty applied above is the cost of it.
        if (self.terminate_on_fall_below_start
                and not terminated
                and level < self._episode_start_level):
            terminated = True

        truncated = self._steps >= self.max_steps

        if terminated or truncated:
            # success = reached the GOAL, not merely 'episode ended'. A fall
            # below the start level terminates the episode but is NOT a success.
            self._record_episode(success=reached_goal)

        banked = self._maybe_bank_frontier()   # grounded-gated; None unless a new spot

        info = {"level": self.levels.current_level,
                "altitude": alt,
                "x": self.king.rect_x,
                "y": self.king.rect_y,
                "success": reached_goal,
                "from_checkpoint": self._episode_is_curric,
                "frontier": banked,
                "curriculum": self.curriculum_status()}
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