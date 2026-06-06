#!/usr/bin/env python
"""
Subprocess vectorized environment.

Why subprocesses (not threads / in-process loop)? pygame keeps the display,
the mixer, os.environ, and our pygame.key.get_pressed monkeypatch as
PROCESS-GLOBAL state. Two JumpKingEnv objects in one process would stomp on
each other. One process per env keeps them fully isolated.

Autoreset semantics (match Gymnasium): when an episode ends (terminated or
truncated) the worker immediately resets and returns the FIRST obs of the new
episode as the step obs; the true terminal obs and the episode stats are put in
info under 'terminal_obs' and 'episode'.
"""

import numpy as np
import multiprocessing as mp
import cloudpickle


class _Wrapper:
    """Lets us send an env-constructor (a closure) across the process boundary."""
    def __init__(self, fn):
        self.data = cloudpickle.dumps(fn)

    def __call__(self):
        return cloudpickle.loads(self.data)()


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    ep_ret, ep_len = 0.0, 0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                ep_ret += reward
                ep_len += 1
                if term or trunc:
                    info = dict(info)
                    info["terminal_obs"] = obs
                    info["episode"] = {"r": ep_ret, "l": ep_len,
                                       "level": info.get("level", -1)}
                    obs, _ = env.reset()
                    ep_ret, ep_len = 0.0, 0
                remote.send((obs, reward, term, trunc, info))
            elif cmd == "reset":
                obs, info = env.reset(**(data or {}))
                ep_ret, ep_len = 0.0, 0
                remote.send((obs, info))
            elif cmd == "spec":
                remote.send((env.obs_dim, env.num_actions,
                             getattr(env, "grid_shape", None),
                             getattr(env, "n_scalars", 0)))
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.n = len(env_fns)
        ctx = mp.get_context("spawn")          # 'spawn' is safest with pygame/SDL
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.procs = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, _Wrapper(fn)), daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()

        self.remotes[0].send(("spec", None))
        (self.obs_dim, self.num_actions,
         self.grid_shape, self.n_scalars) = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        outs = [r.recv() for r in self.remotes]
        obs = np.stack([o for o, _ in outs])
        return obs

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        outs = [r.recv() for r in self.remotes]
        obs = np.stack([o[0] for o in outs])
        rew = np.array([o[1] for o in outs], dtype=np.float32)
        term = np.array([o[2] for o in outs], dtype=np.float32)
        trunc = np.array([o[3] for o in outs], dtype=np.float32)
        infos = [o[4] for o in outs]
        return obs, rew, term, trunc, infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)