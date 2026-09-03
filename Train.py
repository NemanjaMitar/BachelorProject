#!/usr/bin/env python
"""
Train a PPO agent on headless Jump King.

Run from the game root (folder containing King.py and the asset folders).

Validate the environment first (single in-process env, no subprocesses):
    python train.py --smoke

Then train:
    python train.py --num-envs 8 --rollout 256 --goal-level 1

Start small: --goal-level 1 trains the agent just to clear level 0 (reach
level 1). Only widen the goal once that works -- that's your correctness check
that the whole stack (env macro-step, reward sign, PPO math) is sound.

Warm-start onto a harder goal (the "staircase"): once level 0 is solved, load
that checkpoint and bump the goal. Use --reset-opt when switching tasks:
    python train.py --num-envs 8 --rollout 64 --goal-level 2 \
        --start-states starts/start_states.json --p-bottom 0.5 \
        --resume checkpoints/ppo_XXXX.pt --reset-opt
"""

import os
import time
import json

def _merge_frontier_to_json(path, states, grid=12):
    """Merge auto-captured frontier states into starts/start_states.json, in the same
    {level_str: [[x, y], ...]} format capture.py writes. ONLY the main training
    process calls this (workers just report states via info), so there is no
    multi-writer race. Dedups by a coarse pixel grid and never drops existing
    points -- this augments the manual pool, it does not replace it."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}
    seen = set()
    for lvl_str, pts in data.items():
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                seen.add((str(lvl_str), int(pt[0]) // grid, int(pt[1]) // grid))
    added = 0
    for st in states:
        lvl = str(int(st['level'])); x = int(st['x']); y = int(st['y'])
        key = (lvl, x // grid, y // grid)
        if key in seen:
            continue
        data.setdefault(lvl, []).append([x, y])
        seen.add(key); added += 1
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)        # atomic write so a crash can't corrupt the file
    return added


import argparse
import numpy as np
import torch

from PPO import PPO, RolloutBuffer, get_device
from Occupancy import resolve_channels


def _parse_charges(s):
    return tuple(int(c) for c in s.split(",")) if s else ()


def _parse_approach(s):
    """'220:0:left:32' -> (220,0,'left',32); comma-separate multiple."""
    out = []
    for item in (s.split(",") if s else []):
        tx, b, d, c = item.split(":")
        out.append((int(tx), int(b), d, int(c)))
    return tuple(out)


def _parse_combo(s):
    """'0:up:12;0:left:32;4:up:32' -> one route of (bucket,dir,charge) steps;
    separate multiple routes with '|'."""
    routes = []
    for route in (s.split("|") if s else []):
        steps = []
        for st in route.split(";"):
            b, d, c = st.split(":")
            steps.append((int(b), d, int(c)))
        routes.append(tuple(steps))
    return tuple(routes)


def _expand_policy_head(state, net):
    """Allow resuming a checkpoint with FEWER actions than the current net
    (extra actions are appended to the table, so old indices keep their
    meaning): copy the old policy-head rows, leave the new rows at their
    fresh init so the new actions start near-uniform."""
    w_old = state.get("policy_head.weight")
    b_old = state.get("policy_head.bias")
    cur = net.state_dict()
    w_new = cur["policy_head.weight"]
    if (w_old is not None and w_old.shape[0] < w_new.shape[0]
            and w_old.shape[1] == w_new.shape[1]):
        w = w_new.clone(); w[:w_old.shape[0]] = w_old
        b = cur["policy_head.bias"].clone(); b[:b_old.shape[0]] = b_old
        state = dict(state)
        state["policy_head.weight"] = w
        state["policy_head.bias"] = b
        print(f"resume: expanded policy head {w_old.shape[0]} -> "
              f"{w_new.shape[0]} actions (new actions start at init)")
        return state, True
    return state, False


def maybe_resume(agent, args, device):
    """Warm-start from a saved checkpoint. Returns the step to continue from.

    Loads the policy/value weights always. By default also restores the Adam
    optimizer state so a continued run resumes cleanly; pass --reset-opt when
    warm-starting onto a NEW task (e.g. bumping --goal-level) where fresh
    optimizer moments usually train better. The step counter is carried forward
    so new checkpoint files don't overwrite the old ones."""
    if not args.resume:
        return 0
    ckpt = torch.load(args.resume, map_location=device)
    model_state, expanded = _expand_policy_head(ckpt["model"], agent.net)
    agent.net.load_state_dict(model_state)
    if expanded:
        # Adam's load_state_dict does NOT validate tensor shapes: stale
        # moments for the old head "load" fine and crash at opt.step().
        # An expanded head therefore always starts with a fresh optimizer.
        print("resume: head was expanded -> starting with a fresh optimizer")
    elif not args.reset_opt and "opt" in ckpt:
        try:
            agent.opt.load_state_dict(ckpt["opt"])
        except Exception as e:
            print("warning: could not load optimizer state, continuing fresh:", e)
    step = int(ckpt.get("step", 0))
    print(f"resumed from {args.resume} at step {step} "
          f"(opt {'reset' if args.reset_opt else 'restored'})")
    return step


def _goal_boxes(args):
    """Ordered (y, x_lo, x_hi) route waypoints from a RouteSearch result."""
    if not args.route_waypoints:
        return ()
    with open(args.route_waypoints, encoding="utf-8") as f:
        d = json.load(f)
    return tuple(tuple(int(v) for v in w) for w in d["waypoints"])


def _goal_ys(args):
    """The staircase of subgoal heights for --goal-ys.

    "auto" derives it from the level's own platform tops (every distinct height
    a king can stand on, above the start), which is what the agent must climb
    anyway. A comma list overrides it."""
    if not args.goal_ys:
        return ()
    if args.goal_ys != "auto":
        return tuple(int(v) for v in args.goal_ys.split(","))
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from JK_Env import JumpKingEnv
    probe = JumpKingEnv(max_steps=10)
    if args.world:
        import CustomWorld
        CustomWorld.apply_world(probe, CustomWorld.load_world(args.world))
    lvl = args.goal_level - 1 if args.goal_level else 0
    kh = probe.king.rect_height
    ys = sorted({int(p.y) - kh for p in (probe.levels.levels[lvl].platforms or [])
                 if p.width >= 8 and p.height >= 4 and p.y - kh > 0},
                reverse=True)
    probe.close()
    below = [y for y in ys if y < args.goal_y_from] if args.goal_y_from else ys
    return tuple(below)


def apply_world(env, args, worker=0):
    """Swap the engine's level data for a custom / generated world.

    --world       one hand-built world, fixed for the whole run.
    --world-pool  a directory of PROVED procedurally generated worlds; a random
                  one is drawn on every reset, so the agent climbs a screen it
                  has (almost) never seen. This is the generalization test.
    --world-gen   the same, but generating on the fly and without the proof --
                  cheaper, and a few % of the screens are unsolvable.

    Must run AFTER the env exists; Levels.reset() does not undo it."""
    if getattr(args, "world_pool", None):
        import WorldGen
        WorldGen.attach_pool(env, args.world_pool,
                             seed=args.world_gen_seed + 1000 * worker,
                             curriculum=not args.world_fixed_curriculum,
                             cur_target=args.world_cur_target,
                             cur_window=args.world_cur_window,
                             mix_below=args.world_mix_below)
    elif getattr(args, "world_gen", False):
        import WorldGen
        WorldGen.attach_generator(env, seed=args.world_gen_seed + 1000 * worker,
                                  curriculum=not args.world_fixed_curriculum,
                                  cur_target=args.world_cur_target,
                                  cur_window=args.world_cur_window,
                                  mix_below=args.world_mix_below)
    elif getattr(args, "world", None):
        import CustomWorld
        CustomWorld.apply_world(env, CustomWorld.load_world(args.world))
    return env


def env_factory(args, worker=0):
    """Returns a zero-arg callable that builds one env (picklable via cloudpickle)."""
    def _make():
        from JK_Env import JumpKingEnv
        env = JumpKingEnv(
            max_steps=args.max_steps,
            goal_level=args.goal_level,
            transition_fail_y=args.transition_fail_y,
            level_reward=args.level_reward,
            level_penalty=args.level_penalty,
            level_penalty_cap=args.level_penalty_cap,
            altitude_breadcrumb=args.altitude_breadcrumb,
            step_cost=args.step_cost,
            start_states=args.start_states,   # curriculum checkpoint pool
            p_bottom=args.p_bottom,           # fraction of resets that start at the bottom
            curriculum=args.curriculum,       # reverse easy->hard curriculum
            cur_advance_rate=args.cur_advance_rate,
            cur_window=args.cur_window,
            cur_target_p_bottom=args.cur_target_p_bottom,
            terminate_on_fall_below_start=args.terminate_on_fall,
            auto_frontier=args.auto_frontier,
            frontier_min_level=args.frontier_min_level,
            fine_walk_frames=args.fine_walk_frames,
            extra_charges=_parse_charges(args.extra_charges),
            wait_frames=_parse_charges(args.wait_frames),
            wind_obs=args.wind_obs, vel_obs=args.vel_obs,
            grid_channels=args.grid_channels,
            wind_jump=_parse_charges(args.wind_jump),
            approach_jump=_parse_approach(args.approach_jump),
            wind_combo=_parse_combo(args.wind_combo),
            quarantine_after=args.quarantine_after,
            start_vel_jitter=args.start_vel_jitter,
            goal_ys=_goal_ys(args),
            goal_boxes=_goal_boxes(args),
            goal_y_reward=args.goal_y_reward,
            settle_action=args.settle_action,
            goal_x_min=args.goal_x_min,
            goal_x_max=args.goal_x_max,
        )
        return apply_world(env, args, worker)
    return _make


def smoke(args):
    """Single env, in-process. Confirms the env loads assets, the macro-step
    runs, rewards have the right sign (up => positive), and PPO can do one
    update. If this passes, scaling to subprocesses is the easy part."""
    from JK_Env import JumpKingEnv
    device = get_device()
    print("device:", device)

    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      transition_fail_y=args.transition_fail_y,
                      level_reward=args.level_reward, level_penalty=args.level_penalty,
                      level_penalty_cap=args.level_penalty_cap,
                      altitude_breadcrumb=args.altitude_breadcrumb,
                      step_cost=args.step_cost,
                      start_states=args.start_states, p_bottom=args.p_bottom,
                      fine_walk_frames=args.fine_walk_frames,
                      extra_charges=_parse_charges(args.extra_charges),
                      wait_frames=_parse_charges(args.wait_frames),
                      wind_obs=args.wind_obs, vel_obs=args.vel_obs,
                      grid_channels=args.grid_channels,
                      wind_jump=_parse_charges(args.wind_jump),
                      approach_jump=_parse_approach(args.approach_jump),
                      wind_combo=_parse_combo(args.wind_combo),
                      quarantine_after=args.quarantine_after,
                      start_vel_jitter=args.start_vel_jitter,
                      goal_ys=_goal_ys(args),
                      goal_boxes=_goal_boxes(args),
                      goal_y_reward=args.goal_y_reward,
                      settle_action=args.settle_action,
                      goal_x_min=args.goal_x_min,
                      goal_x_max=args.goal_x_max)
    apply_world(env, args)
    agent = PPO(env.obs_dim, env.num_actions, device=device,
                grid_shape=env.grid_shape, n_scalars=env.n_scalars,
                extra_conv=args.extra_conv, scalar_embed=args.scalar_embed)
    maybe_resume(agent, args, device)
    print(f"obs_dim={env.obs_dim} num_actions={env.num_actions} "
          f"params={sum(p.numel() for p in agent.net.parameters())} "
          f"| checkpoints={len(env._start_pool)} p_bottom={env.p_bottom}")

    obs, _ = env.reset()
    T = 64
    buf = RolloutBuffer(T, 1, env.obs_dim, device)
    for t in range(T):
        ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
        a, lp, v = agent.net.act(ob)
        nobs, r, term, trunc, info = env.step(a.item())
        buf.add(ob.squeeze(0), a.squeeze(0), lp.squeeze(0),
                torch.tensor(r, device=device),
                v.squeeze(0),
                torch.tensor(float(term), device=device),
                torch.tensor(float(trunc), device=device))
        print(f"t={t:2d} act={a.item():2d} {str(env.actions[a.item()]):24s} "
              f"r={r:+.3f} lvl={info['level']} alt={info['altitude']:.0f}")
        obs = nobs
        if term or trunc:
            obs, _ = env.reset()
    with torch.no_grad():
        last_v = agent.net.forward(
            torch.as_tensor(obs, device=device).float().unsqueeze(0))[1]
    logs = agent.update(buf, last_v)
    print("one PPO update ok:", {k: round(v, 4) for k, v in logs.items()})
    env.close()
    print("SMOKE TEST PASSED")


def train(args):
    from VecEnv import SubprocVecEnv
    device = get_device()
    print("device:", device, "| num_envs:", args.num_envs)

    envs = SubprocVecEnv([env_factory(args, i) for i in range(args.num_envs)])
    obs_dim, num_actions = envs.obs_dim, envs.num_actions
    agent = PPO(obs_dim, num_actions, device=device,
                lr=args.lr, gamma=args.gamma, lam=args.lam,
                clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
                ent_coef=args.ent_coef,
                grid_shape=envs.grid_shape, n_scalars=envs.n_scalars,
                extra_conv=args.extra_conv, scalar_embed=args.scalar_embed)

    os.makedirs(args.save_dir, exist_ok=True)
    obs = envs.reset()
    global_step = maybe_resume(agent, args, device)
    run_start_step = global_step   # sps must count THIS run's steps only
    recent_returns, recent_levels, recent_success = [], [], []
    cur_stat = None
    frontier_buffer = []
    start = time.time()

    n_updates = args.total_steps // (args.rollout * args.num_envs)
    for update in range(1, n_updates + 1):
        buf = RolloutBuffer(args.rollout, args.num_envs, obs_dim, device)

        for _ in range(args.rollout):
            ob = torch.as_tensor(obs, device=device).float()
            with torch.no_grad():
                action, logprob, value = agent.net.act(ob)
            a_np = action.cpu().numpy()
            nobs, rew, term, trunc, infos = envs.step(a_np)

            buf.add(ob, action, logprob,
                    torch.as_tensor(rew, device=device),
                    value,
                    torch.as_tensor(term, device=device),
                    torch.as_tensor(trunc, device=device))

            for info in infos:
                if "episode" in info:
                    recent_returns.append(info["episode"]["r"])
                    recent_levels.append(info["episode"]["level"])
                    recent_success.append(1.0 if info.get("success") else 0.0)
                    if info.get("curriculum"):
                        cur_stat = info["curriculum"]
                if info.get("frontier"):
                    frontier_buffer.append(info["frontier"])

            obs = nobs
            global_step += args.num_envs

        with torch.no_grad():
            last_value = agent.net.forward(
                torch.as_tensor(obs, device=device).float())[1]
        logs = agent.update(buf, last_value)

        # ADAPTIVE ENTROPY FLOOR. A curriculum rung that is solved by ONE action
        # drives the policy deterministic (entropy -> 0); the next rung then
        # needs a two-action discovery that a collapsed policy can never sample,
        # so the ladder stalls with kl ~ 0. Push the entropy coefficient up
        # whenever measured entropy falls under the floor, and relax it back
        # once exploration recovers.
        if args.ent_floor > 0.0:
            if logs["entropy"] < args.ent_floor:
                agent.ent_coef = min(agent.ent_coef * 1.4, args.ent_coef_max)
            elif logs["entropy"] > args.ent_floor * 1.6:
                agent.ent_coef = max(agent.ent_coef / 1.2, args.ent_coef)

        if args.auto_frontier and frontier_buffer and update % args.frontier_dump_every == 0:
            added = _merge_frontier_to_json(args.frontier_out, frontier_buffer, grid=12)
            print(f"  frontier: +{added} new start spots -> {args.frontier_out} "
                  f"({len(frontier_buffer)} reported this window)")
            frontier_buffer = []

        if update % args.log_every == 0:
            sps = int((global_step - run_start_step) / (time.time() - start))
            mr = np.mean(recent_returns[-100:]) if recent_returns else float("nan")
            ml = np.mean(recent_levels[-100:]) if recent_levels else float("nan")
            mx = max(recent_levels[-100:]) if recent_levels else -1
            sr = np.mean(recent_success[-100:]) if recent_success else float("nan")
            cur = ""
            if cur_stat:
                cur = (f" | cur {cur_stat['unlocked']}/{cur_stat['total']} "
                       f"pb {cur_stat['p_bottom']}")
                if cur_stat.get("focus"):
                    fx, fy, fw = cur_stat["focus"]
                    cur += f" | focus ({fx},{fy}) f={fw}"
            print(f"upd {update:4d} | step {global_step:8d} | {sps:5d} sps | "
                  f"ret {mr:8.2f} | succ {sr:4.2f} | mean_lvl {ml:4.1f} | "
                  f"max_lvl {mx:2d}{cur} | "
                  f"kl {logs['approx_kl']:+.4f} | ent {logs['entropy']:.3f}")

        ckpt_payload = {"model": agent.net.state_dict(),
                        "opt": agent.opt.state_dict(),
                        "step": global_step,
                        # lets Play/Handoff auto-configure the env action set
                        "action_cfg": {
                            "fine_walk_frames": args.fine_walk_frames,
                            "extra_charges": list(_parse_charges(args.extra_charges)),
                            "wait_frames": list(_parse_charges(args.wait_frames)),
                            "wind_jump": list(_parse_charges(args.wind_jump)),
                            "approach_jump": [list(a) for a in _parse_approach(args.approach_jump)],
                            "wind_combo": [[list(s) for s in r] for r in _parse_combo(args.wind_combo)],
                            "settle_action": args.settle_action,
                            "wind_obs": args.wind_obs, "vel_obs": args.vel_obs,
                            "grid_channels": list(resolve_channels(args.grid_channels)),
                            "extra_conv": args.extra_conv,
                            "scalar_embed": args.scalar_embed}}

        # Early stop for the automated pipeline: curriculum fully unlocked and
        # recent success high enough -> the level is solved, don't burn budget.
        if (args.stop_at_succ is not None and cur_stat
                and cur_stat.get("unlocked") == cur_stat.get("total")
                and len(recent_success) >= 100
                and float(np.mean(recent_success[-100:])) >= args.stop_at_succ):
            path = os.path.join(args.save_dir, f"ppo_{global_step}.pt")
            torch.save(ckpt_payload, path)
            print(f"EARLY STOP at update {update}: succ "
                  f"{np.mean(recent_success[-100:]):.2f} >= {args.stop_at_succ} "
                  f"with curriculum fully unlocked; saved {path}")
            break

        if update % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ppo_{global_step}.pt")
            torch.save(ckpt_payload, path)
            print("saved", path)

    envs.close()


def train_visible(args):
    """Single in-process env in a REAL window, so you can WATCH training.

    This is for diagnosis, not speed: it runs ONE env (N=1), samples actions
    stochastically (so you see exploration), and renders the settled frame
    after every macro-action. If the policy has collapsed you will literally
    see the king repeat the same jump; healthy training shows varied actions
    and the king working its way upward. Subprocess vec-envs can't render (each
    has its own SDL), which is why visible mode is single-env."""
    # Must override the headless driver BEFORE JK_Env imports pygame.
    os.environ["SDL_VIDEODRIVER"] = ""
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    from JK_Env import JumpKingEnv

    device = get_device()
    print("device:", device, "| VISIBLE single-env training (N=1)")
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      transition_fail_y=args.transition_fail_y,
                      level_reward=args.level_reward, level_penalty=args.level_penalty,
                      level_penalty_cap=args.level_penalty_cap,
                      altitude_breadcrumb=args.altitude_breadcrumb,
                      step_cost=args.step_cost,
                      start_states=args.start_states, p_bottom=args.p_bottom,
                      curriculum=args.curriculum,
                      cur_advance_rate=args.cur_advance_rate,
                      cur_window=args.cur_window,
                      cur_target_p_bottom=args.cur_target_p_bottom,
                      terminate_on_fall_below_start=args.terminate_on_fall,
                      auto_frontier=args.auto_frontier,
                      frontier_min_level=args.frontier_min_level,
                      fine_walk_frames=args.fine_walk_frames,
                      extra_charges=_parse_charges(args.extra_charges),
                      wait_frames=_parse_charges(args.wait_frames),
                      wind_obs=args.wind_obs, vel_obs=args.vel_obs,
                      grid_channels=args.grid_channels,
                      wind_jump=_parse_charges(args.wind_jump),
                      approach_jump=_parse_approach(args.approach_jump),
                      wind_combo=_parse_combo(args.wind_combo),
                      quarantine_after=args.quarantine_after,
                      start_vel_jitter=args.start_vel_jitter,
                      goal_ys=_goal_ys(args),
                      goal_boxes=_goal_boxes(args),
                      goal_y_reward=args.goal_y_reward,
                      settle_action=args.settle_action,
                      goal_x_min=args.goal_x_min,
                      goal_x_max=args.goal_x_max)
    apply_world(env, args)
    agent = PPO(env.obs_dim, env.num_actions, device=device,
                lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip,
                epochs=args.epochs, minibatches=args.minibatches, ent_coef=args.ent_coef,
                grid_shape=env.grid_shape, n_scalars=env.n_scalars,
                extra_conv=args.extra_conv, scalar_embed=args.scalar_embed)
    global_step = maybe_resume(agent, args, device)
    print(f"checkpoints={len(env._start_pool)} curriculum={env.curriculum}")

    SCALE = 2
    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King - VISIBLE training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    def draw(update, a, r, info):
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1(); env.king.blitme(); env.babe.blitme(); env.levels.blit2()
        except Exception:
            pass
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))
        cs = info.get("curriculum", {})
        l1 = font.render(f"upd {update}  act {a} {env.actions[a]}  r={r:+.2f}",
                         True, (255, 255, 0))
        l2 = font.render(f"lvl={info['level']} y={int(info['y'])} "
                         f"cur={cs.get('unlocked','-')}/{cs.get('total','-')} "
                         f"pb={cs.get('p_bottom','-')} succ={cs.get('succ_rate','-')}",
                         True, (120, 255, 120))
        window.blit(l1, (6, 6)); window.blit(l2, (6, 28))
        pygame.display.flip()
        clock.tick(args.fps)

    os.makedirs(args.save_dir, exist_ok=True)
    obs, _ = env.reset()
    n_updates = args.total_steps // args.rollout
    for update in range(1, n_updates + 1):
        buf = RolloutBuffer(args.rollout, 1, env.obs_dim, device)
        for t in range(args.rollout):
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN
                                             and e.key == pygame.K_ESCAPE):
                    env.close(); return
            ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                a, lp, v = agent.net.act(ob)
            nobs, r, term, trunc, info = env.step(a.item())
            buf.add(ob.squeeze(0), a.squeeze(0), lp.squeeze(0),
                    torch.tensor(r, device=device), v.squeeze(0),
                    torch.tensor(float(term), device=device),
                    torch.tensor(float(trunc), device=device))
            draw(update, a.item(), r, info)
            obs = nobs
            global_step += 1
            if term or trunc:
                obs, _ = env.reset()
        with torch.no_grad():
            last_v = agent.net.forward(
                torch.as_tensor(obs, device=device).float().unsqueeze(0))[1]
        logs = agent.update(buf, last_v)
        print(f"upd {update:4d} | step {global_step:7d} | ent {logs['entropy']:.3f} "
              f"| kl {logs['approx_kl']:+.4f}")
        if update % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ppo_visible_{global_step}.pt")
            torch.save({"model": agent.net.state_dict(),
                        "opt": agent.opt.state_dict(), "step": global_step,
                        "action_cfg": {
                            "fine_walk_frames": args.fine_walk_frames,
                            "extra_charges": list(_parse_charges(args.extra_charges)),
                            "wait_frames": list(_parse_charges(args.wait_frames)),
                            "wind_jump": list(_parse_charges(args.wind_jump)),
                            "approach_jump": [list(a) for a in _parse_approach(args.approach_jump)],
                            "wind_combo": [[list(s) for s in r] for r in _parse_combo(args.wind_combo)],
                            "settle_action": args.settle_action,
                            "wind_obs": args.wind_obs, "vel_obs": args.vel_obs,
                            "grid_channels": list(resolve_channels(args.grid_channels)),
                            "extra_conv": args.extra_conv,
                            "scalar_embed": args.scalar_embed}},
                       path)
            print("saved", path)
    env.close()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="single-env sanity check")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--max-steps", type=int, default=600, help="env step limit per episode")
    p.add_argument("--world", type=str, default=None,
                   help="train on a custom world built with LevelEditor.py "
                        "(levels/<name>.json) instead of the original game's "
                        "screens. --goal-level then defaults to its top screen.")
    p.add_argument("--world-pool", type=str, default=None,
                   help="directory of PROVED generated worlds (WorldGen.py "
                        "--out): a random one is drawn every episode, so the "
                        "policy is trained on screens instead of on ONE screen. "
                        "Evaluate on a pool built with a disjoint seed.")
    p.add_argument("--world-gen", action="store_true",
                   help="generate an unverified random world every episode")
    p.add_argument("--world-gen-seed", type=int, default=0)
    p.add_argument("--world-fixed-curriculum", action="store_true",
                   help="use the old SCHEDULED start curriculum instead of the "
                        "adaptive one (kept for the ablation: scheduled was "
                        "measured to stall at 1/120 bottom starts)")
    p.add_argument("--world-cur-target", type=float, default=0.65,
                   help="success rate a stage must hold before the generated "
                        "reverse curriculum drops one rung lower")
    p.add_argument("--world-mix-below", type=float, default=0.3,
                   help="fraction of episodes drawn from an EASIER already "
                        "cleared curriculum stage, so the skill is not "
                        "forgotten while the deepest stage is learnt")
    p.add_argument("--world-cur-window", type=int, default=150,
                   help="episodes per env the stage success is measured over")
    p.add_argument("--world-anneal-episodes", type=int, default=0,
                   help="anneal the generated reverse curriculum over this "
                        "many episodes PER ENV: start on the top rung only, "
                        "widen downward, end on bottom-only starts. 0 = the "
                        "fixed --world-start-rung-p mix.")
    p.add_argument("--world-start-rung-p", type=float, default=0.5,
                   help="fraction of --world-pool episodes that start on a "
                        "random rung instead of the floor (the generated "
                        "screen's own reverse curriculum). Evaluation always "
                        "starts at the bottom.")
    p.add_argument("--goal-level", type=int, default=None,
                   help="terminate+reward on reaching this level (curriculum)")
    p.add_argument("--transition-fail-y", type=int, default=None,
                   help="treat a level transition into the goal level with rect_y >= this value as a failed episode")
    p.add_argument("--level-reward", type=float, default=10.0)
    p.add_argument("--level-penalty", type=float, default=10.0,
                   help="penalty per level fallen. MUST equal --level-reward "
                        "whenever episodes can span multiple levels (merged "
                        "models), or climb-fall-climb reward farming collapses "
                        "the run; harmless for per-level models (falls "
                        "terminate anyway)")
    p.add_argument("--level-penalty-cap", type=int, default=None,
                   help="charge at most N levels per fall (e.g. 1). Without it "
                        "the cost of failing grows with how far the king "
                        "tumbles, so climbing raises the downside and the agent "
                        "learns to fail early instead of going up.")
    p.add_argument("--altitude-breadcrumb", type=float, default=0.0)
    p.add_argument("--step-cost", type=float, default=0.0,
                   help="charge this for every action. Needed once falling is "
                        "capped: otherwise standing still costs nothing and the "
                        "risk-averse optimum is to idle until truncation.")
    p.add_argument("--start-states", type=str, default=None,
                   help="path to starts/start_states.json; enables the checkpoint curriculum")
    p.add_argument("--p-bottom", type=float, default=0.2,
                   help="fraction of episode resets that start at the bottom "
                        "instead of a checkpoint (only matters with --start-states)")
    p.add_argument("--curriculum", action="store_true",
                   help="reverse curriculum: start only from the easiest (near-exit) "
                        "checkpoint and unlock harder/lower ones as success rises")
    p.add_argument("--cur-advance-rate", type=float, default=0.6,
                   help="recent success rate that unlocks the next curriculum stage")
    p.add_argument("--cur-window", type=int, default=30,
                   help="episodes per curriculum success-rate window (PER WORKER; "
                        "smaller = faster but noisier unlocking)")
    p.add_argument("--auto-frontier", action="store_true",
                   help="bank grounded high-level states the agent reaches as new "
                        "start states (augments the manual pool; no manual capture needed)")
    p.add_argument("--frontier-min-level", type=int, default=1,
                   help="only auto-bank states on this level or above (set to the level "
                        "you are currently pushing from to avoid re-banking mastered ones)")
    p.add_argument("--frontier-out", type=str, default="starts/start_states.json",
                   help="file the auto-captured start states are merged into")
    p.add_argument("--frontier-dump-every", type=int, default=25,
                   help="write newly banked frontier states to disk every N updates")
    p.add_argument("--cur-target-p-bottom", type=float, default=0.6,
                   help="bottom-start fraction reached once all checkpoints are mastered")
    p.add_argument("--terminate-on-fall", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="end the episode (as a failure) if the king falls below the "
                        "level it started this attempt on (default ON for per-level "
                        "models; pass --no-terminate-on-fall for full-climb runs)")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint .pt to warm-start from")
    p.add_argument("--reset-opt", action="store_true",
                   help="when resuming, start with a fresh optimizer "
                        "(recommended when switching tasks, e.g. a new --goal-level)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--fine-walk-frames", type=int, default=0,
                   help="if > 0, add a second micro-walk action pair of this many "
                        "frames (~1.4 px/frame; try 3) so the agent can position "
                        "into launch windows narrower than the 14 px normal walk. "
                        "CHANGES the action count: incompatible with old checkpoints, "
                        "train a fresh model when enabling")
    p.add_argument("--goal-x-min", type=int, default=None,
                   help="delivery region: success on the goal level only "
                        "counts at x >= this (steer where a model hands off)")
    p.add_argument("--goal-x-max", type=int, default=None,
                   help="delivery region: success only at x <= this")
    p.add_argument("--wait-frames", type=str, default="",
                   help="comma list of 'stand still N frames' actions appended "
                        "to the table (wind levels; e.g. 30,60)")
    p.add_argument("--settle-action", action="store_true",
                   help="add a standalone 'settle' action (damps momentum to "
                        "rest). On ice this recovers the whole penalty that "
                        "sliding arrivals cause. Appended LAST, so old "
                        "checkpoints keep their action indices.")
    p.add_argument("--start-vel-jitter", type=float, default=0.0,
                   help="randomise the momentum curriculum starts begin with "
                        "(observation units, e.g. 0.3). Teleports place the king "
                        "at REST, but the relay hands him a SLIDING king -- on "
                        "ice that is a different state. 0 = old behaviour.")
    p.add_argument("--quarantine-after", type=int, default=150,
                   help="consecutive failures before a solvable-looking start "
                        "state gets benched; raise (e.g. 600) when a state "
                        "needs a rare multi-action discovery")
    p.add_argument("--wind-jump", type=str, default="",
                   help="comma list of wind buckets 0..4 to add atomic "
                        "'wait for bucket B then jump' combos (wind crossings), "
                        "e.g. 0,4")
    p.add_argument("--approach-jump", type=str, default="",
                   help="atomic 'walk to target_x, then wind-timed jump' macros "
                        "for crossings whose launch window the conv grid can't "
                        "perceive. Format target_x:bucket:dir:charge, comma-"
                        "separated. e.g. 220:0:left:32")
    p.add_argument("--wind-combo", type=str, default="",
                   help="atomic multi-step wind route(s): steps 'bucket:dir:"
                        "charge' joined by ';', multiple routes by '|'. Encodes "
                        "a whole short windy level as one action. e.g. "
                        "0:up:12;0:left:32;4:up:32;4:up:32")
    p.add_argument("--wind-obs", action="store_true",
                   help="append wind-phase sin/cos to the observation and "
                        "randomize the phase at every reset (levels 25-31). "
                        "CHANGES obs size: needs a fresh model.")
    p.add_argument("--vel-obs", action="store_true",
                   help="append the king's (vx,vy) velocity to the observation "
                        "so momentum is observable (icy levels 36-38). "
                        "CHANGES obs size: needs a fresh/warm-started model.")
    p.add_argument("--extra-conv", action="store_true",
                   help="one more stride-2 conv layer, shrinking the flattened "
                        "grid feature 1536 -> 384 so the scalars are not "
                        "drowned by it. CHANGES the network: needs a fresh model.")
    p.add_argument("--scalar-embed", type=int, default=0,
                   help="project the observation scalars through a Linear(N) "
                        "before concatenating them to the grid feature (0 = "
                        "append them raw, the historical behaviour). With "
                        "--extra-conv this takes the scalars' share of the "
                        "trunk input from ~0.4%% to ~14%% -- what makes (vx,vy) "
                        "actually visible on ice. CHANGES the network.")
    p.add_argument("--grid-channels", default=None,
                   help="comma-separated occupancy planes the observation "
                        "carries, e.g. 'solid,king,hazard' (the default) or "
                        "'solid,hazard' for the fully-icy levels 36-38 where "
                        "hazard duplicates solid. CHANGES obs size: needs a "
                        "fresh model. The choice is stored in the checkpoint, "
                        "so relays keep serving mixed families.")
    p.add_argument("--stop-at-succ", type=float, default=None,
                   help="stop training early once the curriculum is fully "
                        "unlocked and recent success reaches this rate "
                        "(used by AutoPilot; e.g. 0.85)")
    p.add_argument("--extra-charges", type=str, default="",
                   help="comma list of extra jump charges APPENDED to the "
                        "action table (e.g. 22,24,28,30). Old checkpoints "
                        "warm-start via automatic policy-head expansion.")
    p.add_argument("--ent-coef", type=float, default=0.04)
    p.add_argument("--route-waypoints", type=str, default=None,
                   help="JSON from RouteSearch: ordered (y, x_lo, x_hi) boxes "
                        "along a VERIFIED route, each sized to its measured "
                        "basin. Paid in order, potential-based, and charged "
                        "back on a fall -- so a step that gains no height "
                        "(sliding out from under an overhang) still pays.")
    p.add_argument("--goal-ys", type=str, default=None,
                   help="WAYPOINTS: 'auto' (every platform height on the "
                        "level) or a comma list of king rect_y values. Each is "
                        "paid ONCE per episode the first time it is reached. "
                        "Height is not route order (routes go up and down), so "
                        "these are rewards, not staged goals -- the episode "
                        "still ends only at the level exit or on a fall.")
    p.add_argument("--goal-y-from", type=int, default=None,
                   help="ignore staircase steps at or below this y (i.e. keep "
                        "only platforms ABOVE the entry)")
    p.add_argument("--goal-y-reward", type=float, default=5.0,
                   help="paid once per newly reached waypoint height")
    p.add_argument("--ent-floor", type=float, default=0.0,
                   help="keep policy entropy above this by raising --ent-coef "
                        "adaptively (0 = off). A rung solved by a single action "
                        "collapses the policy to ~0 entropy, which kills the "
                        "exploration the NEXT rung needs. For a 37-38 action "
                        "table, 0.8-1.2 keeps a usable spread.")
    p.add_argument("--ent-coef-max", type=float, default=0.30,
                   help="ceiling for the adaptive entropy coefficient")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--render", action="store_true",
                   help="watch training in a real window (single env, slow, for diagnosis)")
    p.add_argument("--fps", type=int, default=60, help="render cap in --render mode")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.goal_level is None:
        if args.world_pool or args.world_gen:
            args.goal_level = 1          # generated worlds are 2 screens
        elif args.world:
            import CustomWorld
            args.goal_level = len(CustomWorld.load_world(args.world)["levels"]) - 1
            print(f"world {args.world}: goal-level defaults to its top screen "
                  f"({args.goal_level})")
        else:
            args.goal_level = 1
    if args.smoke:
        smoke(args)
    elif args.render:
        train_visible(args)
    else:
        train(args)