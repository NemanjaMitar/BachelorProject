"""
Procedural level generator + self-contained PPO  (thesis: does the ALGORITHM
generalize, or did it just memorize the 43 hand-made Jump King screens?)

Every episode is a BRAND-NEW randomly generated screen: a stack of ledges, each
placed within jump reach of the one below, bottom -> top. The agent never sees
the same level twice, so a rising success curve here is generalization, not
memorization. Physics mirror Jump King in spirit: a jump is (direction, charge)
-> a projectile arc under gravity; you land on the first ledge you fall onto,
or slide back to the ground if you miss. Pure numpy + torch, no pygame, headless.

Run:   python RandomLevel.py --updates 400 --log logs/randlevel.log
Chart: python ChartLogs.py randlevel     (writes charts/randlevel.png)
"""
import argparse, math, time, json, os
import numpy as np
import torch
import torch.nn as nn

W, H = 480.0, 360.0            # screen (same scale as the real game)
G = 0.55                       # gravity per tick
VX_MAX, VY_MAX = 4.6, 12.5     # max horizontal / vertical launch speed at full charge
N_PLAT = 5                     # ledges above the ground => 6 levels of progress
PLAT_W = (70.0, 110.0)         # ledge width range
DY = (48.0, 68.0)              # vertical gap between consecutive ledges
NEXT_K = 3                     # how many upcoming ledges the agent perceives

# action = (direction in {-1,0,+1}) x (charge in 5 buckets)  -> 15 jumps
CHARGES = [0.35, 0.55, 0.72, 0.86, 1.0]
DIRS = [-1, 0, 1]
ACTIONS = [(d, c) for d in DIRS for c in CHARGES]
N_ACT = len(ACTIONS)
OBS_DIM = 4 + NEXT_K * 3        # king(x,y,vy_room,cur_idx) + K*(rel_x,rel_dy,halfw)


def gen_level(rng, hard=False):
    """Stack solvable ledges bottom->top; ground is ledge 0 (full width).
    hard=True -> narrower ledges, bigger gaps and large horizontal offsets so the
    king MUST aim left/right (a straight-up jump no longer reaches the next ledge)."""
    pw  = (42.0, 68.0) if hard else PLAT_W
    dyr = (52.0, 74.0) if hard else DY
    fac = 0.34 if hard else 0.11                 # horizontal-offset budget
    plats = [(0.0, W, H - 12.0)]                 # ground: (xl, xr, y)
    x = W * 0.5
    for _ in range(N_PLAT):
        w = rng.uniform(*pw)
        dy = rng.uniform(*dyr)
        y = plats[-1][2] - dy
        max_dx = VX_MAX * 2.0 * (VY_MAX / G) * fac
        cx = np.clip(x + rng.uniform(-max_dx, max_dx), w * 0.5 + 6, W - w * 0.5 - 6)
        plats.append((cx - w * 0.5, cx + w * 0.5, y))
        x = cx
    return plats


class VecRandEnv:
    """Vectorized batch of independent random-level episodes."""
    def __init__(self, batch, seed=0, hard=False):
        self.B = batch
        self.hard = hard
        self.rng = np.random.default_rng(seed)
        self.reset_all()

    def reset_all(self):
        self.levels = [gen_level(self.rng, self.hard) for _ in range(self.B)]
        self.idx = np.zeros(self.B, dtype=np.int64)      # ledge the king stands on
        self.x = np.array([lv[0][0] * 0 + W * 0.5 for lv in self.levels])
        self.y = np.array([lv[0][2] for lv in self.levels])
        self.steps = np.zeros(self.B, dtype=np.int64)
        return self._obs()

    def _reset_one(self, i):
        self.levels[i] = gen_level(self.rng, self.hard)
        self.idx[i] = 0
        self.x[i] = W * 0.5
        self.y[i] = self.levels[i][0][2]
        self.steps[i] = 0

    def _obs(self):
        out = np.zeros((self.B, OBS_DIM), dtype=np.float32)
        for i in range(self.B):
            lv = self.levels[i]
            ci = self.idx[i]
            out[i, 0] = self.x[i] / W
            out[i, 1] = self.y[i] / H
            out[i, 2] = 1.0                              # charge headroom (const here)
            out[i, 3] = ci / (N_PLAT + 1)
            for k in range(NEXT_K):
                j = ci + 1 + k
                base = 4 + k * 3
                if j < len(lv):
                    xl, xr, yy = lv[j]
                    cx = 0.5 * (xl + xr)
                    out[i, base] = (cx - self.x[i]) / W
                    out[i, base + 1] = (self.y[i] - yy) / H
                    out[i, base + 2] = (xr - xl) * 0.5 / W
        return out

    def _simulate(self, i, d, charge):
        """Projectile from current spot; return landing (x, ledge_idx)."""
        lv = self.levels[i]
        x, y = float(self.x[i]), float(self.y[i])
        vx, vy = d * charge * VX_MAX, -charge * VY_MAX     # y grows downward
        start_idx = self.idx[i]
        px, py = x, y
        for _ in range(240):
            vy += G
            x += vx; y += vy
            if x < 0 or x > W:                              # hit a wall -> drop straight
                x = min(max(x, 0.0), W); vx = 0.0
            # descending: did we cross a ledge top between py and y?
            if vy > 0:
                for j, (xl, xr, yy) in enumerate(lv):
                    if py <= yy <= y and xl <= x <= xr:
                        return x, j
            px, py = x, y
            if y > H:                                       # fell off the bottom
                return x, 0
        return x, start_idx

    def step(self, actions):
        rew = np.zeros(self.B, dtype=np.float32)
        done = np.zeros(self.B, dtype=np.float32)
        for i in range(self.B):
            d, c = ACTIONS[int(actions[i])]
            prev = self.idx[i]
            nx, nidx = self._simulate(i, d, c)
            self.x[i] = nx; self.y[i] = self.levels[i][nidx][2]; self.idx[i] = nidx
            self.steps[i] += 1
            rew[i] = 1.5 * (nidx - prev) - 0.05            # progress up, small time cost
            if nidx < prev:
                rew[i] -= 0.5                              # extra penalty for falling
            if nidx >= N_PLAT:                             # reached the top ledge
                rew[i] += 12.0; done[i] = 1.0
            elif self.steps[i] >= 30:
                done[i] = 1.0
        obs_next = self._obs()
        succ = np.array([1.0 if self.idx[i] >= N_PLAT else 0.0 for i in range(self.B)],
                        dtype=np.float32)
        for i in range(self.B):
            if done[i]:
                self._reset_one(i)
        return obs_next, rew, done, succ


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 96), nn.Tanh(),
                                  nn.Linear(96, 96), nn.Tanh())
        self.pi = nn.Linear(96, n_act)
        self.v = nn.Linear(96, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def train(args):
    torch.manual_seed(args.seed)
    dev = "cpu"
    env = VecRandEnv(args.batch, seed=args.seed, hard=getattr(args, "hard", False))
    net = ActorCritic(OBS_DIM, N_ACT).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    obs = torch.tensor(env.reset_all(), device=dev)
    T = args.rollout
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    logf = open(args.log, "w")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    log(f"# RandomLevel PPO  batch={args.batch} rollout={T} updates={args.updates} "
        f"obs_dim={OBS_DIM} n_act={N_ACT}")
    succ_hist = []
    for upd in range(1, args.updates + 1):
        mb_obs, mb_act, mb_logp, mb_val, mb_rew, mb_done = [], [], [], [], [], []
        ep_succ = []
        for _ in range(T):
            with torch.no_grad():
                logits, val = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            nobs, rew, done, succ = env.step(act.cpu().numpy())
            mb_obs.append(obs); mb_act.append(act); mb_logp.append(logp)
            mb_val.append(val); mb_rew.append(torch.tensor(rew, device=dev))
            mb_done.append(torch.tensor(done, device=dev))
            for i in range(env.B):
                if done[i]:
                    ep_succ.append(succ[i])
            obs = torch.tensor(nobs, device=dev)
        with torch.no_grad():
            _, last_val = net(obs)
        # GAE
        adv = torch.zeros(env.B, device=dev)
        advs = [None] * T
        for t in reversed(range(T)):
            nextval = last_val if t == T - 1 else mb_val[t + 1]
            nonterm = 1.0 - mb_done[t]
            delta = mb_rew[t] + 0.99 * nextval * nonterm - mb_val[t]
            adv = delta + 0.99 * 0.95 * nonterm * adv
            advs[t] = adv.clone()
        obs_b = torch.cat(mb_obs); act_b = torch.cat(mb_act)
        logp_b = torch.cat(mb_logp); val_b = torch.cat(mb_val)
        adv_b = torch.cat(advs); ret_b = adv_b + val_b
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        ent_last = 0.0
        for _ in range(4):
            idx = torch.randperm(obs_b.shape[0])
            for s in range(0, obs_b.shape[0], 512):
                b = idx[s:s + 512]
                logits, val = net(obs_b[b])
                dist = torch.distributions.Categorical(logits=logits)
                nlogp = dist.log_prob(act_b[b])
                ratio = torch.exp(nlogp - logp_b[b])
                a = adv_b[b]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 0.8, 1.2) * a).mean()
                vl = ((val - ret_b[b]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - 0.01 * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                ent_last = float(ent)
        step = upd * T * env.B
        sr = float(np.mean(ep_succ)) if ep_succ else 0.0
        succ_hist.append(sr)
        ret = float(torch.stack(mb_rew).sum(0).mean())
        if upd % 5 == 0 or upd == 1:
            log(f"upd {upd:4d} | step {step:8d} | ret {ret:7.2f} | "
                f"succ {sr:0.2f} | ent {ent_last:0.3f} | eps {len(ep_succ)}")
    log(f"# DONE final_succ={np.mean(succ_hist[-10:]):0.3f}")
    torch.save({"model": net.state_dict()}, args.save)
    log(f"# saved policy -> {args.save}")
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=str, default="logs/randlevel.log")
    ap.add_argument("--save", type=str, default="checkpoints/experiments/randlevel/ppo.pt")
    ap.add_argument("--hard", action="store_true", help="harder levels (diagonal jumps required)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    t0 = time.time()
    train(args)
    print(f"elapsed {time.time()-t0:0.1f}s")
