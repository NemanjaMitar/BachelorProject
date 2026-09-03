# -*- coding: utf-8 -*-
"""Train / watch a PPO agent on a HAND-DESIGNED level (levels/*.json).

The fixed level becomes an environment (same jump physics as RandomLevel). Each
platform is a resting "graph state" on the path; a reverse curriculum starts the
king on platforms near the goal and expands downward, sampling failing start
states more often (+ random x on each platform = the "random states"). The agent
must end up solving the level from the very bottom.

    python CustomLevel.py --train --level levels/complex.json
    python CustomLevel.py --watch --level levels/complex.json
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np, torch, torch.nn as nn

spec = importlib.util.spec_from_file_location("RL", os.path.join(os.path.dirname(__file__), "RandomLevel.py"))
RL = importlib.util.module_from_spec(spec); spec.loader.exec_module(RL)
W, H, G, VX, VY = RL.W, RL.H, RL.G, RL.VX_MAX, RL.VY_MAX


def load_level(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict) and d.get("format") == "jumpking-world":
        raise SystemExit(
            f"{path} is a jumpking-world file (a level SEQUENCE for the real "
            f"engine), not this script's single-screen toy format. "
            f"Use:  python Train.py --world {path} ...   (see CustomWorld.py)")
    plats = [(x, x + w, y) for (x, y, w, h) in d["platforms"]]
    plats.sort(key=lambda p: -p[2])                 # ground (largest y) first -> top last
    return d, plats


class FixedEnv(RL.VecRandEnv):
    """RandomLevel physics on a fixed platform stack with a start-state curriculum."""
    def __init__(self, batch, plats, seed=0):
        self.plats = plats
        self.N = len(plats) - 1                      # goal = top platform index
        self.unlocked = 1                            # reverse-curriculum breadth (from top)
        self.failema = np.ones(len(plats))           # priority: failing start states sampled more
        self.startidx = np.zeros(batch, dtype=np.int64)
        RL.N_PLAT = self.N                           # RandomLevel uses this global for obs/success
        super().__init__(batch, seed)

    def _sample_start(self):
        lo = max(0, self.N - self.unlocked)
        idxs = np.arange(lo, self.N)                 # start on a platform below the goal
        w = self.failema[lo:self.N] + 0.05
        return int(np.random.choice(idxs, p=w / w.sum()))

    def _place(self, i, si):
        xl, xr, y = self.plats[si]
        self.idx[i] = si; self.x[i] = np.random.uniform(xl + 8, xr - 8)
        self.y[i] = y; self.steps[i] = 0; self.startidx[i] = si

    def reset_all(self):
        self.levels = [list(self.plats) for _ in range(self.B)]
        self.idx = np.zeros(self.B, dtype=np.int64); self.x = np.full(self.B, W / 2.0)
        self.y = np.zeros(self.B); self.steps = np.zeros(self.B, dtype=np.int64)
        self.startidx = np.zeros(self.B, dtype=np.int64)
        for i in range(self.B): self._place(i, self._sample_start())
        return self._obs()

    def _reset_one(self, i):
        si = int(self.startidx[i]); reached = 1.0 if self.idx[i] >= self.N else 0.0
        self.failema[si] = 0.9 * self.failema[si] + 0.1 * (1.0 - reached)
        self._place(i, self._sample_start())


def sim_arc(plats, x0, y0, d, ch):
    vx = d * ch * VX; vy = -ch * VY; x, y = x0, y0; py = y0; arc = [(x, y)]
    for _ in range(240):
        vy += G; x += vx; y += vy
        if x < 0 or x > W: x = min(max(x, 0.0), W)
        arc.append((x, y))
        if vy > 0:
            for j, (xl, xr, yy) in enumerate(plats):
                if py <= yy <= y and xl <= x <= xr: return arc, j
        py = y
        if y > H: return arc, 0
    return arc, 0


def train(args):
    torch.manual_seed(args.seed)
    d, plats = load_level(args.level)
    env = FixedEnv(args.batch, plats, seed=args.seed)
    net = RL.ActorCritic(RL.OBS_DIM, RL.N_ACT); opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    obs = torch.tensor(env.reset_all()); T = args.rollout
    name = os.path.splitext(os.path.basename(args.level))[0]
    os.makedirs("logs", exist_ok=True); os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    lf = open(args.log or f"logs/custom_{name}.log", "w")
    def log(m): print(m); lf.write(m + "\n"); lf.flush()
    log(f"# CustomLevel PPO  level={name}  platforms={env.N} (goal idx {env.N})  batch={args.batch} rollout={T}")
    recent = []
    for upd in range(1, args.updates + 1):
        MO, MA, ML, MV, MR, MD, es = [], [], [], [], [], [], []
        for _ in range(T):
            with torch.no_grad():
                lg, v = net(obs); dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample(); lp = dist.log_prob(a)
            no, r, dn, sc = env.step(a.numpy())
            MO.append(obs); MA.append(a); ML.append(lp); MV.append(v)
            MR.append(torch.tensor(r)); MD.append(torch.tensor(dn))
            for i in range(env.B):
                if dn[i]: es.append(sc[i])
            obs = torch.tensor(no)
        with torch.no_grad(): _, lastv = net(obs)
        adv = torch.zeros(env.B); A = [None] * T
        for t in reversed(range(T)):
            nv = lastv if t == T - 1 else MV[t + 1]; nt = 1.0 - MD[t]
            delta = MR[t] + 0.99 * nv * nt - MV[t]; adv = delta + 0.99 * 0.95 * nt * adv; A[t] = adv.clone()
        ob = torch.cat(MO); ac = torch.cat(MA); lpb = torch.cat(ML); vb = torch.cat(MV)
        ab = torch.cat(A); rb = ab + vb; ab = (ab - ab.mean()) / (ab.std() + 1e-8)
        ent_last = 0.0
        for _ in range(4):
            idx = torch.randperm(ob.shape[0])
            for s in range(0, ob.shape[0], 512):
                b = idx[s:s + 512]; lg, v = net(ob[b]); dist = torch.distributions.Categorical(logits=lg)
                ratio = torch.exp(dist.log_prob(ac[b]) - lpb[b]); aa = ab[b]
                pg = -torch.min(ratio * aa, torch.clamp(ratio, 0.8, 1.2) * aa).mean()
                loss = pg + 0.5 * ((v - rb[b]) ** 2).mean() - 0.02 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                ent_last = float(dist.entropy().mean())
        sr = float(np.mean(es)) if es else 0.0; recent.append(sr)
        if len(recent) > 15: recent.pop(0)
        if np.mean(recent) > 0.6 and env.unlocked < env.N:
            env.unlocked += 1; recent = []                    # expand curriculum one platform deeper
        ret = float(torch.stack(MR).sum(0).mean())
        if upd % 5 == 0 or upd == 1:
            log(f"upd {upd:4d} | step {upd*T*env.B:8d} | ret {ret:7.2f} | succ {sr:0.2f} | "
                f"ent {ent_last:0.3f} | cur {env.unlocked}/{env.N} | eps {len(es)}")
        if upd % 25 == 0:
            torch.save({"model": net.state_dict(), "level": args.level}, args.save)
    torch.save({"model": net.state_dict(), "level": args.level}, args.save)
    log(f"# DONE final_succ={np.mean(recent[-5:]) if recent else 0:.2f}  saved -> {args.save}")
    lf.close()


def watch(args):
    import pygame
    d, plats = load_level(args.level)
    env = FixedEnv(1, plats); env.unlocked = env.N
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = RL.ActorCritic(RL.OBS_DIM, RL.N_ACT); net.load_state_dict(ck["model"]); net.eval()
    sys.argv = ["LevelView.py", "--file", args.level]
    spec2 = importlib.util.spec_from_file_location("LVW", "LevelView.py")
    V = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(V)
    kingx = d["king"][0]; groundy = max(p[2] for p in plats)
    def new_run():
        return 0, float(kingx), float(groundy), 0
    idx, x, y, jumps = new_run()
    clock = pygame.time.Clock(); running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE): running = False
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_r, pygame.K_n): idx, x, y, jumps = new_run()
        env.idx[0] = idx; env.x[0] = x; env.y[0] = y
        with torch.no_grad(): a = int(net(torch.tensor(env._obs()))[0].argmax(1).item())
        dd, cc = RL.ACTIONS[a]
        arc, land = sim_arc(plats, x, y, dd, cc)
        for (ax, ay) in arc[::2]:
            V.LV["king"] = [ax, max(ay, 4)]; V.draw(False); clock.tick(90)
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
        idx = land; x = arc[-1][0]; y = plats[land][2]; jumps += 1
        if idx >= env.N:
            V.LV["king"] = [x, y]; V.draw(False); pygame.time.wait(1200); idx, x, y, jumps = new_run()
        elif jumps > 40:
            idx, x, y, jumps = new_run()
    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true"); ap.add_argument("--watch", action="store_true")
    ap.add_argument("--level", default="levels/complex.json")
    ap.add_argument("--batch", type=int, default=96); ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--checkpoint", default=None); ap.add_argument("--save", default=None)
    a = ap.parse_args()
    nm = os.path.splitext(os.path.basename(a.level))[0]
    if a.save is None: a.save = f"checkpoints/custom_{nm}/ppo.pt"
    if a.checkpoint is None: a.checkpoint = a.save
    if a.watch: watch(a)
    else: train(a)
