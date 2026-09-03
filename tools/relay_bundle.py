#!/usr/bin/env python
"""The fully learned relay as ONE file: every screen of the 0 -> 42 climb driven
by a trained PPO policy, packed into `checkpoints/relay_bundle.pt`.

    python tools/relay_bundle.py                 # play it: 3 headless trials
    python tools/relay_bundle.py --trials 10
    python tools/relay_bundle.py --list          # what is inside, screen by screen
    python tools/relay_bundle.py --play          # watch it in a real window
    python tools/relay_bundle.py --build         # repack it from checkpoints/

Running it reads NOTHING from `checkpoints/L<N>/` -- the archive carries every
network and the action-table config it was trained with, so the bundle on its
own is the deployable agent.

Only the weights and the action config are packed. Optimizer state and training
bookkeeping are dropped: they are several times the size of the policy and
nothing plays the game with them.

Screen 41 has no entry and that is correct -- it is crossed *through* by L40's
route, never stood on, so no model was ever trained for it. `FullRelay.run_trial`
carries the mid-route model across it.
"""
import os, sys, glob, argparse, datetime, subprocess

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
import FullRelay as FR

BUNDLE = "checkpoints/relay_bundle.pt"
FORMAT = "jk-relay-bundle-v2"

# The screens whose model lives in its own training directory rather than in
# checkpoints/L<N>/. The ice three are named explicitly because their runs are
# not in the L<N>_* naming scheme.
ICE = {36: "checkpoints/L36_ice_pure/best_rung0.pt",
       37: "checkpoints/L37_ice_walk/best_rung0.pt",
       38: "checkpoints/L38_ice_relay/best_rung0.pt"}
OWN_DIR = (23, 26, 28, 29, 30, 31, 32, 33, 34, 35, 39, 40, 42)


def newest_own(level):
    """The most recently written best_rung0.pt among this level's own dirs.

    rung 0 is the level's real entry, so a directory holding only best_rung4.pt
    is a ladder that never reached the entry and is NOT picked up. The glob is
    L<N>_* rather than one fixed suffix: the directories are named after the run
    that produced them (_real, _real3, _king, _king_j3, _swap ...) and pinning
    the pattern to one of those silently drops finished models."""
    cands = [p for p in glob.glob(f"checkpoints/L{level}_*/best_rung0.pt")
             if os.path.isfile(p)]
    return max(cands, key=os.path.getmtime) if cands else None


def learned_models():
    """{level: path} for the screens trained outside checkpoints/L<N>/."""
    picked = {L: p for L, p in ICE.items() if os.path.isfile(p)}
    for L in OWN_DIR:
        p = newest_own(L)
        if p:
            picked[L] = p
    return picked


def _key(src):
    """The name a model is filed under: its path below checkpoints/.

    Levels 0-3 share one network, so keying by source path packs it once and
    all four levels point at the same entry."""
    return os.path.relpath(src, "checkpoints").replace("\\", "/")


def sources():
    """{level: checkpoint path} for the whole chain.

    A screen's own training directory wherever there is one, the model in
    checkpoints/L<N>/ everywhere else."""
    picked = learned_models()
    out = {}
    for L in range(0, 43):
        src = picked.get(L) or FR.latest(L)
        if src:
            out[L] = src
    return out


def build(path=BUNDLE):
    levels, models = {}, {}
    for L, src in sorted(sources().items()):
        k = _key(src)
        levels[L] = k
        if k not in models:
            ck = torch.load(src, map_location="cpu", weights_only=False)
            cfg = ck.get("action_cfg", {}) or {}
            models[k] = dict(model=ck["model"], action_cfg=cfg,
                             src=src.replace("\\", "/"),
                             step=int(ck.get("step", 0)))
    torch.save(dict(format=FORMAT, chain="fully-learned",
                    built=datetime.date.today().isoformat(),
                    levels=levels, models=models), path)
    mb = os.path.getsize(path) / 1e6
    print(f"wrote {path}  ({mb:.1f} MB, {len(levels)} screens, "
          f"{len(models)} distinct models)")
    return 0


def open_bundle(path=BUNDLE):
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found -- build it with --build")
    b = torch.load(path, map_location="cpu", weights_only=False)
    b["levels"] = {int(L): k for L, k in b["levels"].items()}
    return b


def show(path=BUNDLE):
    b = open_bundle(path)
    print(f"{path}  format={b.get('format')}  chain={b.get('chain','?')}  "
          f"built={b.get('built','?')}")
    print(f"{'lvl':>4}  {'model':34s} {'steps':>10}  verdict")
    scripted = []
    for L in sorted(b["levels"]):
        e = b["models"][b["levels"][L]]
        combo = len(e["action_cfg"].get("wind_combo", []) or [])
        step = int(e.get("step", 0))
        verdict = "learned" if (step > 0 and not combo) else "SCRIPTED MACRO"
        if verdict != "learned":
            scripted.append(L)
        print(f"{L:>4}  {b['levels'][L]:34s} {step:>10}  {verdict}")
    print(f"\n{len(b['levels'])} screens, "
          f"{len(b['models'])} distinct models, "
          f"{os.path.getsize(path)/1e6:.1f} MB")
    print("scripted macros: " + (str(scripted) if scripted else "none -- "
          "every screen is a trained policy"))
    print("screen 41 carries no model: L40's route crosses it without landing.")
    return 0


def unpack(path, stage):
    """Write the bundle back out as a `checkpoints/`-shaped tree.

    Play.py resolves <model-dir>/L<N>/ and takes the newest file there, so
    watching the bundle in a window means laying it out that way first. The real
    checkpoints/ is never touched."""
    b = open_bundle(path)
    os.makedirs(stage, exist_ok=True)
    base = None
    for L in sorted(b["levels"]):
        e = b["models"][b["levels"][L]]
        ck = {"model": e["model"], "action_cfg": e["action_cfg"],
              "step": e.get("step", 0)}
        if L <= 3:
            base = os.path.join(stage, "ppo_1868800.pt")
            if not os.path.exists(base):
                torch.save(ck, base)
            continue
        d = os.path.join(stage, f"L{L}")
        os.makedirs(d, exist_ok=True)
        torch.save(ck, os.path.join(d, "ppo_9999999.pt"))
    return b, base


def play(path, stage, viz=False, extra=()):
    b, base = unpack(path, stage)
    print(f"unpacked {len(b['levels'])} screens into {stage}/\n")
    env = dict(os.environ)
    for k in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        env.pop(k, None)
    # VizPlay puts the game on the left and the network on the right: the action
    # probabilities, the scalars the model actually reads, V(s), and the conv /
    # trunk activations -- of whichever screen's policy is driving right now.
    script = "VizPlay.py" if viz else "Play.py"
    return subprocess.call([sys.executable, script,
                            "--checkpoint", base, "--model-dir", stage,
                            *extra], env=env)


def run(path, trials, seed, frm):
    b = open_bundle(path)
    e = FR.build_env(seed=seed)
    hw = (e.grid_h, e.grid_w)
    models = FR.bind(e, {L: FR.load(L, hw, ck=b["models"][k], name=k)
                         for L, k in b["levels"].items() if L >= frm})
    print(f"{path}: {len(models)} screens, "
          f"{len(b['models'])} distinct models\n")
    wins = 0
    for t in range(trials):
        r = FR.run_trial(e, models, frm)
        wins += r["ok"]
        traj, falls = r["traj"], r["falls"]
        tag = ("CLEAN" if r["ok"] else
               ("FELL@" + str(falls[:2]) if falls else r["result"]))
        span = r.get("spans") or []
        print(f"trial {t}: {tag} | max={max(traj)} | traj="
              f"{traj if len(traj) < 22 else traj[:22] + ['...']}"
              + (f" | spanned {span} on the model below" if span else ""))
    print(f"\n{wins}/{trials} clean top-outs from L{frm}")
    e.close()
    return 0 if wins == trials else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--build", action="store_true", help="repack from checkpoints/")
    ap.add_argument("--list", action="store_true", help="show what is inside")
    ap.add_argument("--play", action="store_true", help="watch it in a window")
    ap.add_argument("--viz", action="store_true",
                    help="watch it with the live network panel beside the game")
    ap.add_argument("--unpack", metavar="DIR", default=None,
                    help="write the bundle out as a checkpoints/-shaped tree")
    ap.add_argument("--stage-dir", default="checkpoints_bundle")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--from", dest="frm", type=int, default=0)
    args, rest = ap.parse_known_args()   # anything else is passed to Play/VizPlay

    if args.build:
        return build(args.bundle)
    if args.list:
        return show(args.bundle)
    if args.unpack:
        b, _ = unpack(args.bundle, args.unpack)
        print(f"unpacked {len(b['levels'])} screens into {args.unpack}/")
        return 0
    if args.play or args.viz:
        return play(args.bundle, args.stage_dir, viz=args.viz, extra=rest)
    return run(args.bundle, args.trials, args.seed, args.frm)


if __name__ == "__main__":
    sys.exit(main())
