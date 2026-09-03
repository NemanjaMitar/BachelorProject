# Jump King — Reinforcement Learning (bachelor thesis)

PPO agents that play the pygame re-implementation of *Jump King* from the bottom
screen to the top. The deployed relay chains one policy per level and completes
**levels 0 → 42** (the babe) without ever falling to a lower screen.

Reproduce the run:

```bash
python tools/relay_bundle.py           # headless: plays 0 -> top, flags any FALL or STALL
python tools/relay_bundle.py --play    # the same climb, in a real window
python tools/relay_bundle.py --viz     # ...with the network's output beside the game
```

`--viz` opens the game on the left and a live panel on the right: the action
probabilities of the screen's own policy, the observation scalars it actually
reads (the ones it does not are greyed out), `V(s)`, and the conv / trunk
activations. It follows the handoff, so the panel switches to the next screen's
model at the seam.

## The relay

One trained PPO policy per screen, handed off at the seam. **Every screen is a
learned model — there is no hand-written route anywhere in the chain.** Measured:
3/3 clean top-outs from level 0, no fall to a lower screen.

The whole chain is packed into a single file, `checkpoints/relay_bundle.pt` —
every network plus the action-table config it was trained with, 42 screens and 39
distinct models in 28 MB. Running it reads nothing else, so that one file is the
deployable agent.

```bash
python tools/relay_bundle.py --list    # what is inside, screen by screen
python tools/relay_bundle.py --build   # repack it from checkpoints/
```

`FullRelay.py` is the relay itself -- the env, the per-screen model handoff and
the FALL/STALL verdict; `relay_bundle.py` drives it from the packed file.

Two screens, **24 and 41, are never stood on**: the routes for 23 and 40 climb
*through* them inside a single crossing, so no model was ever trained there and
`FullRelay.run_trial` carries the mid-route model across.

**How a screen's model is built.** A screen is trained from a demonstration of
one crossing, under a backward curriculum: the demonstration's states become
rungs, training starts at the rung nearest the exit, and a rung only opens once
the one above it is solved — so the policy meets the level's real entry last,
when it can already finish from everywhere above.

The demonstration has to start from the arrival the chain *really* makes, and
that arrival moves whenever a lower screen changes. So the route is **searched**
from the actual handover state, not replayed from an older one:

```bash
python tools/route_from_entry.py --level 39                    # search from the real arrival
python tools/train_ice.py --level 39 --demo starts/demo_L39_swap.json     --save-dir checkpoints/L39_swap --channels solid,king,slope --subgoal ...
```

Measured on L39: a route searched for an arrival at x=443.0 falls three screens
when the chain below delivers x=449.7 instead. The route found from 449.7 is five
actions, and the policy trained on it crosses 40/40 greedy from the arrival (0.65
under +/-10 px / +/-20 % perturbation). Fixing L39 moved L40's arrival in turn,
so L40 was rebuilt the same way: ten actions spanning screen 41 into 42, 40/40
exact, 0.80 at +/-3 px, 0.23 at +/-10 px -- a ten-action chain is only as robust
as its least forgiving link.

Two settings matter for every screen whose demonstration puts several states on
one screen: `--channels solid,king,slope` (without the `king` plane the occupancy
grid is screen-absolute and identical at every state, so two rungs 175 px apart
become near-identical inputs needing opposite actions) and `--guard-prev`, which
keeps a solved rung from rotting while the next one trains.

## Start here

`JumpKing_PPO_walkthrough.ipynb` is a step-by-step walkthrough of the core
implementation — the environment, the observation, the macro-action table, the reward
rule, the reverse curriculum, the actor-critic network, GAE, the clipped PPO objective,
and a short training run that visibly learns in about a minute on a CPU.

```bash
pip install jupyter
jupyter notebook JumpKing_PPO_walkthrough.ipynb
```

`checkpoints/RUN_MANIFEST.txt` records the exact checkpoint each screen is
driven by, and `python tools/relay_bundle.py --list` prints the same table out
of the packed relay.

## Layout

| Folder | Contents |
|---|---|
| `JumpKing_PPO_walkthrough.ipynb` | step-by-step walkthrough of the core implementation |
| `*.py` (root) | the game (`JumpKing.py`, `King.py`, `Level.py`, …) and the RL toolchain (`JK_Env.py`, `PPO.py`, `Train.py`, `AutoPilot.py`, `FullRelay.py`, …) |
| `starts/` | start-state pools — the reverse curricula (`starts_L<N>.json`, `_demo`, `_broad`, plus the global `start_states.json`) |
| `checkpoints/relay_bundle.pt` | the whole 0→42 chain in one file — every screen's policy and its action config |
| `checkpoints/L<N>/` | the model of screen N, one folder per screen; the relay takes the highest-numbered `.pt` |
| `checkpoints/L<N>_real*/ _king*/ _ice*/ _swap/` | the screens trained with the demonstration + backward-curriculum pipeline (`tools/train_ice.py`), one folder per training run |
| `checkpoints/experiments/` | side experiments and ablations (`abl_*`, `*_ppo`, `*_broad`, `*_fix*`, ice/wind variants, `worldgen*` generated-level runs) |
| `charts/` | per-run training curves + their CSVs, written by `ChartLogs.py` |
| `figures/` | figures used in the thesis, plus `screenshots/` and the `data/` behind them |
| `logs/` | raw training / handoff / graph logs (the source for `charts/`) |
| `thesis/` | the current thesis documents, `drafts/` of earlier versions, `ABLATION_SUMMARY.md` |
| `levels/` | custom worlds built with `LevelEditor.py`, plus the generated pools `gen_train/` and `gen_eval/` (`WorldGen.py`) |
| `tools/` | the relay (`relay_bundle.py`), route search (`route_from_entry.py`), the curriculum trainer (`train_ice.py`), the generated-screen tools (`gen_*.py`), run scripts and `thesis/` figure generators |
| game assets | `Audio/ BG/ MG/ FG/ Fonts/ images/ props/ weather/ Scrolling/ gui/ hiddenwalls/ Saves/ Cro/` — unchanged from the game |

## Custom worlds (levels the game never had)

`LevelEditor.py` draws new screens; they run on the **real engine**, so King.py's
collision, ice slip and wind act on them and the same conv policy trains on them.
That makes a custom world a genuine generalisation test, not a second physics model.

```bash
python LevelEditor.py levels/tower.json
```

Drag blocks from the palette onto the canvas, or drag on empty canvas to paint one.
Land / ice / snow, wind per screen, and a stack of screens (screen 0 is the bottom;
the king climbs into higher indices; the top screen is the goal).

**Play it yourself first** — same engine, your keyboard, so you find out whether the
level is possible before spending an hour training on it:

```bash
python PlayWorld.py levels/tower.json
```

Hold SPACE to charge, arrows to steer, R to respawn, TAB to skip a screen, H toggles
the platform overlay. The engine's original background art still shows behind custom
screens and its painted ledges are *not* solid -- the overlay (on by default) is the
real collision geometry.

Then seed start states from the geometry and train:

```bash
python CustomWorld.py --seed levels/tower.json
python Train.py --world levels/tower.json --start-states starts/world_tower.json --curriculum --save-dir checkpoints/world_tower
```

`--world` also works with `Play.py` (watch it) and `LevelGraph.py` (BFS-prove a
screen is solvable before training it). `levels/tower.json` is a worked example:
a land climb, an ice shelf, and a precise seam crossing into the goal screen.

**Designing screens that work.** Two facts about the engine constrain the layout:
a full-charge jump rises ~169px and travels ~240px sideways, and crossing the seam
into the next screen costs an extra ~20px. So the king enters the screen above near
its bottom edge — a screen that is walled off across its bottom cannot be entered at
all, and the column he ascends through must be clear while he lands on a ledge to
one side. `CustomWorld.py --info <file>` reports obvious problems.

**A safety floor removes the pressure to commit.** In `tower.json` screen 0 has a
full-width floor, so a missed jump just drops the king back onto it and he retries.
Nothing punishes a wasted action, so PPO has no reason to make the winning jump the
*most likely* one: the trained model reaches the goal 20/20 when sampling, but its
argmax is a jump into the right wall (p=0.79) while the actual crossing sits at p=0.21
-- so `Play.py` (greedy by default) loops forever and needs `--stochastic`. The
original game does not have this problem because a miss makes you FALL a screen, which
ends the episode as a failure. If you want a policy that commits, build screens where
missing costs a level.

## Procedurally generated screens (the generalisation test)

`RandomLevel.py` is a *toy* generator — its own numpy parabola, 13 scalars, a
96x96 MLP. It says something about PPO as an algorithm and nothing about the
agent that plays this game. `WorldGen.py` is the real version: it emits a
`jumpking-world` whose platforms go straight into `Platforms.Platform`, so
King.py's collision acts on them, `JK_Env` builds its usual occupancy grid, and
the **same conv policy with the same macro-action table** trains and is measured
on them.

```bash
python WorldGen.py --preview --seed 3           # describe one generated screen
python WorldGen.py --selftest --n 20            # engine-prove random screens (~85% pass)
python WorldGen.py --out levels/gen_eval --n 160 --seed 900000 --jobs 8
```

Every screen in a pool is **proved in the engine** before it is used: for each
rung `prove_ladder` searches the real action table for one action that reaches a
higher rung, and from the top rung one that leaves the screen. A pool is
therefore a set of screens with a known solution chain, not an assumption.

Train on an endless stream (a brand-new screen every episode, nothing to
memorise) or on a fixed proved pool, and score on a held-out pool:

```bash
python Train.py --world-gen --num-envs 10 --max-steps 32 --grid-channels solid,king     --altitude-breadcrumb 0.02 --step-cost 0.15 --ent-coef 0.015 --ent-floor 0.8     --save-dir checkpoints/experiments/worldgen
python GenEval.py --checkpoint checkpoints/experiments/worldgen/ppo_final.pt     --worlds levels/gen_eval --starts 3
python tools/gen_harvest.py --dir checkpoints/experiments/worldgen9   # pick the best ckpt
python tools/gen_stages.py --checkpoint <ckpt>   # score per curriculum stage
python tools/gen_try.py --world levels/moj.json  # one screen YOU drew
python tools/gen_render.py --n 3 --combine       # a GIF of unseen screens
```

**The measured result.** `checkpoints/worldgen_best.pt` (worldgen9, 10.6 M
steps) on **80 held-out generated screens x 3 bottom starts, none seen in
training**: greedy 79.6 % of starts / 71.2 % of screens from all three;
sampled 91.2 % / 86.2 %.

One greedy pass is not what "reliable" means here, though, and
`tools/gen_reliability.py` measures the number that is. Every generated screen
has a full-width floor, so a missed jump drops the king back onto it and he
tries again -- exactly like a human. Over 40 unseen screens x 3 starts:

| | climbed |
|---|---|
| one greedy pass | 83.3 % |
| sampled, 1 attempt | 95.0 % |
| sampled, within 2 | 96.7 % |
| sampled, within 5 | **97.5 %** |

and the greedy weakness is **concentrated in screen height**, not spread evenly:
3 rungs 93.9 %, 4 rungs 86.3 %, 5 rungs 69.4 % greedy -- while the 5-rung
screens reach 100 % once retries are allowed.

The greedy/sampled gap is the catch-floor effect described above, and it is
*not* the entropy bonus: a fine-tune resumed from this checkpoint with the
entropy floor removed (`--ent-floor 0 --ent-coef 0.002`, 460 k steps) left
rollout entropy at 0.52 and scored greedy **78.3 %**, i.e. no better
(`checkpoints/experiments/worldgen10`, `logs/gen_harvest_ent.json`).

```bash
python tools/gen_reliability.py --limit 40          # the table above
python tools/gen_render.py --n 12 --greedy --keep-failures --combine --no-art --colors 32
```

`--no-art` matters for any picture of a generated screen: the engine's painted
terraces are **not** solid and the generated platforms are, so a frame with the
background art in it shows the wrong geometry.

On hand-drawn screens it climbs `levels/moj.json` from 3/3 starts; it fails
`levels/tower.json`, whose climb runs over an **ice** shelf the generator never
produces, and `levels/mine.json` is not solvable at all (`--prove` reports rung
5 a dead end), so neither is evidence against it.

The reverse curriculum is derived from the screen itself: its rungs are the
checkpoints, so no capture pass is needed, and it is **adaptive** — a stage only
opens once the last 150 episodes clear the target rate. A scheduled version of
the same curriculum was measured to stall (bottom-start success 1/120); the
adaptive one walks all the way down to floor starts.

**Two engine facts the generator had to be built around.** A king who climbs out
of a screen arrives in the next one at `rect_y ~ 371..379`, i.e. *below* its
bottom edge, so a catch floor drawn anywhere on-screen is a ceiling he bonks on:
with a floor at y=352 a full-charge up-jump crossed only 53% of the time, and
thickness made no difference (every variant's underside sits at y=360, which is
what he hits). A floor whose top is just below the entry point — y in 366..384 —
catches him on the way down instead: 45/45. And a rung roofed over by the rung
above it is a dead end, so rungs are placed staggered, each keeping a strip of
open sky to launch from.

## Training pipeline

```bash
python AutoPilot.py --levels 10-14    # handoff -> autoseed -> level graph -> train, per level
python ChartLogs.py                   # logs/*.log -> charts/*.png + .csv
```

Start-state pools are read from `starts/`. Older commands that name a pool
without the folder (`--start-states starts_L12.json`) still resolve — see
`resolve_start_states` in `JK_Env.py`.

---

## About the game

Jump King with Pygame — run `JumpKing.exe` to play. A near-perfect replica;
extras and end-game content are partly missing. Hitboxes can be shown from the
graphics menu, and "C" toggles a flying mode.
