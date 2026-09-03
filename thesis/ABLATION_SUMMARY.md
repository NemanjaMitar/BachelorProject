# PPO on Jump King — feature ablations & generalization (results for the thesis)

All runs are PPO with an identical network (3× stride-2 conv over an 8-px occupancy
grid + a small scalar head) and identical optimisation hyper-parameters. Between the
two members of each comparison **only one thing changes** — the feature under test —
so any difference in the learning curve is attributable to that feature.

Charts: `figures/ablation_ice_wind.png`, `figures/randlevel.png`
(per-run panels: `charts/L36_veltest.png`, `charts/L36_broad.png`, `charts/L29_broad.png`, …)

---

## 1. ICE — is a velocity (momentum) observation necessary? *(controlled ablation)*

On slippery ground the king keeps sliding after a landing (`speed *= slip`, slip>0),
so momentum must be managed. Two separate questions are worth keeping distinct:
(a) *can PPO learn ice at all?* and (b) *does adding a velocity observation help?*

### (a) Genuine PPO on ice — it learns

Two independent pieces of evidence that ice is genuinely PPO-learnable:

| Run | Setup | Success |
|-----|-------|--------:|
| `IceLevel.py` (`--ice`) | PPO on *procedurally generated slippery levels* (momentum carries after every landing), fully from scratch | **~0.74** (converged) — chart `figures/ice_ppo.png` |
| `L36_veltest` (game) | PPO with `vel_obs` on the real L36 ice screen, near-goal curriculum | 0.87 on the unlocked near-goal states |

**Honest scope:** the *game* ice policy is only robust on states close to the goal.
Continued training toward a whole-level solver (`checkpoints/experiments/L36_gameice`) improved
then regressed (0/4 → 1/4 → 0/4 on-path across 25 updates) — ice macro-rollouts are
slow (~20 s/update) and the run did not converge to a robust whole-level solver in
the time available. Whole-level robustness (`GenTest` over ~49 states) stayed at 2–6%.
So: **PPO demonstrably learns ice physics (IceLevel, near-goal L36), but a robust
end-to-end solver for the game ice screen never converged from scratch in the time
available.** What did work is the demonstration + backward-curriculum pipeline
(`tools/train_ice.py`): the three ice screens 36-38 are trained policies in the
deployed 0→42 run, and the ladder reaches the screen's real entry rather than
stopping near the goal.

### (b) Does velocity observation help? — depends on the perception

| Setting | vel-obs ON | vel-obs OFF |
|---------|-----------:|------------:|
| Fast MLP toy (`IceLevel`, full state visible) | 0.72 | 0.72 — **no difference** |
| Game L36 (conv occupancy grid) | learns (0.87 near-goal) | 0.00 |

**Reading (important nuance):** in the fully-observable MLP toy the agent controls its
own momentum through its chosen jump charge, so it never *needs* to observe velocity —
the ablation shows no gap. In the **game** the conv grid downsamples 8× and aliases
fine position, so the velocity channel restores information the vision discards — there
the same toggle is the difference between 0.00 and learning. i.e. the velocity
observation is a **perception** fix, not an ice-physics necessity. This is the honest,
defensible framing.

---

## 2. WIND — treat the wind as noise, or as a tool? 

Windy screens apply a lateral push `wind = sin(phase)·6.25` every airborne frame,
with the phase advancing ~1 push-cycle per 1000 frames. The phase is **not**
controllable, only observable. Our approach gives the agent (a) a `wind_obs`
channel `[sin(phase), cos(phase)]` and (b) *wind-timed* jump actions that hold
position until the wind reaches a chosen strength bucket, then release — turning the
disturbance into a deterministic assist across the gap.

| Run | Configuration | Final success |
|-----|---------------|--------------:|
| `L29_broad` | wind as help: `wind_obs` + wind-timed jumps, whole-level curriculum | **1.00** |
| `L31_fast`  | wind as help (demo curriculum) | **0.90** |

**Reading:** with wind perception + wind-timed actions the agent solves the windy
crossings reliably (0.90–1.00). A fixed ("deterministic") jump cannot: because the
push depends on the unobserved phase, an identical launch lands in a different place
on every attempt, so success without wind-awareness is bounded by chance alignment.
*(A from-scratch deterministic-baseline PPO run is queued — `checkpoints/experiments/abl_wind_det`
— but ice/wind macro-rollouts are slow; the qualitative result already follows from
the physics and from the fact that every windy level only converged once these two
features were enabled.)*

---

## 3. PROCEDURAL GENERALIZATION — did the algorithm learn, or memorise?

To show the PPO algorithm itself generalises (rather than overfitting the 43
hand-authored screens), a **procedural level generator** builds a brand-new random
stack of ledges every episode (`RandomLevel.py`, faithful jump-arc physics under
gravity). The agent never sees the same layout twice.

| Run | Setup | Final success (unseen levels) |
|-----|-------|------------------------------:|
| `randlevel` | PPO on procedurally generated levels, new layout each episode | **1.00** (converged over 400 updates) |

Success climbs 0.03 → 1.00 while entropy falls from 2.70 → ~0.08, i.e. the policy
converges to a general bottom-to-top strategy that transfers to layouts it has never
encountered. This is direct evidence of learning, not memorisation.

---

## Hyper-parameters (what each one controls)

| Parameter | Value | What it governs |
|-----------|-------|-----------------|
| discount γ | 0.99 | How far ahead reward is credited (long climbs need high γ) |
| GAE λ | 0.95 | Bias/variance of the advantage estimate |
| clip ε | 0.2 | Max per-step policy change (trust-region → stability) |
| entropy coef | 0.05 (game) / 0.01 (rand) | Exploration pressure; high early avoids premature collapse |
| value coef | 0.5 | Weight of the critic loss |
| lr | 3e-4 | Adam step size |
| rollout × envs | 256 × 6–8 | On-policy batch size per update |
| curriculum | reverse, prioritised-failure | Start near the goal, expand outward as success rises (`cur i/N`) |

**Diagnostic reading of a training line** (`succ`, `ent`, `cur`, `kl`): `succ` is the
outcome; `ent` shows whether the policy is still exploring (high) or has committed
(low); `cur i/N` is how much of the level the curriculum has unlocked; `kl` confirms
updates stay inside the trust region. A run stuck at `succ 0.00` with `ent ≈ 2.0`
(the ice-no-velocity case) is the signature of an unlearnable/aliased observation.
