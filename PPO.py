#!/usr/bin/env python
"""
From-scratch PPO (discrete actions), CUDA-enabled.

Implements the pieces that most commonly break in handwritten PPO:
  * GAE with correct terminal vs. truncation bootstrap masking
  * old log-probs stored detached from the graph (used in the ratio)
  * K epochs * minibatches over each rollout (the whole point of PPO)
  * per-batch advantage normalisation
  * entropy bonus (critical for hard-exploration games like Jump King)
  * optional value-function clipping and global grad-norm clipping

The policy head is a softmax (Categorical). To switch to a continuous
charge later, only ActorCritic's head + the dist/log_prob calls change;
the buffer, GAE, and update loop stay identical.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """Shared trunk -> policy + value heads.

    Two input modes:
      * MLP (grid_shape=None): obs is a flat scalar vector (legacy behaviour).
      * CNN (grid_shape=(C,H,W)): obs is a flat vector that PACKS a flattened
        occupancy grid (first C*H*W entries) followed by `n_scalars` scalars.
        We unpack, run a small conv stack over the grid, and concat the scalars
        before the dense trunk. Packing keeps the env/buffer/vec-env pipeline
        identical to the scalar agent -- only this class knows about the grid.

    Two optional knobs, both OFF by default so every existing checkpoint builds
    the identical module and loads unchanged:
      * extra_conv    -- one more stride-2 conv, shrinking the flattened grid
                         feature (45x60 grid: 32*6*8=1536 -> 32*3*4=384).
      * scalar_embed  -- project the scalars through a small MLP before the
                         concat instead of appending them raw.
    They exist for the same reason: with a raw concat the scalars are a handful
    of numbers against a 1536-wide grid feature, so the king's velocity -- the
    ONLY thing that distinguishes two icy states at the same position -- has
    almost no share of the trunk's input. Together they take that share from
    ~0.4% to ~14%.
    """
    def __init__(self, obs_dim, num_actions, hidden=128, grid_shape=None, n_scalars=0,
                 extra_conv=False, scalar_embed=0, explore_eps=0.0):
        super().__init__()
        # Floor on every action's probability, mixed into the policy ITSELF so
        # sampling and evaluation use the same distribution and PPO stays
        # on-policy. The entropy coefficient only controls AVERAGE entropy: a
        # policy can sit at entropy 0.8 overall and still be at p=1.00 on the
        # one state that matters, which is exactly what stalled L36 rung 5 --
        # 3 of 38 actions there hand off to a win and the policy sampled none
        # of them. argmax is unchanged, so greedy evaluation and the ladder
        # gate are unaffected.
        self.explore_eps = float(explore_eps)
        self.grid_shape = grid_shape
        self.n_scalars = int(n_scalars)
        self.extra_conv = bool(extra_conv)
        self.scalar_embed = int(scalar_embed or 0)
        if grid_shape is not None:
            C, H, W = grid_shape
            self.grid_flat = C * H * W
            layers = [nn.Conv2d(C, 16, 3, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()]
            if self.extra_conv:
                layers += [nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU()]
            layers.append(nn.Flatten())
            self.conv = nn.Sequential(*layers)
            with torch.no_grad():
                conv_out = self.conv(torch.zeros(1, C, H, W)).shape[1]
            if self.scalar_embed and self.n_scalars:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(self.n_scalars, self.scalar_embed), nn.Tanh())
                scal_out = self.scalar_embed
            else:
                self.scalar_mlp = None
                scal_out = self.n_scalars
            trunk_in = conv_out + scal_out
        else:
            self.grid_flat = 0
            self.conv = None
            self.scalar_mlp = None
            trunk_in = obs_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)
        self.apply(self._init)
        # Small policy-head init keeps the initial policy near-uniform.
        nn.init.orthogonal_(self.policy_head.weight, 0.01)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _features(self, x):
        if self.conv is None:
            return self.trunk(x)
        C, H, W = self.grid_shape
        grid = x[:, :self.grid_flat].reshape(-1, C, H, W)
        scal = x[:, self.grid_flat:]
        if self.scalar_mlp is not None:
            scal = self.scalar_mlp(scal)
        h = self.conv(grid)
        return self.trunk(torch.cat([h, scal], dim=1))

    def forward(self, x):
        h = self._features(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def _dist(self, logits):
        if self.explore_eps <= 0.0:
            return Categorical(logits=logits)
        p = torch.softmax(logits, dim=-1)
        n = logits.shape[-1]
        return Categorical(probs=(1.0 - self.explore_eps) * p + self.explore_eps / n)

    @torch.no_grad()
    def act(self, obs):
        """obs: (N, obs_dim) tensor -> action, logprob, value (all (N,))."""
        logits, value = self.forward(obs)
        dist = self._dist(logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = self._dist(logits)
        return dist.log_prob(actions), dist.entropy(), value


class RolloutBuffer:
    """Stores a (T, N) rollout from N parallel envs and computes GAE."""

    def __init__(self, T, N, obs_dim, device):
        self.T, self.N, self.device = T, N, device
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, N, device=device)
        self.rewards = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, N, device=device)
        self.terminated = torch.zeros(T, N, device=device)   # real episode end
        self.truncated = torch.zeros(T, N, device=device)    # time-limit cut
        self.t = 0

    def add(self, obs, action, logprob, reward, value, terminated, truncated):
        i = self.t
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.values[i] = value
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.t += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """Generalised Advantage Estimation.

        Key subtlety: at a *terminated* step there is no future, so we zero the
        bootstrap (next value). At a *truncated* step the episode was cut by the
        time limit but physically continues, so we DO bootstrap from the value
        of the state we were cut at. We approximate the latter using the stored
        value at that step (standard practical handling)."""
        adv = torch.zeros(self.T, self.N, device=self.device)
        last_gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            done = self.terminated[t]                  # only true termination kills bootstrap
            nonterminal = 1.0 - done

            # If the step was truncated (not terminated), treat the transition
            # as ending here for GAE purposes (don't propagate across the cut),
            # but still bootstrap from this step's own value via next_value.
            cut = torch.clamp(self.terminated[t] + self.truncated[t], max=1.0)

            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * lam * (1.0 - cut) * last_gae
            adv[t] = last_gae

        returns = adv + self.values
        return adv, returns


class PPO:
    def __init__(self, obs_dim, num_actions, device=None,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, minibatches=4,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 clip_value=True, hidden=128,
                 grid_shape=None, n_scalars=0,
                 extra_conv=False, scalar_embed=0, explore_eps=0.0,
                 grad_split=False):
        self.device = device or get_device()
        self.net = ActorCritic(obs_dim, num_actions, hidden,
                               grid_shape=grid_shape, n_scalars=n_scalars,
                               extra_conv=extra_conv,
                               scalar_embed=scalar_embed,
                               explore_eps=explore_eps).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm, self.clip_value = max_grad_norm, clip_value
        # CLIP THE POLICY AND VALUE GRADIENTS SEPARATELY. The trunk is shared,
        # so one global clip_grad_norm_ splits the budget in proportion to the
        # two losses -- and the value loss is a SQUARED return. Measured on the
        # stuck L37 rung 5 (returns spread -51..+48, value loss 485):
        #   ||grad policy_loss||  =   0.55
        #   ||grad 0.5*value_loss|| = 443.6
        # so the global clip to 0.5 scaled everything by 0.0011 and the policy
        # moved at ~1/800 of its nominal learning rate -- exactly the frozen
        # approx_kl ~ 5e-4 and the entropy pinned near log(38) seen for hundreds
        # of updates. Clipping the two contributions to max_grad_norm each and
        # summing bounds them equally, so the reward scale can no longer decide
        # how fast the policy learns.
        self.grad_split = bool(grad_split)

    def update(self, buf: RolloutBuffer, last_value):
        adv, returns = buf.compute_gae(last_value, self.gamma, self.lam)

        # Flatten (T, N, ...) -> (T*N, ...)
        b_obs = buf.obs.reshape(-1, buf.obs.shape[-1])
        b_act = buf.actions.reshape(-1)
        b_logp = buf.logprobs.reshape(-1)
        b_val = buf.values.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)

        n = b_obs.shape[0]
        mb_size = n // self.minibatches
        idx = np.arange(n)

        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clipfrac": 0.0}
        updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                if len(mb) == 0:
                    continue
                mb = torch.as_tensor(mb, device=self.device)

                # advantage normalisation (per minibatch)
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_logp, entropy, new_val = self.net.evaluate(b_obs[mb], b_act[mb])
                ratio = torch.exp(new_logp - b_logp[mb])     # old logp is detached (no_grad at collection)

                # clipped surrogate policy loss
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value loss (optionally clipped)
                if self.clip_value:
                    v_clipped = b_val[mb] + torch.clamp(
                        new_val - b_val[mb], -self.clip, self.clip)
                    v_loss = torch.max((new_val - b_ret[mb]) ** 2,
                                       (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = ((new_val - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                if self.grad_split:
                    self.opt.zero_grad()
                    (policy_loss - self.ent_coef * entropy_loss).backward(
                        retain_graph=True)
                    nn.utils.clip_grad_norm_(self.net.parameters(),
                                             self.max_grad_norm)
                    pol_grad = [None if p.grad is None else p.grad.detach().clone()
                                for p in self.net.parameters()]
                    self.opt.zero_grad()
                    (self.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(),
                                             self.max_grad_norm)
                    for p, g in zip(self.net.parameters(), pol_grad):
                        if g is None:
                            continue
                        p.grad = g if p.grad is None else p.grad + g
                    self.opt.step()
                else:
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                    self.opt.step()

                with torch.no_grad():
                    approx_kl = (b_logp[mb] - new_logp).mean().item()
                    clipfrac = ((ratio - 1).abs() > self.clip).float().mean().item()
                logs["policy_loss"] += policy_loss.item()
                logs["value_loss"] += v_loss.item()
                logs["entropy"] += entropy_loss.item()
                logs["approx_kl"] += approx_kl
                logs["clipfrac"] += clipfrac
                updates += 1

        for k in logs:
            logs[k] /= max(updates, 1)
        return logs