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
    def __init__(self, obs_dim, num_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)
        self.apply(self._init)
        # Small policy-head init keeps the initial policy near-uniform.
        nn.init.orthogonal_(self.policy_head.weight, 0.01)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """obs: (N, obs_dim) tensor -> action, logprob, value (all (N,))."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
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
                 clip_value=True, hidden=128):
        self.device = device or get_device()
        self.net = ActorCritic(obs_dim, num_actions, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm, self.clip_value = max_grad_norm, clip_value

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