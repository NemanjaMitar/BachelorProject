# -*- coding: utf-8 -*-
"""
Real-time Neural Network Visualizer for PPO playback (adapted from the early-
project panel to the current CNN architecture).

Shows, per decision point:
  • top action probabilities with readable macro-action names (selected ringed)
  • the scalar part of the observation (x, y, level, altitude, wind, velocity)
  • critic value V(s)
  • activation heatmaps: conv features (1536), trunk H1 (128), trunk H2 (128)
  • rolling episode-return graph + live stats
"""
import pygame
import numpy as np

ARROW = {"left": "<", "right": ">", "up": "^"}


def short_name(a):
    """Compact readable label for one macro-action table entry."""
    kind = a[0]
    if kind == "jump":
        return f"J{ARROW.get(a[1], '?')}{a[2]:02d}"
    if kind == "walk":
        return f"Hod{ARROW.get(a[1], '?')}{a[2]}"
    if kind == "wait":
        return f"Cek{a[2]}"
    if kind == "wait_wind":
        return f"CekV{a[1]}"
    if kind == "jump_wind":
        b, d = a[1]
        return f"V{b}{ARROW.get(d, '?')}{a[2]:02d}"
    if kind == "approach_jump":
        tx, b, d = a[1]
        return f"Pr{tx}{ARROW.get(d, '?')}"
    if kind == "wind_combo":
        return f"Kombo{a[1]}"
    return str(a)[:8]


def action_color(a):
    kind = a[0]
    if kind == "jump":
        return {"left": (52, 152, 219), "up": (155, 89, 182),
                "right": (46, 204, 113)}.get(a[1], (149, 165, 166))
    if kind == "walk":
        return (149, 165, 166)
    if kind in ("wait", "wait_wind"):
        return (241, 196, 15)
    if kind in ("jump_wind", "approach_jump"):
        return (26, 188, 156)
    if kind == "wind_combo":
        return (230, 126, 34)
    return (120, 120, 140)


class NNVisualizer:
    """Pygame side-panel that visualises PPO internals in real time."""

    PANEL_WIDTH = 400

    BG          = (18, 18, 24)
    BG_SECTION  = (28, 28, 38)
    TEXT        = (200, 200, 210)
    TEXT_DIM    = (120, 120, 140)
    ACCENT      = (80, 160, 255)
    POS_COLOR   = (46, 204, 113)
    NEG_COLOR   = (231, 76, 60)

    STATE_NAMES = ["X", "Y", "Nivo", "Vis", "V sin", "V cos", "vx", "vy"]

    def __init__(self, surface, x_offset, height):
        self.surface = surface
        self.x = x_offset
        self.w = self.PANEL_WIDTH
        self.h = height
        try:
            self.font_title = pygame.font.SysFont("consolas", 18, bold=True)
            self.font       = pygame.font.SysFont("consolas", 13)
            self.font_small = pygame.font.SysFont("consolas", 11)
        except Exception:
            self.font_title = pygame.font.Font(None, 22)
            self.font       = pygame.font.Font(None, 16)
            self.font_small = pygame.font.Font(None, 14)

        self.action_probs    = np.array([1.0])
        self.actions         = [("jump", "up", 4)]
        self.selected_action = 0
        self.exec_label      = None     # live primitive being executed (macros)
        self.state           = [0.0] * 8
        self.state_mask      = None     # per-scalar: True = model sees it
        self.value           = 0.0
        self.reward          = 0.0
        self.episode         = 0
        self.ep_reward       = 0.0
        self.altitude        = 0.0
        self.level           = 0
        self.model_name      = ""

        self.conv_out = None      # (1536,)
        self.actor_h1 = None      # (128,)
        self.actor_h2 = None      # (128,)

        self._reward_history  = []
        self._max_history_len = 200

    # ── public API ────────────────────────────────────────────────
    def update(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def add_episode_reward(self, reward):
        self._reward_history.append(reward)
        if len(self._reward_history) > self._max_history_len:
            self._reward_history.pop(0)

    def render(self):
        pygame.draw.rect(self.surface, self.BG, (self.x, 0, self.w, self.h))
        pygame.draw.line(self.surface, self.ACCENT,
                         (self.x, 0), (self.x, self.h), 2)
        y = 5
        y = self._draw_title(y)
        y = self._draw_action_probs(y)
        y = self._draw_state(y)
        y = self._draw_activations(y)
        y = self._draw_reward_graph(y)
        self._draw_stats(y)

    # ── sections ──────────────────────────────────────────────────
    def _draw_title(self, y):
        t = self.font_title.render("PPO Network Monitor", True, self.ACCENT)
        self.surface.blit(t, (self.x + 10, y))
        if self.model_name:
            m = self.font_small.render(self.model_name[-40:], True, self.TEXT_DIM)
            self.surface.blit(m, (self.x + 10, y + 20))
            y += 12
        y += 25
        pygame.draw.line(self.surface, (50, 50, 70),
                         (self.x + 10, y), (self.x + self.w - 10, y))
        return y + 5

    def _draw_action_probs(self, y):
        n = len(self.action_probs)
        hdr = self.font.render(f"Akcije (top 8 od {n})", True, self.TEXT)
        self.surface.blit(hdr, (self.x + 10, y))
        y += 18

        order = list(np.argsort(self.action_probs)[::-1][:8])
        if self.selected_action not in order:
            order[-1] = self.selected_action

        bar_w = self.w - 130
        bar_h = 14
        for i in order:
            prob = float(self.action_probs[i])
            a = self.actions[i] if i < len(self.actions) else ("?",)
            name = short_name(a)
            if i == self.selected_action and self.exec_label:
                name = self.exec_label          # live primitive inside a macro
            color = action_color(a)

            lbl = self.font_small.render(f"{name:>7s}", True, self.TEXT_DIM)
            self.surface.blit(lbl, (self.x + 10, y + 1))

            bx = self.x + 78
            pygame.draw.rect(self.surface, self.BG_SECTION, (bx, y, bar_w, bar_h))
            fill = int(bar_w * min(prob, 1.0))
            c = color if i == self.selected_action else tuple(v // 2 for v in color)
            pygame.draw.rect(self.surface, c, (bx, y, fill, bar_h))
            if i == self.selected_action:
                pygame.draw.rect(self.surface, (255, 255, 255),
                                 (bx, y, max(fill, 2), bar_h), 1)
            vt = self.font_small.render(f"{prob:.3f}", True, self.TEXT)
            self.surface.blit(vt, (bx + bar_w + 5, y + 1))
            y += bar_h + 3
        return y + 5

    def _draw_state(self, y):
        hdr = self.font.render("Opservacija (skalari)", True, self.TEXT)
        self.surface.blit(hdr, (self.x + 10, y))
        y += 18

        bar_w = self.w - 150
        bar_h = 12
        names = self.STATE_NAMES[:len(self.state)]
        mask = self.state_mask or [True] * len(self.state)
        for (val, name, seen) in zip(self.state, names, mask):
            lbl = self.font_small.render(f"{name:>6s}", True,
                                         self.TEXT_DIM if seen else (70, 70, 88))
            self.surface.blit(lbl, (self.x + 10, y))
            bx = self.x + 75
            pygame.draw.rect(self.surface, self.BG_SECTION, (bx, y, bar_w, bar_h))
            center = bx + bar_w // 2
            fv = float(val)
            fw = int((bar_w / 2) * min(abs(fv), 1.0))
            col = self.POS_COLOR if fv >= 0 else self.NEG_COLOR
            if not seen:
                col = tuple(v // 3 for v in col)
            xx = center if fv >= 0 else center - fw
            pygame.draw.rect(self.surface, col, (xx, y, fw, bar_h))
            pygame.draw.line(self.surface, (80, 80, 100),
                             (center, y), (center, y + bar_h))
            vt = self.font_small.render(f"{fv:+.3f}", True,
                                        self.TEXT if seen else (90, 90, 108))
            self.surface.blit(vt, (bx + bar_w + 5, y))
            y += bar_h + 2

        y += 3
        vc = self.POS_COLOR if self.value >= 0 else self.NEG_COLOR
        vt = self.font.render(f"V(s) = {self.value:+.3f}", True, vc)
        self.surface.blit(vt, (self.x + 10, y))
        return y + 20

    def _draw_activations(self, y):
        hdr = self.font.render("Aktivacije slojeva", True, self.TEXT)
        self.surface.blit(hdr, (self.x + 10, y))
        y += 16
        vis_w = self.w - 30
        for label, acts in [("Conv 1536", self.conv_out),
                            ("FC H1 128", self.actor_h1),
                            ("FC H2 128", self.actor_h2)]:
            lbl = self.font_small.render(label, True, self.TEXT_DIM)
            self.surface.blit(lbl, (self.x + 10, y))
            y += 13
            if acts is not None and len(acts) > 0:
                acts = np.asarray(acts).ravel()
                if len(acts) > vis_w:
                    step = len(acts) / vis_w
                    sampled = acts[(np.arange(vis_w) * step).astype(int)]
                    cell = 1
                else:
                    sampled = acts
                    cell = max(1, vis_w // len(acts))
                mx = max(abs(float(sampled.max())), abs(float(sampled.min())), 1e-8)
                for j, a in enumerate(sampled):
                    nrm = float(a) / mx
                    if nrm > 0:
                        c = (0, min(int(nrm * 200), 200), min(int(nrm * 255), 255))
                    else:
                        c = (min(int(-nrm * 255), 255), 0, 0)
                    pygame.draw.rect(self.surface, c,
                                     (self.x + 15 + j * cell, y, cell, 8))
            else:
                pygame.draw.rect(self.surface, self.BG_SECTION,
                                 (self.x + 15, y, vis_w, 8))
            y += 12
        return y + 3

    def _draw_reward_graph(self, y):
        hdr = self.font.render("Dobit po epizodi", True, self.TEXT)
        self.surface.blit(hdr, (self.x + 10, y))
        y += 16
        gw = self.w - 30
        gh = max(60, min(150, self.h - y - 75))
        gx = self.x + 15
        pygame.draw.rect(self.surface, self.BG_SECTION, (gx, y, gw, gh))
        pygame.draw.rect(self.surface, (50, 50, 70), (gx, y, gw, gh), 1)
        rr = self._reward_history
        if len(rr) > 1:
            mn, mx = min(rr), max(rr)
            span = mx - mn if mx != mn else 1.0
            if mn < 0 < mx:
                zy = y + gh - int((0 - mn) / span * gh)
                pygame.draw.line(self.surface, (80, 80, 100),
                                 (gx, zy), (gx + gw, zy), 1)
            pts = []
            for i, r in enumerate(rr):
                px = gx + int(i / max(len(rr) - 1, 1) * gw)
                py = y + gh - int((r - mn) / span * (gh - 4)) - 2
                pts.append((px, py))
            pygame.draw.lines(self.surface, self.ACCENT, False, pts, 2)
            pygame.draw.circle(self.surface, (255, 255, 255), pts[-1], 3)
            self.surface.blit(self.font_small.render(f"{mx:.1f}", True, self.TEXT_DIM),
                              (gx + 2, y + 2))
            self.surface.blit(self.font_small.render(f"{mn:.1f}", True, self.TEXT_DIM),
                              (gx + 2, y + gh - 13))
        else:
            t = self.font_small.render("Prikupljanje...", True, self.TEXT_DIM)
            self.surface.blit(t, (gx + gw // 2 - 45, y + gh // 2 - 6))
        return y + gh + 5

    def _draw_stats(self, y):
        pygame.draw.line(self.surface, (50, 50, 70),
                         (self.x + 10, y), (self.x + self.w - 10, y))
        y += 5
        sel = self.actions[self.selected_action] \
            if self.selected_action < len(self.actions) else ("?",)
        sel_name = self.exec_label or short_name(sel)
        for s in [
            f"Epizoda: {self.episode}   Akcija: {sel_name}",
            f"Dobit ep: {self.ep_reward:+.2f}  |  korak: {self.reward:+.2f}",
            f"Visina: {self.altitude:.0f}  |  Nivo: {self.level}",
        ]:
            txt = self.font_small.render(s, True, self.TEXT)
            self.surface.blit(txt, (self.x + 10, y))
            y += 14
