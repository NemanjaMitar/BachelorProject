#!/usr/bin/env python
"""
Visual editor for custom Jump King worlds (see CustomWorld.py for the format).

A world is a STACK of screens: screen 0 is the bottom, and the king climbs into
higher indices by leaving through the top of the screen -- so the editor shows
the stack top-down in the screen list, the way you would look at the mountain.

    python LevelEditor.py                     # start a new world
    python LevelEditor.py levels/tower.json   # edit an existing one

Mouse
    drag from a BLOCK swatch onto the canvas ... drop a new block there
    drag on empty canvas ....................... paint a block of the active type
    click a block .............................. select it
    drag a selected block ...................... move it
    drag its bottom-right corner ............... resize it
    right-click a block ........................ delete it

Keys
    1 / 2 / 3 .. land / ice / snow      W .. toggle wind on this screen
    [ / ] ...... previous / next screen  G .. toggle 8px snap
    A .......... add a screen above      D .. delete this screen
    Del ........ delete selection        Ctrl+Z .. undo
    Ctrl+S ..... save                    Ctrl+O .. open      Ctrl+N .. new
    Esc ........ deselect, then quit
"""

import os
import sys
import copy
import argparse

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # the editor makes no sound

import pygame

import CustomWorld as CW

# ---------------------------------------------------------------- appearance
BG        = (24, 26, 32)
PANEL     = (34, 37, 46)
PANEL_HI  = (46, 50, 62)
CANVAS_BG = (16, 18, 22)
GRID      = (38, 42, 52)
TEXT      = (222, 226, 234)
TEXT_DIM  = (140, 148, 162)
ACCENT    = (108, 170, 255)
WARN      = (240, 176, 84)
DANGER    = (232, 104, 104)
OK        = (126, 208, 140)

BLOCK_COL = {"land": (108, 122, 96), "ice": (128, 196, 232), "snow": (226, 232, 240)}
BLOCK_EDGE = {"land": (152, 172, 132), "ice": (176, 226, 250), "snow": (255, 255, 255)}

PANEL_W = 232
TOPBAR_H = 34
PAD = 8
MIN_SIZE = 6            # smallest block you can paint, in game pixels
HANDLE = 10             # resize handle, in screen pixels


class Button:
    def __init__(self, rect, label, action, tip="", toggle=None, danger=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
        self.tip = tip
        self.toggle = toggle          # callable -> bool, renders an on/off pill
        self.danger = danger
        self.hot = False

    def draw(self, surf, font):
        on = self.toggle() if self.toggle else False
        if on:
            bg = ACCENT
        elif self.hot:
            bg = PANEL_HI
        else:
            bg = (44, 48, 59)
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, (60, 66, 80), self.rect, 1, border_radius=4)
        col = (18, 20, 26) if on else (DANGER if self.danger else TEXT)
        t = font.render(self.label, True, col)
        surf.blit(t, t.get_rect(center=self.rect.center))


class Editor:
    def __init__(self, path=None, scale=2):
        self.scale = scale
        self.path = path
        if path and os.path.exists(path):
            self.world = CW.load_world(path)
            started = f"opened {path}"
        elif path:
            # naming a file that does not exist yet starts a new world there,
            # so `LevelEditor.py levels/mine.json` just works and Ctrl+S saves it
            self.world = CW.new_world(os.path.splitext(os.path.basename(path))[0])
            started = f"new world -> {path}"
        else:
            self.world = CW.new_world("untitled")
            started = "new world"
        self.level = 0
        self.kind = "land"
        self.snap = True
        self.selected = None            # index into the current screen's platforms
        self.undo_stack = []
        self.dirty = False
        self.status = started
        self.status_col = TEXT_DIM

        # interaction state
        self.drag_mode = None           # 'paint' | 'move' | 'resize' | 'from_palette'
        self.drag_from = (0, 0)
        self.drag_rect = None
        self.prompt = None              # (title, text, callback) modal text input

        self.canvas = pygame.Rect(PAD, TOPBAR_H + PAD,
                                  CW.SCREEN_W * scale, CW.SCREEN_H * scale)
        w = self.canvas.right + PAD + PANEL_W + PAD
        h = max(self.canvas.bottom + PAD, TOPBAR_H + 640)
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Jump King — level editor")
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 17)
        self.big = pygame.font.Font(None, 24)
        self.buttons = []
        self.palette_rects = {}
        self._build_panel()

    # ------------------------------------------------------------- world data
    @property
    def levels(self):
        return self.world["levels"]

    @property
    def plats(self):
        return self.levels[self.level]["platforms"]

    def push_undo(self):
        self.undo_stack.append((self.level, copy.deepcopy(self.world["levels"])))
        if len(self.undo_stack) > 60:
            self.undo_stack.pop(0)
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            return self.say("nothing to undo", TEXT_DIM)
        lvl, levels = self.undo_stack.pop()
        self.world["levels"] = levels
        self.level = min(lvl, len(levels) - 1)
        self.selected = None
        self.say("undo")

    def say(self, msg, col=TEXT_DIM):
        self.status, self.status_col = msg, col

    # ---------------------------------------------------------------- actions
    def set_kind(self, k):
        self.kind = k
        if self.selected is not None:
            self.push_undo()
            self.plats[self.selected]["type"] = k
            self.say(f"block -> {k}")

    def toggle_wind(self):
        self.push_undo()
        lvl = self.levels[self.level]
        lvl["wind"] = not lvl["wind"]
        self.say(f"wind {'ON' if lvl['wind'] else 'off'} on screen {self.level}",
                 ACCENT if lvl["wind"] else TEXT_DIM)

    def add_level(self):
        self.push_undo()
        self.levels.insert(self.level + 1, CW.new_level())
        self.level += 1
        self.selected = None
        self.say(f"added screen {self.level}")

    def delete_level(self):
        if len(self.levels) <= 1:
            return self.say("a world needs at least one screen", WARN)
        self.push_undo()
        self.levels.pop(self.level)
        self.level = min(self.level, len(self.levels) - 1)
        self.selected = None
        self.say("screen deleted")

    def clear_level(self):
        if not self.plats:
            return
        self.push_undo()
        self.levels[self.level]["platforms"] = []
        self.selected = None
        self.say("screen cleared")

    def go(self, d):
        self.level = max(0, min(len(self.levels) - 1, self.level + d))
        self.selected = None

    def delete_selected(self):
        if self.selected is None:
            return
        self.push_undo()
        self.plats.pop(self.selected)
        self.selected = None
        self.say("block deleted")

    # ------------------------------------------------------------- file stuff
    def do_new(self):
        self.ask("new world name:", "untitled", self._new_named)

    def _new_named(self, name):
        self.world = CW.new_world(name.strip() or "untitled")
        self.path = None
        self.level, self.selected, self.undo_stack = 0, None, []
        self.dirty = False
        self.say("new world")

    def do_open(self):
        self.ask("open path:", self.path or "levels/", self._open_path)

    def _open_path(self, path):
        path = path.strip()
        try:
            self.world = CW.load_world(path)
        except Exception as e:
            return self.say(f"open failed: {e}", DANGER)
        self.path = path
        self.level, self.selected, self.undo_stack = 0, None, []
        self.dirty = False
        self.say(f"opened {path}", OK)

    def do_save(self):
        if self.path:
            return self._save_path(self.path)
        self.ask("save as:", f"levels/{self.world['name']}.json", self._save_path)

    def _save_path(self, path):
        path = path.strip()
        if not path.endswith(".json"):
            path += ".json"
        self.world["name"] = os.path.splitext(os.path.basename(path))[0]
        try:
            CW.save_world(self.world, path)
        except Exception as e:
            return self.say(f"save failed: {e}", DANGER)
        self.path = path
        self.dirty = False
        self.say(f"saved {path}", OK)

    def ask(self, title, initial, cb):
        self.prompt = [title, initial, cb]

    # ------------------------------------------------------------ coordinates
    def to_game(self, pos):
        return ((pos[0] - self.canvas.x) / self.scale,
                (pos[1] - self.canvas.y) / self.scale)

    def to_screen(self, x, y):
        return (self.canvas.x + x * self.scale, self.canvas.y + y * self.scale)

    def snapv(self, v):
        return round(v / 8) * 8 if self.snap else round(v)

    def plat_rect(self, p):
        return pygame.Rect(*self.to_screen(p["x"], p["y"]),
                           max(1, p["w"] * self.scale), max(1, p["h"] * self.scale))

    def hit(self, pos):
        """Topmost platform under the cursor (last drawn wins)."""
        for i in range(len(self.plats) - 1, -1, -1):
            if self.plat_rect(self.plats[i]).collidepoint(pos):
                return i
        return None

    def add_block(self, x, y, w, h, kind=None):
        p = {"x": int(self.snapv(x)), "y": int(self.snapv(y)),
             "w": max(MIN_SIZE, int(self.snapv(w))), "h": max(MIN_SIZE, int(self.snapv(h))),
             "type": kind or self.kind}
        self.plats.append(p)
        self.selected = len(self.plats) - 1
        return p

    # ----------------------------------------------------------------- events
    def handle(self, e):
        if self.prompt is not None:
            return self._handle_prompt(e)

        if e.type == pygame.MOUSEMOTION:
            for b in self.buttons:
                b.hot = b.rect.collidepoint(e.pos)
            self._drag_motion(e.pos)

        elif e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1:
                self._mouse_down(e.pos)
            elif e.button == 3 and self.canvas.collidepoint(e.pos):
                i = self.hit(e.pos)
                if i is not None:
                    self.push_undo()
                    self.plats.pop(i)
                    self.selected = None
                    self.say("block deleted")

        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self._mouse_up(e.pos)

        elif e.type == pygame.KEYDOWN:
            self._key(e)

    def _mouse_down(self, pos):
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                b.action()
                return
        for kind, r in self.palette_rects.items():
            if r.collidepoint(pos):
                self.kind = kind
                self.drag_mode = "from_palette"
                self.drag_from = pos
                return
        if not self.canvas.collidepoint(pos):
            return

        # resize handle of the current selection?
        if self.selected is not None:
            r = self.plat_rect(self.plats[self.selected])
            grip = pygame.Rect(r.right - HANDLE, r.bottom - HANDLE, HANDLE * 2, HANDLE * 2)
            if grip.collidepoint(pos):
                self.push_undo()
                self.drag_mode, self.drag_from = "resize", pos
                return

        i = self.hit(pos)
        if i is not None:
            self.selected = i
            self.push_undo()
            gx, gy = self.to_game(pos)
            p = self.plats[i]
            self.drag_mode = "move"
            self.drag_from = (gx - p["x"], gy - p["y"])
        else:
            self.selected = None
            self.drag_mode = "paint"
            self.drag_from = pos
            self.drag_rect = pygame.Rect(pos, (0, 0))

    def _drag_motion(self, pos):
        if self.drag_mode == "paint":
            x0, y0 = self.drag_from
            self.drag_rect = pygame.Rect(min(x0, pos[0]), min(y0, pos[1]),
                                         abs(pos[0] - x0), abs(pos[1] - y0))
        elif self.drag_mode == "move" and self.selected is not None:
            gx, gy = self.to_game(pos)
            p = self.plats[self.selected]
            p["x"] = int(self.snapv(gx - self.drag_from[0]))
            p["y"] = int(self.snapv(gy - self.drag_from[1]))
            self.dirty = True
        elif self.drag_mode == "resize" and self.selected is not None:
            gx, gy = self.to_game(pos)
            p = self.plats[self.selected]
            p["w"] = max(MIN_SIZE, int(self.snapv(gx - p["x"])))
            p["h"] = max(MIN_SIZE, int(self.snapv(gy - p["y"])))
            self.dirty = True

    def _mouse_up(self, pos):
        mode, self.drag_mode = self.drag_mode, None
        if mode == "paint" and self.drag_rect:
            r = self.drag_rect
            self.drag_rect = None
            if r.width > 4 and r.height > 4:
                gx, gy = self.to_game(r.topleft)
                self.push_undo()
                self.add_block(gx, gy, r.width / self.scale, r.height / self.scale)
                self.say(f"painted {self.kind}")
        elif mode == "from_palette":
            if self.canvas.collidepoint(pos):
                gx, gy = self.to_game(pos)
                self.push_undo()
                self.add_block(gx - 32, gy - 8, 64, 16)
                self.say(f"placed {self.kind}")
        self.drag_rect = None

    def _key(self, e):
        ctrl = e.mod & pygame.KMOD_CTRL
        if ctrl and e.key == pygame.K_z:
            self.undo()
        elif ctrl and e.key == pygame.K_s:
            self.do_save()
        elif ctrl and e.key == pygame.K_o:
            self.do_open()
        elif ctrl and e.key == pygame.K_n:
            self.do_new()
        elif e.key in (pygame.K_1, pygame.K_KP1):
            self.set_kind("land")
        elif e.key in (pygame.K_2, pygame.K_KP2):
            self.set_kind("ice")
        elif e.key in (pygame.K_3, pygame.K_KP3):
            self.set_kind("snow")
        elif e.key == pygame.K_w:
            self.toggle_wind()
        elif e.key == pygame.K_g:
            self.snap = not self.snap
            self.say(f"snap {'on' if self.snap else 'off'}")
        elif e.key == pygame.K_a:
            self.add_level()
        elif e.key == pygame.K_d:
            self.delete_level()
        elif e.key in (pygame.K_LEFTBRACKET, pygame.K_PAGEDOWN):
            self.go(-1)
        elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_PAGEUP):
            self.go(+1)
        elif e.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            self.delete_selected()

    def _handle_prompt(self, e):
        if e.type != pygame.KEYDOWN:
            return
        if e.key == pygame.K_ESCAPE:
            self.prompt = None
        elif e.key == pygame.K_RETURN:
            title, text, cb = self.prompt
            self.prompt = None
            cb(text)
        elif e.key == pygame.K_BACKSPACE:
            self.prompt[1] = self.prompt[1][:-1]
        elif e.unicode and e.unicode.isprintable():
            self.prompt[1] += e.unicode

    # ------------------------------------------------------------------ panel
    def _build_panel(self):
        x = self.canvas.right + PAD
        w = PANEL_W
        y = TOPBAR_H + PAD
        b = self.buttons

        def row(h=26, gap=6):
            nonlocal y
            r = pygame.Rect(x, y, w, h)
            y += h + gap
            return r

        def half(h=26, gap=6):
            nonlocal y
            a = pygame.Rect(x, y, w // 2 - 3, h)
            c = pygame.Rect(x + w // 2 + 3, y, w // 2 - 3, h)
            y += h + gap
            return a, c

        y += 46                                    # "SCREEN i / n" header
        a, c = half()
        b.append(Button(a, "< lower", lambda: self.go(-1), "[ "))
        b.append(Button(c, "upper >", lambda: self.go(+1), " ]"))
        a, c = half()
        b.append(Button(a, "+ screen", self.add_level))
        b.append(Button(c, "- screen", self.delete_level, danger=True))
        b.append(Button(row(), "wind on this screen", self.toggle_wind,
                        toggle=lambda: self.levels[self.level]["wind"]))
        y += 22                                    # "BLOCKS" header
        pr = row(34)
        third = (pr.width - 12) // 3
        for i, k in enumerate(CW.TYPES):
            self.palette_rects[k] = pygame.Rect(pr.x + i * (third + 6), pr.y, third, pr.height)
        y += 20                                    # "TOOLS" header
        b.append(Button(row(), "snap to 8px grid", lambda: setattr(self, "snap", not self.snap),
                        toggle=lambda: self.snap))
        a, c = half()
        b.append(Button(a, "undo", self.undo))
        b.append(Button(c, "clear", self.clear_level, danger=True))
        y += 20                                    # "FILE" header
        a, c = half()
        b.append(Button(a, "new", self.do_new))
        b.append(Button(c, "open", self.do_open))
        b.append(Button(row(), "save", self.do_save))
        self._panel_x, self._panel_bottom = x, y

    # ----------------------------------------------------------------- render
    def render(self):
        s = self.screen
        s.fill(BG)
        self._draw_topbar(s)
        self._draw_canvas(s)
        self._draw_panel(s)
        if self.prompt:
            self._draw_prompt(s)

    def _draw_topbar(self, s):
        pygame.draw.rect(s, PANEL, (0, 0, s.get_width(), TOPBAR_H))
        name = self.path or f"{self.world['name']} (unsaved)"
        t = self.big.render(f"{name}{' *' if self.dirty else ''}", True, TEXT)
        s.blit(t, (PAD, 7))
        st = self.small.render(self.status, True, self.status_col)
        s.blit(st, (s.get_width() - st.get_width() - PAD, 10))

    def _draw_canvas(self, s):
        pygame.draw.rect(s, CANVAS_BG, self.canvas)
        if self.snap:
            for gx in range(0, CW.SCREEN_W + 1, 8):
                px = self.canvas.x + gx * self.scale
                pygame.draw.line(s, GRID, (px, self.canvas.y), (px, self.canvas.bottom))
            for gy in range(0, CW.SCREEN_H + 1, 8):
                py = self.canvas.y + gy * self.scale
                pygame.draw.line(s, GRID, (self.canvas.x, py), (self.canvas.right, py))

        for i, p in enumerate(self.plats):
            r = self.plat_rect(p)
            pygame.draw.rect(s, BLOCK_COL[p["type"]], r)
            pygame.draw.rect(s, BLOCK_EDGE[p["type"]], r, 1)
            if p["type"] == "ice":
                pygame.draw.line(s, (255, 255, 255), (r.x + 2, r.y + 2), (r.right - 3, r.y + 2))
            if i == self.selected:
                pygame.draw.rect(s, ACCENT, r.inflate(4, 4), 2)
                grip = pygame.Rect(r.right - HANDLE, r.bottom - HANDLE, HANDLE, HANDLE)
                pygame.draw.rect(s, ACCENT, grip)

        if self.drag_rect and self.drag_mode == "paint":
            pygame.draw.rect(s, BLOCK_EDGE[self.kind], self.drag_rect, 2)

        # the seam: leaving through the top puts the king on the next screen up
        top_label = (f"^ up to screen {self.level + 1}"
                     if self.level < len(self.levels) - 1 else "^ TOP (goal)")
        t = self.small.render(top_label, True, ACCENT if self.level < len(self.levels) - 1 else OK)
        s.blit(t, (self.canvas.x + 6, self.canvas.y + 4))
        if self.level > 0:
            t = self.small.render(f"v down to screen {self.level - 1}", True, DANGER)
            s.blit(t, (self.canvas.x + 6, self.canvas.bottom - 18))
        else:
            t = self.small.render("v king spawns here (screen 0 needs a floor)", True, TEXT_DIM)
            s.blit(t, (self.canvas.x + 6, self.canvas.bottom - 18))
            kx, ky = self.to_screen(231, 320)
            pygame.draw.rect(s, WARN, (kx - 4, ky - 8, 8, 16), 2)

        pygame.draw.rect(s, (70, 76, 92), self.canvas, 1)

    def _draw_panel(self, s):
        x, w = self._panel_x, PANEL_W

        def header(text, y):
            t = self.small.render(text, True, TEXT_DIM)
            s.blit(t, (x, y))

        top = TOPBAR_H + PAD
        t = self.big.render(f"SCREEN {self.level}", True, TEXT)
        s.blit(t, (x, top))
        t = self.small.render(f"of {len(self.levels)}   "
                              f"({len(self.plats)} blocks)", True, TEXT_DIM)
        s.blit(t, (x, top + 24))

        header("BLOCKS  (drag onto canvas)", self.palette_rects["land"].y - 18)
        for k, r in self.palette_rects.items():
            pygame.draw.rect(s, BLOCK_COL[k], r, border_radius=3)
            pygame.draw.rect(s, ACCENT if k == self.kind else (60, 66, 80), r,
                             2 if k == self.kind else 1, border_radius=3)
            col = (20, 22, 28) if k != "land" else TEXT
            t = self.small.render(k, True, col)
            s.blit(t, t.get_rect(center=r.center))

        for b in self.buttons:
            b.draw(s, self.font)

        # the stack, drawn the way you look at a mountain: top screen on top
        y = self._panel_bottom + 6
        header("STACK", y); y += 18
        for i in range(len(self.levels) - 1, -1, -1):
            r = pygame.Rect(x, y, w, 19)
            if i == self.level:
                pygame.draw.rect(s, PANEL_HI, r, border_radius=3)
                pygame.draw.rect(s, ACCENT, r, 1, border_radius=3)
            lv = self.levels[i]
            tag = "GOAL" if i == len(self.levels) - 1 else ("start" if i == 0 else "")
            label = f"{i:>2}  {len(lv['platforms']):>2} blk"
            if lv["wind"]:
                label += "  wind"
            t = self.small.render(label, True, TEXT if i == self.level else TEXT_DIM)
            s.blit(t, (r.x + 6, r.y + 3))
            if tag:
                t = self.small.render(tag, True, OK if tag == "GOAL" else TEXT_DIM)
                s.blit(t, (r.right - t.get_width() - 6, r.y + 3))
            y += 21
            if y > s.get_height() - 90:
                break

        warns = []
        try:
            warns = CW.validate(copy.deepcopy(self.world))
        except Exception as e:
            warns = [str(e)]
        y = s.get_height() - 74
        for wmsg in warns[:3]:
            for line in _wrap(wmsg, 34):
                t = self.small.render(line, True, WARN)
                s.blit(t, (x, y)); y += 15

    def _draw_prompt(self, s):
        title, text, _ = self.prompt
        box = pygame.Rect(0, 0, 460, 96)
        box.center = (s.get_width() // 2, s.get_height() // 2)
        pygame.draw.rect(s, PANEL, box.inflate(16, 16), border_radius=6)
        pygame.draw.rect(s, ACCENT, box.inflate(16, 16), 2, border_radius=6)
        t = self.font.render(title, True, TEXT)
        s.blit(t, (box.x + 8, box.y + 8))
        field = pygame.Rect(box.x + 8, box.y + 36, box.width - 16, 28)
        pygame.draw.rect(s, (20, 22, 28), field, border_radius=4)
        t = self.font.render(text + "_", True, TEXT)
        s.blit(t, (field.x + 6, field.y + 6))
        t = self.small.render("Enter = ok    Esc = cancel", True, TEXT_DIM)
        s.blit(t, (box.x + 8, box.bottom - 16))


def _wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None, help="world file to edit")
    ap.add_argument("--scale", type=int, default=2, help="canvas zoom (1-3)")
    args = ap.parse_args()

    pygame.init()
    ed = Editor(args.path, scale=max(1, min(3, args.scale)))
    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                  and ed.prompt is None):
                if ed.selected is not None:
                    ed.selected = None
                else:
                    running = False
            else:
                ed.handle(e)
        ed.render()
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

    if ed.dirty:
        print("NOTE: you quit with unsaved changes.")
    elif ed.path:
        print(f"saved world: {ed.path}")
        print(f"next:\n  python CustomWorld.py --seed {ed.path}\n"
              f"  python Train.py --world {ed.path} "
              f"--start-states {CW.default_pool_path(ed.path)} --curriculum")


if __name__ == "__main__":
    main()
