# -*- coding: utf-8 -*-
"""Render a hand-designed level from levels/level.json so you can see it.

    python LevelView.py                 # shows levels/level.json
    python LevelView.py --file levels/level2.json

Keys:  R = reload from disk    G = toggle coordinate grid    ESC = quit

Level format (level coordinates, origin top-left):
    {"name": "...", "size": [W, H],
     "platforms": [[x, y, w, h], ...],   black rectangles in the drawing
     "king": [x, y],       red  -> start (feet point)
     "princess": [x, y]}   green -> goal  (feet point)
"""
import argparse, json, os, pygame

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="levels/level.json")
ap.add_argument("--max", type=int, default=760, help="max window height in px")
args = ap.parse_args()

STONE=(120,114,106); STONE_T=(162,156,146); STONE_B=(76,70,64); SEAM=(94,88,82)
KING=(196,54,48); KING_D=(150,38,34); CAPE=(120,28,26); CROWN=(242,204,78)
SKIN=(234,200,164); EYE=(38,38,52)
PDRESS=(60,170,90); PDRESS_D=(40,130,66); PHAIR=(232,196,120); HUD=(238,238,244); MUT=(150,155,168); GRID=(255,255,255,26)

def load():
    with open(args.file, encoding="utf-8") as f: d = json.load(f)
    if isinstance(d, dict) and d.get("format") == "jumpking-world":
        raise SystemExit(
            f"{args.file} is a jumpking-world file (a level SEQUENCE for the "
            f"real engine), not this viewer's single-screen format. "
            f"Use:  python LevelEditor.py {args.file}")
    return d

pygame.init()
LV = load()
W, H = LV.get("size", [480, 360])
SC = min(args.max / H, 1400 / W)
SW, SH = int(W * SC), int(H * SC)
screen = pygame.display.set_mode((SW, SH))
pygame.display.set_caption("Jump King — приказ нивоа")
clock = pygame.time.Clock()
def font(sz, b=True): return pygame.font.SysFont("consolas,couriernew,monospace", int(sz), bold=b)
FS, FM = font(15), font(20)
def sx(v): return int(v * SC)

def make_bg():
    bg = pygame.Surface((SW, SH)); top, bot = (92,108,140), (30,34,52)
    for yy in range(SH):
        t = yy / max(SH,1); bg.fill(tuple(int(top[i]+(bot[i]-top[i])*t) for i in range(3)), (0, yy, SW, 1))
    br = pygame.Surface((SW, SH), pygame.SRCALPHA); bh, bw = sx(22), sx(40)
    for r, yy in enumerate(range(0, SH, bh)):
        off = (bw // 2) if r % 2 else 0
        pygame.draw.line(br, (255,255,255,10), (0, yy), (SW, yy))
        for xx in range(-bw, SW, bw): pygame.draw.line(br, (0,0,0,16), (xx+off, yy), (xx+off, yy+bh))
    bg.blit(br, (0, 0)); return bg
BG = make_bg()

def draw_platform(x, y, w, h):
    X, Y, Wd, Ht = sx(x), sx(y), sx(w), max(sx(h), 6)
    pygame.draw.rect(screen, STONE_B, (X, Y+2, Wd, Ht+3))
    pygame.draw.rect(screen, STONE, (X, Y, Wd, Ht))
    pygame.draw.rect(screen, STONE_T, (X, Y, Wd, max(3, sx(3))))
    for bx in range(X+sx(10), X+Wd-sx(4), sx(20)): pygame.draw.line(screen, SEAM, (bx, Y), (bx, Y+Ht), 1)

def draw_king(x, y):
    cx, cy = sx(x), sx(y); u = max(2, sx(1.6))
    pygame.draw.polygon(screen, CAPE, [(cx-5*u, cy-16*u),(cx-9*u, cy-2*u),(cx-2*u, cy-6*u)])
    pygame.draw.rect(screen, KING_D, (cx-5*u, cy-15*u, 10*u, 13*u), border_radius=max(1,u))
    pygame.draw.rect(screen, KING, (cx-5*u, cy-15*u, 10*u, 8*u), border_radius=max(1,u))
    pygame.draw.circle(screen, SKIN, (cx, cy-18*u), 5*u)
    pygame.draw.rect(screen, EYE, (cx+u, cy-19*u, 2*u, 2*u))
    pts=[(cx-5*u,cy-22*u),(cx-3*u,cy-26*u),(cx-u,cy-22*u),(cx+u,cy-26*u),(cx+3*u,cy-22*u),(cx+5*u,cy-26*u),(cx+5*u,cy-21*u),(cx-5*u,cy-21*u)]
    pygame.draw.polygon(screen, CROWN, pts)

def draw_princess(x, y):
    cx, cy = sx(x), sx(y); u = max(2, sx(1.6))
    pygame.draw.polygon(screen, PDRESS_D, [(cx-6*u, cy),(cx+6*u, cy),(cx+3*u, cy-12*u),(cx-3*u, cy-12*u)])
    pygame.draw.polygon(screen, PDRESS, [(cx-6*u, cy),(cx+5*u, cy-1*u),(cx+2*u, cy-11*u),(cx-3*u, cy-12*u)])
    pygame.draw.circle(screen, SKIN, (cx, cy-16*u), 4*u)
    pygame.draw.rect(screen, PHAIR, (cx-4*u, cy-20*u, 8*u, 4*u), border_radius=u)
    pygame.draw.rect(screen, EYE, (cx-2*u, cy-16*u, 2*u, 2*u)); pygame.draw.rect(screen, EYE, (cx+u, cy-16*u, 2*u, 2*u))

def draw(grid):
    global LV, W, H
    screen.blit(BG, (0, 0))
    if grid:
        for gx in range(0, W+1, 40):
            pygame.draw.line(screen, (255,255,255), (sx(gx),0),(sx(gx),SH), 1)
            screen.blit(FS.render(str(gx), True, MUT), (sx(gx)+2, 2))
        for gy in range(0, H+1, 40):
            pygame.draw.line(screen, (255,255,255), (0,sx(gy)),(SW,sx(gy)), 1)
            screen.blit(FS.render(str(gy), True, MUT), (2, sx(gy)+2))
    for dy in LV.get("dividers", []):
        ys = sx(dy)
        for xx in range(0, SW, sx(10)): pygame.draw.line(screen, (120, 130, 150), (xx, ys), (xx + sx(5), ys), 1)
    for p in LV["platforms"]: draw_platform(*p)
    if "princess" in LV: draw_princess(*LV["princess"])
    if "king" in LV: draw_king(*LV["king"])
    bar = pygame.Surface((SW, 26), pygame.SRCALPHA); bar.fill((0,0,0,130)); screen.blit(bar,(0,0))
    screen.blit(FM.render(LV.get("name","ниво"), True, HUD), (10, 3))
    screen.blit(FS.render("R:освежи  G:мрежа  ESC:излаз", True, MUT), (SW-260, 6))
    pygame.display.flip()

def run():
    global LV
    grid = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: pygame.quit(); return
                if e.key == pygame.K_g: grid = not grid
                if e.key == pygame.K_r:
                    try: LV = load()
                    except Exception as ex: print("reload error:", ex)
        draw(grid); clock.tick(30)

if __name__ == "__main__":
    run()
