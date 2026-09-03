# -*- coding: utf-8 -*-
"""Jump-King-styled viewer for the procedural level generator + trained PPO agent.

    python RandomView.py            # watch the trained agent climb harder random levels
    python RandomView.py --play     # play the generator yourself
    python RandomView.py --easy     # gentler levels

Keys:  R/N = new level   P = agent / play   ESC = quit
Play:  1-5 = jump strength    LEFT / UP / RIGHT = jump that way
"""
import argparse, importlib.util, os, numpy as np, torch, pygame

spec = importlib.util.spec_from_file_location("RL", os.path.join(os.path.dirname(__file__), "RandomLevel.py"))
RL = importlib.util.module_from_spec(spec); spec.loader.exec_module(RL)
W, H, N, G, VX, VY = RL.W, RL.H, RL.N_PLAT, RL.G, RL.VX_MAX, RL.VY_MAX
CH = [0.35, 0.55, 0.72, 0.86, 1.0]

ap = argparse.ArgumentParser()
ap.add_argument("--play", action="store_true")
ap.add_argument("--easy", action="store_true", help="gentler levels (straight-up jumps)")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--scale", type=float, default=2.0)
args = ap.parse_args()
HARD = not args.easy

ckpath = args.checkpoint or ("checkpoints/experiments/randhard/ppo.pt" if HARD else "checkpoints/experiments/randlevel/ppo.pt")
if not os.path.exists(ckpath):
    alt = "checkpoints/experiments/randlevel/ppo.pt"
    ckpath = alt if os.path.exists(alt) else None
net = None
if ckpath:
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    net = RL.ActorCritic(RL.OBS_DIM, RL.N_ACT); net.load_state_dict(ck["model"]); net.eval()
elif not args.play:
    args.play = True

S = args.scale; SW, SH = int(W * S), int(H * S)
pygame.init()
screen = pygame.display.set_mode((SW, SH)); pygame.display.set_caption("Jump King — процедурални ниво")
clock = pygame.time.Clock()
def font(sz, b=True): return pygame.font.SysFont("consolas,couriernew,monospace", int(sz * S / 2), bold=b)
FS, FM, FB = font(13), font(16), font(24)
def sx(v): return int(v * S)

STONE=(122,116,108); STONE_T=(164,158,148); STONE_B=(78,72,66); SEAM=(96,90,84)
GOLD=(214,178,74); FLAG=(178,54,48); CAPE=(150,42,42); ARMOR=(176,184,196)
ARMOR_D=(120,130,148); SKIN=(232,198,162); CROWN=(242,204,78); EYE=(38,38,52)
HUD=(238,238,244); MUT=(150,155,168)

def make_bg():
    bg = pygame.Surface((SW, SH))
    top, bot = (92, 108, 140), (32, 36, 54)
    for yy in range(SH):
        t = yy / SH
        c = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        pygame.draw.line(bg, c, (0, yy), (SW, yy))
    brick = pygame.Surface((SW, SH), pygame.SRCALPHA)
    bh = sx(22); bw = sx(40)
    for r, yy in enumerate(range(0, SH, bh)):
        off = (bw // 2) if r % 2 else 0
        pygame.draw.line(brick, (255, 255, 255, 10), (0, yy), (SW, yy))
        for xx in range(-bw, SW, bw):
            pygame.draw.line(brick, (0, 0, 0, 16), (xx + off, yy), (xx + off, yy + bh))
    bg.blit(brick, (0, 0))
    return bg
BG = make_bg()

def draw_platform(xl, xr, y, goal=False):
    x0, w = sx(xl), sx(xr - xl); yy = sx(y); th = sx(11)
    body = GOLD if goal else STONE
    pygame.draw.rect(screen, STONE_B, (x0, yy + 2, w, th + 4))
    pygame.draw.rect(screen, body, (x0, yy, w, th))
    pygame.draw.rect(screen, (STONE_T if not goal else (240, 214, 130)), (x0, yy, w, sx(3)))
    for bx in range(x0 + sx(10), x0 + w - sx(4), sx(18)):
        pygame.draw.line(screen, SEAM if not goal else (196, 158, 60), (bx, yy), (bx, yy + th), 1)
    if goal:
        px = x0 + w // 2
        pygame.draw.line(screen, (60, 50, 40), (px, yy), (px, yy - sx(26)), max(2, sx(1.5)))
        pygame.draw.polygon(screen, FLAG, [(px, yy - sx(26)), (px + sx(20), yy - sx(20)), (px, yy - sx(14))])

def draw_king(x, y, face=1):
    cx, cy = sx(x), sx(y)             # feet point
    u = max(2, sx(1))
    pygame.draw.polygon(screen, CAPE, [(cx - 5*u*face, cy - 16*u), (cx - 9*u*face, cy - 2*u), (cx - 2*u*face, cy - 6*u)])
    pygame.draw.rect(screen, ARMOR_D, (cx - 5*u, cy - 15*u, 10*u, 13*u), border_radius=max(1, u))
    pygame.draw.rect(screen, ARMOR, (cx - 5*u, cy - 15*u, 10*u, 8*u), border_radius=max(1, u))
    pygame.draw.rect(screen, ARMOR_D, (cx - 4*u, cy - 2*u, 3*u, 3*u))
    pygame.draw.rect(screen, ARMOR_D, (cx + 1*u, cy - 2*u, 3*u, 3*u))
    pygame.draw.circle(screen, SKIN, (cx, cy - 18*u), 5*u)
    pygame.draw.rect(screen, EYE, (cx + (1 if face>0 else -3)*u, cy - 19*u, 2*u, 2*u))
    pts = [(cx - 5*u, cy - 22*u), (cx - 3*u, cy - 26*u), (cx - u, cy - 22*u),
           (cx + u, cy - 26*u), (cx + 3*u, cy - 22*u), (cx + 5*u, cy - 26*u), (cx + 5*u, cy - 21*u), (cx - 5*u, cy - 21*u)]
    pygame.draw.polygon(screen, CROWN, pts)

def draw_hud(mode, idx, jumps, charge_i):
    bar = pygame.Surface((SW, sx(20)), pygame.SRCALPHA); bar.fill((0, 0, 0, 130)); screen.blit(bar, (0, 0))
    screen.blit(FS.render("R:нови  P:агент/играј  ESC:излаз", True, MUT), (10, sx(3)))
    tag = "АГЕНТ" if mode == "agent" else "ИГРАЧ"
    txt = f"{tag}   висина {idx}/{N}   скокови {jumps}"
    t = FM.render(txt, True, HUD); screen.blit(t, (SW - t.get_width() - 12, sx(2)))
    if mode == "play":
        mx, my, mw, mh = SW - sx(16), sx(30), sx(9), sx(120)
        pygame.draw.rect(screen, (0, 0, 0, 0), (mx, my, mw, mh))
        pygame.draw.rect(screen, (20, 22, 30), (mx, my, mw, mh), border_radius=sx(2))
        fill = int(mh * CH[charge_i])
        pygame.draw.rect(screen, GOLD, (mx, my + mh - fill, mw, fill), border_radius=sx(2))
        screen.blit(FS.render("снага", True, MUT), (mx - sx(6), my + mh + sx(3)))

def draw_scene(plats, king, trail, mode, idx, jumps, charge_i, face=1, banner="", bcol=GOLD):
    screen.blit(BG, (0, 0))
    for i, (xl, xr, y) in enumerate(plats):
        if i == 0:
            pygame.draw.rect(screen, STONE_B, (0, sx(y), SW, SH)); pygame.draw.rect(screen, STONE, (0, sx(y), SW, sx(6)))
        else:
            draw_platform(xl, xr, y, goal=(i == len(plats) - 1))
    if trail:
        for arc in trail:
            pts = [(sx(px), sx(max(py, 2))) for px, py in arc]
            if len(pts) > 1: pygame.draw.lines(screen, (235, 220, 120), False, pts, max(1, sx(1)))
    if king: draw_king(king[0], king[1], face)
    draw_hud(mode, idx, jumps, charge_i)
    if banner:
        b = pygame.Surface((SW, sx(34)), pygame.SRCALPHA); b.fill((0, 0, 0, 150)); screen.blit(b, (0, sx(24)))
        t = FB.render(banner, True, bcol); screen.blit(t, (SW // 2 - t.get_width() // 2, sx(28)))
    pygame.display.flip()

def pump():
    for e in pygame.event.get():
        if e.type == pygame.QUIT: return "quit"
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: return "quit"
            if e.key in (pygame.K_r, pygame.K_n): return "new"
            if e.key == pygame.K_p: return "toggle"
            if e.key == pygame.K_LEFT: return ("jump", -1)
            if e.key == pygame.K_UP: return ("jump", 0)
            if e.key == pygame.K_RIGHT: return ("jump", 1)
            if pygame.K_1 <= e.key <= pygame.K_5: return ("charge", e.key - pygame.K_1)
    return None

def new_env():
    env = RL.VecRandEnv(1, seed=int(np.random.default_rng().integers(1_000_000_000)), hard=HARD)
    return env, env.levels[0]

def run():
    env, lv = new_env(); idx = 0; x = W/2; y = lv[0][2]; jumps = 0; trail = []
    play = args.play; charge_i = 4; face = 1; banner = ""; bcol = GOLD; solved = False
    while True:
        mode = "play" if play else "agent"
        if play or solved or net is None:
            draw_scene(lv, (x, y), trail, mode, idx, jumps, charge_i, face, banner, bcol)
            ev = pump(); clock.tick(60)
            if ev == "quit": break
            if ev == "new":
                env, lv = new_env(); idx=0; x=W/2; y=lv[0][2]; jumps=0; trail=[]; banner=""; solved=False; continue
            if ev == "toggle": play = not play; solved=False; banner=""; continue
            if not play or solved: continue
            if isinstance(ev, tuple) and ev[0] == "charge": charge_i = ev[1]; continue
            if isinstance(ev, tuple) and ev[0] == "jump":
                d = ev[1]; face = 1 if d >= 0 else -1
                res = animate(lv, x, y, d, CH[charge_i], trail, "play", idx, jumps, charge_i, face)
                if res == "quit": break
                (nx, ni), arc = res; trail.append(arc); idx = ni; x = nx; y = lv[ni][2]; jumps += 1
                if idx >= N: banner, bcol, solved = "СТИГАО СИ НА ВРХ!", (120, 220, 120), True
            continue
        env.idx[0]=idx; env.x[0]=x; env.y[0]=y
        with torch.no_grad(): a = int(net(torch.tensor(env._obs()))[0].argmax(1).item())
        d, c = RL.ACTIONS[a]; face = 1 if d >= 0 else -1
        res = animate(lv, x, y, d, c, trail, "agent", idx, jumps, charge_i, face)
        if res == "quit": break
        (nx, ni), arc = res; trail.append(arc); idx = ni; x = nx; y = lv[ni][2]; jumps += 1
        if idx >= N:
            draw_scene(lv, (x, y), trail, "agent", idx, jumps, charge_i, face, "АГЕНТ РЕШИО НИВО", (120, 220, 120)); pygame.time.wait(1000)
            env, lv = new_env(); idx=0; x=W/2; y=lv[0][2]; jumps=0; trail=[]
        elif jumps > 34:
            draw_scene(lv, (x, y), trail, "agent", idx, jumps, charge_i, face, "нови ниво", MUT); pygame.time.wait(500)
            env, lv = new_env(); idx=0; x=W/2; y=lv[0][2]; jumps=0; trail=[]

def animate(plats, x0, y0, d, charge, trail, mode, idx, jumps, charge_i, face):
    vx = d * charge * VX; vy = -charge * VY; x, y = x0, y0; py = y0; arc = [(x, y)]
    for _ in range(240):
        if pump() == "quit": return "quit"
        vy += G; x += vx; y += vy
        if x < 0 or x > W: x = min(max(x, 0.0), W)
        arc.append((x, y))
        draw_scene(plats, (x, y), trail + [arc], mode, idx, jumps, charge_i, face)
        clock.tick(120)
        if vy > 0:
            for j, (xl, xr, yy) in enumerate(plats):
                if py <= yy <= y and xl <= x <= xr: return (x, j), arc
        py = y
        if y > H: return (x, 0), arc
    return (x, 0), arc

if __name__ == "__main__":
    run(); pygame.quit()
