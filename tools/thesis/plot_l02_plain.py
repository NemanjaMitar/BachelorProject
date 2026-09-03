# -*- coding: utf-8 -*-
import re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = r"""
upd 1 ret -8.29 succ 0.00 mean_lvl 0.2 cur 1/17 ent 3.135
upd 2 ret -7.12 succ 0.00 mean_lvl 0.2 cur 1/17 ent 3.134
upd 3 ret -7.00 succ 0.00 mean_lvl 0.2 cur 1/17 ent 3.132
upd 4 ret -6.50 succ 0.00 mean_lvl 0.2 cur 1/17 ent 3.130
upd 5 ret -6.70 succ 0.01 mean_lvl 0.2 cur 1/17 ent 3.127
upd 6 ret -7.30 succ 0.01 mean_lvl 0.3 cur 1/17 ent 3.123
upd 7 ret -7.00 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.122
upd 8 ret -6.90 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.121
upd 9 ret -7.40 succ 0.00 mean_lvl 0.2 cur 1/17 ent 3.122
upd 10 ret -7.40 succ 0.01 mean_lvl 0.2 cur 1/17 ent 3.112
upd 11 ret -7.70 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.108
upd 12 ret -7.30 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.111
upd 13 ret -6.30 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.109
upd 14 ret -6.60 succ 0.02 mean_lvl 0.2 cur 1/17 ent 3.106
upd 15 ret -6.70 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.101
upd 16 ret -5.90 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.105
upd 17 ret -6.00 succ 0.02 mean_lvl 0.1 cur 1/17 ent 3.088
upd 18 ret -5.40 succ 0.01 mean_lvl 0.1 cur 1/17 ent 3.071
upd 19 ret -5.80 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.038
upd 20 ret -5.50 succ 0.06 mean_lvl 0.4 cur 1/17 ent 3.005
upd 21 ret -6.10 succ 0.05 mean_lvl 0.3 cur 1/17 ent 3.031
upd 22 ret -5.70 succ 0.05 mean_lvl 0.3 cur 1/17 ent 3.075
upd 23 ret -6.60 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.042
upd 24 ret -6.30 succ 0.04 mean_lvl 0.3 cur 1/17 ent 3.060
upd 25 ret -6.30 succ 0.04 mean_lvl 0.2 cur 1/17 ent 3.041
upd 26 ret -5.70 succ 0.05 mean_lvl 0.2 cur 1/17 ent 3.071
upd 27 ret -5.30 succ 0.06 mean_lvl 0.3 cur 1/17 ent 3.065
upd 28 ret -6.20 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.068
upd 29 ret -6.30 succ 0.02 mean_lvl 0.2 cur 1/17 ent 3.074
upd 30 ret -6.10 succ 0.02 mean_lvl 0.2 cur 1/17 ent 3.088
upd 31 ret -6.00 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.028
upd 32 ret -6.10 succ 0.05 mean_lvl 0.3 cur 1/17 ent 3.017
upd 33 ret -6.10 succ 0.05 mean_lvl 0.4 cur 1/17 ent 3.060
upd 34 ret -5.00 succ 0.07 mean_lvl 0.4 cur 1/17 ent 3.078
upd 35 ret -5.50 succ 0.06 mean_lvl 0.3 cur 1/17 ent 3.062
upd 36 ret -5.90 succ 0.06 mean_lvl 0.4 cur 1/17 ent 3.037
upd 37 ret -7.20 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.041
upd 38 ret -6.50 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.095
upd 39 ret -6.10 succ 0.04 mean_lvl 0.3 cur 1/17 ent 3.095
upd 40 ret -5.40 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.080
upd 41 ret -5.40 succ 0.04 mean_lvl 0.4 cur 1/17 ent 3.050
upd 42 ret -6.30 succ 0.05 mean_lvl 0.4 cur 1/17 ent 3.039
upd 43 ret -7.30 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.054
upd 44 ret -6.80 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.059
upd 45 ret -7.10 succ 0.03 mean_lvl 0.4 cur 1/17 ent 3.060
upd 46 ret -7.30 succ 0.02 mean_lvl 0.3 cur 1/17 ent 3.095
upd 47 ret -6.50 succ 0.03 mean_lvl 0.3 cur 1/17 ent 3.082
upd 48 ret -5.10 succ 0.08 mean_lvl 0.5 cur 1/17 ent 3.079
upd 49 ret -5.70 succ 0.07 mean_lvl 0.4 cur 1/17 ent 3.069
upd 50 ret -7.60 succ 0.01 mean_lvl 0.2 cur 1/17 ent 3.087
upd 51 ret -6.80 succ 0.03 mean_lvl 0.2 cur 1/17 ent 3.066
upd 52 ret -6.10 succ 0.05 mean_lvl 0.3 cur 1/17 ent 3.069
upd 53 ret -6.70 succ 0.04 mean_lvl 0.3 cur 1/17 ent 3.055
upd 54 ret -6.60 succ 0.04 mean_lvl 0.3 cur 1/17 ent 3.072
upd 55 ret -5.80 succ 0.06 mean_lvl 0.3 cur 1/17 ent 3.071
upd 56 ret -5.70 succ 0.06 mean_lvl 0.3 cur 1/17 ent 3.076
upd 57 ret -5.60 succ 0.05 mean_lvl 0.3 cur 1/17 ent 3.085
upd 58 ret -5.40 succ 0.06 mean_lvl 0.4 cur 1/17 ent 3.069
upd 59 ret -5.60 succ 0.06 mean_lvl 0.3 cur 1/17 ent 3.084
upd 60 ret -4.80 succ 0.08 mean_lvl 0.3 cur 1/17 ent 3.066
"""
U,R,S,C,E=[],[],[],[],[]
for ln in DATA.splitlines():
    m=re.search(r"upd (\d+) ret (-?[\d.]+) succ ([\d.]+) mean_lvl [\d.]+ cur (\d+)/17 ent ([\d.]+)",ln)
    if m: U.append(int(m[1]));R.append(float(m[2]));S.append(float(m[3]));C.append(int(m[4]));E.append(float(m[5]))
plt.rcParams.update({
    "figure.facecolor":"#181a20","axes.facecolor":"#20232b","savefig.facecolor":"#181a20",
    "text.color":"#c9cdd6","axes.labelcolor":"#aeb3bd","axes.titlecolor":"#e4e7ec",
    "xtick.color":"#868c98","ytick.color":"#868c98","axes.edgecolor":"#3a3f4a","axes.linewidth":0.8,
    "axes.grid":True,"grid.color":"#ffffff","grid.alpha":0.05,"grid.linewidth":0.7,
    "axes.axisbelow":True,"axes.titleweight":"normal","axes.titlesize":13,"font.size":11})
COL={"s":"#4fd07a","r":"#54a8e6","e":"#f0a94e","c":"#b184ea"}
fig,ax=plt.subplots(2,2,figsize=(13.5,7.8))
a=ax[0,0]; a.plot(U,S,color=COL["s"],lw=2.2); a.set_ylim(0,1.02); a.set_title("Успешност")
a=ax[0,1]; a.plot(U,R,color=COL["r"],lw=2.2); a.axhline(0,color="#454b57",lw=0.9); a.set_title("Добит")
a=ax[1,0]; a.plot(U,E,color=COL["e"],lw=2.2); a.set_title("Ентропија")
a=ax[1,1]; a.step(U,C,color=COL["c"],lw=2.2,where="post"); a.set_ylim(0,17); a.set_title("Курикулум")
for a in ax.flat: a.set_xlabel("ажурирање"); a.margins(x=0.01)
fig.suptitle("Нивои 0–2 · обичан курикулум",fontweight="normal",fontsize=12,color="#c9cdd6")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig("charts/L0-2_plain.png",dpi=130)
print(f"parsed {len(U)} | succ {S[0]:.2f}->{max(S):.2f}(peak) | cur {C[0]}->{C[-1]}/17 | ent {E[0]:.2f}->{E[-1]:.2f}")
