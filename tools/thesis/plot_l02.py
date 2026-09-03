# -*- coding: utf-8 -*-
import re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA = r"""
upd    1 | step     2048 |    96 sps | ret    -9.40 | succ 0.07 | mean_lvl  1.0 | max_lvl  3 | cur 1/17 pb 0.0 | ent 3.135
upd    2 | step     4096 | ret    -8.30 | succ 0.10 | mean_lvl  1.1 | cur 1/17 | ent 3.130
upd    3 | ret    -7.10 | succ 0.14 | mean_lvl  1.1 | cur 1/17 | ent 3.122
upd    4 | ret    -6.40 | succ 0.16 | mean_lvl  1.2 | cur 1/17 | ent 3.106
upd    5 | ret    -5.10 | succ 0.20 | mean_lvl  1.3 | cur 1/17 | ent 3.077
upd    6 | ret    -1.70 | succ 0.31 | mean_lvl  1.5 | cur 1/17 | ent 3.030
upd    7 | ret    -0.30 | succ 0.35 | mean_lvl  1.6 | cur 1/17 | ent 2.947
upd    8 | ret     5.00 | succ 0.52 | mean_lvl  2.0 | cur 1/17 | ent 2.876
upd    9 | ret     4.70 | succ 0.52 | mean_lvl  1.9 | cur 1/17 | ent 2.793
upd   10 | ret     9.80 | succ 0.68 | mean_lvl  2.3 | cur 1/17 | ent 2.696
upd   11 | ret    14.50 | succ 0.83 | mean_lvl  2.6 | cur 1/17 | ent 2.577
upd   12 | ret    14.70 | succ 0.83 | mean_lvl  2.6 | cur 1/17 | ent 2.460
upd   13 | ret    11.40 | succ 0.72 | mean_lvl  2.4 | cur 1/17 | ent 2.296
upd   14 | ret     6.00 | succ 0.54 | mean_lvl  2.1 | cur 2/17 | ent 2.347
upd   15 | ret    -6.70 | succ 0.14 | mean_lvl  1.2 | cur 2/17 | ent 2.549
upd   16 | ret    -4.70 | succ 0.21 | mean_lvl  1.3 | cur 2/17 | ent 2.732
upd   17 | ret    -3.70 | succ 0.23 | mean_lvl  1.4 | cur 2/17 | ent 2.773
upd   18 | ret    -6.20 | succ 0.14 | mean_lvl  1.2 | cur 2/17 | ent 2.859
upd   19 | ret    -6.30 | succ 0.15 | mean_lvl  1.2 | cur 2/17 | ent 2.850
upd   20 | ret    -1.60 | succ 0.29 | mean_lvl  1.6 | cur 2/17 | ent 2.869
upd   21 | ret    -2.10 | succ 0.28 | mean_lvl  1.5 | cur 2/17 | ent 2.928
upd   22 | ret    -2.30 | succ 0.26 | mean_lvl  1.5 | cur 2/17 | ent 2.953
upd   23 | ret    -1.40 | succ 0.30 | mean_lvl  1.6 | cur 2/17 | ent 2.943
upd   24 | ret    -3.90 | succ 0.21 | mean_lvl  1.4 | cur 2/17 | ent 2.941
upd   25 | ret    -0.50 | succ 0.33 | mean_lvl  1.6 | cur 2/17 | ent 2.938
upd   26 | ret    -0.10 | succ 0.35 | mean_lvl  1.6 | cur 2/17 | ent 2.890
upd   27 | ret     1.20 | succ 0.38 | mean_lvl  1.7 | cur 2/17 | ent 2.828
upd   28 | ret     2.70 | succ 0.43 | mean_lvl  1.8 | cur 2/17 | ent 2.754
upd   29 | ret     3.00 | succ 0.44 | mean_lvl  1.9 | cur 2/17 | ent 2.723
upd   30 | ret     2.70 | succ 0.44 | mean_lvl  1.8 | cur 2/17 | ent 2.642
upd   31 | ret     2.40 | succ 0.43 | mean_lvl  1.8 | cur 2/17 | ent 2.639
upd   32 | ret     7.00 | succ 0.58 | mean_lvl  2.1 | cur 2/17 | ent 2.619
upd   33 | ret     3.50 | succ 0.47 | mean_lvl  1.9 | cur 2/17 | ent 2.618
upd   34 | ret     7.20 | succ 0.59 | mean_lvl  2.1 | cur 2/17 | ent 2.543
upd   35 | ret     5.80 | succ 0.53 | mean_lvl  2.0 | cur 2/17 | ent 2.540
upd   36 | ret     6.40 | succ 0.56 | mean_lvl  2.1 | cur 2/17 | ent 2.524
upd   37 | ret     6.90 | succ 0.58 | mean_lvl  2.1 | cur 2/17 | ent 2.494
upd   38 | ret     7.40 | succ 0.58 | mean_lvl  2.2 | cur 2/17 | ent 2.508
upd   39 | ret    10.80 | succ 0.70 | mean_lvl  2.4 | cur 2/17 | ent 2.429
upd   40 | ret     4.30 | succ 0.49 | mean_lvl  1.9 | cur 3/17 | ent 2.413
upd   41 | ret     4.00 | succ 0.48 | mean_lvl  1.9 | cur 2/17 | ent 2.347
upd   42 | ret     3.00 | succ 0.45 | mean_lvl  1.9 | cur 2/17 | ent 2.352
upd   43 | ret    -0.40 | succ 0.34 | mean_lvl  1.6 | cur 3/17 | ent 2.344
upd   44 | ret     3.00 | succ 0.44 | mean_lvl  1.9 | cur 3/17 | ent 2.307
upd   45 | ret     3.00 | succ 0.45 | mean_lvl  1.9 | cur 3/17 | ent 2.296
upd   46 | ret     7.30 | succ 0.59 | mean_lvl  2.1 | cur 3/17 | ent 2.241
upd   47 | ret     1.50 | succ 0.41 | mean_lvl  1.7 | cur 4/17 | ent 2.247
upd   48 | ret    -3.30 | succ 0.27 | mean_lvl  1.4 | cur 4/17 | ent 2.367
upd   49 | ret    -0.50 | succ 0.34 | mean_lvl  1.6 | cur 4/17 | ent 2.413
upd   50 | ret    -3.70 | succ 0.24 | mean_lvl  1.4 | cur 4/17 | ent 2.471
upd   51 | ret    -0.90 | succ 0.31 | mean_lvl  1.6 | cur 4/17 | ent 2.476
upd   52 | ret    -2.10 | succ 0.27 | mean_lvl  1.5 | cur 4/17 | ent 2.441
upd   53 | ret    -3.90 | succ 0.21 | mean_lvl  1.4 | cur 4/17 | ent 2.453
upd   54 | ret    -4.50 | succ 0.19 | mean_lvl  1.4 | cur 4/17 | ent 2.444
upd   55 | ret    -3.80 | succ 0.23 | mean_lvl  1.4 | cur 4/17 | ent 2.426
upd   56 | ret    -3.90 | succ 0.22 | mean_lvl  1.4 | cur 4/17 | ent 2.381
upd   57 | ret    -3.70 | succ 0.23 | mean_lvl  1.4 | cur 4/17 | ent 2.406
upd   58 | ret    -2.70 | succ 0.26 | mean_lvl  1.5 | cur 4/17 | ent 2.402
upd   59 | ret    -2.90 | succ 0.25 | mean_lvl  1.5 | cur 4/17 | ent 2.316
upd   60 | ret    -1.90 | succ 0.28 | mean_lvl  1.5 | cur 4/17 | ent 2.263
upd   61 | ret    -1.00 | succ 0.31 | mean_lvl  1.6 | cur 4/17 | ent 2.240
upd   62 | ret    -1.10 | succ 0.33 | mean_lvl  1.6 | cur 4/17 | ent 2.206
upd   63 | ret     0.80 | succ 0.38 | mean_lvl  1.7 | cur 4/17 | ent 2.136
upd   64 | ret    -1.10 | succ 0.32 | mean_lvl  1.6 | cur 4/17 | ent 2.143
upd   65 | ret     0.50 | succ 0.38 | mean_lvl  1.7 | cur 4/17 | ent 2.099
upd   66 | ret     3.10 | succ 0.45 | mean_lvl  1.9 | cur 4/17 | ent 2.040
upd   67 | ret     2.40 | succ 0.43 | mean_lvl  1.8 | cur 4/17 | ent 2.031
upd   68 | ret     4.40 | succ 0.50 | mean_lvl  1.9 | cur 4/17 | ent 1.982
upd   69 | ret     6.20 | succ 0.55 | mean_lvl  2.1 | cur 4/17 | ent 1.953
upd   70 | ret     3.50 | succ 0.46 | mean_lvl  1.9 | cur 4/17 | ent 1.906
upd   71 | ret     5.40 | succ 0.53 | mean_lvl  2.0 | cur 4/17 | ent 1.853
upd   72 | ret     3.30 | succ 0.46 | mean_lvl  1.9 | cur 4/17 | ent 1.835
upd   73 | ret     8.30 | succ 0.62 | mean_lvl  2.2 | cur 4/17 | ent 1.784
upd   74 | ret     7.60 | succ 0.59 | mean_lvl  2.2 | cur 4/17 | ent 1.772
upd   75 | ret     7.70 | succ 0.60 | mean_lvl  2.2 | cur 4/17 | ent 1.753
upd   76 | ret     5.50 | succ 0.53 | mean_lvl  2.0 | cur 4/17 | ent 1.763
upd   77 | ret     7.20 | succ 0.58 | mean_lvl  2.1 | cur 4/17 | ent 1.718
upd   78 | ret     7.30 | succ 0.58 | mean_lvl  2.1 | cur 4/17 | ent 1.726
upd   79 | ret     5.10 | succ 0.51 | mean_lvl  2.0 | cur 4/17 | ent 1.700
"""
U,R,S,M,C,E=[],[],[],[],[],[]
for ln in DATA.splitlines():
    m=re.search(r"upd\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+).*?mean_lvl\s+([\d.]+).*?cur\s+(\d+)/17.*?ent\s+([\d.]+)",ln)
    if m: U.append(int(m[1]));R.append(float(m[2]));S.append(float(m[3]));M.append(float(m[4]));C.append(int(m[5]));E.append(float(m[6]))
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
fig.suptitle("Нивои 0–2",fontweight="normal",fontsize=12,color="#c9cdd6")
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig("charts/L0-2_train.png",dpi=130)
print(f"parsed {len(U)} updates -> charts/L0-2_train.png | succ {S[0]:.2f}->{S[-1]:.2f} | cur {C[0]}->{C[-1]}/17 | ent {E[0]:.2f}->{E[-1]:.2f}")
