# -*- coding: utf-8 -*-
"""Schematic: a Jump King level (left) and the same as a directed state graph (right).
Nodes = resting positions on platforms; edges = macro-actions (jumps)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
plt.rcParams.update({"figure.dpi":140,"font.size":11})

C={"plat":"#c8b18f","platedge":"#8a6d47","node":"#2b6cb0","start":"#dd8452",
   "exit":"#2ca02c","trap":"#d62728","path":"#2ca02c","edge":"#9aa4b0","king":"#c0392b"}

# node positions in level coords (x:0-480, y:0-360, y up)
N={'s':(240,32),'a':(112,112),'b':(352,118),'c':(206,188),
   'd':(402,205),'e':(122,262),'f':(286,300),'x':(286,352)}
EDGES=[('s','a'),('s','b'),('a','c'),('b','d'),('c','e'),('c','b'),
       ('e','f'),('b','f'),('f','x')]
PATH=[('s','a'),('a','c'),('c','e'),('e','f'),('f','x')]  # proven route
PW=46  # half-width of a ledge

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.4,5.0))

# ---------- LEFT: the level ----------
axL.add_patch(Rectangle((0,0),480,20,color=C["plat"]))            # ground
axL.add_patch(Rectangle((0,20),480,2,color=C["platedge"]))
for k,(x,y) in N.items():
    if k in ('s','x'): continue
    axL.add_patch(Rectangle((x-PW,y-12),2*PW,12,color=C["plat"]))
    axL.add_patch(Rectangle((x-PW,y-1),2*PW,2,color=C["platedge"]))
# king at start
kx,ky=N['s']
axL.plot([kx],[ky+6],marker="^",ms=15,color=C["king"],zorder=5)
axL.text(kx,ky+22,"краљ",ha="center",fontsize=9,color=C["king"])
# a couple of dashed jump arcs (illustrative)
import numpy as np
def arc(ax,p,q,col,rad=0.25,lw=1.6,ls="--",alpha=0.8,z=2):
    a=FancyArrowPatch(p,q,connectionstyle=f"arc3,rad={rad}",arrowstyle="-|>",
                      mutation_scale=13,lw=lw,ls=ls,color=col,alpha=alpha,zorder=z)
    ax.add_patch(a)
for (u,v) in [('s','a'),('s','b'),('c','e'),('f','x')]:
    arc(axL,N[u],N[v],C["king"],rad=0.28,ls="--",alpha=0.55)
# exit marker
axL.annotate("излаз → следећи ниво",xy=(N['x'][0],358),xytext=(N['x'][0],372),
             ha="center",fontsize=9,color=C["exit"],
             arrowprops=dict(arrowstyle="-|>",color=C["exit"]))
axL.axhline(360,color=C["exit"],lw=1.4,ls=":")
axL.set_xlim(0,480); axL.set_ylim(0,392); axL.set_aspect("equal")
axL.set_title("а) Ниво (платформе, краљ, скокови)",fontweight="bold")
axL.set_xticks([]); axL.set_yticks([])
for s in axL.spines.values(): s.set_edgecolor("#bbb")

# ---------- RIGHT: the state graph ----------
def draw_edge(ax,u,v,col,lw,rad,z=2,alpha=1.0):
    a=FancyArrowPatch(N[u],N[v],connectionstyle=f"arc3,rad={rad}",arrowstyle="-|>",
                      mutation_scale=15,lw=lw,color=col,alpha=alpha,zorder=z,
                      shrinkA=14,shrinkB=14)
    ax.add_patch(a)
for (u,v) in EDGES:
    if (u,v) in PATH: continue
    draw_edge(axR,u,v,C["edge"],1.8,0.22,z=2,alpha=0.9)
for (u,v) in PATH:
    draw_edge(axR,u,v,C["path"],3.0,0.22,z=4)
for k,(x,y) in N.items():
    col=C["node"]; r=15
    if k=='s': col=C["start"]
    elif k=='x': col=C["exit"]
    elif k=='d': col=C["trap"]
    axR.add_patch(Circle((x,y),r,color=col,zorder=5,ec="white",lw=2))
    axR.text(x,y,k,ha="center",va="center",color="white",fontsize=10,fontweight="bold",zorder=6)
# labels
axR.text(N['s'][0]+22,N['s'][1]-4,"почетно стање",fontsize=9,color=C["start"],va="center")
axR.text(N['x'][0]+22,N['x'][1],"излаз (следећи ниво)",fontsize=9,color=C["exit"],va="center")
axR.text(N['d'][0]+22,N['d'][1],"замка (нема излаза)",fontsize=9,color=C["trap"],va="center")
axR.annotate("доказана путања",xy=((N['c'][0]+N['e'][0])/2-8,(N['c'][1]+N['e'][1])/2),
             xytext=(30,250),fontsize=9,color=C["path"],
             arrowprops=dict(arrowstyle="-|>",color=C["path"]))
axR.annotate("грана = макро-акција (скок)",xy=((N['s'][0]+N['b'][0])/2+10,(N['s'][1]+N['b'][1])/2-8),
             xytext=(150,40),fontsize=9,color="#555",
             arrowprops=dict(arrowstyle="-|>",color="#888"))
axR.set_xlim(0,480); axR.set_ylim(0,392); axR.set_aspect("equal")
axR.set_title("б) Граф стања (чворови = стања мировања, гране = акције)",fontweight="bold")
axR.set_xticks([]); axR.set_yticks([])
for s in axR.spines.values(): s.set_edgecolor("#bbb")

fig.tight_layout(pad=1.4)
fig.savefig("figures/fig_level_graph.png")
print("saved figures/fig_level_graph.png")
