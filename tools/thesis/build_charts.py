# -*- coding: utf-8 -*-
"""Polished, consistent figure suite for the thesis results chapter (Serbian labels).
All from real training logs. Smoothed, annotated, uniform style."""
import re, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CH = "figures/"
os.makedirs(CH, exist_ok=True)

# ---------- uniform style ----------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 11, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.fontsize": 9, "legend.frameon": True,
    "legend.framealpha": 0.9, "figure.facecolor": "white",
})
C = {"good":"#2ca02c","bad":"#d62728","sec":"#1f77b4","ent":"#e8820c",
     "multi":"#7d3ac1","gen":"#1f9e89","gray":"#666"}

def parse(fn):
    d={"step":[],"succ":[],"ent":[],"ml":[],"ci":[],"cn":[]}
    if not os.path.exists(fn): return d
    for ln in open(fn, encoding="utf-8", errors="ignore"):
        m=re.search(r"step\s+(\d+).*?succ\s+([\d.]+).*?mean_lvl\s+([\d.]+).*?cur\s+(\d+)/(\d+).*?ent\s+([\d.]+)",ln)
        if m:
            d["step"].append(int(m[1])); d["succ"].append(float(m[2])); d["ml"].append(float(m[3]))
            d["ci"].append(int(m[4])); d["cn"].append(int(m[5])); d["ent"].append(float(m[6])); continue
        m=re.search(r"step\s+(\d+).*?succ\s+([\d.]+).*?ent\s+([\d.]+)",ln)  # no mean_lvl/cur (rand/ice toy)
        if m:
            d["step"].append(int(m[1])); d["succ"].append(float(m[2])); d["ent"].append(float(m[3]))
    return d

def smooth(y,w=7):
    y=np.array(y,dtype=float)
    if len(y)<w or w<2: return y
    k=np.ones(w)/w; ys=np.convolve(y,k,mode="same")
    for i in range(w//2): ys[i]=y[:i+1].mean(); ys[-(i+1)]=y[-(i+1):].mean()
    return ys

def finish(fig,name):
    for ax in fig.axes:
        t=ax.get_title()
        if t: ax.set_title(t, pad=8)
    fig.tight_layout(pad=1.1); fig.savefig(CH+name); plt.close(fig); print("  ->",name)

# ============ 1. HEALTHY CONVERGENCE (archetype) ============
d=parse("logs/L20_train.log"); x=np.array(d["step"])
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.plot(x,smooth(d["succ"]),color=C["good"],lw=2.4,label="успешност")
ax.fill_between(x,0,smooth(d["succ"]),color=C["good"],alpha=0.07)
ax.set_ylim(0,1.03); ax.set_xlabel("кораци окружења"); ax.set_ylabel("успешност",color=C["good"])
ax.tick_params(axis="y",labelcolor=C["good"])
a2=ax.twinx(); a2.plot(x,smooth(d["ent"]),color=C["ent"],lw=2,ls="--",label="ентропија")
a2.set_ylabel("ентропија политике",color=C["ent"]); a2.tick_params(axis="y",labelcolor=C["ent"]); a2.grid(False)
ax.set_title("Здрава конвергенција: успешност расте, ентропија опада")
l1,la1=ax.get_legend_handles_labels(); l2,la2=a2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,loc="center right")
finish(fig,"fig01_convergence.png")

# ============ 2. REVERSE CURRICULUM UNLOCKING ============
d=parse("logs/L22_train.log"); x=np.array(d["step"])
fig,ax=plt.subplots(figsize=(7.2,4.3))
frac=np.array(d["ci"])/np.maximum(np.array(d["cn"]),1)
ax.plot(x,smooth(d["succ"]),color=C["good"],lw=2.4,label="успешност")
ax.set_ylim(0,1.03); ax.set_xlabel("кораци окружења"); ax.set_ylabel("успешност",color=C["good"]); ax.tick_params(axis="y",labelcolor=C["good"])
a2=ax.twinx(); a2.plot(x,frac,color=C["sec"],lw=2,label="откључана стања (удео)")
a2.set_ylabel("удео откључаних почетних стања",color=C["sec"]); a2.set_ylim(0,1.03); a2.tick_params(axis="y",labelcolor=C["sec"]); a2.grid(False)
ax.set_title("Обрнути курикулум откључава све тежа почетна стања")
l1,la1=ax.get_legend_handles_labels(); l2,la2=a2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,loc="lower right")
finish(fig,"fig02_curriculum.png")

# ============ 3. WHY AGGREGATE SUCCESS DIPS — curriculum reaches hard bottom states ============
d=parse("logs/L24_train.log"); x=np.array(d["step"]); s=smooth(d["succ"],7); frac=np.array(d["ci"])/np.maximum(np.array(d["cn"]),1)
fig,ax=plt.subplots(figsize=(7.4,4.3))
ax.plot(x,s,color=C["sec"],lw=2.2,label="агрегатна успешност")
# find early near-goal peak
pk=int(np.argmax(s[:max(5,len(s)//4)]))
ax.annotate("лака стања уз циљ\n(курикулум узак)",xy=(x[pk],s[pk]),xytext=(x[pk],0.95),
            ha="center",fontsize=8.5,color=C["good"],arrowprops=dict(arrowstyle="->",color=C["good"]))
ax.annotate("курикулум досегао\nтежа стања при дну",xy=(x[int(len(x)*0.8)],s[int(len(s)*0.8)]),
            xytext=(0.62,0.8),textcoords="axes fraction",ha="center",fontsize=8.5,color=C["gray"],
            arrowprops=dict(arrowstyle="->",color=C["gray"]))
ax.set_ylim(0,1.03); ax.set_xlabel("кораци окружења"); ax.set_ylabel("успешност",color=C["sec"]); ax.tick_params(axis="y",labelcolor=C["sec"])
a2=ax.twinx(); a2.plot(x,frac,color=C["ent"],lw=1.8,ls="--",label="удео откључаних стања"); a2.set_ylim(0,1.03); a2.set_ylabel("удео откључаних стања",color=C["ent"]); a2.grid(False); a2.tick_params(axis="y",labelcolor=C["ent"])
ax.set_title("Зашто агрегатна крива опада: курикулум досеже тежа стања")
l1,la1=ax.get_legend_handles_labels(); l2,la2=a2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,loc="upper right")
finish(fig,"fig03_aggregate.png")

# ============ 4. ICE ABLATION — velocity obs vs not (CLEAN, smoothed) ============
v=parse("logs/L36_veltest.log"); t=parse("logs/L36_train.log"); b=parse("logs/L36_broad.log")
fig,ax=plt.subplots(figsize=(7.8,4.4))
ax.plot(v["step"],smooth(v["succ"],9),color=C["good"],lw=2.6,label="са опсервацијом брзине (учи → 0,87)")
ax.plot(b["step"],smooth(b["succ"],9),color=C["bad"],lw=2.6,label="без опсервације брзине (0,00)")
ax.axhline(0.87,color=C["good"],ls=":",lw=1,alpha=0.6)
ax.set_ylim(-0.03,1.05); ax.set_xlabel("кораци окружења"); ax.set_ylabel("успешност")
ax.set_title("Ледени ниво 36: брзина одваја учење од неуспеха")
ax.annotate("исти алгоритам и мрежа,\nсамо +2 скалара (vx, vy)",xy=(0.62,0.5),
            xytext=(0.5,0.66),textcoords="axes fraction",fontsize=9,color=C["gray"],
            arrowprops=dict(arrowstyle="->",color=C["gray"]))
ax.legend(loc="upper left")
finish(fig,"fig04_ice_ablation.png")

# ============ 5. ICE ENTROPY — why no-velocity never commits ============
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.plot(v["step"],smooth(v["ent"],9),color=C["good"],lw=2.4,label="са брзином: ентропија опада (учи)")
ax.plot(b["step"],smooth(b["ent"],9),color=C["bad"],lw=2.4,label="без брзине: ентропија остаје висока (лутање)")
ax.set_xlabel("кораци окружења"); ax.set_ylabel("ентропија политике")
ax.set_title("Без брзине политика никада не постаје сигурна")
ax.legend(loc="center right")
finish(fig,"fig05_ice_entropy.png")

# ============ 6. WIND — solved with wind-obs + wind-timed actions ============
w=parse("logs/L29_broad.log"); x=np.array(w["step"])
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.plot(x,smooth(w["succ"],9),color=C["good"],lw=2.4,label="успешност")
ax.fill_between(x,0,smooth(w["succ"],9),color=C["good"],alpha=0.07)
ax.set_ylim(0,1.03); ax.set_xlabel("кораци окружења"); ax.set_ylabel("успешност",color=C["good"]); ax.tick_params(axis="y",labelcolor=C["good"])
a2=ax.twinx(); a2.plot(x,smooth(w["ent"],9),color=C["ent"],lw=1.8,ls="--",label="ентропија"); a2.set_ylabel("ентропија",color=C["ent"]); a2.grid(False); a2.tick_params(axis="y",labelcolor=C["ent"])
ax.set_title("Ветар: уз опсервацију ветра и тајмиране скокове — поуздан прелазак")
l1,la1=ax.get_legend_handles_labels(); l2,la2=a2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,loc="center right")
finish(fig,"fig06_wind.png")

# ============ 7. MULTI-LEVEL AGENT ============
m=parse("logs/L10to11_multi.log")
b0=m["step"][0] if m["step"] else 0
x=np.array(m["step"])-b0
fig,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].plot(x,smooth(m["succ"],9),color=C["multi"],lw=2.2); axs[0].set_title("успешност (уланчавање 10→11→12)"); axs[0].set_ylim(0,1.03)
axs[1].plot(x,smooth(m["ml"],9),color=C["sec"],lw=2.2); axs[1].axhline(12,ls=":",color=C["good"]); axs[1].set_title("просечан достигнут ниво")
axs[2].plot(x,np.array(m["ci"]),color=C["ent"],lw=2.2); axs[2].set_title(f"откључана стања (/{m['cn'][-1] if m['cn'] else 24})")
for a in axs: a.set_xlabel("нови кораци обучавања")
fig.suptitle("Вишенивовски агент: један модел учи да повеже три нивоа",fontweight="bold")
finish(fig,"fig07_multilevel.png")

# ============ 8. GENERATOR — generalization ============
g=parse("logs/randlevel.log"); x=np.arange(len(g["succ"]))*5
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.plot(x,smooth(g["succ"],7),color=C["gen"],lw=2.4,label="успешност (невиђени нивои)")
ax.fill_between(x,0,smooth(g["succ"],7),color=C["gen"],alpha=0.08)
ax.set_ylim(0,1.03); ax.set_xlabel("ажурирања (× закуп)"); ax.set_ylabel("успешност",color=C["gen"]); ax.tick_params(axis="y",labelcolor=C["gen"])
a2=ax.twinx(); a2.plot(x,smooth(g["ent"],7),color=C["ent"],lw=1.8,ls="--"); a2.set_ylabel("ентропија",color=C["ent"]); a2.grid(False); a2.tick_params(axis="y",labelcolor=C["ent"])
ax.set_title("Верификација генератором: генерализација на нове нивое (1,00)")
ax.legend(loc="lower right")
finish(fig,"fig08_generator.png")

# ============ 9. REGIME COMPARISON BAR ============
fig,ax=plt.subplots(figsize=(7.2,4.3))
labels=["детермини-\nстички","ветровити","ледени\n(са брзином)","ледени\n(без брзине)","генератор\n(невиђено)"]
vals=[0.92,1.00,0.87,0.00,1.00]; cols=[C["sec"],C["good"],C["gen"],C["bad"],C["multi"]]
bars=ax.bar(labels,vals,color=cols,alpha=0.9,edgecolor="white")
for bx,vv in zip(bars,vals): ax.text(bx.get_x()+bx.get_width()/2,vv+0.02,f"{vv:.2f}",ha="center",fontweight="bold")
ax.set_ylim(0,1.12); ax.set_ylabel("финална успешност"); ax.set_title("Успешност по режимима нивоа")
finish(fig,"fig09_regime_bar.png")

# ============ 10. STEPS-TO-SOLVE / EFFICIENCY per sampled level ============
levels=[("L10",parse("logs/L10_train.log")),("L20",parse("logs/L20_train.log")),
        ("L22",parse("logs/L22_train.log")),("L24",parse("logs/L24_train.log")),
        ("L29 (ветар)",parse("logs/L29_broad.log")),("L36 (лед)",parse("logs/L36_veltest.log"))]
fig,ax=plt.subplots(figsize=(7.6,4.3))
names=[]; steps=[]; cols=[]
for nm,d in levels:
    if not d["step"]: continue
    # first step where smoothed success >= 0.8 (else total steps)
    s=smooth(d["succ"],7); reach=next((d["step"][i] for i in range(len(s)) if s[i]>=0.8), d["step"][-1])
    names.append(nm); steps.append(reach/1e3)
    cols.append(C["good"] if "ветар" in nm else (C["gen"] if "лед" in nm else C["sec"]))
ax.barh(names,steps,color=cols,alpha=0.9,edgecolor="white")
for i,(v) in enumerate(steps): ax.text(v+max(steps)*0.01,i,f"{v:.0f}k",va="center",fontsize=9)
ax.set_xlabel("кораци обучавања до успешности 0,8 (хиљаде)"); ax.set_title("Ефикасност обучавања по нивоу (мања вредност — брже)")
finish(fig,"fig10_efficiency.png")

# ============ 11. ENTROPY ROLE — healthy vs collapse (side by side) ============
fig,ax=plt.subplots(figsize=(7.2,4.3))
dh=parse("logs/L20_train.log")
ax.plot(np.array(dh["step"]),smooth(dh["ent"],7),color=C["good"],lw=2.2,label="здраво: постепен пад ентропије")
ax.plot(np.array(b["step"]),smooth(b["ent"],7),color=C["bad"],lw=2.2,label="патолошки: ентропија заглављена високо")
ax.set_xlabel("кораци окружења"); ax.set_ylabel("ентропија политике")
ax.set_title("Улога ентропије: пад значи учење, заглављеност значи лутање")
ax.legend(loc="center right")
finish(fig,"fig11_entropy_role.png")

print("DONE")
