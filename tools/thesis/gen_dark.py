import re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
U,S,E=[],[],[]
for ln in open("logs/randlevel.log",encoding="utf-8",errors="ignore"):
    m=re.search(r"upd\s+(\d+).*?succ\s+([\d.]+).*?ent\s+([\d.]+)",ln)
    if m: U.append(int(m[1]));S.append(float(m[2]));E.append(float(m[3]))
def sm(y,w=5):
    y=np.array(y,float)
    if len(y)<w: return y
    z=np.convolve(y,np.ones(w)/w,mode="same")
    for i in range(w//2): z[i]=y[:i+1].mean(); z[-(i+1)]=y[-(i+1):].mean()
    return z
plt.rcParams.update({"figure.facecolor":"#181a20","axes.facecolor":"#20232b","savefig.facecolor":"#181a20",
 "text.color":"#c9cdd6","axes.labelcolor":"#aeb3bd","axes.titlecolor":"#e4e7ec","xtick.color":"#868c98",
 "ytick.color":"#868c98","axes.edgecolor":"#3a3f4a","axes.linewidth":0.8,"axes.grid":True,
 "grid.color":"#ffffff","grid.alpha":0.05,"axes.axisbelow":True,"axes.titleweight":"normal","axes.titlesize":13,"font.size":12})
GRN="#2fae94"; ORG="#f0a94e"
fig,ax=plt.subplots(figsize=(9.2,4.9))
ax.plot(U,sm(S),color=GRN,lw=2.4); ax.fill_between(U,0,sm(S),color=GRN,alpha=0.08)
ax.set_ylim(0,1.02); ax.set_xlabel("ажурирања"); ax.set_ylabel("успешност",color=GRN); ax.tick_params(axis="y",labelcolor=GRN)
a2=ax.twinx(); a2.plot(U,sm(E),color=ORG,lw=2.0,ls="--"); a2.set_ylabel("ентропија",color=ORG); a2.tick_params(axis="y",labelcolor=ORG); a2.grid(False)
ax.set_title("Верификација генератором · генерализација на нове нивое (1,00)")
fig.tight_layout(); fig.savefig("figures/fig08_generator.png",dpi=130)
