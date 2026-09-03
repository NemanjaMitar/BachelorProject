import re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
st,S,R,E=[],[],[],[]
for ln in open("logs/L36_veltest.log",encoding="utf-8",errors="ignore"):
    m=re.search(r"step\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+).*?ent\s+([\d.]+)",ln)
    if m: st.append(int(m[1])/1e3);R.append(float(m[2]));S.append(float(m[3]));E.append(float(m[4]))
def sm(y,w=5):
    y=np.array(y,float)
    if len(y)<w: return y
    z=np.convolve(y,np.ones(w)/w,mode="same")
    for i in range(w//2): z[i]=y[:i+1].mean(); z[-(i+1)]=y[-(i+1):].mean()
    return z
plt.rcParams.update({"figure.facecolor":"#181a20","axes.facecolor":"#20232b","savefig.facecolor":"#181a20",
 "text.color":"#c9cdd6","axes.labelcolor":"#aeb3bd","axes.titlecolor":"#e4e7ec","xtick.color":"#868c98",
 "ytick.color":"#868c98","axes.edgecolor":"#3a3f4a","axes.linewidth":0.8,"axes.grid":True,
 "grid.color":"#ffffff","grid.alpha":0.05,"axes.axisbelow":True,"axes.titleweight":"normal","axes.titlesize":13,"font.size":11})
fig,ax=plt.subplots(1,3,figsize=(15,4.7))
ax[0].plot(st,sm(S),color="#4fd07a",lw=2.2);ax[0].set_ylim(0,1.02);ax[0].set_title("Успешност")
ax[1].plot(st,sm(R),color="#54a8e6",lw=2.2);ax[1].axhline(0,color="#454b57",lw=0.9);ax[1].set_title("Добит")
ax[2].plot(st,sm(E),color="#f0a94e",lw=2.2);ax[2].set_title("Ентропија")
for a in ax: a.set_xlabel("кораци окружења (хиљаде)"); a.margins(x=0.01)
fig.suptitle("Л36 · лед",fontweight="normal",fontsize=12,color="#c9cdd6")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("charts/L36_veltest.png",dpi=130)
