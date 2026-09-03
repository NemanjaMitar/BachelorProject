"""
ICE variant of the procedural env  ->  genuine PPO trained on slippery levels,
and the ablation that proves WHY the real game needs a velocity observation.

On ice a landing does NOT stop the king: he keeps his horizontal speed and SLIDES
(x += vx each tick, friction decays it). If he slides past the ledge edge he falls.
So the agent must (a) SETTLE to bleed off momentum before lining up the next jump,
or (b) account for the residual slide when it launches. The only way to do either
is to PERCEIVE the momentum -> the ablation toggles whether vx is in the observation.

  python IceLevel.py --ice --vel-obs   --updates 300 --log logs/ice_velobs.log
  python IceLevel.py --ice --no-vel-obs --updates 300 --log logs/ice_novel.log
Chart both -> velocity obs is the difference between learning ice and not.
"""
import argparse, os, re, time
import numpy as np
import torch, torch.nn as nn

W, H = 480.0, 360.0
G = 0.55
VX_MAX, VY_MAX = 4.6, 12.5
N_PLAT = 5
PLAT_W = (48.0, 74.0)    # narrow ledges: a slide of a few px off-centre = a fall
DY = (46.0, 64.0)
NEXT_K = 3
FRICTION = 0.72          # per-tick horizontal decay while sliding on ice (slow = persists)
SLIDE_TICKS = 6          # the post-landing slide is integrated before the next action

CHARGES = [0.35, 0.55, 0.72, 0.86, 1.0]
DIRS = [-1, 0, 1]
JUMPS = [(d, c) for d in DIRS for c in CHARGES]
N_ACT = len(JUMPS)                     # jumps only — no blind "settle" escape hatch
OBS_DIM = 5 + NEXT_K * 3               # x,y,vx,charge,cur + K*(rel_x,rel_dy,halfw)


def gen_level(rng):
    plats = [(0.0, W, H - 12.0)]
    x = W * 0.5
    for _ in range(N_PLAT):
        w = rng.uniform(*PLAT_W); dy = rng.uniform(*DY)
        y = plats[-1][2] - dy
        max_dx = VX_MAX * 2.0 * (VY_MAX / G) * 0.11
        cx = np.clip(x + rng.uniform(-max_dx, max_dx), w*0.5+6, W-w*0.5-6)
        plats.append((cx - w*0.5, cx + w*0.5, y)); x = cx
    return plats


class VecIce:
    def __init__(self, batch, ice=True, vel_obs=True, seed=0):
        self.B, self.ice, self.vel_obs = batch, ice, vel_obs
        self.rng = np.random.default_rng(seed)
        self.reset_all()

    def reset_all(self):
        self.lv = [gen_level(self.rng) for _ in range(self.B)]
        self.idx = np.zeros(self.B, np.int64)
        self.x = np.full(self.B, W*0.5)
        self.vx = np.zeros(self.B)
        self.y = np.array([self.lv[i][0][2] for i in range(self.B)])
        self.steps = np.zeros(self.B, np.int64)
        return self._obs()

    def _reset_one(self, i):
        self.lv[i] = gen_level(self.rng); self.idx[i]=0
        self.x[i]=W*0.5; self.vx[i]=0.0; self.y[i]=self.lv[i][0][2]; self.steps[i]=0

    def _plat_at(self, lv, x, top_idx):
        """highest ledge at/below top_idx whose span contains x (where a slide-off lands)."""
        for j in range(top_idx-1, -1, -1):
            xl, xr, _ = lv[j]
            if xl <= x <= xr:
                return j
        return 0

    def _obs(self):
        o = np.zeros((self.B, OBS_DIM), np.float32)
        for i in range(self.B):
            lv, ci = self.lv[i], self.idx[i]
            o[i,0]=self.x[i]/W; o[i,1]=self.y[i]/H
            o[i,2]=(self.vx[i]/VX_MAX) if self.vel_obs else 0.0   # <-- ablation switch
            o[i,3]=1.0; o[i,4]=ci/(N_PLAT+1)
            for k in range(NEXT_K):
                j=ci+1+k; b=5+k*3
                if j < len(lv):
                    xl,xr,yy=lv[j]; cx=0.5*(xl+xr)
                    o[i,b]=(cx-self.x[i])/W; o[i,b+1]=(self.y[i]-yy)/H; o[i,b+2]=(xr-xl)*0.5/W
        return o

    def _launch(self, lv, x0, y0, d, charge):
        vx0 = d*charge*VX_MAX; vy=-charge*VY_MAX; x,y=x0,y0; py=y0
        for _ in range(240):
            vy+=G; x+=vx0; y+=vy
            if x<0 or x>W: x=min(max(x,0.0),W)
            if vy>0:
                for j,(xl,xr,yy) in enumerate(lv):
                    if py<=yy<=y and xl<=x<=xr:
                        return x, j, vx0            # landing x, ledge, horiz speed carried
            py=y
            if y>H: return x,0,0.0
        return x0,0,0.0

    def step(self, actions):
        rew=np.zeros(self.B,np.float32); done=np.zeros(self.B,np.float32)
        for i in range(self.B):
            lv=self.lv[i]; prev=self.idx[i]; a=int(actions[i])
            # (1) ice slide from any residual momentum, integrated over SLIDE_TICKS
            if self.ice and abs(self.vx[i])>1e-3:
                for _ in range(SLIDE_TICKS):
                    self.x[i]+=self.vx[i]; self.vx[i]*=FRICTION
                    xl,xr,_=lv[self.idx[i]]
                    if self.x[i]<xl or self.x[i]>xr:        # slid off the edge -> fall
                        self.x[i]=min(max(self.x[i],0.0),W)
                        nj=self._plat_at(lv,self.x[i],self.idx[i])
                        self.idx[i]=nj; self.y[i]=lv[nj][2]; self.vx[i]=0.0
                        break
            # (2) the chosen jump (launched from the post-slide position)
            d,c=JUMPS[a]
            nx,nj,vxl=self._launch(lv,self.x[i],self.y[i],d,c)
            self.idx[i]=nj; self.x[i]=nx; self.y[i]=lv[nj][2]
            self.vx[i]= vxl if self.ice else 0.0            # ice carries speed; else stop
            self.steps[i]+=1
            cur=self.idx[i]
            rew[i]=1.5*(cur-prev)-0.05
            if cur<prev: rew[i]-=0.6
            if cur>=N_PLAT: rew[i]+=12.0; done[i]=1.0
            elif self.steps[i]>=36: done[i]=1.0
        succ=np.array([1.0 if self.idx[i]>=N_PLAT else 0.0 for i in range(self.B)],np.float32)
        obs=self._obs()
        for i in range(self.B):
            if done[i]: self._reset_one(i)
        return obs, rew, done, succ


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(obs_dim,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh())
        self.pi=nn.Linear(96,n_act); self.v=nn.Linear(96,1)
    def forward(self,x):
        h=self.body(x); return self.pi(h), self.v(h).squeeze(-1)


def train(args):
    torch.manual_seed(args.seed)
    env=VecIce(args.batch, ice=args.ice, vel_obs=args.vel_obs, seed=args.seed)
    net=ActorCritic(OBS_DIM,N_ACT); opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    obs=torch.tensor(env.reset_all()); T=args.rollout
    os.makedirs(os.path.dirname(args.log) or ".",exist_ok=True); logf=open(args.log,"w")
    def log(m): print(m); logf.write(m+"\n"); logf.flush()
    log(f"# IceLevel PPO ice={args.ice} vel_obs={args.vel_obs} batch={args.batch} "
        f"rollout={T} updates={args.updates} obs_dim={OBS_DIM} n_act={N_ACT}")
    hist=[]
    for upd in range(1,args.updates+1):
        MO,MA,ML,MV,MR,MD=[],[],[],[],[],[]; es=[]
        for _ in range(T):
            with torch.no_grad():
                lg,v=net(obs); dist=torch.distributions.Categorical(logits=lg)
                act=dist.sample(); lp=dist.log_prob(act)
            no,r,d,sc=env.step(act.numpy())
            MO.append(obs);MA.append(act);ML.append(lp);MV.append(v)
            MR.append(torch.tensor(r));MD.append(torch.tensor(d))
            for i in range(env.B):
                if d[i]: es.append(sc[i])
            obs=torch.tensor(no)
        with torch.no_grad(): _,lastv=net(obs)
        adv=torch.zeros(env.B); A=[None]*T
        for t in reversed(range(T)):
            nv=lastv if t==T-1 else MV[t+1]; nt=1.0-MD[t]
            delta=MR[t]+0.99*nv*nt-MV[t]; adv=delta+0.99*0.95*nt*adv; A[t]=adv.clone()
        ob=torch.cat(MO);ac=torch.cat(MA);lpb=torch.cat(ML);vb=torch.cat(MV)
        ab=torch.cat(A);rb=ab+vb; ab=(ab-ab.mean())/(ab.std()+1e-8)
        ent_last=0.0
        for _ in range(4):
            idx=torch.randperm(ob.shape[0])
            for s in range(0,ob.shape[0],512):
                b=idx[s:s+512]; lg,v=net(ob[b]); dist=torch.distributions.Categorical(logits=lg)
                nlp=dist.log_prob(ac[b]); ratio=torch.exp(nlp-lpb[b]); a=ab[b]
                pg=-torch.min(ratio*a,torch.clamp(ratio,0.8,1.2)*a).mean()
                vl=((v-rb[b])**2).mean(); ent=dist.entropy().mean()
                (pg+0.5*vl-0.01*ent).backward()
                nn.utils.clip_grad_norm_(net.parameters(),0.5); opt.step(); opt.zero_grad()
                ent_last=float(ent)
        sr=float(np.mean(es)) if es else 0.0; hist.append(sr)
        ret=float(torch.stack(MR).sum(0).mean())
        if upd%5==0 or upd==1:
            log(f"upd {upd:4d} | step {upd*T*env.B:8d} | ret {ret:7.2f} | succ {sr:0.2f} | "
                f"ent {ent_last:0.3f} | eps {len(es)}")
    log(f"# DONE final_succ={np.mean(hist[-10:]):0.3f}")
    torch.save({"model":net.state_dict()},args.save); log(f"# saved -> {args.save}")
    logf.close()


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ice",action="store_true")
    ap.add_argument("--vel-obs",dest="vel_obs",action="store_true",default=True)
    ap.add_argument("--no-vel-obs",dest="vel_obs",action="store_false")
    ap.add_argument("--batch",type=int,default=96)
    ap.add_argument("--rollout",type=int,default=32)
    ap.add_argument("--updates",type=int,default=300)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--log",type=str,default="logs/ice.log")
    ap.add_argument("--save",type=str,default="checkpoints/experiments/icelevel/ppo.pt")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.save) or ".",exist_ok=True)
    t0=time.time(); train(args); print(f"elapsed {time.time()-t0:0.1f}s")
