#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_goal_predictor_with_neighbors.py
- Input: (history, lane context, neighbor context) in ego frame
- Output: temporal goal (P step ahead)
- Model: HistoryEncoder + LaneEncoder + NeighborEncoder + MDN head
- Train: all npz under train_dir
- Test: one npz file, visualize predictions
"""

import os, glob, math, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import cKDTree as KDTree
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_dir = str(os.path.dirname(os.path.abspath(__file__)))
# ===== Utility =====
def safe_heading(xy: np.ndarray) -> float:
    diffs = np.diff(xy, axis=0)
    for d in diffs[::-1]:
        if np.linalg.norm(d) > 1e-6:
            return float(np.arctan2(d[1], d[0]))
    return 0.0

def rotmat(theta: float) -> np.ndarray:
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s],[s,c]],dtype=np.float32)

def build_lane_feature(lane_points, pos, psi, radius=20.0, kmax=32):
    if lane_points.shape[0]==0:
        flat = np.zeros((2*kmax,),np.float32)
        return np.concatenate([flat,[np.cos(psi),np.sin(psi)]],0)
    kdt = KDTree(lane_points)
    idxs = kdt.query_ball_point(pos,r=radius)
    neigh = lane_points[idxs] if len(idxs)>0 else np.empty((0,2))
    if neigh.shape[0]>0:
        d2 = np.sum((neigh-pos)**2,axis=1)
        order = np.argsort(d2)[:kmax]
        neigh = neigh[order]
    else:
        neigh = np.zeros((0,2))
    R = rotmat(-psi); rel=(neigh-pos)@R.T
    out = np.zeros((kmax,2),np.float32); k=min(kmax,rel.shape[0])
    if k>0: out[:k]=rel[:k]
    flat=out.reshape(-1)
    return np.concatenate([flat,[np.cos(psi),np.sin(psi)]],0).astype(np.float32)

def build_neighbor_feature(X, V, s, m, pos, psi, radius=20.0, kmax=6):
    """
    Build neighbor feature in ego frame.
    Return (2*kmax,) vector.
    """
    T, M, _ = X.shape
    R = rotmat(-psi)
    feats = []
    for j in range(M):
        if j == m: continue
        if not V[s, j]: continue
        nbr_pos = X[s, j, :]
        if np.linalg.norm(nbr_pos - pos) <= radius:
            rel = (nbr_pos - pos) @ R.T
            feats.append(rel.astype(np.float32))
    feats.sort(key=lambda v: np.linalg.norm(v))
    feats = feats[:kmax]
    while len(feats) < kmax:
        feats.append(np.zeros(2, np.float32))
    return np.concatenate(feats, axis=0)  # (2*kmax,)

# ===== Dataset =====
class GoalDataset(Dataset):
    def __init__(self, files, H=10, P=50, lane_radius=20, lane_k=32, neighbor_radius=20, neighbor_k=6):
        self.samples=[]
        for f in files:
            D=np.load(f,allow_pickle=True)
            X=D['X'].astype(np.float32) # (T,M,2)
            V=D['V'].astype(bool)
            lane_points=D.get('lane_points',np.zeros((0,2),np.float32))
            T,M,_=X.shape
            for m in range(M):
                vcol=V[:,m]
                if np.sum(vcol)<H+P: continue
                for s in range(H-1, T-P):
                    if not (np.all(vcol[s-H+1:s+1]) and vcol[s+P]): continue
                    hist=X[s-H+1:s+1,m,:]
                    goal=X[s+P,m,:]
                    pos=hist[-1]; psi=safe_heading(hist)
                    R=rotmat(-psi)
                    hist_ego=(hist-pos)@R.T
                    goal_ego=(goal-pos)@R.T
                    hvec=hist_ego.reshape(-1)
                    cvec=build_lane_feature(lane_points,pos,psi,radius=lane_radius,kmax=lane_k)
                    nvec=build_neighbor_feature(X,V,s,m,pos,psi,radius=neighbor_radius,kmax=neighbor_k)
                    inp=np.concatenate([hvec,cvec,nvec],0).astype(np.float32)
                    out=goal_ego.astype(np.float32)
                    self.samples.append((inp,out,pos,psi))
        print(f"[DATA] {len(self.samples)} samples loaded")

    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx]

# ===== Model =====
class HistoryEncoder(nn.Module):
    def __init__(self, H, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2*H
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, h): return self.net(h)

class LaneEncoder(nn.Module):
    def __init__(self, lane_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2*lane_k + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, c): return self.net(c)

class NeighborEncoder(nn.Module):
    def __init__(self, nbr_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2*nbr_k
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, n): return self.net(n)

class MDNHead(nn.Module):
    def __init__(self, in_dim, n_comp=5, sigma_min=0.05, sigma_max=20.0):
        super().__init__()
        self.n_comp = n_comp
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        hidden = 128
        self.core = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_comp)
        self.mu = nn.Linear(hidden, n_comp*2)
        self.log_sigma = nn.Linear(hidden, n_comp*2)

    def forward(self, z):
        h = self.core(z)
        logit_pi = self.pi(h)
        mu = self.mu(h).view(-1, self.n_comp, 2)
        log_sigma = self.log_sigma(h).view(-1, self.n_comp, 2)
        min_log, max_log = math.log(self.sigma_min), math.log(self.sigma_max)
        log_sigma = torch.clamp(log_sigma, min=min_log, max=max_log)
        log_pi = torch.log_softmax(logit_pi, dim=-1)
        return log_pi, mu, log_sigma

class MDNGoalPredictorWithNeighbors(nn.Module):
    def __init__(self, H, lane_k, nbr_k, n_comp=5, sigma_min=0.05,
                 h_hidden=128,h_out=64,c_hidden=128,c_out=64,n_hidden=128,n_out=64):
        super().__init__()
        self.H=H; self.K=lane_k; self.M=nbr_k
        self.f_h=HistoryEncoder(H,hidden=h_hidden,out_dim=h_out)
        self.f_c=LaneEncoder(lane_k,hidden=c_hidden,out_dim=c_out)
        self.f_n=NeighborEncoder(nbr_k,hidden=n_hidden,out_dim=n_out)
        self.head=MDNHead(h_out+c_out+n_out,n_comp=n_comp,sigma_min=sigma_min)

    def forward(self,x):
        H2=2*self.H; Cdim=2*self.K+2; Ndim=2*self.M
        h_vec=x[:,:H2]; c_vec=x[:,H2:H2+Cdim]; n_vec=x[:,H2+Cdim:]
        h_star=self.f_h(h_vec); c_star=self.f_c(c_vec); n_star=self.f_n(n_vec)
        z=torch.cat([h_star,c_star,n_star],dim=-1)
        return self.head(z)

# ===== Loss =====
def mdn_nll_loss(log_pi, mu, log_sigma, y):
    y_exp=y.unsqueeze(1).expand_as(mu)
    inv_var=torch.exp(-2.0*log_sigma)
    inv_var=torch.clamp(inv_var,max=1e6)
    quad=torch.sum((y_exp-mu)**2*inv_var,dim=-1)
    log_det=torch.sum(2.0*log_sigma,dim=-1)
    log_prob=-0.5*(quad+log_det)
    log_mix=torch.logsumexp(log_pi+log_prob,dim=-1)
    nll=-torch.mean(log_mix)
    return torch.nan_to_num(nll,nan=1e6,posinf=1e6,neginf=1e6)

# ===== Train =====
def train_model(train_files,H=10,P=50,epochs=5,batch_size=128,lr=1e-3,
                lane_radius=20,lane_k=32,neighbor_radius=20,neighbor_k=6,
                device='cpu',save_path="goal_mlp_neighbors.pt"):
    dataset=GoalDataset(train_files,H,P,lane_radius,lane_k,neighbor_radius,neighbor_k)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    model=MDNGoalPredictorWithNeighbors(H=H,lane_k=lane_k,nbr_k=neighbor_k).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for ep in range(epochs):
        tot=0; cnt=0
        for inp,out,_,_ in loader:
            x=torch.from_numpy(np.stack(inp)).float().to(device)
            y=torch.from_numpy(np.stack(out)).float().to(device)
            opt.zero_grad()
            log_pi,mu,log_sigma=model(x)
            loss=mdn_nll_loss(log_pi,mu,log_sigma,y)
            loss.backward(); opt.step()
            tot+=loss.item()*x.size(0); cnt+=x.size(0)
        print(f"[EPOCH {ep}] nll={tot/cnt:.4f}")
    torch.save({'model_state':model.state_dict(),'H':H,'lane_k':lane_k,'neighbor_k':neighbor_k},save_path)
    print(f"[SAVE] model saved to {save_path}")
    return model

# ===== Test + Visualization =====
def test_and_visualize(model,test_file,H=10,P=30,lane_radius=20,lane_k=32,neighbor_radius=20,neighbor_k=6,outfile="goal_pred_vis.mp4"):
    D=np.load(test_file,allow_pickle=True)
    X=D['X'].astype(np.float32); V=D['V'].astype(bool)
    lane_points=D.get('lane_points',np.zeros((0,2),np.float32))
    T,M,_=X.shape
    m=None
    for j in range(M):
        if np.sum(V[:,j])>=H+P: m=j; break
    if m is None: return
    print(f"[TEST] agent={m}")
    frames=[]
    for s in range(H-1,T-P):
        if not (np.all(V[s-H+1:s+1,m]) and V[s+P,m]): continue
        hist=X[s-H+1:s+1,m,:]; goal=X[s+P,m,:]
        pos=hist[-1]; psi=safe_heading(hist); R=rotmat(-psi)
        hist_ego=(hist-pos)@R.T; goal_ego=(goal-pos)@R.T
        hvec=hist_ego.reshape(-1)
        cvec=build_lane_feature(lane_points,pos,psi,lane_radius,lane_k)
        nvec=build_neighbor_feature(X,V,s,m,pos,psi,neighbor_radius,neighbor_k)
        device=next(model.parameters()).device
        inp=np.concatenate([hvec,cvec,nvec],0)[None,:]
        with torch.no_grad():
            log_pi,mu,log_sigma=model(torch.from_numpy(inp).float().to(device))
            j=torch.argmax(torch.softmax(log_pi,dim=-1),dim=-1).item()
            pred_ego=mu[0,j].cpu().numpy()
        pred_global=pred_ego@R+pos
        frames.append((hist,goal,pred_global,hist_ego,goal_ego,pred_ego))
    fig,(axG,axE)=plt.subplots(1,2,figsize=(12,6))
    axG.set_title("Global"); axE.set_title("Ego")
    axG.set_aspect('equal'); axE.set_aspect('equal')
    all_x=X[:,m,0][V[:,m]]; all_y=X[:,m,1][V[:,m]]
    axG.set_xlim(all_x.min()-10,all_x.max()+10)
    axG.set_ylim(all_y.min()-10,all_y.max()+10)
    axE.set_xlim(-60,60); axE.set_ylim(-60,60)
    if lane_points.shape[0]>0:
        axG.scatter(lane_points[:,0],lane_points[:,1],s=4,alpha=0.2,c='gray',label='lanes')
    ln_histG,=axG.plot([],[],'-o',label="hist")
    sc_goalG=axG.scatter([],[],c='r',marker='x',s=80,label="GT goal")
    sc_predG=axG.scatter([],[],c='g',marker='*',s=80,label="Pred goal")
    ln_histE,=axE.plot([],[],'-o',label="hist")
    sc_goalE=axE.scatter([],[],c='r',marker='x',s=80,label="GT goal")
    sc_predE=axE.scatter([],[],c='g',marker='*',s=80,label="Pred goal")
    def init():
        ln_histG.set_data([],[]); ln_histE.set_data([],[])
        sc_goalG.set_offsets(np.empty((0,2))); sc_predG.set_offsets(np.empty((0,2)))
        sc_goalE.set_offsets(np.empty((0,2))); sc_predE.set_offsets(np.empty((0,2)))
        return ln_histG,sc_goalG,sc_predG,ln_histE,sc_goalE,sc_predE
    def animate(i):
        hist,goal,predG,histE,goalE,predE=frames[i]
        ln_histG.set_data(hist[:,0],hist[:,1])
        sc_goalG.set_offsets(goal[None,:]); sc_predG.set_offsets(predG[None,:])
        ln_histE.set_data(histE[:,0],histE[:,1])
        sc_goalE.set_offsets(goalE[None,:]); sc_predE.set_offsets(predE[None,:])
        return ln_histG,sc_goalG,sc_predG,ln_histE,sc_goalE,sc_predE
    ani=animation.FuncAnimation(fig,animate,init_func=init,frames=len(frames),interval=200,blit=False)
    ani.save(outfile,writer='ffmpeg')
    print(f"[SAVE] {outfile}")

# ===== Main =====
def main():
    train_dir=current_dir+"/waymo_one_shard_npz_all"
    train_files=glob.glob(os.path.join(train_dir,"*.npz"))[:400]
    test_file=current_dir+"/waymo_one_shard_npz_all/1a0c14835a811166.npz"
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=train_model(train_files,H=10,P=30,lane_radius=50,neighbor_radius=20,neighbor_k=6,
                      lr=1e-3,epochs=100,device=device,save_path="goal_mlp_neighbors.pt")
    test_and_visualize(model,test_file,H=10,P=30,lane_radius=50,neighbor_radius=20,neighbor_k=6,
                       outfile="goal_pred_neighbors_vis.mp4")

if __name__=="__main__":
    main()
