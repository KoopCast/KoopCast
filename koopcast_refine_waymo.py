#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal-Conditioned Koopman Predictor (EDMD) — standalone (now supports Neighbor context)
- Uses a pretrained MDN goal predictor:
  * Automatically infers architecture (mixture count, H, lane_k, neighbor_k, hidden sizes) from checkpoint.
  * Supports models WITH or WITHOUT Neighbor encoder.
  * Supports GAUSSIAN-30 checkpoints (e.g., goal_mlp_gaussian_30.pt).

What this script does
---------------------
1) Train K (EDMD) on windows from TRAIN_DIR:
   psi_t = [ x_t, x_{t-1}, ..., x_{t-H+1}, g_t, 1 ]  (latest-first), with z-score normalization (optional).
2) Test on TEST_FILE:
   - For each valid window, predict temporal goal with the MDN goal model (History/Lane[/Neighbor]).
   - Roll out P steps by multiplying K in normalized observable space.
   - Render lanes + history + GT future + predicted traj + GT/Pred goals to a video.
3) Save K (and normalization stats) to KOOPMAN_NPZ. If it already exists, just load and use.

Author: ChatGPT (Jungjin's assistant)
"""

import os, glob, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
from scipy.spatial import cKDTree as KDTree

# =========================
# Config
# =========================
current_dir = str(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR   = current_dir+"/waymo_one_shard_npz_all"
TEST_FILE   = current_dir+"/waymo/val/fe2dc1e4a17a2c05.npz"
GOAL_CKPT   = current_dir+"/goal_mlp_neighbors.pt"
KOOPMAN_NPZ = os.environ.get("KOOPMAN_NPZ", "koopman_K_goal_global.npz")

DEFAULTS = dict(
    H=10,
    P=int(os.environ.get("P", 30)),        # freely changeable: 20/50/80...
    ridge_lambda=1,
    max_train_files=100,                   # use first N files for speed
    include_bias=True,                     # psi = [..., g, 1]
    normalize=True,                        # z-score features for EDMD
    max_step_thresh=80.0,                  # skip windows with huge teleports
    outfile="koopman_goal_rollout.mp4",
    lane_radius=50.0,
    neighbor_radius=20.0                   # radius for neighbor features (in global frame, then ego-rotated)
)

# =========================
# Minimal helpers (global frame)
# =========================
def safe_heading(xy: np.ndarray) -> float:
    diffs = np.diff(xy, axis=0)
    for d in diffs[::-1]:
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return float(np.arctan2(d[1], d[0]))
    return 0.0

def rotmat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s,  c]], dtype=np.float32)

def build_lane_feature(lane_points, pos, psi, radius=50.0, kmax=32):
    """k-NN (by distance) lane points around 'pos', rotated into ego frame, flattened + [cos psi, sin psi]."""
    if lane_points.shape[0] == 0 or kmax <= 0:
        flat = np.zeros((2 * max(kmax,0),), np.float32)
        return np.concatenate([flat, [np.cos(psi), np.sin(psi)]], 0)

    kdt = KDTree(lane_points)
    idxs = kdt.query_ball_point(pos, r=radius)
    neigh = lane_points[idxs] if len(idxs) > 0 else np.empty((0, 2))
    if neigh.shape[0] > 0:
        d2 = np.sum((neigh - pos) ** 2, axis=1)
        order = np.argsort(d2)[:kmax]
        neigh = neigh[order]
    else:
        neigh = np.zeros((0, 2))

    R = rotmat(-psi)
    rel = (neigh - pos) @ R.T
    out = np.zeros((kmax, 2), np.float32)
    k = min(kmax, rel.shape[0])
    if k > 0:
        out[:k] = rel[:k]
    flat = out.reshape(-1)
    return np.concatenate([flat, [np.cos(psi), np.sin(psi)]], 0).astype(np.float32)

def build_neighbor_feature(X, V, s, m, pos, psi, radius=20.0, kmax=6):
    """
    Build neighbor feature in ego frame at time index 's'.
    - X: (T, M, 2), V: (T, M) validity mask
    - m: target agent index
    Returns (2*kmax,) vector of nearest neighbors in ego coordinates (by L2).
    """
    if kmax <= 0:
        return np.zeros((0,), np.float32)

    T, M, _ = X.shape
    R = rotmat(-psi)
    feats = []
    for j in range(M):
        if j == m: 
            continue
        if not V[s, j]:
            continue
        nbr_pos = X[s, j, :]
        if np.linalg.norm(nbr_pos - pos) <= radius:
            rel = (nbr_pos - pos) @ R.T
            feats.append(rel.astype(np.float32))
    feats.sort(key=lambda v: np.linalg.norm(v))
    feats = feats[:kmax]
    while len(feats) < kmax:
        feats.append(np.zeros(2, np.float32))
    return np.concatenate(feats, axis=0)  # (2*kmax,)

# =========================
# MDN goal model (robust loader — supports Split or Neighbor variants)
# =========================
class HistoryEncoder(nn.Module):
    def __init__(self, H, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * H
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, h): return self.net(h)

class LaneEncoder(nn.Module):
    def __init__(self, lane_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * lane_k + 2  # (2K) + (cos psi, sin psi)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, c): return self.net(c)

class NeighborEncoder(nn.Module):
    def __init__(self, nbr_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * nbr_k
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, n): return self.net(n)

class MDNHead(nn.Module):
    def __init__(self, in_dim, n_comp=5, sigma_min=0.05, sigma_max=20.0, hidden=128):
        super().__init__()
        self.n_comp = n_comp
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.core = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_comp)
        self.mu = nn.Linear(hidden, n_comp * 2)
        self.log_sigma = nn.Linear(hidden, n_comp * 2)

    def forward(self, z):
        h = self.core(z)
        logit_pi = self.pi(h)
        mu = self.mu(h).view(-1, self.n_comp, 2)
        log_sigma = self.log_sigma(h).view(-1, self.n_comp, 2)
        min_log = math.log(self.sigma_min); max_log = math.log(self.sigma_max)
        log_sigma = torch.clamp(log_sigma, min=min_log, max=max_log)
        log_pi = torch.log_softmax(logit_pi, dim=-1)
        return log_pi, mu, log_sigma

class MDNGoalPredictorSplit(nn.Module):
    """History + Lane"""
    def __init__(self, H, lane_k, n_comp=5, sigma_min=0.05,
                 h_hidden=128, h_out=64, c_hidden=128, c_out=64, head_hidden=128):
        super().__init__()
        self.H = H
        self.K = lane_k
        self.f_h = HistoryEncoder(H, hidden=h_hidden, out_dim=h_out)
        self.f_c = LaneEncoder(lane_k, hidden=c_hidden, out_dim=c_out)
        self.head = MDNHead(h_out + c_out, n_comp=n_comp, sigma_min=sigma_min, hidden=head_hidden)
    def forward(self, x):
        H2 = 2 * self.H
        h_vec = x[:, :H2]
        c_vec = x[:, H2:]
        z = torch.cat([self.f_h(h_vec), self.f_c(c_vec)], dim=-1)
        return self.head(z)

class MDNGoalPredictorWithNeighbors(nn.Module):
    """History + Lane + Neighbor"""
    def __init__(self, H, lane_k, nbr_k, n_comp=5, sigma_min=0.05,
                 h_hidden=128, h_out=64, c_hidden=128, c_out=64,
                 n_hidden=128, n_out=64, head_hidden=128):
        super().__init__()
        self.H = H; self.K = lane_k; self.M = nbr_k
        self.f_h = HistoryEncoder(H, hidden=h_hidden, out_dim=h_out)
        self.f_c = LaneEncoder(lane_k, hidden=c_hidden, out_dim=c_out)
        self.f_n = NeighborEncoder(nbr_k, hidden=n_hidden, out_dim=n_out)
        self.head = MDNHead(h_out + c_out + n_out, n_comp=n_comp, sigma_min=sigma_min, hidden=head_hidden)
    def forward(self, x):
        H2 = 2 * self.H; Cdim = 2 * self.K + 2; Ndim = 2 * self.M
        h_vec = x[:, :H2]
        c_vec = x[:, H2:H2 + Cdim]
        n_vec = x[:, H2 + Cdim: H2 + Cdim + Ndim]
        z = torch.cat([self.f_h(h_vec), self.f_c(c_vec), self.f_n(n_vec)], dim=-1)
        return self.head(z)

def mdn_nll_loss(log_pi, mu, log_sigma, y):
    y_exp = y.unsqueeze(1).expand_as(mu)
    inv_var = torch.exp(-2.0 * log_sigma)
    inv_var = torch.clamp(inv_var, max=1e6)
    quad = torch.sum((y_exp - mu) ** 2 * inv_var, dim=-1)
    log_det = torch.sum(2.0 * log_sigma, dim=-1)
    log_prob = -0.5 * (quad + log_det)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)
    nll = -torch.mean(log_mix)
    if not torch.isfinite(nll):
        nll = torch.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=1e6)
    return nll

# Robust loader that infers architecture sizes from checkpoint shapes, supports neighbor variants
def load_goal_predictor(device="cpu"):
    if not os.path.exists(GOAL_CKPT):
        raise FileNotFoundError(f"Goal checkpoint not found: {GOAL_CKPT}")
    ckpt = torch.load(GOAL_CKPT, map_location=device)
    state = ckpt.get('model_state', ckpt)  # allow raw state_dict or wrapped

    # Detect presence of neighbor encoder
    has_neighbor = any(k.startswith('f_n.') for k in state.keys())

    # Infer dimensions from layer shapes
    in_h = state['f_h.net.0.weight'].shape[1]             # 2*H
    H_used = in_h // 2
    in_c = state['f_c.net.0.weight'].shape[1]             # 2*lane_k + 2
    lane_k = (in_c - 2) // 2
    if has_neighbor:
        in_n = state['f_n.net.0.weight'].shape[1]         # 2*neighbor_k
        neighbor_k = in_n // 2
        n_hidden = state['f_n.net.0.weight'].shape[0]
        n_out    = state['f_n.net.2.weight'].shape[0]
    else:
        neighbor_k = 0
        n_hidden = n_out = None

    n_comp      = state['head.pi.weight'].shape[0]
    h_hidden    = state['f_h.net.0.weight'].shape[0]
    h_out       = state['f_h.net.2.weight'].shape[0]
    c_hidden    = state['f_c.net.0.weight'].shape[0]
    c_out       = state['f_c.net.2.weight'].shape[0]
    head_hidden = state['head.core.0.weight'].shape[0]

    # Instantiate the right model
    if has_neighbor:
        model = MDNGoalPredictorWithNeighbors(
            H=H_used, lane_k=lane_k, nbr_k=neighbor_k, n_comp=n_comp,
            h_hidden=h_hidden, h_out=h_out, c_hidden=c_hidden, c_out=c_out,
            n_hidden=n_hidden, n_out=n_out, head_hidden=head_hidden
        ).to(device)
    else:
        model = MDNGoalPredictorSplit(
            H=H_used, lane_k=lane_k, n_comp=n_comp,
            h_hidden=h_hidden, h_out=h_out, c_hidden=c_hidden, c_out=c_out,
            head_hidden=head_hidden
        ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[LOAD] goal ckpt={GOAL_CKPT} | H={H_used} lane_k={lane_k} "
          f"{'neighbor_k='+str(neighbor_k) if has_neighbor else '(no neighbor)'} | mixtures={n_comp}")

    meta = dict(H=H_used, lane_k=lane_k, neighbor_k=neighbor_k, has_neighbor=has_neighbor)
    return model, meta

# =========================
# Utilities for EDMD & data windows
# =========================
OBS_KIND = os.environ.get("OBS_KIND", "poly2")

def make_hist_latest_first(X: np.ndarray, s: int, H: int, m: int) -> np.ndarray:
    """Return (H,2): [x_s, x_{s-1}, ..., x_{s-H+1}]"""
    hist = X[s - H + 1 : s + 1, m, :]
    return hist[::-1].copy()

def flatten_hist(latest_first_hist: np.ndarray) -> np.ndarray:
    return latest_first_hist.reshape(-1)

def window_is_sane(X: np.ndarray, V: np.ndarray, m: int, s: int, H: int, P: int, max_step: float) -> bool:
    # All required indices must be valid
    idxs = list(range(s - H + 1, s + 1)) + [s + P, s + P + 1]
    for t in idxs:
        if t < 0 or t >= X.shape[0] or not V[t, m]:
            return False
    # history smoothness
    for t0 in range(s - H + 1, s):
        d = np.linalg.norm(X[t0 + 1, m] - X[t0, m])
        if d > max_step:
            return False
    # goal step not a teleport
    d_goal = np.linalg.norm(X[s + P + 1, m] - X[s + P, m])
    if d_goal > max_step:
        return False
    return True

# =========================
# Observable (lifting) API
# =========================
def observable(hist_lf: np.ndarray, g: np.ndarray, kind: str = OBS_KIND) -> np.ndarray:
    """Return phi_t (1D np.float32). Keep x at the FIRST TWO entries of phi."""
    if kind == "default":
        return np.concatenate([hist_lf.reshape(-1), g], 0).astype(np.float32)
    elif kind == "poly2":
        x_t = hist_lf[0]
        phi_x = np.array([x_t[0], x_t[1], x_t[0]**2, x_t[1]**2, x_t[0]*x_t[1]], np.float32)
        return np.concatenate([phi_x, hist_lf.reshape(-1), g], 0).astype(np.float32)
    elif kind == "fourier3":
        x_t = hist_lf[0].astype(np.float32)
        ks = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        f = np.concatenate([
            np.sin(ks * x_t[0]), np.cos(ks * x_t[0]),
            np.sin(ks * x_t[1]), np.cos(ks * x_t[1])
        ]).astype(np.float32)
        return np.concatenate([x_t, f, hist_lf.reshape(-1), g], 0).astype(np.float32)
    else:
        raise ValueError(f"Unknown OBS_KIND={kind}")

def decode_x_from_psi(psi: np.ndarray, kind: str = OBS_KIND, H: int = 10) -> np.ndarray:
    """ Recover x_{t+1} from psi_{t+1}. For all defined observables, x is stored in the first two dims. """
    return psi[:2]

# =========================
# EDMD fit / save / load
# =========================
def collect_edmd_pairs(train_files, H, P, include_bias=True, max_step=80.0):
    Psis, Psis_next = [], []
    for f in train_files:
        try:
            D = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if not ("X" in D and "V" in D):
            continue
        X = D["X"].astype(np.float32)  # (T,M,2)
        V = D["V"].astype(bool)
        # Per-file origin shift to reduce numeric range
        xmin = np.nanmin(X[:,:,0]); ymin = np.nanmin(X[:,:,1])
        X[:,:,0] -= xmin; X[:,:,1] -= ymin

        T, M, _ = X.shape
        for m in range(M):
            for s in range(H - 1, T - P - 1):  # need s+P+1
                if not window_is_sane(X, V, m, s, H, P, max_step):
                    continue
                # psi_t
                hist_lf = make_hist_latest_first(X, s, H, m)
                g_t = X[s + P, m]
                psi = observable(hist_lf, g_t, kind=OBS_KIND)
                if include_bias:
                    psi = np.concatenate([psi, [1.0]], 0)
                # psi_{t+1}
                hist_lf_next = make_hist_latest_first(X, s + 1, H, m)
                g_t1 = X[s + P + 1, m]
                psi_nxt = observable(hist_lf_next, g_t1, kind=OBS_KIND)
                if include_bias:
                    psi_nxt = np.concatenate([psi_nxt, [1.0]], 0)
                if (not np.isfinite(psi).all()) or (not np.isfinite(psi_nxt).all()):
                    continue
                Psis.append(psi); Psis_next.append(psi_nxt)

    if len(Psis) == 0:
        raise RuntimeError("No EDMD training pairs were collected. Check data/thresholds.")
    Psi = np.stack(Psis).astype(np.float32)
    Psi_next = np.stack(Psis_next).astype(np.float32)
    return Psi, Psi_next

def fit_koopman(Psi: np.ndarray, Psi_next: np.ndarray, ridge_lambda=1e-3, normalize=True):
    X = Psi; Y = Psi_next
    if normalize:
        mu = X.mean(axis=0)
        sig = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sig
        Yn = (Y - mu) / sig
    else:
        mu = np.zeros(X.shape[1], np.float32)
        sig = np.ones(X.shape[1], np.float32)
        Xn, Yn = X, Y

    d = Xn.shape[1]
    XtX = Xn.T @ Xn
    XtY = Xn.T @ Yn
    A = XtX + ridge_lambda * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, XtY)
    K = W.T
    return dict(K=K.astype(np.float32), mu=mu.astype(np.float32), sig=sig.astype(np.float32))

def save_K(pack, path, H, P, include_bias, normalize):
    np.savez(path, K=pack["K"], mu=pack["mu"], sig=pack["sig"],
             H=H, P=P, include_bias=include_bias, normalize=normalize)
    print(f"[K] Estimated and saved to {path}, shape={pack['K'].shape}")

def load_K(path):
    Z = np.load(path)
    pack = dict(K=Z["K"].astype(np.float32), mu=Z["mu"].astype(np.float32), sig=Z["sig"].astype(np.float32))
    meta = dict(H=int(Z["H"]), P=int(Z["P"]), include_bias=bool(Z["include_bias"]), normalize=bool(Z["normalize"]))
    print(f"[K] Loaded from {path}, shape={pack['K'].shape}, H={meta['H']}, P={meta['P']}, bias={meta['include_bias']}")
    return pack, meta

# =========================
# Rollout with K
# =========================
def rollout_with_K(history_global: np.ndarray, g0_global: np.ndarray, pack: dict, include_bias=True, P=50):
    H = history_global.shape[0]
    hist_lf = history_global[::-1].copy()
    psi = observable(hist_lf, g0_global, kind=OBS_KIND)
    if include_bias:
        psi = np.concatenate([psi, [1.0]], 0)

    K = pack["K"]; mu = pack["mu"]; sig = pack["sig"]
    traj = []
    for _ in range(P):
        x_n = (psi - mu) / sig
        psi_next_n = K @ x_n
        psi = psi_next_n * sig + mu
        x_next = decode_x_from_psi(psi, kind=OBS_KIND, H=H).copy()
        traj.append(x_next)
    traj = np.stack(traj)
    # Try to read back goal slice if design permits; otherwise NaNs
    try:
        g_last = psi[2*H:2*H+2].copy()
    except Exception:
        g_last = np.full(2, np.nan, dtype=np.float32)
    return traj, g_last

# =========================
# Agent selection helper
# =========================
def pick_best_agent(V: np.ndarray, H: int, P: int):
    T, M = V.shape
    best_m, best_count = None, -1
    for m in range(M):
        cnt = 0
        for s in range(H - 1, T - P):
            if np.all(V[s - H + 1 : s + 1, m]) and np.all(V[s + 1 : s + P + 1, m]) and V[s + P, m]:
                cnt += 1
        if cnt > best_count:
            best_m, best_count = m, cnt
    return best_m, best_count

# =========================
# Visualization
# =========================
def visualize_sequence(ax, lane_points, x_hist, x_future, x_pred, g_gt, g_pred):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title('Global frame — history / GT future / K rollout')
    if lane_points is not None and lane_points.shape[0] > 0:
        ax.scatter(lane_points[:,0], lane_points[:,1], s=3, alpha=0.15, c='gray', label='lanes')
    ax.plot(x_hist[:,0],   x_hist[:,1],   '-o', lw=2, ms=3, label='history')
    ax.plot(x_future[:,0], x_future[:,1], '-o', lw=2, ms=3, label='GT future')
    ax.plot(x_pred[:,0],   x_pred[:,1],   '-o', lw=2, ms=3, label='K rollout')
    ax.scatter([g_gt[0]],[g_gt[1]], c='r', marker='x', s=80,  label='GT goal')
    ax.scatter([g_pred[0]],[g_pred[1]], c='g', marker='*', s=100, label='Pred goal (MDN)')
    ax.legend(loc='best')

# =========================
# Main test pipeline
# =========================
def test_with_video(K_pack, H, P, include_bias, outfile):
    D = np.load(TEST_FILE, allow_pickle=True)
    X = D['X'].astype(np.float32); V = D['V'].astype(bool)
    lane_points = D.get('lane_points', np.zeros((0,2), np.float32))

    # origin shift
    xmin = np.nanmin(X[:,:,0]); ymin = np.nanmin(X[:,:,1])
    X[:,:,0] -= xmin; X[:,:,1] -= ymin
    if lane_points.shape[0] > 0:
        lane_points = lane_points - np.array([xmin, ymin], dtype=np.float32)

    T, M, _ = X.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, meta = load_goal_predictor(device)
    H_goal = meta['H']; lane_k = meta['lane_k']
    has_neighbor = meta['has_neighbor']; neighbor_k = meta['neighbor_k']

    # agent
    m, count = pick_best_agent(V, H, P)
    if m is None or count <= 0:
        raise RuntimeError("No suitable agent found in TEST_FILE for given H,P.")
    print(f"[TEST] agent={m}, valid windows={count}")

    valid_xy = X[V[:,m], m, :]
    xmin_v, ymin_v = valid_xy.min(axis=0) - 10
    xmax_v, ymax_v = valid_xy.max(axis=0) + 10

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    frames = []
    ADE_list, FDE_list = [], []

    for s in range(H - 1, T - P):
        if not (np.all(V[s - H + 1 : s + 1, m]) and V[s + P, m] and np.all(V[s + 1 : s + P + 1, m])):
            continue
        # history (global -> ego features for goal net)
        x_hist = X[s - H + 1 : s + 1, m, :].copy()
        pos = x_hist[-1]
        psi_heading = safe_heading(x_hist)
        R = rotmat(-psi_heading)
        x_hist_ego = (x_hist - pos) @ R.T
        h_vec = x_hist_ego.reshape(-1)

        c_vec = build_lane_feature(lane_points, pos, psi_heading,
                                   radius=DEFAULTS['lane_radius'], kmax=lane_k)

        if has_neighbor and neighbor_k > 0:
            n_vec = build_neighbor_feature(X, V, s, m, pos, psi_heading,
                                           radius=DEFAULTS['neighbor_radius'], kmax=neighbor_k)
            inp = np.concatenate([h_vec, c_vec, n_vec], 0)[None, :].astype(np.float32)
        else:
            inp = np.concatenate([h_vec, c_vec], 0)[None, :].astype(np.float32)

        # goal predictor (ego → global)
        with torch.no_grad():
            log_pi, mu, log_sigma = model(torch.from_numpy(inp).float().to(device))
            pi = torch.softmax(log_pi, dim=-1)
            # Use mixture EXPECTATION (works robustly for any mixture count, e.g., 30)
            pred_ego = torch.sum(pi.unsqueeze(-1) * mu, dim=1)[0].cpu().numpy()
        g_pred = (pred_ego @ R) + pos  # ego -> global

        # Koopman rollout
        x_hist_for_K = x_hist[-H:, :] if H_goal != H else x_hist
        x_pred, _ = rollout_with_K(x_hist_for_K, g_pred, K_pack, include_bias=include_bias, P=P)

        # GT future
        x_future = X[s + 1 : s + P + 1, m, :].copy()
        ade = float(np.linalg.norm(x_pred - x_future, axis=1).mean())
        fde = float(np.linalg.norm(x_pred[-1] - x_future[-1]))
        ADE_list.append(ade); FDE_list.append(fde)

        # restore origin for rendering
        frames.append((
            x_hist + np.array([xmin, ymin]),
            x_future + np.array([xmin, ymin]),
            x_pred + np.array([xmin, ymin]),
            X[s + P, m, :] + np.array([xmin, ymin]),
            g_pred + np.array([xmin, ymin]),
        ))

    # animation
    def init():
        ax.set_xlim(xmin_v + xmin, xmax_v + xmin)
        ax.set_ylim(ymin_v + ymin, ymax_v + ymin)
        ax.set_aspect('equal')
        return []

    def animate(i):
        x_hist, x_future, x_pred, g_gt, g_pred = frames[i]
        ax.set_xlim(xmin_v + xmin, xmax_v + xmin)
        ax.set_ylim(ymin_v + ymin, ymax_v + ymin)
        visualize_sequence(ax, lane_points + np.array([xmin, ymin]),
                           x_hist, x_future, x_pred, g_gt, g_pred)
        return []

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(frames), interval=200, blit=False)
    try:
        ani.save(outfile, writer='ffmpeg')
        print(f"[SAVE] {outfile}")
    except Exception as e:
        fout = outfile.replace('.mp4', '.gif')
        ani.save(fout, writer='pillow')
        print(f"[SAVE] {fout} (ffmpeg not available: {e})")

    if len(ADE_list) > 0:
        print(f"[METRIC] ADE={np.mean(ADE_list):.3f} | FDE={np.mean(FDE_list):.3f} | windows={len(ADE_list)}")
    else:
        print("[WARN] No windows evaluated.")

def evaluate_dir(test_dir, K_pack, H, P, include_bias=True):
    """Evaluate ADE/FDE over all npz files in test_dir using predicted goals (MDN)."""
    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {test_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, meta = load_goal_predictor(device)
    H_goal = meta['H']; lane_k = meta['lane_k']
    has_neighbor = meta['has_neighbor']; neighbor_k = meta['neighbor_k']

    all_ADE, all_FDE = [], []
    total_windows = 0

    for f in files:
        try:
            D = np.load(f, allow_pickle=True)
            X = D['X'].astype(np.float32)
            V = D['V'].astype(bool)
            lane_points = D.get('lane_points', np.zeros((0,2), np.float32))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue

        # origin shift
        xmin, ymin = np.nanmin(X[:,:,0]), np.nanmin(X[:,:,1])
        X[:,:,0] -= xmin; X[:,:,1] -= ymin
        if lane_points.shape[0] > 0:
            lane_points = lane_points - np.array([xmin, ymin], dtype=np.float32)

        T, M, _ = X.shape
        m, count = pick_best_agent(V, H, P)
        if m is None or count <= 0:
            continue

        file_ADE, file_FDE = [], []
        for s in range(H - 1, T - P):
            if not (np.all(V[s - H + 1 : s + 1, m]) and V[s + P, m] and np.all(V[s + 1 : s + P + 1, m])):
                continue

            x_hist = X[s - H + 1 : s + 1, m, :].copy()
            pos = x_hist[-1]
            psi_heading = safe_heading(x_hist)
            R = rotmat(-psi_heading)
            x_hist_ego = (x_hist - pos) @ R.T
            h_vec = x_hist_ego.reshape(-1)
            c_vec = build_lane_feature(lane_points, pos, psi_heading,
                                       radius=DEFAULTS['lane_radius'], kmax=lane_k)

            if has_neighbor and neighbor_k > 0:
                n_vec = build_neighbor_feature(X, V, s, m, pos, psi_heading,
                                               radius=DEFAULTS['neighbor_radius'], kmax=neighbor_k)
                inp = np.concatenate([h_vec, c_vec, n_vec], 0)[None, :].astype(np.float32)
            else:
                inp = np.concatenate([h_vec, c_vec], 0)[None, :].astype(np.float32)

            with torch.no_grad():
                log_pi, mu, log_sigma = model(torch.from_numpy(inp).float().to(device))
                pi = torch.softmax(log_pi, dim=-1)
                pred_ego = torch.sum(pi.unsqueeze(-1) * mu, dim=1)[0].cpu().numpy()
            g_pred = (pred_ego @ R) + pos

            x_hist_for_K = x_hist[-H:, :] if H_goal != H else x_hist
            x_pred, _ = rollout_with_K(x_hist_for_K, g_pred, K_pack, include_bias=include_bias, P=P)

            x_future = X[s + 1 : s + P + 1, m, :].copy()
            ade = float(np.linalg.norm(x_pred - x_future, axis=1).mean())
            fde = float(np.linalg.norm(x_pred[-1] - x_future[-1]))
            file_ADE.append(ade); file_FDE.append(fde)

        if len(file_ADE) > 0:
            mean_ADE, mean_FDE = np.mean(file_ADE), np.mean(file_FDE)
            print(f"[RESULT] {os.path.basename(f)} | ADE={mean_ADE:.3f} FDE={mean_FDE:.3f} | windows={len(file_ADE)}")
            all_ADE.extend(file_ADE); all_FDE.extend(file_FDE)
            total_windows += len(file_ADE)

    if total_windows > 0:
        print("=============================================")
        print(f"[SUMMARY] Files={len(files)} Windows={total_windows}")
        print(f"[SUMMARY] ADE={np.mean(all_ADE):.3f} | FDE={np.mean(all_FDE):.3f}")
    else:
        print("[WARN] No valid windows evaluated in test_dir")

# =========================
# End-to-end
# =========================
def main():
    H = DEFAULTS['H']
    P = DEFAULTS['P']
    include_bias = DEFAULTS['include_bias']
    normalize = DEFAULTS['normalize']

    # 1) K: load or fit
    if os.path.exists(KOOPMAN_NPZ):
        K_pack, meta = load_K(KOOPMAN_NPZ)
        if meta['H'] != H or meta['P'] != P:
            print(f"[WARN] Loaded K was fit for H={meta['H']}, P={meta['P']} but you set H={H}, P={P}.")
    else:
        print("[EDMD] Collecting training pairs ...")
        all_files = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.npz')))
        use_files = all_files[: DEFAULTS['max_train_files']]
        Psi, Psi_next = collect_edmd_pairs(
            use_files, H, P, include_bias=include_bias, max_step=DEFAULTS['max_step_thresh']
        )
        print(f"[EDMD] Pairs: {Psi.shape[0]}, dim={Psi.shape[1]}")
        pack = fit_koopman(Psi, Psi_next, ridge_lambda=DEFAULTS['ridge_lambda'], normalize=normalize)
        save_K(pack, KOOPMAN_NPZ, H=H, P=P, include_bias=include_bias, normalize=normalize)
        K_pack = pack

    if 'K_pack' not in locals():
        K_pack = K_pack if 'K_pack' in globals() else pack if 'pack' in locals() else K_pack

    # 2) Test + video
    test_with_video(K_pack, H, P, include_bias, DEFAULTS['outfile'])

    # 3) Eval on a directory
    VAL_DIR = current_dir+"/waymo/val"
    evaluate_dir(VAL_DIR, K_pack, H, P, include_bias=include_bias)

if __name__ == "__main__":
    main()
