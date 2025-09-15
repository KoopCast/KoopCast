#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDN + Koopman (stochastic one-shot target) trajectory predictor — Multi-Agent MDN (instant neighbors only)

Pipeline (single file):
  1) Train an MDN p(g_t | h_t^{(i)}, m_t^{(i)}) for the P-step-ahead target g_t^{(i)} = x_{t+P}^{(i)}
     where:
       - h_t^{(i)} = (x_{t-H+1}^{(i)}, ..., x_t^{(i)})  (agent i history)
       - m_t^{(i)} = Positions of neighbors at time s within radius R (relative to agent i).
  2) Estimate a global Koopman-like linear operator K on lifted state
     psi_t = [x_t, x_{t-1}, ..., x_{t-H+1}, g_t, 1]
     so that psi_{t+1} ≈ K psi_t with g_{t+1} = x_{t+1+P} (unchanged).
  3) Evaluate on a test file: use MDN mean and best-of-S MDN samples for g_t,
     roll forward P steps via K, and report ADE / FDE.

Author: ChatGPT (Jungjin's assistant)
"""

import os
import math
import glob
import argparse
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------- Hyperparameters & Defaults ----------------------
HISTORY_LENGTH_DEFAULT = 8
PRED_LENGTH_DEFAULT    = 12
MDN_COMPONENTS_DEFAULT = 6   # mixture components
MDN_HIDDEN_DEFAULT     = 128
MDN_EPOCHS_DEFAULT     = 20
MDN_BATCH_DEFAULT      = 512
MDN_LR_DEFAULT         = 1e-3
SIGMA_MIN_DEFAULT      = 0.05  # meters
RIDGE_LAMBDA_DEFAULT   = 1e-3  # Koopman ridge
SAMPLES_DEFAULT        = 20     # best-of-S samples for metrics

# Multi-agent context defaults
MAX_NEIGHBORS_DEFAULT  = 6      # M (0 => disable)
NEIGHBOR_RADIUS_DEFAULT = 5.0   # meters
NEIGHBOR_RELATIVE_DEFAULT = True  # frame-wise relative coordinates

# ---------------------- Utility ----------------------

def valid_windows(traj: np.ndarray, H: int, P: int):
    T = traj.shape[0]
    mask = np.all(np.isfinite(traj), axis=1)
    for s in range(H-1, T - P):
        if np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+1]) and mask[s+P]:
            yield s

def build_multiagent_context_instant(
    data: np.ndarray,
    s: int,
    i: int,
    neighbor_radius: float,
    max_neighbors: int,
    relative: bool = True
) -> np.ndarray:
    """
    Use ONLY neighbor positions at time s.
    Returns vector shape (2*max_neighbors,)
    """
    if max_neighbors <= 0:
        return np.zeros((0,), dtype=np.float32)

    T, N, _ = data.shape
    pos_i_s = data[s, i, :]  # (2,)

    candidates = []
    for j in range(N):
        if j == i:
            continue
        pos_j_s = data[s, j, :]
        if not np.all(np.isfinite(pos_j_s)):
            continue
        if np.linalg.norm(pos_j_s - pos_i_s) <= neighbor_radius:
            candidates.append((j, pos_j_s))

    candidates.sort(key=lambda item: np.linalg.norm(item[1] - pos_i_s))
    neighbors = candidates[:max_neighbors]

    pieces = []
    for _, pos_j_s in neighbors:
        if relative:
            vec = pos_j_s - pos_i_s
        else:
            vec = pos_j_s
        pieces.append(vec.astype(np.float32))

    while len(pieces) < max_neighbors:
        pieces.append(np.zeros((2,), dtype=np.float32))

    return np.concatenate(pieces, axis=0)  # (2*max_neighbors,)

def collect_mdn_dataset_multiagent(
    train_files: List[str], H: int, P: int,
    max_neighbors: int, neighbor_radius: float, neighbor_relative: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (z, g) for MDN training:
      z = [h_t^{(i)}, m_t^{(i)}], h_t^{(i)} ∈ R^{2H}, m_t^{(i)} ∈ R^{2*max_neighbors}
      g = x_{t+P}^{(i)} ∈ R^2
    """
    H2 = 2 * H
    Mdim = 2 * max_neighbors
    X_list, Y_list = [], []

    for fpath in train_files:
        data = np.load(fpath)  # (T, N, 2)
        T, N, _ = data.shape
        for i in range(N):
            traj_i = data[:, i, :]
            for s in valid_windows(traj_i, H, P):
                hist_i = traj_i[s-H+1:s+1]                 # (H,2)
                g_i    = traj_i[s+P]                       # (2,)
                hvec   = hist_i[::-1].reshape(H2)          # latest-first
                mvec   = build_multiagent_context_instant(
                    data, s, i,
                    neighbor_radius=neighbor_radius,
                    max_neighbors=max_neighbors,
                    relative=neighbor_relative
                ) if max_neighbors > 0 else np.zeros((0,), dtype=np.float32)
                z = np.concatenate([hvec.astype(np.float32), mvec.astype(np.float32)], axis=0)
                X_list.append(z)
                Y_list.append(g_i.astype(np.float32))

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y

def collect_koopman_pairs(train_files: List[str], H: int, P: int, use_bias: bool=True):
    H2 = 2 * H
    D  = H2 + 2 + (1 if use_bias else 0)
    P_cols, F_cols = [], []
    for fpath in train_files:
        data = np.load(fpath)  # (T, N, 2)
        T, N, _ = data.shape
        for ag in range(N):
            traj = data[:, ag, :]
            mask = np.all(np.isfinite(traj), axis=1)
            for s in range(H-1, T - P - 1):
                if not (np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+2])):
                    continue
                hist  = traj[s-H+1:s+1]
                histp = traj[s-H+2:s+2]
                gt_g    = traj[s+P]
                gt_gp   = traj[s+1+P]
                hvec  = hist[::-1].reshape(H2)
                hvecp = histp[::-1].reshape(H2)
                psi   = np.concatenate([hvec, gt_g], axis=0)
                psip  = np.concatenate([hvecp, gt_gp], axis=0)
                if use_bias:
                    psi  = np.concatenate([psi,  [1.0]], axis=0)
                    psip = np.concatenate([psip, [1.0]], axis=0)
                P_cols.append(psi)
                F_cols.append(psip)
    if len(P_cols) == 0:
        raise RuntimeError("No Koopman pairs collected. Check data and masks.")
    Pmat = np.stack(P_cols, axis=1)  # (D,M)
    Fmat = np.stack(F_cols, axis=1)  # (D,M)
    return Pmat.astype(np.float64), Fmat.astype(np.float64)

# ---------------------- MDN Model ----------------------

LOG2PI = math.log(2.0 * math.pi)

class MDN(nn.Module):
    def __init__(self, in_dim: int, n_components: int = MDN_COMPONENTS_DEFAULT,
                 hidden: int = MDN_HIDDEN_DEFAULT, sigma_min: float = SIGMA_MIN_DEFAULT):
        super().__init__()
        self.K = n_components
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_pi   = nn.Linear(hidden, self.K)
        self.head_mu   = nn.Linear(hidden, self.K * 2)
        self.head_logS = nn.Linear(hidden, self.K * 2)

    def forward(self, x):
        h = self.net(x)
        pi_logits = self.head_pi(h)
        mu        = self.head_mu(h).view(-1, self.K, 2)
        log_sigma = self.head_logS(h).view(-1, self.K, 2)
        min_log = math.log(self.sigma_min)
        log_sigma = torch.clamp(log_sigma, min=min_log)
        log_pi = torch.log_softmax(pi_logits, dim=1)
        return log_pi, mu, log_sigma

    @torch.no_grad()
    def mixture_mean(self, x):
        log_pi, mu, _ = self.forward(x)
        pi = torch.exp(log_pi)
        mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        return mean

    @torch.no_grad()
    def sample(self, x, n_samples: int = 1):
        B = x.shape[0]
        log_pi, mu, log_sigma = self.forward(x)
        pi = torch.exp(log_pi)
        comps = torch.multinomial(pi, num_samples=n_samples, replacement=True)
        mu_g = torch.gather(mu, 1, comps.unsqueeze(-1).expand(B, n_samples, 2))
        sigma = torch.exp(torch.gather(log_sigma, 1, comps.unsqueeze(-1).expand(B, n_samples, 2)))
        eps = torch.randn_like(mu_g)
        return mu_g + sigma * eps

def mdn_nll(log_pi, mu, log_sigma, y):
    B, K, _ = mu.shape
    y_exp = y.unsqueeze(1).expand(B, K, 2)
    inv_var = torch.exp(-2.0 * log_sigma)
    log_norm = -0.5 * (
        torch.sum((y_exp - mu)**2 * inv_var, dim=2) +
        2.0 * torch.sum(log_sigma, dim=2) +
        2.0 * LOG2PI
    )
    log_mix = torch.logsumexp(log_pi + log_norm, dim=1)
    return -torch.mean(log_mix)

# ---------------------- Koopman Estimation ----------------------

def estimate_K(Pmat: np.ndarray, Fmat: np.ndarray, ridge: float = RIDGE_LAMBDA_DEFAULT) -> np.ndarray:
    D, M = Pmat.shape
    A = Pmat @ Pmat.T
    if ridge > 0:
        A = A + ridge * np.eye(D, dtype=Pmat.dtype)
    B = Fmat @ Pmat.T
    K = np.linalg.solve(A, B.T).T
    return K

# ---------------------- Build psi and rollout ----------------------

def build_psi_from_hist_g(hist: np.ndarray, g: np.ndarray, use_bias: bool=True) -> np.ndarray:
    H = hist.shape[0]
    hvec = hist[::-1].reshape(2*H)
    psi = np.concatenate([hvec, g], axis=0)
    if use_bias:
        psi = np.concatenate([psi, [1.0]], axis=0)
    return psi

def rollout_with_K(hist: np.ndarray, g: np.ndarray, K: np.ndarray, P: int, use_bias: bool=True) -> np.ndarray:
    psi = build_psi_from_hist_g(hist, g, use_bias)
    preds = []
    for _ in range(P):
        psi = K @ psi
        preds.append(psi[:2].copy())
    return np.stack(preds, axis=0)

# ---------------------- Dataset iterators ----------------------

def iter_test_windows(data: np.ndarray, H: int, P: int):
    T, N, _ = data.shape
    for i in range(N):
        traj = data[:, i, :]
        mask = np.all(np.isfinite(traj), axis=1)
        for s in range(H-1, T - P):
            if not (np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+1])):
                continue
            hist = traj[s-H+1:s+1]
            gt   = traj[s+1:s+P+1]
            yield i, s, hist, gt

# ---------------------- Training / Evaluation ----------------------

def train_mdn(X: np.ndarray, Y: np.ndarray, in_dim: int, K: int, hidden: int,
              epochs: int, batch_size: int, lr: float, device: str='cpu') -> MDN:
    model = MDN(in_dim, n_components=K, hidden=hidden).to(device)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            log_pi, mu, log_sigma = model(xb)
            loss = mdn_nll(log_pi, mu, log_sigma, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(ds)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[MDN] Epoch {ep}/{epochs}  NLL: {avg:.6f}")
    return model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import time

# ---------------------- Visualization ----------------------
def visualize_prediction(test_file: str, H: int, P: int, mdn: MDN, K: np.ndarray,
                         device: str='cpu', outfile: str="mdn_koopman_vis.mp4",
                         max_neighbors: int = 0, neighbor_radius: float = 5.0, neighbor_relative: bool = True):
    data = np.load(test_file)  # (T,N,2)

    T, N, _ = data.shape
    chosen = None
    for i in range(N):
        traj = data[:, i, :]
        mask = np.all(np.isfinite(traj), axis=1)
        if np.sum(mask) > H+P:
            chosen = i
            break
    if chosen is None:
        print("[VIZ] No eligible agent for visualization.")
        return

    traj = data[:, chosen, :]
    mask = np.all(np.isfinite(traj), axis=1)
    hist = gt = None
    g_mean = None

    for s in range(H-1, T-P-1):
        if not (np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+1])):
            continue
        hist = traj[s-H+1:s+1]
        gt   = traj[s+1:s+P+1]
        gt_goal = traj[s+P]

        hvec = hist[::-1].reshape(1, 2*H).astype(np.float32)
        mvec = build_multiagent_context_instant(
            data, s, chosen,
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors,
            relative=neighbor_relative
        ).reshape(1,-1).astype(np.float32) if max_neighbors > 0 else np.zeros((1,0), dtype=np.float32)
        z = np.concatenate([hvec, mvec], axis=1)

        hx = torch.from_numpy(z).to(device)
        with torch.no_grad():
            g_mean = mdn.mixture_mean(hx).cpu().numpy().reshape(2)
        pred = rollout_with_K(hist, g_mean, K, P)
        break

    if hist is None:
        print("[VIZ] No valid sequence to render.")
        return

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect("equal")
    ax.set_title("MDN+Neighbors (instant) → Goal, then Koopman Rollout")

    all_x = np.concatenate([hist[:,0], gt[:,0], pred[:,0], [gt_goal[0]], [g_mean[0]]])
    all_y = np.concatenate([hist[:,1], gt[:,1], pred[:,1], [gt_goal[1]], [g_mean[1]]])
    ax.set_xlim(all_x.min()-1, all_x.max()+1)
    ax.set_ylim(all_y.min()-1, all_y.max()+1)

    ln_hist, = ax.plot(hist[:,0], hist[:,1], "ko-", label="History")
    ln_gt,   = ax.plot([], [], "go-", label="GT future")
    ln_pred, = ax.plot([], [], "ro--", label="Koopman pred")
    pt_gtgoal,  = ax.plot([], [], "g*", markersize=12, label="GT goal")
    pt_predgoal,= ax.plot([], [], "r*", markersize=12, label="Pred goal")
    ax.legend()

    def init():
        ln_gt.set_data([], [])
        ln_pred.set_data([], [])
        pt_gtgoal.set_data([], [])
        pt_predgoal.set_data([], [])
        return ln_gt, ln_pred, pt_gtgoal, pt_predgoal

    def update(frame):
        ln_gt.set_data(gt[:frame,0], gt[:frame,1])
        ln_pred.set_data(pred[:frame,0], pred[:frame,1])
        if frame == P:
            pt_gtgoal.set_data(gt[-1,0], gt[-1,1])
            pt_predgoal.set_data(g_mean[0], g_mean[1])
        return ln_gt, ln_pred, pt_gtgoal, pt_predgoal

    anim = animation.FuncAnimation(
        fig, update, frames=P+1, init_func=init,
        interval=300, blit=True
    )
    try:
        anim.save(outfile, writer="ffmpeg")
    except Exception:
        anim.save(outfile.replace(".mp4",".gif"), writer="pillow")
    plt.close(fig)
    print(f"[VIZ] Saved rollout video to {outfile}")

# ---------------------- Evaluate ----------------------

def evaluate(test_file: str, H: int, P: int, mdn: MDN, K: np.ndarray,
             samples: int = SAMPLES_DEFAULT, device: str='cpu',
             max_neighbors: int = 0, neighbor_radius: float = 5.0, neighbor_relative: bool = True):
    data = np.load(test_file)  # (T,N,2)
    ade_mean_list, fde_mean_list = [], []
    ade_best_list, fde_best_list = [], []

    times_per_sample = []

    for i, s, hist, gt in iter_test_windows(data, H, P):
        hvec = hist[::-1].reshape(1, 2*H).astype(np.float32)
        mvec = build_multiagent_context_instant(
            data, s, i,
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors,
            relative=neighbor_relative
        ).reshape(1,-1).astype(np.float32) if max_neighbors > 0 else np.zeros((1,0), dtype=np.float32)
        z = np.concatenate([hvec, mvec], axis=1)
        hx = torch.from_numpy(z).to(device)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            g_mean = mdn.mixture_mean(hx).cpu().numpy().reshape(2)
        pred_mean = rollout_with_K(hist, g_mean, K, P)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        times_per_sample.append((t1 - t0) * 1000.0)

        ade_mean = float(np.mean(np.linalg.norm(pred_mean - gt, axis=1)))
        fde_mean = float(np.linalg.norm(pred_mean[-1] - gt[-1]))
        ade_mean_list.append(ade_mean)
        fde_mean_list.append(fde_mean)

        with torch.no_grad():
            gs = mdn.sample(hx, n_samples=samples).cpu().numpy().reshape(samples, 2)
        ades, fdes = [], []
        for sidx in range(samples):
            pred = rollout_with_K(hist, gs[sidx], K, P)
            ades.append(float(np.mean(np.linalg.norm(pred - gt, axis=1))))
            fdes.append(float(np.linalg.norm(pred[-1] - gt[-1])))
        ade_best_list.append(float(np.min(ades)))
        fde_best_list.append(float(np.min(fdes)))

    if len(ade_mean_list) == 0:
        print("[EVAL] No valid test windows.")
        return

    print(f"[EVAL|MDN-mean]      ADE: {np.mean(ade_mean_list):.4f}  FDE: {np.mean(fde_mean_list):.4f}")
    print(f"[EVAL|best-of-{samples}] ADE: {np.mean(ade_best_list):.4f}  FDE: {np.mean(fde_best_list):.4f}")

    np.savez(
        'metrics_mdn_koopman.npz',
        ade_mean=np.array(ade_mean_list), fde_mean=np.array(fde_mean_list),
        ade_best=np.array(ade_best_list), fde_best=np.array(fde_best_list)
    )
    print("Saved metrics to metrics_mdn_koopman.npz")

    if times_per_sample:
        mean_t = np.mean(times_per_sample)
        std_t  = np.std(times_per_sample)
        print(f"[EVAL|latency] End-to-end inference: {mean_t:.2f} ± {std_t:.2f} ms/sample "
              f"over {len(times_per_sample)} samples")

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/home/jungbbal/ood/lobby2/biwi_hotel/train')
    parser.add_argument('--test_file', type=str, default='/home/jungbbal/ood/lobby2/biwi_hotel/test/biwi_hotel.npy')
    parser.add_argument('--history', type=int, default=HISTORY_LENGTH_DEFAULT)
    parser.add_argument('--pred', type=int, default=PRED_LENGTH_DEFAULT)
    parser.add_argument('--mdn_K', type=int, default=MDN_COMPONENTS_DEFAULT)
    parser.add_argument('--mdn_hidden', type=int, default=MDN_HIDDEN_DEFAULT)
    parser.add_argument('--mdn_epochs', type=int, default=MDN_EPOCHS_DEFAULT)
    parser.add_argument('--mdn_batch', type=int, default=MDN_BATCH_DEFAULT)
    parser.add_argument('--mdn_lr', type=float, default=MDN_LR_DEFAULT)
    parser.add_argument('--sigma_min', type=float, default=SIGMA_MIN_DEFAULT)
    parser.add_argument('--ridge', type=float, default=RIDGE_LAMBDA_DEFAULT)
    parser.add_argument('--samples', type=int, default=SAMPLES_DEFAULT)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_mdn', type=str, default='mdn.pt')
    parser.add_argument('--save_K', type=str, default='koopman_K_1.npy')
    parser.add_argument('--max_neighbors', type=int, default=MAX_NEIGHBORS_DEFAULT)
    parser.add_argument('--neighbor_radius', type=float, default=NEIGHBOR_RADIUS_DEFAULT)
    parser.add_argument('--neighbor_relative', type=int, default=1)

    args = parser.parse_args()

    H, P = args.history, args.pred
    maxN  = max(0, int(args.max_neighbors))
    R     = float(args.neighbor_radius)
    rel   = bool(int(args.neighbor_relative))

    train_files = glob.glob(os.path.join(args.train_dir, '*.npy'))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .npy files found under {args.train_dir}")
    print(f"Found {len(train_files)} train files.")
    X, Y = collect_mdn_dataset_multiagent(
        train_files, H, P, max_neighbors=maxN, neighbor_radius=R, neighbor_relative=rel
    )
    print(f"MDN dataset: X={X.shape}, Y={Y.shape}")

    in_dim = X.shape[1]
    mdn = train_mdn(
        X, Y, in_dim=in_dim, K=args.mdn_K, hidden=args.mdn_hidden,
        epochs=args.mdn_epochs, batch_size=args.mdn_batch, lr=args.mdn_lr,
        device=args.device
    )
    torch.save(mdn.state_dict(), args.save_mdn)
    print(f"Saved MDN to {args.save_mdn}")

    Pmat, Fmat = collect_koopman_pairs(train_files, H, P, use_bias=True)
    K = estimate_K(Pmat, Fmat, ridge=args.ridge)
    np.save(args.save_K, K)
    print(f"Estimated K with shape {K.shape}, saved to {args.save_K}")

    evaluate(
        args.test_file, H, P, mdn, K,
        samples=args.samples, device=args.device,
        max_neighbors=maxN, neighbor_radius=R, neighbor_relative=rel
    )

    visualize_prediction(
        args.test_file, H, P, mdn, K,
        device=args.device, outfile="mdn_koopman_vis.mp4",
        max_neighbors=maxN, neighbor_radius=R, neighbor_relative=rel
    )

if __name__ == '__main__':
    main()
