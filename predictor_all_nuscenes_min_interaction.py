#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Koopman refinement + MDN goal (neighbors) 평가 스크립트
- Goal ckpt: /raid/users/snowhan227/temporary/goal_mlp_neighbors_nusc.pt
- Dataset: nuScenes NPZ (X: [T,M,2], V: [T,M], lane_points: [N,2])
- 평가: K=1 (argmax component mean), K=5 (5 샘플 중 minADE/minFDE)
"""

import os, glob, math, argparse, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch, torch.nn as nn
from scipy.spatial import cKDTree as KDTree
current_dir = str(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 기본 유틸
# =========================
def seed_all(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_heading(xy: np.ndarray) -> float:
    diffs = np.diff(xy, axis=0)
    for d in diffs[::-1]:
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            return float(np.arctan2(d[1], d[0]))
    return 0.0

def rotmat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

def build_lane_feature(lane_points, pos, psi, radius=50.0, kmax=32):
    """가까운 lane k점 (ego-frame) + [cos psi, sin psi]"""
    if lane_points.shape[0] == 0:
        flat = np.zeros((2*kmax,), np.float32)
        return np.concatenate([flat, [np.cos(psi), np.sin(psi)]], 0)
    kdt = KDTree(lane_points)
    idxs = kdt.query_ball_point(pos, r=radius)
    neigh = lane_points[idxs] if len(idxs) > 0 else np.empty((0, 2), dtype=np.float32)
    if neigh.shape[0] > 0:
        d2 = np.sum((neigh - pos) ** 2, axis=1)
        order = np.argsort(d2)[:kmax]
        neigh = neigh[order]
    else:
        neigh = np.zeros((0, 2), dtype=np.float32)
    R = rotmat(-psi)
    rel = (neigh - pos) @ R.T
    out = np.zeros((kmax, 2), np.float32)
    k = min(kmax, rel.shape[0])
    if k > 0:
        out[:k] = rel[:k]
    flat = out.reshape(-1)
    return np.concatenate([flat, [np.cos(psi), np.sin(psi)]], 0).astype(np.float32)

def build_neighbor_feature(X, V, s, m, pos, psi, radius=20.0, kmax=6):
    """ego-frame에서 반경 내 kmax 이웃 (정지 패딩)"""
    _, M, _ = X.shape
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
# MDN goal (neighbors) 모델 정의
# =========================
class HistoryEncoder(nn.Module):
    def __init__(self, H, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * H
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, h): 
        return self.net(h)

class LaneEncoder(nn.Module):
    def __init__(self, lane_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * lane_k + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, c): 
        return self.net(c)

class NeighborEncoder(nn.Module):
    def __init__(self, nbr_k, hidden=128, out_dim=64):
        super().__init__()
        in_dim = 2 * nbr_k
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
    def forward(self, n): 
        return self.net(n)

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
        min_log, max_log = math.log(self.sigma_min), math.log(self.sigma_max)
        log_sigma = torch.clamp(log_sigma, min=min_log, max=max_log)
        log_pi = torch.log_softmax(logit_pi, dim=-1)
        return log_pi, mu, log_sigma

class MDNGoalPredictorWithNeighbors(nn.Module):
    def __init__(self, H, lane_k, nbr_k, n_comp=5, sigma_min=0.05,
                 h_hidden=128, h_out=64, c_hidden=128, c_out=64, n_hidden=128, n_out=64,
                 head_hidden=128):
        super().__init__()
        self.H = H; self.K = lane_k; self.M = nbr_k
        self.f_h = HistoryEncoder(H, hidden=h_hidden, out_dim=h_out)
        self.f_c = LaneEncoder(lane_k, hidden=c_hidden, out_dim=c_out)
        self.f_n = NeighborEncoder(nbr_k, hidden=n_hidden, out_dim=n_out)
        self.head = MDNHead(h_out + c_out + n_out, n_comp=n_comp, sigma_min=sigma_min, hidden=head_hidden)

    def forward(self, x):
        H2 = 2 * self.H
        Cdim = 2 * self.K + 2
        h_vec = x[:, :H2]
        c_vec = x[:, H2:H2 + Cdim]
        n_vec = x[:, H2 + Cdim:]
        h_star = self.f_h(h_vec)
        c_star = self.f_c(c_vec)
        n_star = self.f_n(n_vec)
        z = torch.cat([h_star, c_star, n_star], dim=-1)
        return self.head(z)

# 체크포인트에서 구조 자동 복원
def load_goal_predictor(ckpt_path: str, device: str = "cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Goal checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"]

    H_used = int(ckpt.get("H"))
    lane_k = int(ckpt.get("lane_k"))
    nbr_k  = int(ckpt.get("neighbor_k"))
    n_comp = int(state["head.pi.weight"].shape[0])

    # hidden/out 차원 유추
    h_hidden = int(state["f_h.net.0.weight"].shape[0])
    h_out    = int(state["f_h.net.2.weight"].shape[0])
    c_hidden = int(state["f_c.net.0.weight"].shape[0])
    c_out    = int(state["f_c.net.2.weight"].shape[0])
    n_hidden = int(state["f_n.net.0.weight"].shape[0])
    n_out    = int(state["f_n.net.2.weight"].shape[0])
    head_hidden = int(state["head.core.0.weight"].shape[0])
    sigma_min = float(ckpt.get("sigma_min", 0.05))

    model = MDNGoalPredictorWithNeighbors(
        H=H_used, lane_k=lane_k, nbr_k=nbr_k, n_comp=n_comp, sigma_min=sigma_min,
        h_hidden=h_hidden, h_out=h_out, c_hidden=c_hidden, c_out=c_out,
        n_hidden=n_hidden, n_out=n_out, head_hidden=head_hidden
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[LOAD] goal ckpt={ckpt_path} | H={H_used} lane_k={lane_k} nbr_k={nbr_k} mixtures={n_comp}")
    return model, H_used, lane_k, nbr_k

# =========================
# EDMD (K) 구성요소
# =========================
def window_is_sane(X: np.ndarray, V: np.ndarray, m: int, s: int, H: int, P: int, max_step: float) -> bool:
    """히스토리/미래 구간에 비정상적인 큰 점프가 있으면 제외"""
    idxs = list(range(s - H + 1, s + 1)) + [s + P, s + P + 1]
    for t in idxs:
        if t < 0 or t >= X.shape[0] or not V[t, m]:
            return False
    for t0 in range(s - H + 1, s):
        d = np.linalg.norm(X[t0 + 1, m] - X[t0, m])
        if d > max_step:
            return False
    d_goal = np.linalg.norm(X[s + P + 1, m] - X[s + P, m])
    if d_goal > max_step:
        return False
    return True

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

        # 파일 단위 원점 이동(안전)
        valid_mask = np.isfinite(X).all(axis=-1).any(axis=1)
        if np.any(valid_mask):
            xmin = np.nanmin(X[:,:,0]); ymin = np.nanmin(X[:,:,1])
            X[:,:,0] -= xmin; X[:,:,1] -= ymin

        T, M, _ = X.shape
        for m in range(M):
            for s in range(H - 1, T - P - 1):  # s+P+1 필요
                if not window_is_sane(X, V, m, s, H, P, max_step):
                    continue

                hist = X[s - H + 1:s + 1, m, :]
                g_t  = X[s + P,     m, :]
                hist_next = X[s - H + 2:s + 2, m, :]
                g_t1 = X[s + P + 1, m, :]

                psi     = np.concatenate([hist[::-1].reshape(-1), g_t], 0)
                psi_nxt = np.concatenate([hist_next[::-1].reshape(-1), g_t1], 0)
                if include_bias:
                    psi     = np.concatenate([psi, [1.0]], 0)
                    psi_nxt = np.concatenate([psi_nxt, [1.0]], 0)

                if (not np.isfinite(psi).all()) or (not np.isfinite(psi_nxt).all()):
                    continue
                Psis.append(psi); Psis_next.append(psi_nxt)

    if len(Psis) == 0:
        raise RuntimeError("No EDMD training pairs were collected. Check data/thresholds.")

    Psi = np.stack(Psis).astype(np.float32)
    Psi_next = np.stack(Psis_next).astype(np.float32)
    return Psi, Psi_next

def fit_koopman(Psi: np.ndarray, Psi_next: np.ndarray, ridge_lambda=1.0, normalize=True):
    X, Y = Psi, Psi_next
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
    np.savez(path, K=pack["K"], mu=pack["mu"], sig=pack["sig"], H=H, P=P,
             include_bias=include_bias, normalize=normalize)
    print(f"[K] Estimated and saved to {path}, shape={pack['K'].shape}")

def load_K(path):
    Z = np.load(path)
    pack = dict(K=Z["K"].astype(np.float32), mu=Z["mu"].astype(np.float32), sig=Z["sig"].astype(np.float32))
    meta = dict(H=int(Z["H"]), P=int(Z["P"]), include_bias=bool(Z["include_bias"]), normalize=bool(Z["normalize"]))
    print(f"[K] Loaded from {path}, shape={pack['K'].shape}, H={meta['H']}, P={meta['P']}, bias={meta['include_bias']}")
    return pack, meta

def rollout_with_K(history_global: np.ndarray, g0_global: np.ndarray, pack: dict, include_bias=True, P=50):
    """히스토리 최신→과거 역순으로 쌓았다고 가정"""
    H = history_global.shape[0]
    hist_lf = history_global[::-1].copy()
    psi = np.concatenate([hist_lf.reshape(-1), g0_global], 0)
    if include_bias:
        psi = np.concatenate([psi, [1.0]], 0)
    K = pack["K"]; mu = pack["mu"]; sig = pack["sig"]

    traj = []
    for _ in range(P):
        x_n = (psi - mu) / sig
        psi_next_n = K @ x_n
        psi = psi_next_n * sig + mu
        traj.append(psi[:2].copy())  # x_{t+1}
    traj = np.stack(traj)
    g_last = psi[2*H:2*H+2].copy()
    return traj, g_last

# =========================
# MDN 샘플링 / 점추정
# =========================
@torch.no_grad()
def mdn_argmax_mean(log_pi, mu):
    """K=1: 가장 확률 높은 컴포넌트의 평균점"""
    idx = torch.argmax(log_pi, dim=-1).item()
    return mu[0, idx, :].cpu().numpy()  # (2,)

@torch.no_grad()
def mdn_sample_goals(log_pi, mu, log_sigma, n_samples=5):
    """K=5: 혼합에서 샘플 n개 (대각 공분산)"""
    pi = torch.softmax(log_pi, dim=-1)[0]              # (K,)
    idxs = torch.multinomial(pi, n_samples, replacement=True)  # (n_samples,)
    mu_sel = mu[0, idxs, :]                            # (n_samples,2)
    sigma_sel = torch.exp(log_sigma[0, idxs, :])       # (n_samples,2)
    eps = torch.randn_like(mu_sel)
    samp = mu_sel + eps * sigma_sel
    return samp.cpu().numpy()                          # (n_samples,2)

def ade_fde(pred_traj, gt_traj):
    d = np.linalg.norm(pred_traj - gt_traj, axis=1)
    ade = float(d.mean())
    fde = float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))
    return ade, fde

# =========================
# 평가 루프 (디렉터리 전체)
# =========================
def evaluate_dir(test_dir, K_pack, H_K, P, include_bias, goal_ckpt_path,
                 lane_radius=50.0, neighbor_radius=20.0, make_video=False, video_out="koopman_goal_rollout.mp4"):
    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {test_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, H_goal, lane_k, nbr_k = load_goal_predictor(goal_ckpt_path, device)

    all_ADE_K1, all_FDE_K1 = [], []
    all_ADE_K5, all_FDE_K5 = [], []
    total_windows = 0

    fig, ax = (None, None)
    if make_video:
        fig, ax = plt.subplots(1,1, figsize=(7,7))

    for f in files:
        try:
            D = np.load(f, allow_pickle=True)
            X = D["X"].astype(np.float32)  # (T,M,2)
            V = D["V"].astype(bool)
            lane_points = D.get("lane_points", np.zeros((0,2), np.float32))
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
            continue

        # 파일 원점 이동 (수치 안정)
        xmin, ymin = np.nanmin(X[:,:,0]), np.nanmin(X[:,:,1])
        X[:,:,0] -= xmin; X[:,:,1] -= ymin
        if lane_points.shape[0] > 0:
            lane_points = lane_points - np.array([xmin, ymin], dtype=np.float32)

        T, M, _ = X.shape
        file_ADE_K1, file_FDE_K1 = [], []
        file_ADE_K5, file_FDE_K5 = [], []

        # 모든 agent/windows 평가
        for m in range(M):
            if not np.any(V[:, m]): 
                continue
            for s in range(H_goal - 1, T - P):
                # 히스토리/Horizon 유효성
                if not (np.all(V[s - H_goal + 1 : s + 1, m]) and 
                        V[s + P, m] and 
                        np.all(V[s + 1 : s + P + 1, m])):
                    continue

                # history & ego 변환
                x_hist = X[s - H_goal + 1 : s + 1, m, :].copy()
                pos = x_hist[-1]
                psi_heading = safe_heading(x_hist)
                R = rotmat(-psi_heading)

                x_hist_ego = (x_hist - pos) @ R.T
                h_vec = x_hist_ego.reshape(-1)
                c_vec = build_lane_feature(lane_points, pos, psi_heading, radius=lane_radius, kmax=lane_k)
                n_vec = build_neighbor_feature(X, V, s, m, pos, psi_heading, radius=neighbor_radius, kmax=nbr_k)
                inp = np.concatenate([h_vec, c_vec, n_vec], 0)[None, :].astype(np.float32)

                with torch.no_grad():
                    log_pi, mu, log_sigma = model(torch.from_numpy(inp).float().to(device))

                # GT future (global)
                x_future = X[s + 1 : s + P + 1, m, :].copy()

                # ---------- K=1 (argmax mean) ----------
                g_ego_k1 = mdn_argmax_mean(log_pi, mu)                         # (2,)
                g_glb_k1 = (g_ego_k1 @ R) + pos
                # K가 학습된 H와 goal H가 다르면, K 입력용 history를 맞춰준다
                x_hist_for_K = x_hist[-H_K:, :] if H_goal != H_K else x_hist
                x_pred_k1, _ = rollout_with_K(x_hist_for_K, g_glb_k1, K_pack, include_bias=include_bias, P=P)
                ade1, fde1 = ade_fde(x_pred_k1, x_future)
                file_ADE_K1.append(ade1); file_FDE_K1.append(fde1)

                # ---------- K=5 (sample 5 → minADE/FDE) ----------
                ego_samples = mdn_sample_goals(log_pi, mu, log_sigma, n_samples=5)  # (5,2)
                best_ade, best_fde = float("inf"), float("inf")
                for i in range(ego_samples.shape[0]):
                    g_pred = (ego_samples[i] @ R) + pos
                    x_hist_for_K = x_hist[-H_K:, :] if H_goal != H_K else x_hist
                    x_pred, _ = rollout_with_K(x_hist_for_K, g_pred, K_pack, include_bias=include_bias, P=P)
                    ade, fde = ade_fde(x_pred, x_future)
                    if (ade < best_ade) or (abs(ade - best_ade) < 1e-9 and fde < best_fde):
                        best_ade, best_fde = ade, fde
                        best_pack_for_vid = (x_hist + np.array([xmin,ymin]),
                                             x_future + np.array([xmin,ymin]),
                                             x_pred + np.array([xmin,ymin]),
                                             X[s + P, m, :] + np.array([xmin,ymin]),
                                             g_pred + np.array([xmin,ymin]))

                file_ADE_K5.append(best_ade); file_FDE_K5.append(best_fde)

                # (옵션) 비디오 한두 프레임 저장
                if make_video and ax is not None and len(file_ADE_K5) <= 120:
                    x_hist_v, x_future_v, x_pred_v, g_gt_v, g_pred_v = best_pack_for_vid
                    visualize_frame(ax, lane_points + np.array([xmin,ymin]), x_hist_v, x_future_v, x_pred_v, g_gt_v, g_pred_v)

        # 파일 단위 리포트
        if len(file_ADE_K1) > 0:
            print(f"[RESULT K=1] {os.path.basename(f)} | ADE={np.mean(file_ADE_K1):.3f} FDE={np.mean(file_FDE_K1):.3f} | windows={len(file_ADE_K1)}")
            all_ADE_K1.extend(file_ADE_K1); all_FDE_K1.extend(file_FDE_K1)
            total_windows += len(file_ADE_K1)
        if len(file_ADE_K5) > 0:
            print(f"[RESULT K=5] {os.path.basename(f)} | minADE={np.mean(file_ADE_K5):.3f} minFDE={np.mean(file_FDE_K5):.3f} | windows={len(file_ADE_K5)}")
            all_ADE_K5.extend(file_ADE_K5); all_FDE_K5.extend(file_FDE_K5)

    # 전체 요약
    print("=============================================")
    if len(all_ADE_K1) > 0:
        print(f"[SUMMARY K=1] Files={len(files)} Windows={len(all_ADE_K1)} | ADE={np.mean(all_ADE_K1):.3f} FDE={np.mean(all_FDE_K1):.3f}")
    else:
        print("[SUMMARY K=1] No valid windows.")
    if len(all_ADE_K5) > 0:
        print(f"[SUMMARY K=5] Files={len(files)} Windows={len(all_ADE_K5)} | minADE={np.mean(all_ADE_K5):.3f} minFDE={np.mean(all_FDE_K5):.3f}")
    else:
        print("[SUMMARY K=5] No valid windows.")

    if make_video and fig is not None:
        try:
            ani = animation.FuncAnimation(fig, lambda i: [], frames=1)
            fig.savefig(video_out.replace(".mp4",".png"), dpi=150)
            print(f"[SAVE] Preview saved to {video_out.replace('.mp4','.png')} (quick snapshot)")
        except Exception as e:
            print(f"[WARN] Video save failed: {e}")

# 간단 프레임 렌더 (디버그/프리뷰용)
def visualize_frame(ax, lane_points, x_hist, x_future, x_pred, g_gt, g_pred):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title('History / GT future / K rollout (best-of-5)')
    if lane_points is not None and lane_points.shape[0] > 0:
        ax.scatter(lane_points[:,0], lane_points[:,1], s=2, alpha=0.15, c='gray', label='lanes')
    ax.plot(x_hist[:,0], x_hist[:,1], '-o', lw=2, ms=3, label='history')
    ax.plot(x_future[:,0], x_future[:,1], '-o', lw=2, ms=3, label='GT future')
    ax.plot(x_pred[:,0], x_pred[:,1], '-o', lw=2, ms=3, label='K rollout (best)')
    ax.scatter([g_gt[0]],[g_gt[1]], c='r', marker='x', s=80, label='GT goal')
    ax.scatter([g_pred[0]],[g_pred[1]], c='g', marker='*', s=100, label='Pred goal (best)')
    ax.legend(loc='best')

# =========================
# 메인
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default=current_dir+"/nusc_npz/train")
    ap.add_argument("--val_dir",   type=str, default=current_dir+"/nusc_npz/val")
    ap.add_argument("--goal_ckpt", type=str, default=current_dir+"/goal_mlp_neighbors_nusc.pt")
    ap.add_argument("--koopman_npz", type=str, default="koopman_K_goal_global_nuscenes.npz")

    ap.add_argument("--H", type=int, default=4, help="EDMD 히스토리 길이 (K의 psi 구성)")
    ap.add_argument("--P", type=int, default=12, help="롤아웃 길이")
    ap.add_argument("--ridge_lambda", type=float, default=1.0)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--include_bias", action="store_true", default=True)
    ap.add_argument("--max_train_files", type=int, default=100)
    ap.add_argument("--max_step_thresh", type=float, default=1.0)  # 훈련 페어 수집용 텔레포트 임계 (전역 프레임)
    ap.add_argument("--lane_radius", type=float, default=50.0)
    ap.add_argument("--neighbor_radius", type=float, default=20.0)

    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--make_video", action="store_true", help="프리뷰 이미지 저장")
    return ap.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)

    # 1) K 로드/학습
    if os.path.exists(args.koopman_npz):
        K_pack, meta = load_K(args.koopman_npz)
        if meta['H'] != args.H or meta['P'] != args.P:
            print(f"[WARN] Loaded K was fit for H={meta['H']}, P={meta['P']} but you set H={args.H}, P={args.P}.")
    else:
        print("[EDMD] Collecting training pairs ...")
        all_files = sorted(glob.glob(os.path.join(args.train_dir, '*.npz')))
        use_files = all_files[: args.max_train_files]
        Psi, Psi_next = collect_edmd_pairs(
            use_files, args.H, args.P, include_bias=args.include_bias,
            max_step=80.0  # EDMD 페어 수집엔 완화된 임계 사용
        )
        print(f"[EDMD] Pairs: {Psi.shape[0]}, dim={Psi.shape[1]}")
        pack = fit_koopman(Psi, Psi_next, ridge_lambda=args.ridge_lambda, normalize=args.normalize)
        save_K(pack, args.koopman_npz, H=args.H, P=args.P, include_bias=args.include_bias, normalize=args.normalize)
        K_pack = pack

    # 2) 밸리데이션 디렉터리 평가 (K=1, K=5 모두)
    evaluate_dir(
        test_dir=args.val_dir,
        K_pack=K_pack if 'K_pack' in locals() else K_pack,  # 정의 보정
        H_K=(args.H if 'meta' not in locals() else meta['H']),
        P=args.P,
        include_bias=args.include_bias,
        goal_ckpt_path=args.goal_ckpt,
        lane_radius=args.lane_radius,
        neighbor_radius=args.neighbor_radius,
        make_video=args.make_video,
        video_out="koopman_goal_rollout.mp4"
    )

if __name__ == "__main__":
    main()
