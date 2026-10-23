"""Synthetic training data for the two simulator-trained networks (Caporali, AdaptiGraph).

Targets always come from the project's analytical mass-spring simulator
(wire_model.dlo_model.DLOSimulator: inextensible chain, bending springs with a
rest angle per joint, damping, Coulomb board friction; 1 s per action = 0.6 s
moving the grasped end edges + 0.4 s settling).

generate_paper(mode)
    The ORIGINAL generator behind the table numbers (Caporali 56.8, AdaptiGraph
    63.9 / 26.1): every chain starts from an artificial constant-curvature arc of
    0.25-0.60 m and is driven by random end actions.
generate_real(mode)
    REAL-ANCHORED data (2026-10-23): chains start from real tracked frames of
    the 103 X-Planar clips (never the 14 evaluation clips), the two ends follow
    the measured end motion of that recording (scaled 0.6-2.0x), the rest shape
    is the recording's frame-0 shape, and m, k_b, k_d, k_c are sampled around
    the measured shared set (0.04, 1.0, 20, 0.15). Motivation: the original data
    never contains a cable as long (0.60-0.89 m) or as straight as ours.

mode "step": one transition per 1 s action  -> Caporali's MLP
     ("V_in", "V_out", "action" (8), "params" [m, k_b, k_d, k_c, L], "rest_angle")
mode "seq":  one transition per video frame (0.25 s) with the 4 preceding
     frames as history -> AdaptiGraph (+ "hist", params with k_b per joint)
"""
from __future__ import annotations

import glob
import time

import numpy as np
import torch

from common import ROOT, turning_angles
from wire_model.actions import rollout, terminal_edge_action
from wire_model.dataset import SamplingRanges, SimConfig, generate, generate_seq
from wire_model.dlo_model import DLOParams, DLOSimulator
from wire_model.geometry import arc_length

N = 16


def _drop_stretched(d, max_stretch=0.01):
    """a real wire is near-inextensible: drop transitions stretched by more than 1 %"""
    keep = (arc_length(d["V_out"]) / d["params"][:, -1] - 1).abs() <= max_stretch
    return {k: (v[keep] if torch.is_tensor(v) else v) for k, v in d.items()}


def generate_paper(mode, chains, seed=0, kb=(0.01, 1.0), flat_kb=True):
    """mode "step": 40 000 chains x 5 actions, k_b 0.01-1.0 uniform along the wire (Caporali's dataset);
    mode "seq": 20 000 chains x 5 actions x 4 frames, default ranges incl. a k_b profile (AdaptiGraph's)."""
    cfg, r = SimConfig(), SamplingRanges()
    if mode == "step":
        r.k_b = tuple(kb)
        if flat_kb:
            r.k_b_p_flat = 1.0
        d = _drop_stretched(generate(chains, 5, cfg=cfg, r=r, device="cuda", seed=seed, verbose=True))
        if flat_kb and d["params"].shape[1] > 5:   # [m, k_b x14 (equal), k_d, k_c, L] -> [m, k_b, k_d, k_c, L]
            P = d["params"]; d["params"] = torch.cat([P[:, :1], P[:, 1:-3].mean(1, keepdim=True), P[:, -3:]], 1)
        return d
    return _drop_stretched(generate_seq(chains, 5, n_hist=4, n_sub=4, cfg=cfg, r=r, device="cuda", seed=seed, verbose=True))


def _edge_action(Pa, Pb, at_end, scale):
    """measured edge motion from frame a to b: midpoint shift + rotation about the midpoint, scaled
    (exactly what the evaluation feeds the networks)"""
    def pose(P):
        a, b = (P[:, -2], P[:, -1]) if at_end else (P[:, 0], P[:, 1])
        return 0.5 * (a + b), np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
    m0, a0 = pose(Pa); m1, a1 = pose(Pb)
    return (m1 - m0) * scale[:, None], np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)) * scale


def generate_real(mode, chains, batch=10000, seed=0, tracks="data/tracks_xplanar", mass=(0.02, 0.10), kb=(0.2, 3.0),
                  kd=(5.0, 40.0), kc=(0.05, 0.40), scale=(0.6, 2.0), rest=(0.6, 1.2), rest_cap=0.6, max_stretch=0.01):
    stride = 4 if mode == "step" else 1
    chain_len = 3 if mode == "step" else 12
    H = 4
    rng = np.random.default_rng(seed); dev = "cuda"; cfg = SimConfig()
    sim = DLOSimulator(n_nodes=N, dt=cfg.dt, device=dev)
    n_move, n_settle = (cfg.n_move, cfg.n_settle) if mode == "step" else ((cfg.n_move + cfg.n_settle) // 4, 0)
    trajs = []
    for f in sorted(glob.glob(str(ROOT / tracks / "*.npz"))):
        z = np.load(f)
        for k in range(4):
            P = z[f"c{k}"].astype(np.float32)
            if np.isfinite(P).all() and len(P) > stride * chain_len + H + 1:
                trajs.append(P)
    logu = lambda lo, hi, n: np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)
    out = {k: [] for k in ("hist", "V_in", "V_out", "action", "params", "rest_angle")}
    t_start = time.time()
    for b0 in range(0, chains, batch):
        B = min(batch, chains - b0)
        ti = rng.integers(0, len(trajs), B)
        t0 = np.array([rng.integers(H, len(trajs[i]) - stride * chain_len) for i in ti])
        seg = np.stack([trajs[i][t - H:t + stride * chain_len + 1] for i, t in zip(ti, t0)])
        f0 = np.stack([trajs[i][0] for i in ti])
        L = np.linalg.norm(np.diff(seg[:, H], axis=1), axis=2).sum(1).astype(np.float32)
        rst = np.clip(turning_angles(f0) * rng.uniform(*rest, (B, 1)), -rest_cap, rest_cap).astype(np.float32)
        m, k_b, k_d = logu(*mass, B), logu(*kb, B), logu(*kd, B)
        k_c = rng.uniform(*kc, B).astype(np.float32); sc = rng.uniform(*scale, B).astype(np.float32)
        T = lambda x: torch.tensor(x, device=dev)
        prm = DLOParams(m=T(m), k_b=T(np.repeat(k_b[:, None], N - 2, 1)), k_d=T(k_d), k_c=T(k_c), k_s=torch.full((B,), 1e5, device=dev),
                        length=T(L), rest_angle=T(rst)).to_batched(B, dev, n_nodes=N)
        p = T(seg[:, H].copy()); static = torch.zeros(B, N, dtype=torch.bool, device=dev)
        window = [seg[:, j].copy() for j in range(H)]
        par = np.concatenate([m[:, None], np.repeat(k_b[:, None], N - 2, 1), k_d[:, None], k_c[:, None], L[:, None]], 1).astype(np.float32)
        alive = np.ones(B, dtype=bool)
        for s in range(chain_len):
            Pa, Pb = seg[:, H + stride * s], seg[:, H + stride * (s + 1)]
            dE, thE = _edge_action(Pa, Pb, True, sc); dH, thH = _edge_action(Pa, Pb, False, sc)
            act = terminal_edge_action(N, T(dE[:, 0]), T(dE[:, 1]), T(thE.astype(np.float32)), at_end=True)
            hub = terminal_edge_action(N, T(dH[:, 0]), T(dH[:, 1]), T(thH.astype(np.float32)), at_end=False)
            with torch.no_grad():
                pf = rollout(sim, p, prm, act, static_mask=static, n_move=n_move, n_settle=n_settle, action2=hub)
            alive &= (torch.isfinite(pf).all(dim=(1, 2)) & ((arc_length(pf) / T(L) - 1).abs() <= max_stretch)).cpu().numpy()
            vin, vout = p.cpu().numpy(), pf.cpu().numpy()
            vec = np.concatenate([np.stack([np.full(B, N - 2), dE[:, 0], dE[:, 1], thE], 1),
                                  np.stack([np.zeros(B), dH[:, 0], dH[:, 1], thH], 1)], 1).astype(np.float32)
            k = alive.copy()
            out["hist"].append(np.stack(window, 1)[k]); out["V_in"].append(vin[k]); out["V_out"].append(vout[k])
            out["action"].append(vec[k]); out["params"].append(par[k]); out["rest_angle"].append(rst[k])
            window = window[1:] + [vin]
            p = torch.where(torch.isfinite(pf), pf, p)
        print(f"  chains {b0 + B}/{chains}: {sum(len(x) for x in out['V_in'])} transitions ({time.time() - t_start:.0f}s)", flush=True)
    d = {k: torch.tensor(np.concatenate(v)) for k, v in out.items()}
    if mode == "step":
        P = d["params"]; d["params"] = torch.cat([P[:, :1], P[:, 1:-3].mean(1, keepdim=True), P[:, -3:]], 1)
        del d["hist"]
    d["config"] = {"generator": "real-anchored", "mode": mode, "tracks": tracks, "mass": mass, "k_b": kb, "k_d": kd, "k_c": kc,
                   "scale": scale, "rest": rest, "rest_cap": rest_cap, "stride": stride, "chain_len": chain_len}
    return d
