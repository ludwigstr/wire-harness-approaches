"""Demonstration synthesis from a fitted spring-mass twin (PhysWorld Sec. 3.2).

Various Motion Pattern Generation: each driven end follows a cubic Bezier
curve from its start to a random target (curvature control points randomised),
traversed with a three-phase velocity profile (accelerate, cruise, decelerate,
C1-continuous) over the episode.
Part-aware Physical Property Perturbation: the per-spring stiffness is scaled
by exp(n_i + n_j)/... where n ~ N(0, sigma^2 S), S_ij = exp(-(i-j)^2 / (2 l^2))
over the node index along the cable (the cable's "parts" are its segments;
nearby nodes get correlated perturbations, as PartField features do on a mesh).
Phi per node = log of the mean stiffness of its incident springs.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def bezier_paths(p0: torch.Tensor, T: int, max_disp: float, rng: np.random.Generator, r=None) -> torch.Tensor:
    """p0 (B, 2) start points -> (B, T, 2) end-node paths: cubic Bezier to a random
    target within max_disp (or at the given distances r (B,), e.g. drawn from the measured
    end displacements), three-phase velocity profile."""
    B = p0.shape[0]; dev = p0.device
    ang = rng.uniform(0, 2 * math.pi, B)
    r = rng.uniform(0.2, 1.0, B) * max_disp if r is None else np.asarray(r, dtype=float)
    p3 = p0 + torch.tensor(np.stack([np.cos(ang) * r, np.sin(ang) * r], -1), dtype=p0.dtype, device=dev)
    ctrl = lambda f: p0 + f * (p3 - p0) + torch.tensor(rng.normal(0, 0.3, (B, 2)) * r[:, None], dtype=p0.dtype, device=dev)
    p1, p2 = ctrl(1 / 3), ctrl(2 / 3)
    # three-phase speed: ramp up over a, cruise, ramp down over a (smoothstep integral)
    tt = np.linspace(0, 1, T); a = rng.uniform(0.15, 0.35)
    v = np.clip(np.minimum(tt / a, (1 - tt) / a), 0, 1); v = v * v * (3 - 2 * v) + 1e-3
    s = np.cumsum(v); s = (s - s[0]) / (s[-1] - s[0])
    u = torch.tensor(s, dtype=p0.dtype, device=dev).view(1, T, 1)
    P0, P1, P2, P3 = (x.unsqueeze(1) for x in (p0, p1, p2, p3))
    return (1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * P1 + 3 * (1 - u) * u ** 2 * P2 + u ** 3 * P3


def part_aware_noise(B: int, N: int, sigma: float, length: float, rng: np.random.Generator) -> np.ndarray:
    """(B, N) samples of N(0, sigma^2 S), S_ij = exp(-(i-j)^2 / (2 length^2))"""
    i = np.arange(N); S = np.exp(-((i[:, None] - i[None, :]) ** 2) / (2 * length ** 2)) + 1e-6 * np.eye(N)
    return rng.standard_normal((B, N)) @ np.linalg.cholesky(sigma ** 2 * S).T


def node_phi(k: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """(B, N, N) stiffness, (B, N, N) adjacency -> (B, N) log mean incident stiffness"""
    a = adj.float(); return torch.log((k * a).sum(-1) / a.sum(-1).clamp_min(1))


def real_end_paths(P: torch.Tensor, cab: np.ndarray, t0: np.ndarray, T: int, rng: np.random.Generator,
                   warp: bool) -> torch.Tensor:
    """The measured motion of both end nodes of cable `cab` from frame `t0` (the actions the model
    gets at test time, not the cable's response). warp: time scale 0.5..2 (linear interpolation
    along the track) and displacement scaled 0.5..1.5 about the start. -> (B, T, 2, 2)."""
    out = []
    Tc = P.shape[1]
    for c, t in zip(cab, t0):
        E = P[c][:, [0, -1]]                                                        # (Tc, 2, 2)
        s = rng.uniform(0.5, 2.0) if warp else 1.0
        s = min(s, (Tc - 1 - t) / max(T - 1, 1)) if (Tc - 1 - t) > 0 else 0.0
        q = t + s * torch.arange(T, dtype=torch.float32, device=P.device)
        i0 = q.floor().long().clamp(max=Tc - 1); i1 = (i0 + 1).clamp(max=Tc - 1); w = (q - i0.float()).view(T, 1, 1)
        path = (1 - w) * E[i0] + w * E[i1]
        if warp:
            path = E[t] + rng.uniform(0.5, 1.5) * (path - E[t])
        out.append(path)
    return torch.stack(out)
