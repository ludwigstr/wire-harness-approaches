"""PhysTwin's two-stage sparse-to-dense optimisation (paper Sec. 4.2.1), one
scenario = one clip with its four cables (same material, same board):
stage 1 (zero-order, CMA-ES) on the homogeneous stiffness k, dashpot gamma,
drag delta, board friction mu and the topology (connection radius in
segments; k_max = 6), cost = mean node error of the resimulation over the
fitting frames (C_motion; the Chamfer term C_geometry is redundant with full
node tracks); stage 2 (first-order, Adam through the simulator) on the dense
per-spring log k of every cable plus log gamma/delta/mu, on random windows of
`win` frames started from the measured state.
"""
from __future__ import annotations

import numpy as np
import torch

from .sim import SpringMassTwin, build_topology

LO = [np.log(1.0), np.log(1e-3), np.log(1e-4), np.log(1e-3), 1.0]
HI = [np.log(1500.0), np.log(1.5), np.log(20.0), np.log(5.0), 4.0]  # 5000 diverged: several springs per node add up


def velocity0(P: torch.Tensor, frame_s: float) -> torch.Tensor:
    return (P[:, 1] - P[:, 0]) / frame_s if P.shape[1] > 1 else torch.zeros_like(P[:, 0])


def resim_error(twin, P, k, gamma, delta, mu, t0=0, t1=None, frame_s=0.25):
    """P (C, T, N, 2); k (pop*C, N, N) or (pop*C, 1, 1); gamma/delta/mu (pop*C,)
    -> mean node error (m) per candidate and cable, (pop, C)"""
    C = P.shape[0]; B = gamma.shape[0]; pop = B // C; t1 = P.shape[1] if t1 is None else t1
    Pf = P[:, t0:t1]
    X0 = Pf[:, 0].repeat(pop, 1, 1); V0 = velocity0(Pf, frame_s).repeat(pop, 1, 1)
    ends = Pf[:, :, [0, -1]].repeat(pop, 1, 1, 1)
    sim = twin.rollout(X0, V0, ends, k, gamma, delta, mu)
    err = (sim[:, :, 1:-1] - Pf[:, :, 1:-1].repeat(pop, 1, 1, 1)).norm(dim=-1)
    err = torch.nan_to_num(err, nan=1e3, posinf=1e3).mean((1, 2))
    return err.view(pop, C)


def stage1(P, fit_frames, device, gens=20, popsize=16, seed=0, log=print, frame_s=0.25, dt=1e-3):
    """CMA-ES over [log k, log gamma, log delta, log mu, radius_segments], shared by the clip's cables."""
    import cma
    x0 = [np.log(200.0), np.log(0.3), np.log(0.05), np.log(0.3), 2.0]
    es = cma.CMAEvolutionStrategy(x0, 0.6, {"popsize": popsize, "seed": seed + 1, "verbose": -9, "bounds": [LO, HI]})
    best = (float("inf"), None)
    for g in range(gens):
        xs = es.ask(); costs = np.zeros(len(xs))
        by_r = {}
        for i, x in enumerate(xs):
            by_r.setdefault(round(float(x[4]) * 2) / 2, []).append(i)           # topology in half-segment steps
        for r, idx in by_r.items():
            twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), dt=dt, frame_s=frame_s)
            xb = torch.tensor(np.array([xs[i] for i in idx]), dtype=torch.float32, device=device)   # (pop_r, 5)
            C = P.shape[0]; rep = lambda v: v.repeat_interleave(C)
            with torch.no_grad():
                e = resim_error(twin, P, torch.exp(rep(xb[:, 0])).view(-1, 1, 1), torch.exp(rep(xb[:, 1])), torch.exp(rep(xb[:, 2])), torch.exp(rep(xb[:, 3])), 0, fit_frames, frame_s)
            costs[idx] = e.mean(1).cpu().numpy() * 1000
        es.tell(xs, costs.tolist())
        i = int(np.argmin(costs))
        if costs[i] < best[0]:
            best = (float(costs[i]), np.array(xs[i]))
        if g % 5 == 0 or g == gens - 1:
            b = best[1]; log(f"    stage 1 gen {g:2d}: best {best[0]:.2f} mm  k {np.exp(b[0]):.0f} gamma {np.exp(b[1]):.3f} delta {np.exp(b[2]):.3f} mu {np.exp(b[3]):.3f} radius {b[4]:.1f} seg")
    return best


def stage2(P, fit_frames, x1, device, iters=100, win=8, lr=0.05, log=print, frame_s=0.25, dt=1e-3, seed=0, check=10):
    """dense per-spring log k per cable (initialised at the homogeneous k) + shared log gamma/delta/mu"""
    r = round(float(x1[4]) * 2) / 2
    twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), dt=dt, frame_s=frame_s)
    C, N = P.shape[0], P.shape[2]
    logk = torch.full((C, N, N), float(x1[0]), device=device, requires_grad=True)
    glob = torch.tensor(x1[1:4], dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([logk, glob], lr=lr); rng = np.random.default_rng(seed)
    # 2026-10-23: the 8-frame windows never see the whole fitting span, and on long clips (134613, 125 frames)
    # the dense k drifted into a set whose full rollout diverged (1e6 mm). Keep the iterate with the lowest
    # error over ALL fitting frames from frame 0 (the reported protocol), checked every `check` iterations;
    # iterate 0 is the stage-1 result, so stage 2 can never make the reported fit worse.
    best = (float("inf"), None, None, -1)
    def check_full(it):
        nonlocal best
        with torch.no_grad():
            k_ = torch.exp(0.5 * (logk + logk.transpose(1, 2))); gl_ = torch.exp(glob).clamp(max=torch.tensor([1.5, 20.0, 5.0], device=device))
            e_ = resim_error(twin, P, k_, gl_[0].expand(C), gl_[1].expand(C), gl_[2].expand(C), 0, fit_frames, frame_s).mean().item() * 1000
        if e_ < best[0]:
            best = (e_, k_.clone(), gl_.clone(), it)
        log(f"    stage 2 it {it:3d}: full fitting-span error {e_:.2f} mm (best {best[0]:.2f} at it {best[3]})")
    for it in range(iters):
        if it % check == 0:
            check_full(it)
        t0 = int(rng.integers(0, max(1, fit_frames - win))); t1 = min(fit_frames, t0 + win)
        k = torch.exp(0.5 * (logk + logk.transpose(1, 2)))
        gl = torch.exp(glob).clamp(max=torch.tensor([1.5, 20.0, 5.0], device=device))
        e = resim_error(twin, P, k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C), t0, t1, frame_s)
        opt.zero_grad(); e.mean().backward(); torch.nn.utils.clip_grad_norm_([logk, glob], 1.0); opt.step()
        with torch.no_grad():
            logk.clamp_(LO[0], HI[0]); glob.clamp_(torch.tensor(LO[1:4], device=device), torch.tensor(HI[1:4], device=device))
        if it % 25 == 0 or it == iters - 1:
            log(f"    stage 2 it {it:3d}: window error {e.mean().item()*1000:.2f} mm")
    check_full(iters)
    return twin, best[1], best[2]
