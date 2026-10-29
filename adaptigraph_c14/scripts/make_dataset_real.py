"""Synthetic training data ANCHORED IN REAL STATES (2026-10-23).

The old generators start every chain from an artificial constant-curvature arc
of 0.25-0.60 m and drive it with random actions. The real cables are 0.60-0.89 m
long and lie almost straight (end-to-end / length 0.89 vs 0.70 in the data), so
the networks never saw a state like ours. Here the targets still come from the
analytical mass-spring simulator ONLY, but:

  * every chain starts from a REAL tracked frame (X-Planar set, 103 clips x 4
    cables -- never the 14 evaluation clips), length = that frame's arc length;
  * the two ends follow the REAL measured end motion of that trajectory
    (edge-midpoint shift + edge rotation, exactly what the evaluation feeds the
    nets), scaled by one factor per chain in [s_lo, s_hi];
  * the rest shape is the trajectory's frame-0 turning angles times a factor in
    [rest_lo, rest_hi], clipped to +-rest_cap (the cable's plastic set as
    recorded);
  * m, k_b (uniform along the wire), k_d, k_c sampled log/uniformly around the
    measured shared set (0.04, 1.0, 20, 0.15).

mode "step": one transition per 1 s action (n_move 0.6 s + n_settle 0.4 s, the
    generator's own protocol), actions from frames 4 apart -> Caporali's MLP.
mode "seq":  one transition per video frame (0.25 s, ends ramped over the whole
    frame), with the 4 preceding frames as history (real frames before the start,
    simulated ones after) -> AdaptiGraph / sequence models.

Output format identical to data/dlo_dataset_twoend.pt (step) and
data/dlo_dataset_seq.pt (seq), params [m, k_b x (N-2), k_d, k_c, length]
(step: k_b collapsed to one column like --flat-kb).

    python3 scripts/make_dataset_real.py --mode step --chains 50000 --out data/dlo_dataset_real_step.pt
    python3 scripts/make_dataset_real.py --mode seq  --chains 30000 --out data/dlo_dataset_real_seq.pt
"""
import argparse, glob, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wire_model.dlo_model import DLOSimulator, DLOParams
from wire_model.dataset import SimConfig
from wire_model.actions import rollout, terminal_edge_action
from wire_model.geometry import arc_length

ROOT = Path(__file__).resolve().parents[1]
N = 16


def turning(P):
    u = np.diff(P, axis=-2)
    return np.arctan2(u[..., :-1, 0] * u[..., 1:, 1] - u[..., :-1, 1] * u[..., 1:, 0], (u[..., :-1, :] * u[..., 1:, :]).sum(-1))


def edge_pose(P, at_end):
    a, b = (P[:, -2], P[:, -1]) if at_end else (P[:, 0], P[:, 1])
    return 0.5 * (a + b), np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])


def edge_action(Pa, Pb, at_end, scale):
    """measured edge motion from frame a to b (midpoint shift, rotation about the midpoint), scaled"""
    m0, a0 = edge_pose(Pa, at_end); m1, a1 = edge_pose(Pb, at_end)
    d = (m1 - m0) * scale[:, None]; th = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)) * scale
    return d, th


def logu(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["step", "seq"], required=True)
    ap.add_argument("--chains", type=int, default=50000); ap.add_argument("--chain-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracks", default="X-Planar_vids/mujoco_videos/tracks")
    ap.add_argument("--mass", type=float, nargs=2, default=[0.02, 0.10]); ap.add_argument("--kb", type=float, nargs=2, default=[0.2, 3.0])
    ap.add_argument("--kd", type=float, nargs=2, default=[5.0, 40.0]); ap.add_argument("--kc", type=float, nargs=2, default=[0.05, 0.40])
    ap.add_argument("--scale", type=float, nargs=2, default=[0.6, 1.4]); ap.add_argument("--rest", type=float, nargs=2, default=[0.6, 1.2])
    ap.add_argument("--rest-cap", type=float, default=0.6); ap.add_argument("--max-stretch", type=float, default=0.01)
    ap.add_argument("--stride", type=int, default=None,
                    help="video frames per transition (default 4 = 1 s for step, 1 for seq); step with 1 = one "
                         "0.25 s frame per transition, the 1 s move/settle protocol scaled by 1/4")
    ap.add_argument("--speed", type=float, nargs=2, default=None,
                    help="speed-only variation: time-warp the real end paths by one factor per chain (user rule "
                         "2026-10-27, e.g. 0.8 1.2); --scale is then forced to 1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stride = a.stride or (4 if a.mode == "step" else 1)
    chain_len = a.chain_len or (12 // stride if a.mode == "step" else 12)
    if a.speed:
        a.scale = [1.0, 1.0]
    H = 4
    rng = np.random.default_rng(a.seed); dev = "cuda"; cfg = SimConfig()
    sim = DLOSimulator(n_nodes=N, dt=cfg.dt, device=dev)
    if a.mode == "step":
        n_move, n_settle = cfg.n_move * stride // 4, cfg.n_settle * stride // 4   # the generator's 1 s protocol per 4 frames
    else:
        n_move, n_settle = (cfg.n_move + cfg.n_settle) // 4, 0       # one 0.25 s frame, ends moving throughout
    trajs = []
    for f in sorted(glob.glob(str(ROOT / a.tracks / "*.npz"))):
        z = np.load(f)
        for k in range(4):
            P = z[f"c{k}"].astype(np.float32)
            if np.isfinite(P).all() and len(P) > int(np.ceil(stride * chain_len * (a.speed[1] if a.speed else 1))) + H + 1:
                trajs.append(P)
    print(f"{len(trajs)} real trajectories from {a.tracks}; mode {a.mode}, stride {stride}, chain_len {chain_len}", flush=True)
    out = {k: [] for k in ("hist", "V_in", "V_out", "action", "params", "rest_angle")}
    t_start = time.time(); n_drop = n_tot = 0
    for b0 in range(0, a.chains, a.batch):
        B = min(a.batch, a.chains - b0)
        ti = rng.integers(0, len(trajs), B)
        if a.speed:                                                     # same geometric path, warped timing
            sp = rng.uniform(*a.speed, B)
            span = stride * chain_len * sp
            t0 = np.array([rng.integers(H, len(trajs[i]) - int(np.ceil(w))) for i, w in zip(ti, span)])
            def warp(P, t, v):
                tt = t + v * np.arange(stride * chain_len + 1); k = np.minimum(np.floor(tt).astype(int), len(P) - 2)
                f = (tt - k)[:, None, None]
                return np.concatenate([P[t - H:t], (1 - f) * P[k] + f * P[k + 1]])
            seg = np.stack([warp(trajs[i], t, v) for i, t, v in zip(ti, t0, sp)]).astype(np.float32)
        else:
            t0 = np.array([rng.integers(H, len(trajs[i]) - stride * chain_len) for i in ti])
            seg = np.stack([trajs[i][t - H:t + stride * chain_len + 1] for i, t in zip(ti, t0)])   # (B, H + S*L + 1, N, 2)
        f0 = np.stack([trajs[i][0] for i in ti])
        L = np.linalg.norm(np.diff(seg[:, H], axis=1), axis=2).sum(1).astype(np.float32)
        rest = np.clip(turning(f0) * rng.uniform(*a.rest, (B, 1)), -a.rest_cap, a.rest_cap).astype(np.float32)
        m, kb, kd = logu(rng, *a.mass, B), logu(rng, *a.kb, B), logu(rng, *a.kd, B)
        kc = rng.uniform(*a.kc, B).astype(np.float32); sc = rng.uniform(*a.scale, B).astype(np.float32)
        T = lambda x: torch.tensor(x, device=dev)
        prm = DLOParams(m=T(m), k_b=T(np.repeat(kb[:, None], N - 2, 1)), k_d=T(kd), k_c=T(kc), k_s=torch.full((B,), 1e5, device=dev),
                        length=T(L), rest_angle=T(rest)).to_batched(B, dev, n_nodes=N)
        p = T(seg[:, H].copy()); static = torch.zeros(B, N, dtype=torch.bool, device=dev)
        window = [seg[:, j].copy() for j in range(H)]                   # real frames before the start
        par = np.concatenate([m[:, None], np.repeat(kb[:, None], N - 2, 1), kd[:, None], kc[:, None], L[:, None]], 1).astype(np.float32)
        alive = np.ones(B, dtype=bool)
        for s in range(chain_len):
            Pa, Pb = seg[:, H + stride * s], seg[:, H + stride * (s + 1)]
            dE, thE = edge_action(Pa, Pb, True, sc); dH, thH = edge_action(Pa, Pb, False, sc)
            act = terminal_edge_action(N, T(dE[:, 0]), T(dE[:, 1]), T(thE.astype(np.float32)), at_end=True)
            hub = terminal_edge_action(N, T(dH[:, 0]), T(dH[:, 1]), T(thH.astype(np.float32)), at_end=False)
            with torch.no_grad():
                pf = rollout(sim, p, prm, act, static_mask=static, n_move=n_move, n_settle=n_settle, action2=hub)
            ok = torch.isfinite(pf).all(dim=(1, 2)).cpu().numpy() & ((arc_length(pf) / T(L) - 1).abs() <= a.max_stretch).cpu().numpy()
            alive &= ok
            vin, vout = p.cpu().numpy(), pf.cpu().numpy()
            vec = np.concatenate([np.stack([np.full(B, N - 2), dE[:, 0], dE[:, 1], thE], 1),
                                  np.stack([np.zeros(B), dH[:, 0], dH[:, 1], thH], 1)], 1).astype(np.float32)
            keep = alive.copy(); n_tot += B; n_drop += int((~keep).sum())
            out["hist"].append(np.stack(window, 1)[keep]); out["V_in"].append(vin[keep]); out["V_out"].append(vout[keep])
            out["action"].append(vec[keep]); out["params"].append(par[keep]); out["rest_angle"].append(rest[keep])
            window = window[1:] + [vin]
            p = torch.where(torch.isfinite(pf), pf, p)
        print(f"  chains {b0 + B}/{a.chains}: {sum(len(x) for x in out['V_in'])} transitions kept, "
              f"{100 * n_drop / max(n_tot, 1):.1f}% dropped (stretch > {100 * a.max_stretch:.0f}% or non-finite)  ({time.time() - t_start:.0f}s)", flush=True)
    d = {k: torch.tensor(np.concatenate(v)) for k, v in out.items()}
    if a.mode == "step":
        P_ = d["params"]; d["params"] = torch.cat([P_[:, :1], P_[:, 1:-3].mean(1, keepdim=True), P_[:, -3:]], 1)   # [m, k_b, k_d, k_c, L]
        del d["hist"]
    else:
        d["t_sub"] = torch.zeros(len(d["V_in"]), dtype=torch.long)
    d["config"] = {"generator": "make_dataset_real", **vars(a), "stride": stride, "chain_len": chain_len, "n_move": n_move, "n_settle": n_settle}
    path = ROOT / a.out; path.parent.mkdir(parents=True, exist_ok=True); torch.save(d, path)
    print(f"-> {a.out}: {len(d['V_in'])} transitions ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
