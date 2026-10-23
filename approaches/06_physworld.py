"""PhysWorld (Yang et al. 2025): real video -> physics-aware demonstration
synthesis with a digital twin -> a graph world model trained on the synthetic
demonstrations. Table: world sheet row 20 -- 31.4 mm/node (median 25.6).

Pipeline (the digital twin is the PhysTwin spring-mass twin of each clip, row 19;
the paper's MPM twin is replaced because a cable is 1-D):
  1. twin per clip: topology radius, stiffness, dashpot, drag, friction from the
     PhysTwin fit (checkpoints/phystwin_twins/<clip>.pt)
  2. synthesis: --episodes per clip, each from a REAL frame of the clip, both
     ends on Bezier paths with a three-phase velocity profile (accelerate, cruise,
     decelerate; max displacement 15 cm over 16 frames), stiffness perturbed
     part-aware per node (sigma 0.5 in log k, correlation length 3 nodes)
  3. GNN world model = the AdaptiGraph GBND with a per-node property Phi (log mean
     stiffness of the node's springs), history h = 3, multi-step loss over
     tau = 3 (paper eq. 8), noise on the history
  4. Phi fine-tuning per node on the real frame transitions of each clip, net frozen
     (the only contact with the real data, in-sample)
  5. evaluation: 5-frame warm-up, open loop from the measured end motions, ends pinned

Variants (2026-10-23; defaults = the table's 31.4):
  --k median|dense      per-cable median stiffness (table) or the full per-spring twin
  --rest start|frame0   episode start frame at rest (table) or the twin's own rest geometry
  --v0 zero|measured    initial velocity of an episode
  --motion bezier|mix   paper Bezier paths (table) or 1/3 Bezier calibrated to the measured
                        end displacements, 1/3 measured end paths, 1/3 measured paths
                        time-warped 0.5-2x and scaled 0.5-1.5x

    python approaches/06_physworld.py                       # train + evaluate, ~20 min on one GPU
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from common import ROOT
from wire_model.phystwin.sim import SpringMassTwin, build_topology
from wire_model.physworld.synth import bezier_paths, part_aware_noise, node_phi, real_end_paths
from wire_model.adaptigraph import AdaptiGraph

H = 3


def load_twins(dev, twins_dir="checkpoints/phystwin_twins"):
    """PhysTwin twins -> {clip: record} with the fields the synthesis reads; the dense
    per-spring stiffness is returned separately per clip"""
    tw, dense = {}, {}
    for f in sorted((ROOT / twins_dir).glob("*.pt")):
        t = torch.load(f, map_location="cpu", weights_only=False)
        k, adj = t["k"], t["adj"].bool()
        tw[f.stem] = {"radius_seg": t["radius_seg"], "k_hom": float(np.exp(t["x1"][0])), "gamma": float(t["gl"][0]),
                      "delta": float(t["gl"][1]), "mu": float(t["gl"][2]),
                      "k_dense_median": [float(k[c][adj[c]].median()) for c in range(k.shape[0])] if k.shape[-1] > 1 else None}
        dense[f.stem] = k
    return tw, dense


def synth_clip(P, prm, n_ep, T_ep, dev, rng, frame_s=0.25, cfg=None, dense=None, d15=None):
    """P (C, T, N, 2) real tracks of the clip; prm = PhysTwin record -> (X (M,T_ep,N,2), phi (M,N))
    cfg (2026-10-23 variants):
      k      "median": per-cable median of the dense stiffness (first run) | "dense": the full per-spring k
      rest   "start": springs at rest in the episode's start frame (first run) | "frame0": the twin's own
             rest geometry = the clip's frame 0, as fitted
      v0     "zero" | "measured": backward difference of the track at the start frame
      motion "bezier": paper, 0.2-1.0 x 15 cm | "mix": 1/3 Bezier with distances drawn from the
             measured 15-frame end displacements, 1/3 measured end paths, 1/3 measured paths
             time-warped 0.5-2x and scaled 0.5-1.5x"""
    cfg = cfg or {"k": "median", "rest": "start", "v0": "zero", "motion": "bezier"}
    C, T, N, _ = P.shape
    r = round(prm["radius_seg"] * 2) / 2
    k_cable = torch.tensor(prm["k_dense_median"] or [prm["k_hom"]] * C, dtype=torch.float32, device=dev)
    adj0 = build_topology(P[:, 0], r, 6)                                               # the twin's topology (clip frame 0)
    Xs, PHI = [], []
    per_batch = 32
    for b0 in range(0, n_ep, per_batch):
        nb = min(per_batch, n_ep - b0)
        cab = rng.integers(0, C, nb)
        t0 = rng.integers(0, max(1, T - T_ep + 1), nb) if cfg["motion"] == "mix" else rng.integers(0, T, nb)
        X0 = P[cab, t0]                                                                 # (nb, N, 2)
        if cfg["rest"] == "frame0":
            twin = SpringMassTwin(P[cab, 0], adj0[cab], dt=1e-3, frame_s=frame_s)       # rest lengths of the fitted twin
        else:
            twin = SpringMassTwin(X0, build_topology(X0, r, 6), dt=1e-3, frame_s=frame_s)
        noise = torch.tensor(part_aware_noise(nb, N, 0.5, 3.0, rng), dtype=torch.float32, device=dev)
        if cfg["k"] == "dense":
            kb = dense[cab].to(dev)                                                     # (nb, N, N), zero off the fitted springs
            kb = torch.where(kb > 0, kb, k_cable[cab].view(nb, 1, 1).expand_as(kb))     # springs the start topology adds
        else:
            kb = k_cable[cab].view(nb, 1, 1).expand(nb, N, N)
        k = kb * torch.exp(0.5 * (noise.unsqueeze(2) + noise.unsqueeze(1)))
        if cfg["motion"] == "mix":
            kind = rng.integers(0, 3, nb); ends = torch.zeros(nb, T_ep, 2, 2, device=dev)
            ib = np.where(kind == 0)[0]
            if len(ib):
                rr = rng.choice(d15, len(ib)); rr2 = rng.choice(d15, len(ib))
                ends[ib, :, 0] = bezier_paths(X0[ib, 0], T_ep, 0.15, rng, r=rr); ends[ib, :, 1] = bezier_paths(X0[ib, -1], T_ep, 0.15, rng, r=rr2)
            for kk, warp in ((1, False), (2, True)):
                ii = np.where(kind == kk)[0]
                if len(ii):
                    ends[ii] = real_end_paths(P, cab[ii], t0[ii], T_ep, rng, warp)
                    ends[ii, :, 0] += X0[ii, 0].unsqueeze(1) - ends[ii, :1, 0]; ends[ii, :, 1] += X0[ii, -1].unsqueeze(1) - ends[ii, :1, 1]
        else:
            e0 = bezier_paths(X0[:, 0], T_ep, 0.15, rng); e1 = bezier_paths(X0[:, -1], T_ep, 0.15, rng)
            ends = torch.stack([e0, e1], 2)
        if cfg["v0"] == "measured":
            V0 = (X0 - P[cab, np.maximum(t0 - 1, 0)]) / frame_s
        else:
            V0 = torch.zeros_like(X0)
        g = torch.full((nb,), prm["gamma"], device=dev); d = torch.full((nb,), prm["delta"], device=dev); mu = torch.full((nb,), prm["mu"], device=dev)
        with torch.no_grad():
            sim = twin.rollout(X0, V0, ends, k, g, d, mu)
        ok = torch.isfinite(sim).all(dim=(1, 2, 3)) & ((sim[:, :, 1:] - sim[:, :, :-1]).norm(dim=-1).amax((1, 2)) < 0.2)
        Xs.append(sim[ok].cpu()); PHI.append(node_phi(k, twin.adj.bool())[ok].cpu())
    return torch.cat(Xs), torch.cat(PHI)


def windows(X, phi, tau):
    """(M, T, N, 2) -> training samples (hist, x, targets tau, dp tau, phi)"""
    M, T, N, _ = X.shape; out = []
    for t in range(H - 1, T - tau):
        hist = X[:, t - H + 1:t + 1][:, :-1] if H > 1 else X[:, t:t]
        hist = torch.cat([X[:, max(0, t - H + j):max(0, t - H + j) + 1] for j in range(H)], 1)
        out.append((hist, X[:, t], X[:, t + 1:t + 1 + tau], phi))
    return out


def dp_of(Xa, Xb):
    d = torch.zeros_like(Xa); d[..., 0, :] = Xb[..., 0, :] - Xa[..., 0, :]; d[..., -1, :] = Xb[..., -1, :] - Xa[..., -1, :]; return d


def train_gnn(data, dev, epochs, log, tau=3, noise_mm=2.0):
    model = AdaptiGraph(16, H).to(dev); opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    hist, x, tgt, phi = (torch.cat([d[i] for d in data]) for i in range(4))
    phi_mu, phi_sd = phi.mean(), phi.std() + 1e-6; phin = (phi - phi_mu) / phi_sd
    n = x.shape[0]; perm = torch.randperm(n); va, tr = perm[: n // 20], perm[n // 20:]
    log(f"  GNN: {n} samples, tau {tau}, Phi mean {phi_mu:.2f} sd {phi_sd:.2f}")
    best = (float("inf"), None)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for i in range(0, len(tr), 256):
            b = tr[i:i + 256]; h = hist[b].to(dev) + torch.randn_like(hist[b].to(dev)) * noise_mm / 1000; xx = x[b].to(dev); loss = 0
            for s in range(tau):
                y = tgt[b, s].to(dev); pred = model(h, xx, phin[b].to(dev), dp_of(xx, y))
                loss = loss + ((pred[:, 1:-1] - y[:, 1:-1]) ** 2).mean()
                h = torch.cat([h[:, 1:], xx.unsqueeze(1)], 1); xx = pred
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        model.eval()
        with torch.no_grad():
            b = va[:4096]; y = tgt[b, 0].to(dev); xx = x[b].to(dev)
            v = (model(hist[b].to(dev), xx, phin[b].to(dev), dp_of(xx, y))[:, 1:-1] - y[:, 1:-1]).norm(dim=-1).mean().item() * 1000
        if v < best[0]: best = (v, {k_: t_.clone() for k_, t_ in model.state_dict().items()})
        log(f"  GNN ep {ep}: val one-step {v:.2f} mm ({time.time()-t0:.0f}s)")
    model.load_state_dict(best[1]); return model, float(phi_mu), float(phi_sd)


def finetune_phi_and_eval(model, P, phi0, warm, dev, iters=200):
    """per cable: per-node Phi (normalised) fitted on the real one-step transitions of the clip, net frozen;
    then open loop from the warm-up. Returns per-frame mean error over the 4 cables (mm)."""
    for p in model.parameters(): p.requires_grad_(False)
    C, T, N, _ = P.shape; errs = []; phis = []
    for c in range(C):
        X = P[c]
        hist = torch.stack([torch.stack([X[max(0, t - H + j)] for j in range(H)]) for t in range(T - 1)])
        dp = dp_of(X[:-1], X[1:])
        phi = torch.full((N,), phi0, device=dev, requires_grad=True); opt = torch.optim.Adam([phi], lr=0.05)
        for it in range(iters):
            pred = model(hist, X[:-1], phi.expand(T - 1, N), dp)
            loss = ((pred[:, 1:-1] - X[1:, 1:-1]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        phis.append(phi.detach().cpu().numpy())
        with torch.no_grad():
            frames = [X[t] for t in range(warm)]; e = []
            for t in range(warm - 1, T - 1):
                h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
                pr = model(h[None], frames[t][None], phi.detach()[None], dp[t][None])[0]; pr[0] = X[t + 1, 0]; pr[-1] = X[t + 1, -1]
                frames.append(pr); e.append((pr - X[t + 1]).norm(dim=-1).mean().item() * 1000)
        errs.append(np.array(e))
    return np.mean(errs, 0), phis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--ep-frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--out", default="results/physworld_14.json"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", default="median", choices=["median", "dense"]); ap.add_argument("--rest", default="start", choices=["start", "frame0"])
    ap.add_argument("--v0", default="zero", choices=["zero", "measured"]); ap.add_argument("--motion", default="bezier", choices=["bezier", "mix"])
    ap.add_argument("--twin-dir", default="checkpoints/phystwin_twins")
    a = ap.parse_args(); dev = a.device; rng = np.random.default_rng(0); torch.manual_seed(0)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    twins, dense_k = load_twins(dev, a.twin_dir); log(f"{len(twins)} PhysTwin twins loaded ({a.twin_dir}); variant k={a.k} rest={a.rest} v0={a.v0} motion={a.motion}")
    data, tracks = [], {}
    for clip, prm in sorted(twins.items()):
        z = np.load(ROOT / f"data/tracks14/{clip}.npz")   # the clip's own tracks
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev); tracks[clip] = P
        cfg = {"k": a.k, "rest": a.rest, "v0": a.v0, "motion": a.motion}
        dense = dense_k[clip] if a.k == "dense" else None
        E = P[:, :, [0, -1]]; d15 = ((E[:, 15:] - E[:, :-15]).norm(dim=-1).flatten().cpu().numpy() if P.shape[1] > 15 else np.array([0.13]))
        t0 = time.time(); X, phi = synth_clip(P, prm, a.episodes, a.ep_frames, dev, rng, cfg=cfg, dense=dense, d15=d15)
        data += windows(X, phi, 3); log(f"{clip}: {X.shape[0]} demonstrations kept ({time.time()-t0:.0f}s)")
    model, phi_mu, phi_sd = train_gnn(data, dev, a.epochs, log)
    (ROOT / "work").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "phi_mu": phi_mu, "phi_sd": phi_sd}, ROOT / "work" / Path(a.out).name.replace(".json", ".pt"))
    res = {}
    for clip, P in tracks.items():
        e, phis = finetune_phi_and_eval(model, P, 0.0, a.warm, dev)
        res[clip] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()), "n_frames": int(len(e)), "video": "",
                     "phi_per_cable_mean": [float(p.mean() * phi_sd + phi_mu) for p in phis]}
        log(f"{clip}: Mittel {e.mean():.1f}")
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True); (ROOT / a.out).write_text(json.dumps(res, indent=1))
    v = np.array([r["mittel"] for r in res.values()])
    log(f"PhysWorld on the 14 clips: Mittel {v.mean():.2f}, Median {np.median(v):.2f} mm/node (frames after a {a.warm}-frame warm-up)")
    log("=== PHYSWORLD_FERTIG ===")


if __name__ == "__main__":
    main()
