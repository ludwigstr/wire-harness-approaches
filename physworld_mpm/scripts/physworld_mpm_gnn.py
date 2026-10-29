"""PhysWorld rebuild (2026-10-28): GNN world model trained on the MPM-twin demonstrations
of ONE clip (one model per scene, paper Sec. 4.1 "for each scene ... 500 episodes to
train the GNN"), then per-vertex Phi fine-tuning on the real clip with the network
frozen (Sec. 3.3, App. B.3), then the table's evaluation.

Training (Eq. 8): multi-step MSE over tau steps from randomly sampled start frames,
noise injected into the history positions and into Phi (paper). One-step validation
error on 10 % held-out demos selects the epoch.
Phi fine-tuning: Phi = per-node (log E, log rho), initialised from the twin's
per-particle values subsampled to the 16 nodes, kept within +-1.5 (log) of it = the
range the demos covered; Adam on the open-loop rollout of the real clip from the
warm-up (default; the one-step variant made the open loop worse: 114019 9.91 -> 14.61
unbounded, 13.58 boxed), loss = MSE + unidirectional squared-distance Chamfer (real ->
predicted nodes), equal weights (App. B.3 gives no weights).
Evaluation (identical to every row of the table): per cable 5-frame warm-up with the
measured states, then open loop driven by the measured end motions, both end nodes
pinned to the measured ends; error = mean node distance (mm) over the predicted frames.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from wire_model.physworld_mpm.gnn import PhysWorldGNN
from wire_model.physworld_mpm.twin import drive_from_nodes

H = 3


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def samples(nodes, yaw, tau):
    """nodes (n, 16, 2), yaw (n, 2) of one cable -> per start frame t: hist (h,16,2), x, targets (tau,16,2),
    dyaw (tau, 2)"""
    n = nodes.shape[0]
    idx = torch.arange(n)
    out = []
    for t in range(0, n - tau):
        hi = torch.clamp(torch.arange(t - H, t), min=0)
        out.append((nodes[hi], nodes[t], nodes[t + 1:t + 1 + tau], yaw[t + 1:t + 1 + tau] - yaw[t:t + tau]))
    return out


def dp_of(xa, xb):
    d = torch.zeros_like(xa); d[..., 0, :] = xb[..., 0, :] - xa[..., 0, :]; d[..., -1, :] = xb[..., -1, :] - xa[..., -1, :]
    return d


def dyaw_nodes(dy, N=16):
    """(..., 2) yaw change of hub/plug -> (..., N) on the end nodes"""
    out = torch.zeros(*dy.shape[:-1], N, device=dy.device, dtype=dy.dtype)
    out[..., 0] = dy[..., 0]; out[..., -1] = dy[..., 1]
    return out


def build(demos, tau, use_yaw):
    hs, xs, ys, dys, phis = [], [], [], [], []
    for d in demos:
        for c in range(d["nodes"].shape[1]):
            for h_, x_, y_, dy_ in samples(d["nodes"][:, c].float(), d["yaw"][c].float(), tau):
                hs.append(h_); xs.append(x_); ys.append(y_); dys.append(dy_ if use_yaw else torch.zeros_like(dy_))
                phis.append(d["phi"][c].float())
    return torch.stack(hs), torch.stack(xs), torch.stack(ys), torch.stack(dys), torch.stack(phis)


def train(demos, dev, epochs, tau, noise_mm, phi_noise, use_yaw, seed, logf):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(demos), generator=g).tolist()
    nva = max(1, len(demos) // 10)
    va_d, tr_d = [demos[i] for i in perm[:nva]], [demos[i] for i in perm[nva:]]
    tr = build(tr_d, tau, use_yaw); va = build(va_d, tau, use_yaw)
    allphi = torch.cat([d["phi"].reshape(-1, 2) for d in demos])
    mu, sd = allphi.mean(0), allphi.std(0) + 1e-6
    model = PhysWorldGNN(16, H).to(dev); opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = tr[0].shape[0]
    logf(f"  GNN: {n} train / {va[0].shape[0]} val samples from {len(tr_d)}/{len(va_d)} demos, tau {tau}, "
         f"Phi mean {mu.tolist()} sd {sd.tolist()}")
    best = (float("inf"), None, -1)
    tr = [t.to(dev) for t in tr]; va = [t.to(dev) for t in va]; mu, sd = mu.to(dev), sd.to(dev)
    for ep in range(epochs):
        model.train(); t0 = time.time(); p = torch.randperm(n, device=dev); tl = 0.0
        for i in range(0, n, 256):
            b = p[i:i + 256]
            hist = tr[0][b] + torch.randn_like(tr[0][b]) * noise_mm / 1000
            x = tr[1][b]; phi = (tr[4][b] - mu) / sd + torch.randn_like(tr[4][b]) * phi_noise
            loss = 0.0
            for s in range(tau):
                y = tr[2][b, s]
                pred = model(hist, x, phi, dp_of(x, y), dyaw_nodes(tr[3][b, s]))
                loss = loss + ((pred[:, 1:-1] - y[:, 1:-1]) ** 2).mean()
                hist = torch.cat([hist[:, 1:], x.unsqueeze(1)], 1); x = pred
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, va[0].shape[0], 2048):
                sl = slice(i, i + 2048); x = va[1][sl]; y = va[2][sl, 0]
                pr = model(va[0][sl], x, (va[4][sl] - mu) / sd, dp_of(x, y), dyaw_nodes(va[3][sl, 0]))
                errs.append((pr[:, 1:-1] - y[:, 1:-1]).norm(dim=-1).mean(-1))
            v = torch.cat(errs).mean().item() * 1000
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, ep)
        if ep % 5 == 0 or ep == epochs - 1:
            logf(f"  GNN ep {ep}: train {tl / n:.3e}  val one-step {v:.3f} mm ({time.time()-t0:.1f}s)")
    model.load_state_dict(best[1])
    logf(f"  GNN best epoch {best[2]} (val one-step {best[0]:.3f} mm)")
    return model, mu, sd, best[0]


def real_yaw(P):
    """P (4, T, 16, 2) torch -> yaw (4, T, 2) of the hub / plug modules (as the twin's drive)"""
    return drive_from_nodes(P[:, :, [0, 1, 14, 15]])[1]


def open_loop(model, X, phi, dp, dy, warm):
    """differentiable open loop from the warm-up (ends pinned) -> predicted frames warm..T-1 (T-warm, N, 2)"""
    T = X.shape[0]
    frames = [X[t] for t in range(warm)]
    for t in range(warm - 1, T - 1):
        h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
        pr = model(h[None], frames[t][None], phi[None], dp[t][None], dy[t][None])[0]
        pr = torch.cat([X[t + 1, :1], pr[1:-1], X[t + 1, -1:]], 0)
        frames.append(pr)
    return torch.stack(frames[warm:])


def finetune_and_eval(model, P, phi0, mu, sd, warm, dev, iters, use_yaw, finetune=True, box=None, objective="rollout"):
    """P (4, T, 16, 2) real; phi0 (4, 16, 2) twin node properties (raw). Returns per-frame mean error over
    the 4 cables (mm, predicted frames) and the fitted Phi (raw units)"""
    for p in model.parameters():
        p.requires_grad_(False)
    C, T, N, _ = P.shape
    Y = real_yaw(P)
    errs, phis = [], []
    for c in range(C):
        X = P[c]
        hist = torch.stack([X[torch.clamp(torch.arange(t - H, t), min=0)] for t in range(T - 1)])
        dp = dp_of(X[:-1], X[1:])
        dy = dyaw_nodes((Y[c, 1:] - Y[c, :-1]) if use_yaw else torch.zeros_like(Y[c, 1:]))
        phi = ((phi0[c] - mu) / sd).clone().requires_grad_(True)
        lo = hi = None
        if box is not None:                       # keep Phi inside the property range the GNN was trained on
            lo = (phi0[c] - box - mu) / sd; hi = (phi0[c] + box - mu) / sd
        if finetune:
            opt = torch.optim.Adam([phi], lr=0.05)
            for it in range(iters):
                if objective == "rollout":
                    pred = open_loop(model, X, phi, dp, dy, warm); tgt = X[warm:]
                else:
                    pred = model(hist, X[:-1], phi.expand(T - 1, N, 2), dp, dy); tgt = X[1:]
                mse = ((pred[:, 1:-1] - tgt[:, 1:-1]) ** 2).mean()
                cd = (torch.cdist(tgt, pred).min(-1).values ** 2).mean()      # unidirectional Chamfer real -> pred (squared)
                loss = mse + cd
                opt.zero_grad(); loss.backward(); opt.step()
                if lo is not None:
                    with torch.no_grad():
                        phi.copy_(torch.maximum(torch.minimum(phi, hi), lo))
        phis.append((phi.detach() * sd + mu).cpu().numpy())
        with torch.no_grad():
            frames = [X[t] for t in range(warm)]; e = []
            for t in range(warm - 1, T - 1):
                h = torch.stack([frames[max(0, t - H + j)] for j in range(H)])
                pr = model(h[None], frames[t][None], phi.detach()[None], dp[t][None], dy[t][None])[0]
                pr[0] = X[t + 1, 0]; pr[-1] = X[t + 1, -1]
                frames.append(pr); e.append((pr - X[t + 1]).norm(dim=-1).mean().item() * 1000)
        errs.append(np.array(e))
    return np.mean(errs, 0), phis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--demos-tag", default="")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--noise-mm", type=float, default=1.0); ap.add_argument("--phi-noise", type=float, default=0.05)
    ap.add_argument("--no-yaw", action="store_true"); ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--phi-iters", type=int, default=100); ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phi-box", type=float, default=1.5,
                    help="fine-tuned Phi kept within +-box (log units) of the twin's value = 3 sigma of the demo "
                         "perturbation (0 = unbounded)")
    ap.add_argument("--phi-objective", default="rollout", choices=["rollout", "onestep"],
                    help="rollout: loss over the open-loop rollout from the warm-up (the quantity the table scores, "
                         "as every in-sample fit of the table); onestep: real one-step transitions")
    ap.add_argument("--load", default="", help="skip training, load this GNN .pt (evaluation only)")
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed)
    use_yaw = not a.no_yaw
    D = torch.load(ROOT / f"data/wm/physworld_mpm_demos/{a.clip}{a.demos_tag}.pt", weights_only=False)
    t0 = time.time()
    if a.load:
        ck = torch.load(ROOT / a.load, weights_only=False)
        model = PhysWorldGNN(16, H).to(dev); model.load_state_dict(ck["state_dict"])
        mu, sd = ck["phi_mu"].to(dev), ck["phi_sd"].to(dev); val = ck.get("val", float("nan"))
        t_train = ck.get("sec_train", float("nan"))
    else:
        model, mu, sd, val = train(D["demos"], dev, a.epochs, a.tau, a.noise_mm, a.phi_noise, use_yaw, a.seed, log)
        t_train = time.time() - t0
    z = np.load(ROOT / f"data/tracks14/{a.clip}.npz")
    P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
    phi0 = D["phi_twin"].to(dev)
    e0, _ = finetune_and_eval(model, P, phi0, mu, sd, a.warm, dev, 0, use_yaw, finetune=False)
    t1 = time.time()
    e, phis = finetune_and_eval(model, P, phi0, mu, sd, a.warm, dev, a.phi_iters, use_yaw, box=a.phi_box or None,
                                objective=a.phi_objective)
    t_phi = time.time() - t1
    res = {"clip": a.clip, "mittel": float(e.mean()), "start": float(e[0]), "ende": float(e[-1]), "n_frames": int(len(e)),
           "per_frame": e.tolist(), "mittel_before_phi_finetune": float(e0.mean()), "val_one_step_mm": val,
           "n_demos": len(D["demos"]), "n_train_frames": int(sum(d["nodes"].shape[0] for d in D["demos"])),
           "phi_twin": phi0.cpu().tolist(), "phi_fitted": [p.tolist() for p in phis],
           "sec_train": t_train, "sec_phi": t_phi, "use_yaw": use_yaw, "epochs": a.epochs, "tau": a.tau,
           "n_gnn_params": sum(p.numel() for p in model.parameters()), "phi_box": a.phi_box, "phi_objective": a.phi_objective, "phi_iters": a.phi_iters}
    tag = f"_{a.tag}" if a.tag else ""
    outd = ROOT / f"data/wm/physworld_mpm_gnn{tag}"; outd.mkdir(parents=True, exist_ok=True)
    (outd / f"{a.clip}.json").write_text(json.dumps(res, indent=1))
    if not a.load:
        torch.save({"state_dict": model.state_dict(), "phi_mu": mu.cpu(), "phi_sd": sd.cpu(), "val": val,
                    "sec_train": t_train}, outd / f"{a.clip}.pt")
    log(f"{a.clip}: PhysWorld (MPM twin) Mittel {e.mean():.2f} mm (before Phi fine-tuning {e0.mean():.2f}); "
        f"train {t_train:.0f}s, Phi fit {t_phi:.0f}s")


if __name__ == "__main__":
    main()
