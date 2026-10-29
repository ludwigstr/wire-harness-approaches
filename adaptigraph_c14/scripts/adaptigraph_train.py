"""Train AdaptiGraph's GBND on the project's synthetic frame-cadence sequences
(dlo_dataset_seq.pt: 4-frame history, per-node k_b, both ends driven), then
optionally fine-tune on the real frame-cadence transitions (in-sample, like
rows 20-28). phi = normalised mean log k_b of the object.

    python3 scripts/adaptigraph_train.py --out data/wm/adaptigraph_sim.pt
    python3 scripts/adaptigraph_train.py --init data/wm/adaptigraph_sim.pt --real --out data/wm/adaptigraph_real.pt
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from wire_model.adaptigraph import AdaptiGraph

KB_LO, KB_HI = np.log(0.004), np.log(1.0)


def phi_from_kb(kb):                                 # (B, 14) -> (B,) in (0, 1)
    return ((np.log(kb).mean(-1) - KB_LO) / (KB_HI - KB_LO)).astype(np.float32)


def load_sim(n_hist, path="data/dlo_dataset_seq.pt", phi_key=None):
    d = torch.load(ROOT / path, weights_only=False)
    hist = d["hist"][:, -n_hist:].float(); X = d["V_in"].float(); Y = d["V_out"].float()
    if phi_key:                                      # phi stored directly (e.g. FleX data: AdaptiGraph's s in [0, 1])
        phi = d[phi_key].float()
    else:
        phi = torch.tensor(phi_from_kb(d["params"][:, 1:15].numpy()))
    return hist, X, Y, phi


def load_real(n_hist, clips=None):
    tr = torch.load(ROOT / "data/transitions_seq_all.pt", weights_only=False)["transitions"]
    if clips: tr = [t for t in tr if t["clip"] in clips]
    hist = torch.tensor(np.stack([np.asarray(t["hist"])[-n_hist:] for t in tr]), dtype=torch.float32)
    X = torch.tensor(np.stack([np.asarray(t["V_in"]) for t in tr]), dtype=torch.float32)
    Y = torch.tensor(np.stack([np.asarray(t["V_out"]) for t in tr]), dtype=torch.float32)
    return hist, X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--init", default=None); ap.add_argument("--real", action="store_true")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-hist", type=int, default=3); ap.add_argument("--radius", type=float, default=0.12); ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default="data/dlo_dataset_seq.pt"); ap.add_argument("--kb-range", type=float, nargs=2, default=[0.004, 1.0])
    ap.add_argument("--phi-key", default=None, help="read phi directly from this dataset key (e.g. 'phi' for make_dataset_flex.py data) instead of from log k_b")
    ap.add_argument("--phi-real", type=float, default=0.5, help="phi used for the real transitions during fine-tuning (unknown; nominal)")
    a = ap.parse_args(); dev = a.device; torch.manual_seed(0)
    log_f = open(ROOT / a.out.replace(".pt", "_log.txt"), "a")
    def log(m): line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()
    model = AdaptiGraph(16, a.n_hist, radius=a.radius).to(dev)
    if a.init:
        model.load_state_dict(torch.load(ROOT / a.init, map_location=dev, weights_only=False)["state_dict"])
    if a.real:
        hist, X, Y = load_real(a.n_hist); phi = torch.full((X.shape[0],), a.phi_real)
        log(f"REAL fine-tuning: {X.shape[0]} transitions from 14 clips, phi fixed at {a.phi_real}")
    else:
        global KB_LO, KB_HI
        KB_LO, KB_HI = np.log(a.kb_range[0]), np.log(a.kb_range[1])
        hist, X, Y, phi = load_sim(a.n_hist, a.data, a.phi_key); log(f"SIM training: {X.shape[0]} transitions, phi range {phi.min():.2f}-{phi.max():.2f}")
    n = X.shape[0]; perm = torch.randperm(n); n_val = max(1, n // 20); va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dp = torch.zeros_like(X); dp[:, 0] = Y[:, 0] - X[:, 0]; dp[:, -1] = Y[:, -1] - X[:, -1]
    best = float("inf"); hist_log = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tl = 0; nb = 0
        for i in range(0, len(tr), a.batch):
            b = tr[i:i + a.batch]
            pred = model(hist[b].to(dev), X[b].to(dev), phi[b].to(dev), dp[b].to(dev))
            loss = F_mse(pred, Y[b].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); tl += float(loss); nb += 1
        sched.step(); model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, len(va), 1024):
                b = va[i:i + 1024]; pred = model(hist[b].to(dev), X[b].to(dev), phi[b].to(dev), dp[b].to(dev))
                errs.append((pred[:, 1:-1] - Y[b].to(dev)[:, 1:-1]).norm(dim=-1).mean(-1))
            val_mm = torch.cat(errs).mean().item() * 1000
            base_mm = (X[va][:, 1:-1] - Y[va][:, 1:-1]).norm(dim=-1).mean().item() * 1000
        hist_log.append({"epoch": ep, "train_loss": tl / nb, "val_mm": val_mm})
        improved = val_mm < best
        if improved:
            best = val_mm; torch.save({"state_dict": model.state_dict(), "n_hist": a.n_hist, "radius": a.radius, "real": a.real, "epoch": ep, "val_mm": val_mm, "hist": hist_log}, ROOT / a.out)
        log(f"ep {ep:3d}: train {tl/nb:.5f} | val one-step {val_mm:.2f} mm (no-change {base_mm:.2f}){' *' if improved else ''} ({time.time()-t0:.0f}s)")
    log(f"=== ADAPTIGRAPH_TRAIN_FERTIG best val {best:.2f} mm -> {a.out} ===")


def F_mse(pred, Y):
    return ((pred[:, 1:-1] - Y[:, 1:-1]) ** 2).mean()


if __name__ == "__main__":
    main()
