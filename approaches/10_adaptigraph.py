"""AdaptiGraph (Zhang et al., RSS 2024): a material-conditioned graph-based neural
dynamics model (GBND). Vertices = the 16 cable nodes with a 3-frame history (as
displacements, which makes the model translation-equivariant), the particle type
(object / end effector), the material and ONE physical property variable phi;
edges = chain neighbours + all pairs within 12 cm; 4 rounds of message passing;
the decoder predicts every node's displacement, the two driven end nodes are set
by the action.
Table: world sheet row 24 -- 24.4 mm/node after fine-tuning on the real transitions of
the 14 clips, 28.4 trained in simulation only (the paper), both with the real-anchored
simulator data (--data real). With the original artificial-arc data: 26.1 / 63.9.

phi = the normalised mean log bending stiffness of the simulated cable. At test
time phi is estimated few-shot (paper Sec. III-C): the value on a grid over the
extended range [-0.2, 1.2] that best predicts the warm-up frames (the paper uses
Bayesian optimisation; for one scalar a dense grid is equivalent).
The paper trains in NVIDIA FleX; here the training data come from the project's
analytical mass-spring simulator (the same one as Caporali, see synthdata.py).

Steps
  data      synthetic sequence data (--data paper: original generator; real: real-anchored)
  train     simulation-only training (20 epochs, one-step MSE)
  finetune  optional, NOT in the paper: the real frame-cadence transitions of the
            14 clips (in-sample, phi fixed at 0.5 because the real stiffness is unknown)
  eval      few-shot phi on the first 5 frames, then open loop from the measured end
            motions with the model's own history; both ends pinned; scored = every
            frame after the 5-frame warm-up

    python approaches/10_adaptigraph.py eval --ckpt checkpoints/adaptigraph_real_real.pt   # 24.4
    python approaches/10_adaptigraph.py eval --ckpt checkpoints/adaptigraph_sim_real.pt    # 28.4
    python approaches/10_adaptigraph.py eval --ckpt checkpoints/adaptigraph_sim.pt         # 63.9 (original data)
    python approaches/10_adaptigraph.py all --data real --tag real && python approaches/10_adaptigraph.py finetune --tag real
"""
import argparse
import time

import numpy as np
import torch

from common import ROOT, load_clips, record, write_results
from wire_model.adaptigraph import AdaptiGraph

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KB_RANGE = {"paper": (0.004, 1.0), "real": (0.2, 3.0)}     # k_b range of each data set -> phi in (0, 1)
H = 3


def make_data(kind, out, seed=0):
    from synthdata import generate_paper, generate_real
    t0 = time.time()
    d = generate_paper("seq", 20000, seed=seed) if kind == "paper" else generate_real("seq", 30000, seed=seed)
    d["kb_range"] = KB_RANGE[kind]
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True); torch.save(d, ROOT / out)
    print(f"{kind} seq data: {len(d['V_in'])} transitions -> {out} ({time.time() - t0:.0f}s)")


def real_transitions(tracks="data/tracks14", n_hist=4):
    """the frame-cadence transitions of the evaluation clips: n_hist preceding frames, V_in, V_out"""
    hist, X, Y = [], [], []
    for c in load_clips(tracks):
        for P in c["P"]:
            for t in range(n_hist, len(P) - 1):
                hist.append(P[t - n_hist:t]); X.append(P[t]); Y.append(P[t + 1])
    return torch.tensor(np.array(hist)), torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def fit(model, hist, X, Y, phi, epochs, lr, out, log, extra):
    n = len(X); perm = torch.randperm(n); n_val = max(1, n // 20); va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dp = torch.zeros_like(X); dp[:, 0] = Y[:, 0] - X[:, 0]; dp[:, -1] = Y[:, -1] - X[:, -1]
    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for i in range(0, len(tr), 256):
            b = tr[i:i + 256]
            pred = model(hist[b].to(DEV), X[b].to(DEV), phi[b].to(DEV), dp[b].to(DEV))
            loss = ((pred[:, 1:-1] - Y[b].to(DEV)[:, 1:-1]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            e = torch.cat([(model(hist[b].to(DEV), X[b].to(DEV), phi[b].to(DEV), dp[b].to(DEV))[:, 1:-1] - Y[b].to(DEV)[:, 1:-1]).norm(dim=-1).mean(-1)
                           for b in va.split(1024)]).mean().item() * 1000
        if e < best:
            best = e; torch.save({"state_dict": model.state_dict(), "n_hist": H, "radius": model.radius, "epoch": ep, "val_mm": e, **extra}, ROOT / out)
        log(f"ep {ep:3d}: val one-step {e:.2f} mm{' *' if e == best else ''} ({time.time() - t0:.0f}s)")


def train_sim(data, out, epochs=20, lr=5e-4):
    torch.manual_seed(0)
    d = torch.load(ROOT / data, weights_only=False)
    lo, hi = np.log(d.get("kb_range", KB_RANGE["paper"]))
    phi = torch.tensor(((np.log(d["params"][:, 1:15].numpy()).mean(-1) - lo) / (hi - lo)).astype(np.float32))
    model = AdaptiGraph(16, H).to(DEV)
    print(f"SIM training: {len(d['V_in'])} transitions, phi {float(phi.min()):.2f}..{float(phi.max()):.2f}")
    fit(model, d["hist"][:, -H:].float(), d["V_in"].float(), d["V_out"].float(), phi, epochs, lr, out, print, {"kb_range": (float(np.exp(lo)), float(np.exp(hi)))})


def finetune_real(init, out, epochs=60, lr=2e-4, phi_real=0.5):
    torch.manual_seed(0)
    ck = torch.load(ROOT / init, map_location=DEV, weights_only=False)
    model = AdaptiGraph(16, ck["n_hist"], radius=ck["radius"]).to(DEV); model.load_state_dict(ck["state_dict"])
    hist, X, Y = real_transitions()
    print(f"REAL fine-tuning: {len(X)} transitions of the 14 clips, phi fixed at {phi_real}")
    fit(model, hist[:, -H:], X, Y, torch.full((len(X),), phi_real), epochs, lr, out, print, {"init": init})


def evaluate(ckpt, tracks, out, warm=5, phi_fixed=None, label=None):
    ck = torch.load(ROOT / ckpt, map_location=DEV, weights_only=False)
    m = AdaptiGraph(16, ck["n_hist"], radius=ck["radius"]).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
    Hh = ck["n_hist"]; grid = torch.linspace(-0.2, 1.2, 29)
    res = {}
    for c in load_clips(tracks):
        errs_c, phis = [], []
        for k in range(4):
            P = torch.tensor(c["P"][k], device=DEV); T = P.shape[0]
            hist_at = lambda t, fr: torch.stack([fr[max(0, t - Hh + j)] for j in range(Hh)])
            def dp_at(t):
                d = torch.zeros(16, 2, device=DEV); d[0] = P[t + 1, 0] - P[t, 0]; d[-1] = P[t + 1, -1] - P[t, -1]; return d
            with torch.no_grad():
                if phi_fixed is None:      # few-shot: one-step error on the warm-up transitions for every phi
                    ts = list(range(0, warm - 1))
                    hb = torch.stack([hist_at(t, P) for t in ts]); db = torch.stack([dp_at(t) for t in ts])
                    cost = [(m(hb, P[ts], torch.full((len(ts),), float(g), device=DEV), db)[:, 1:-1] - P[[t + 1 for t in ts]][:, 1:-1]).norm(dim=-1).mean().item()
                            for g in grid]
                    phi = float(grid[int(np.argmin(cost))])
                else:
                    phi = phi_fixed
                frames = [P[t] for t in range(warm)]; e = []
                for t in range(warm - 1, T - 1):
                    pr = m(hist_at(t, frames)[None], frames[t][None], torch.tensor([phi], device=DEV), dp_at(t)[None])[0]
                    pr[0] = P[t + 1, 0]; pr[-1] = P[t + 1, -1]
                    frames.append(pr); e.append((pr - P[t + 1]).norm(dim=-1).mean().item() * 1000)
            errs_c.append(np.array(e)); phis.append(phi)
        res[c["clip"]] = record(np.mean(errs_c, 0), phi=phis, warmup_frames=warm)
    write_results(res, out, label or f"AdaptiGraph {ckpt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["data", "train", "finetune", "eval", "all"])
    ap.add_argument("--data", default="paper", choices=["paper", "real"]); ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--tracks", default="data/tracks14"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.tag or ("" if a.data == "paper" else a.data); sfx = "_" + tag if tag else ""
    data_f, sim_ck, real_ck = f"work/adaptigraph_data{sfx}.pt", f"checkpoints/adaptigraph_sim{sfx}.pt", f"checkpoints/adaptigraph_real{sfx}.pt"
    if a.step in ("data", "all"):
        make_data(a.data, data_f)
    if a.step in ("train", "all"):
        train_sim(data_f, sim_ck)
    if a.step == "finetune":
        finetune_real(sim_ck, real_ck)
    if a.step in ("eval", "all"):
        ck = a.ckpt or sim_ck
        evaluate(ck, a.tracks, a.out or f"results/{ck.split('/')[-1].replace('.pt', '')}_14.json")


if __name__ == "__main__":
    main()
