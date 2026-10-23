"""Caporali et al., "Deformable Linear Objects Manipulation with Online Model
Parameters Estimation" (IEEE RA-L 2024), as in the paper.

The Fig. 3 MLP (FiLM conditioning, width 256, both cable ends driven = 8-wide
action) learns the one-step response of the analytical mass-spring model from
SYNTHETIC data only. On a real clip the network stays frozen and the model
parameters m, k_b, k_d are estimated ONLINE (Sec. III-D): predict the next
1 s step with the current estimate, observe the real frame, take one Adam step
on the parameters (logit space inside the training box), carry them forward.
The forecast is recorded BEFORE its frame is used, so the score never sees the
frame it predicts.
Table: world sheet row 23 -- 18.4 mm/node on the 14 clips with the real-anchored training
data (--data real, rest from frame 0, online estimate started at the measured 0.04/1.0/20/0.15;
23.0 re-reading the rest shape every step). With the original artificial-arc data: 56.8.

Steps
  data   generate the synthetic training set (--data paper: the original
         generator behind 56.8; --data real: real-anchored, see synthdata.py)
  train  train the MLP (25 epochs, AdamW + OneCycle, MSE on node positions)
  eval   online-estimation rollout on the tracks; scored = every 4th frame
         (1 s steps) plus the last, from frame 0
  Rest shape (the cable's plastic set, an extension of the paper, whose rest
  shape is straight): --rest current re-reads it from the current predicted
  shape every step (the table's 56.8); --rest frame0 reads it once from the
  measured first frame and holds it.

    python approaches/09_caporali.py eval --tag real --rest frame0 --nominal 0.04,1.0,20,0.15   # -> 18.4
    python approaches/09_caporali.py eval                         # original data, checkpoints/caporali_mlp.pt -> 56.8
    python approaches/09_caporali.py all --data real --tag real --rest frame0 --nominal 0.04,1.0,20,0.15
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, load_clips, record, write_results, turning_angles
from wire_model.estimation import load_predictor
from wire_model.nn_model import DLONetwork, DLOPredictor, Normalizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOMINAL = np.array([0.02, 0.1, 10.0, 0.15], np.float32)    # start of the online estimate [m, k_b, k_d, k_c]
REST_CAP = 0.35


# ---------------------------------------------------------------- data + training
def make_data(kind, out, seed=0):
    from synthdata import generate_paper, generate_real
    t0 = time.time()
    d = generate_paper("step", 40000, seed=seed) if kind == "paper" else generate_real("step", 60000, seed=seed)
    (ROOT / out).parent.mkdir(parents=True, exist_ok=True); torch.save(d, ROOT / out)
    print(f"{kind} data: {len(d['V_in'])} transitions -> {out} ({time.time() - t0:.0f}s)")


def normalizer(d, n_nodes=16):
    a, p = d["action"], d["params"]
    lo, hi = a.min(0).values.clone(), a.max(0).values.clone()
    for c in range(0, a.shape[1], 4):              # alpha columns are edge indices
        lo[c], hi[c] = 0.0, float(n_nodes - 2)
    n_in = p.shape[1] - 1                          # [m, k_b, k_d, k_c] -- length is not an input
    log_mask = torch.zeros(n_in, dtype=torch.bool); log_mask[1:n_in - 2] = True     # k_b in log space
    return Normalizer(action_lo=lo, action_hi=hi, param_lo=p[:, :n_in].min(0).values.clone(),
                      param_hi=p[:, :n_in].max(0).values.clone(), param_log=log_mask, pos_scale=0.3)


def train(data, out, epochs=25, width=256, lr=1e-3, batch=1024):
    d = torch.load(ROOT / data, weights_only=False)
    V_in, V_out, action, params, rest = d["V_in"], d["V_out"], d["action"], d["params"][:, :-1], d["rest_angle"]
    norm = normalizer(d).to(DEV)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(V_in), generator=g); n_val = int(0.1 * len(V_in))
    va, tr = perm[:n_val], perm[n_val:]
    torch.manual_seed(0)
    model = DLOPredictor(DLONetwork(n_nodes=16, width=width, n_rest=14, n_params=params.shape[1], combine="film"), norm).to(DEV)
    dl = DataLoader(TensorDataset(V_in[tr], V_out[tr], action[tr], params[tr], rest[tr]), batch_size=batch, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        for vi, vo, a, p, r in dl:
            loss = torch.nn.functional.mse_loss(model(vi.to(DEV), a.to(DEV), p.to(DEV), r.to(DEV)), vo.to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            e = []
            for i in range(0, len(va), 4096):
                b = va[i:i + 4096]
                e.append((model(V_in[b].to(DEV), action[b].to(DEV), params[b].to(DEV), rest[b].to(DEV)) - V_out[b].to(DEV)).norm(dim=-1).mean(-1).cpu())
            v = torch.cat(e).mean().item() * 1000
        if v < best:
            best = v
            torch.save({"state_dict": model.net.state_dict(), "norm": dict(norm.__dict__), "n_nodes": 16, "width": width, "n_rest": 14,
                        "arch": "film", "n_action": action.shape[1], "n_params": params.shape[1]}, ROOT / out)
        print(f"ep {ep + 1:2d}/{epochs}: val one-step {v:.2f} mm/node{' *' if v == best else ''}", flush=True)


# ---------------------------------------------------------------- evaluation
def step_action(s0, real, k):
    """8-vector [alpha, dx, dy, dtheta] x (plug edge, hub edge) from frame s0 to frame real"""
    def edge(V, at_end):
        a, b = (V[-2], V[-1]) if at_end else (V[0], V[1])
        return 0.5 * (a + b), np.arctan2(b[1] - a[1], b[0] - a[0])
    out = []
    for at_end, alpha in ((True, k), (False, 0.0)):
        m0, a0 = edge(s0, at_end); m1, a1 = edge(real, at_end)
        out += [alpha, m1[0] - m0[0], m1[1] - m0[1], np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))]
    return np.array(out, np.float32)


def rollout_online(model, cables, stride=4, lr=0.05, rest_mode="current", rest_cap=REST_CAP, nominal=NOMINAL):
    """cables: list of 4 arrays (T, N, 2) -> list of (frame index, predicted (4, N, 2))"""
    T, N = len(cables[0]), cables[0].shape[1]
    idxs = sorted(set(list(range(0, T - 1, stride)) + [T - 1]))
    V = np.stack([c[0] for c in cables]).astype(np.float32); B = len(cables)
    n_free = len(nominal) - 1                                  # m, k_b, k_d (k_c stays nominal)
    lo = model.norm.param_lo[:n_free].float().to(DEV); hi = model.norm.param_hi[:n_free].float().to(DEV)
    p0 = torch.tensor(nominal[:n_free], device=DEV).expand(B, n_free)
    z = torch.logit(((p0 - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    rest0 = np.clip(turning_angles(np.stack([c[0] for c in cables])), -rest_cap, rest_cap).astype(np.float32)
    trace = [(idxs[0], V.copy())]
    for ai, bi in zip(idxs, idxs[1:]):
        acts = np.stack([step_action(c[ai], c[bi], N - 2) for c in cables])
        p_par = torch.cat([lo + torch.sigmoid(z) * (hi - lo), torch.full((B, 1), float(nominal[-1]), device=DEV)], 1)
        rest = rest0 if rest_mode == "frame0" else np.clip(turning_angles(V), -rest_cap, rest_cap).astype(np.float32)
        pred = model(torch.tensor(V, device=DEV), torch.tensor(acts, device=DEV), p_par, torch.tensor(rest, device=DEV))
        obs = torch.tensor(np.stack([c[bi] for c in cables]).astype(np.float32), device=DEV)
        pf = pred.clone(); pf[:, 0] = obs[:, 0]; pf[:, -1] = obs[:, -1]           # ends stay plugged in
        V = pf.detach().cpu().numpy().astype(np.float32); trace.append((bi, V.copy()))   # forecast recorded first
        loss = ((pf - obs) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()   # then the online update
    return trace


def evaluate(ckpt, tracks, out, rest_mode="current", rest_cap=REST_CAP, lr=0.05, nominal=None, label=None):
    model, _ = load_predictor(ROOT / ckpt, DEV); model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    nom = NOMINAL if nominal is None else np.array([float(x) for x in nominal.split(",")], np.float32)
    res = {}
    for c in load_clips(tracks):
        cables = [c["P"][k] for k in range(4)]
        trace = rollout_online(model, cables, lr=lr, rest_mode=rest_mode, rest_cap=rest_cap, nominal=nom)
        e = [np.mean([np.linalg.norm(V[k] - cables[k][fi], axis=1).mean() for k in range(4)]) * 1000 for fi, V in trace]
        res[c["clip"]] = record(e, stride=4, rest=rest_mode)
        print(f"{c['clip']}: {np.mean(e):.1f} mm/node", flush=True)
    write_results(res, out, label or f"Caporali MLP ({ckpt}, rest {rest_mode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["data", "train", "eval", "all"])
    ap.add_argument("--data", default="paper", choices=["paper", "real"]); ap.add_argument("--tag", default=None)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--tracks", default="data/tracks14")
    ap.add_argument("--rest", default="current", choices=["current", "frame0"]); ap.add_argument("--rest-cap", type=float, default=None)
    ap.add_argument("--nominal", default=None, help="start of the online estimate, e.g. 0.04,1.0,20,0.15 (default 0.02,0.1,10,0.15)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.tag or ("" if a.data == "paper" else a.data)
    data_f = f"work/caporali_data{'_' + tag if tag else ''}.pt"
    ckpt = a.ckpt or ("checkpoints/caporali_mlp.pt" if not tag and a.step == "eval" else f"checkpoints/caporali_mlp{'_' + tag if tag else ''}.pt")
    if a.step in ("data", "all"):
        make_data(a.data, data_f)
    if a.step in ("train", "all"):
        train(data_f, ckpt)
    if a.step in ("eval", "all"):
        cap = a.rest_cap if a.rest_cap is not None else (0.6 if a.rest == "frame0" else REST_CAP)
        evaluate(ckpt, a.tracks, a.out or f"results/caporali{'_' + tag if tag else ''}_{a.rest}.json", a.rest, cap, nominal=a.nominal)


if __name__ == "__main__":
    main()
