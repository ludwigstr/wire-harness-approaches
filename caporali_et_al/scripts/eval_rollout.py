"""8-step rollout of the real step model vs the current 2 s forecast.

The single-step test showed the fine-tuned step model predicts a real 0.25 s
step to 3.5 mm/node on unseen clips, against 7.8 mm for the physics surrogate.
This asks the question that actually matters: rolled out eight times to span the
2 s, does it beat the current forecast (15.3 mm/node) -- or does the per-step
error compound away the advantage?

Honesty is enforced two ways:

  * LEAVE-ONE-CLIP-OUT. For each scene clip, a step model is trained on the
    transitions of every OTHER clip and used to roll out that clip's scenes. A
    scene is never forecast by a model that saw its clip, so a good number
    cannot be clip memorisation.
  * FORECAST conditions. The rollout is driven only by the plug motions, split
    into eight equal steps between t0 and t1 -- the linear plug path a forecaster
    would assume, not the true intermediate frames. Nothing from t1 leaks in
    except the endpoint plug poses, which the mover targets provide anyway.

Scored on exactly the cables the 15.3 mm baseline was scored on.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_model.estimation import load_predictor
from wire_model.nn_model import _turning_angles_batch
from train_step_model import build_tensors, turning_angles, node_err

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NOMINAL = np.array([0.02, 0.1, 10.0, 0.15], np.float32)


def edge_length(V):
    """Total polyline length per sample: (B, N, 2) -> (B,)."""
    return (V[:, 1:] - V[:, :-1]).norm(dim=-1).sum(-1)


def shape_loss(pred, tgt):
    """PIEGraph's shape term (Orozco et al., CoRL 2025, Eq. 3, second summand):
    match the displacement between neighbouring nodes, not just node positions.

    For a chain the neighbourhood N(r) is {r-1, r+1}, so this reduces to
    matching every edge VECTOR. That is strictly more information than the
    total-length term tried earlier tonight, which collapsed all 15 edges into
    one scalar and lost both which edge was wrong and in which direction --
    at weight 1.0 that cost 62% accuracy. Here each edge gets its own 2-D
    residual, so the gradient says where and which way.
    """
    return torch.nn.functional.mse_loss(pred[:, 1:] - pred[:, :-1],
                                        tgt[:, 1:] - tgt[:, :-1])


def arclen_residual(pred, seg_len):
    """C-PINN's global arc-length regularisation (Du et al., IEEE T-ASE 2026,
    Eq. 11): sum_i (||p_i - p_{i-1}|| - ds)^2 with ds = L0/(N-1).

    Unlike the two loss terms tried before this, it is UNSUPERVISED -- every
    segment is compared against the rod's own undeformed segment length, not
    against the target shape. That is what makes it a physics residual in the
    PINN sense rather than a second data term: it constrains predictions the
    training set never covers, which is precisely where a rollout spends most
    of its time.

    `seg_len` is (B, 1): each sample's own ds, taken from V_in's measured arc
    length, so nothing about the future enters.
    """
    seg = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1)      # (B, N-1)
    return ((seg - seg_len) ** 2).sum(-1).mean()


def exit_straight_loss(pred):
    """The wire leaves its connector straight -- a housing prior, measured.

    Both ends of every cable are plugged into a rigid housing, and the plug
    grips a short stretch of wire, so the first and last segments emerge along
    the plug axis instead of bending immediately. In the 851 tracked shapes the
    turning angle at the first interior node has a median of 0.6 deg and at the
    last 1.3 deg, against 7.1 deg across the middle -- and the sharpest bend in
    the whole cable sits at the node right NEXT to those (13-23 deg). So the
    wire really does come out straight and then turn.

    The network does not know this: on the guided-ensemble predictions the same
    two nodes come out at 14.6 and 28.1 deg. Unlike the arc-length and total-
    length residuals tried before -- which constrained a quantity the model was
    already respecting, and so could only cost accuracy -- this one has a real,
    measured violation to correct.
    """
    th = _turning_angles_batch(pred)
    return (th[:, 0] ** 2 + th[:, -1] ** 2).mean()


def pinn_weight(epoch, epochs, lam_max, warmup=0.3, rate=5.0):
    """Exponential ramp for the physics term (C-PINN Sec. IV.E).

    Their staging: train on data alone first, then raise the physics weight
    smoothly rather than switching it on at full strength. The reason given is
    that the loss components produce large, mutually conflicting gradients
    early in training -- which is the most plausible account of why this
    project's earlier length-loss experiment cost 62% accuracy at weight 1.0
    while being harmless at 0.1: not that the constraint was wrong, but that
    it was applied at full weight from step 0, against a network that had not
    yet learned the dynamics it was supposed to constrain.

    Returns 0 through the warmup fraction, then 1-exp(-rate*progress) scaled
    to lam_max.
    """
    if epochs <= 1:
        return lam_max
    t = epoch / (epochs - 1)
    if t <= warmup:
        return 0.0
    p = (t - warmup) / max(1e-9, 1.0 - warmup)
    return lam_max * (1.0 - np.exp(-rate * p)) / (1.0 - np.exp(-rate))


def train_fold(trans, hold_clip, ckpt, epochs, lr, len_weight=0.0,
               shape_weight=0.0, noise_mm=0.0, pinn_lambda=0.0,
               pinn_warmup=0.3, exit_lambda=0.0, nominal=None):
    """len_weight > 0 adds a length-consistency term: the real wire does not
    stretch or shrink between two consecutive tracked frames, so the
    predicted V_out is penalised for a total edge length that drifts from
    V_in's own (self-consistent, no extra stored constant needed). At perfect
    position-MSE this term is redundant -- it only earns its keep as a
    regulariser against overfitting the ~19-clip real dataset, the same
    concern that sank the explicit parameter-fit and history-conditioning
    experiments earlier."""
    tr = [t for t in trans if t["clip"] != hold_clip]
    # The real transitions carry no fitted parameters, so every sample gets one
    # nominal row -- which has to be as wide as the checkpoint expects. The
    # per-joint checkpoints (17 columns) pass their own; the default stays the
    # 4-wide row every earlier result was produced with.
    Vi, Vo, A, P, R = build_tensors(tr, NOMINAL if nominal is None else nominal)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(Vi, Vo, A, P, R), batch_size=128,
                        shuffle=True, drop_last=True)
    model, _ = load_predictor(ROOT / ckpt, DEV)
    if epochs == 0:
        # a checkpoint that is ALREADY fine-tuned (saved once, reused for
        # every clip of a set): nothing to train, OneCycleLR would reject
        # total_steps=0
        model.eval()
        return model
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    for ep in range(epochs):
        model.train()
        lam = pinn_weight(ep, epochs, pinn_lambda, pinn_warmup) if pinn_lambda else 0.0
        for vi, vo, a, p, r in loader:
            vi, vo = vi.to(DEV), vo.to(DEV)
            a, p, r = a.to(DEV), p.to(DEV), r.to(DEV)
            if noise_mm:
                # PhysWorld (Yang et al., arXiv 2510.21447) Sec. 3.3: a model
                # trained only on ground-truth short-term pairs is fed its OWN
                # drifted predictions at rollout time, so it is evaluated off
                # the distribution it was fitted on -- and our rollouts run
                # 19-38 s. Perturbing V_in (and re-deriving the rest angles
                # from the perturbed shape, exactly as the rollout derives
                # them from the drifted one) moves training onto that
                # distribution and additionally teaches error CORRECTION,
                # since the target stays the true V_out.
                vi = vi + torch.randn_like(vi) * (noise_mm / 1000.0)
                r = _turning_angles_batch(vi).clamp(-0.35, 0.35)
            pred = model(vi, a, p, r)
            loss = torch.nn.functional.mse_loss(pred, vo)
            if len_weight:
                loss = loss + len_weight * torch.nn.functional.mse_loss(
                    edge_length(pred), edge_length(vi))
            if shape_weight:
                loss = loss + shape_weight * shape_loss(pred, vo)
            if exit_lambda:
                loss = loss + exit_lambda * exit_straight_loss(pred)
            if lam:
                # ds from V_in's own arc length: causal, per sample, and it
                # tracks each cable's true length instead of assuming one
                # shared constant across four physically different wires.
                ds = (edge_length(vi) / (vi.shape[1] - 1)).unsqueeze(-1)
                loss = loss + lam * arclen_residual(pred, ds)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def build_sequences(trans, hold_clip, tau):
    """Chain single-step transitions into tau-step sequences.

    Transitions carry (clip, k, frame) and the 1 s stride is 4 frames, so a
    sequence exists wherever frames f, f+4, ... f+4(tau-1) are all present for
    the same cable. Chains never cross a clip or cable boundary.
    """
    import collections
    by = collections.defaultdict(dict)
    for t in trans:
        if t["clip"] == hold_clip:
            continue
        by[(t["clip"], t["k"])][t["frame"]] = t

    Vi, A, Vo = [], [], []
    for seq in by.values():
        for f in sorted(seq):
            steps = [seq.get(f + 4 * i) for i in range(tau)]
            if any(s is None for s in steps):
                continue
            Vi.append(steps[0]["V_in"])
            A.append([s["action"] for s in steps])
            Vo.append([s["V_out"] for s in steps])
    if not Vi:
        raise ValueError(f"no tau={tau} sequences outside clip {hold_clip}")
    return (torch.tensor(np.stack(Vi), dtype=torch.float32),
            torch.tensor(np.stack(A), dtype=torch.float32),
            torch.tensor(np.stack(Vo), dtype=torch.float32))


def train_fold_seq(trans, hold_clip, ckpt, epochs, lr, tau=3, noise_mm=0.0):
    """Train on a tau-step rollout instead of a single step.

    PhysWorld (Yang et al., arXiv 2510.21447) Eq. 8 sums the loss over a
    look-forward horizon rather than one step, which optimises the quantity we
    actually report -- rollout error -- instead of a proxy that ignores how
    errors compound. This mirrors the evaluation rollout exactly: the two
    driven ends are re-pinned to their real pose after every step (as
    long_horizon_all.roll_traj does), and the rest angles are re-derived from
    the model's OWN predicted shape, so the network is trained on the states
    it will really be fed.
    """
    Vi, A, Vo = build_sequences(trans, hold_clip, tau)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(Vi, A, Vo), batch_size=64,
                        shuffle=True, drop_last=True)
    model, _ = load_predictor(ROOT / ckpt, DEV)
    for p_ in model.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    for _ in range(epochs):
        model.train()
        for vi, a, vo in loader:
            vi, a, vo = vi.to(DEV), a.to(DEV), vo.to(DEV)
            p_par = torch.tensor(NOMINAL, device=DEV).expand(vi.shape[0], -1)
            V = vi
            if noise_mm:
                V = V + torch.randn_like(V) * (noise_mm / 1000.0)
            loss = 0.0
            for i in range(tau):
                r = _turning_angles_batch(V).clamp(-0.35, 0.35)
                pred = model(V, a[:, i], p_par, r)
                loss = loss + torch.nn.functional.mse_loss(pred, vo[:, i])
                # re-pin the driven ends, exactly as the rollout does
                V = torch.cat([vo[:, i, :1], pred[:, 1:-1], vo[:, i, -1:]], 1)
            opt.zero_grad(set_to_none=True)
            (loss / tau).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def step_action(s0, real, k, n_steps):
    """Per-step 8-vector: total plug motion (both ends) split into n_steps.

    The plug path is assumed linear between t0 and t1 -- a forecaster knows the
    endpoints, not the intermediate frames -- so each step applies the same
    fraction of the total edge translation and rotation.
    """
    def edge(V, at_end):
        a, b = (V[-2], V[-1]) if at_end else (V[0], V[1])
        return 0.5 * (a + b), np.arctan2(b[1] - a[1], b[0] - a[0])
    out = []
    for at_end, alpha in ((True, k), (False, 0.0)):
        m0, a0 = edge(s0, at_end)
        m1, a1 = edge(real, at_end)
        dth = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
        out += [alpha, (m1[0] - m0[0]) / n_steps, (m1[1] - m0[1]) / n_steps,
                dth / n_steps]
    return np.array(out, np.float32)


def rollout(model, s0, real, n_nodes, n_steps, rest_cap=0.35):
    V = s0.copy().astype(np.float32)
    a = step_action(s0, real, n_nodes - 2, n_steps)
    p = NOMINAL[None]
    for _ in range(n_steps):
        r = np.clip(turning_angles(V), -rest_cap, rest_cap)[None].astype(np.float32)
        with torch.no_grad():
            V = model(torch.tensor(V[None], device=DEV),
                      torch.tensor(a[None], device=DEV),
                      torch.tensor(p, device=DEV),
                      torch.tensor(r, device=DEV))[0].cpu().numpy()
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_all.pt")
    ap.add_argument("--scenes", default="data/e2e_twoend/neue_szenen.pt")
    ap.add_argument("--ckpt", default="data/nn_twoend_transformer.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    trans = torch.load(ROOT / args.transitions, weights_only=False)["transitions"]
    recs = torch.load(ROOT / args.scenes, weights_only=False)["recs"]
    by_clip = {}
    for r in recs:
        by_clip.setdefault(r["clip"], []).append(r)
    n_nodes = recs[0]["s0"].shape[0]

    roll_err, base_err = [], []
    for ci, (clip, rs) in enumerate(sorted(by_clip.items())):
        model = train_fold(trans, clip, args.ckpt, args.epochs, args.lr)
        for r in rs:
            pred = rollout(model, r["s0"], r["real"], n_nodes, args.steps)
            roll_err.append(1000 * np.linalg.norm(pred - r["real"],
                                                  axis=1).mean())
            base_err.append(r["err_causal"] if r["fitted"] else r["err_single"])
        re = np.array(roll_err); print(
            f"  [{ci+1}/{len(by_clip)}] {clip}: bisher Median Rollout "
            f"{np.median(re):.1f} mm", flush=True)

    re = np.array(roll_err); be = np.array(base_err, float)
    ok = np.isfinite(re) & np.isfinite(be)
    re, be = re[ok], be[ok]
    print(f"\n{len(re)} Kabel (leave-one-clip-out)\n")
    print(f"{'':<34}{'Median':>9}{'Mittel':>9}{'p75':>8}")
    print(f"{'aktuell (Physik-Prognose)':<34}{np.median(be):>9.1f}"
          f"{be.mean():>9.1f}{np.percentile(be,75):>8.1f}")
    print(f"{'Rollout (reales Schrittmodell)':<34}{np.median(re):>9.1f}"
          f"{re.mean():>9.1f}{np.percentile(re,75):>8.1f}")
    print(f"\nAenderung Median {100*(np.median(re)-np.median(be))/np.median(be):+.1f}%"
          f"   besser bei {int((re<be).sum())}/{len(re)} Kabeln")


if __name__ == "__main__":
    main()
