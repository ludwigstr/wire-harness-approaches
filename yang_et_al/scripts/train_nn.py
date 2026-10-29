"""Train the NN surrogate on the analytical model's data (Sec. II-B)."""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wire_model.nn_model import DLONetwork, DLOPredictor, Normalizer

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_normalizer(d, n_nodes):
    """Ranges taken from the data itself, so nothing silently clips."""
    a, p = d["action"], d["params"]
    # Ranges per action column, whatever the width: 4 for a single grasped end,
    # 8 once the hub is driven as well. Taken from the data so nothing clips.
    lo = a.min(0).values.clone()
    hi = a.max(0).values.clone()
    for c in range(0, a.shape[1], 4):        # the alpha columns are indices
        lo[c], hi[c] = 0.0, float(n_nodes - 2)
    # Column layout is [m, k_b x (N-2), k_d, k_c, length]; everything but the
    # trailing length goes to the net, since V_in's arc length already carries
    # the length and pos_scale is fixed rather than per sample.
    n_in = p.shape[1] - 1
    n_kb = n_in - 3
    # k_b spans two decades and needs log scaling, in every one of its columns;
    # k_c must NOT, because it is sampled down to exactly 0 (the paper's
    # frictionless case) and log of zero is undefined.
    log_mask = torch.zeros(n_in, dtype=torch.bool)
    log_mask[1:1 + n_kb] = True
    return Normalizer(
        action_lo=lo,
        action_hi=hi,
        param_lo=p[:, :n_in].min(0).values.clone(),
        param_hi=p[:, :n_in].max(0).values.clone(),
        param_log=log_mask,
        pos_scale=0.3,
    )


def paper_metric(a, b):
    """D(V1, V2) = sum_i ||V1_i - V2_i||, the paper's Sec. II-C loss."""
    return (a - b).norm(dim=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dlo_dataset.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--combine", default="film", choices=["film", "concat"])
    ap.add_argument("--out", default="data/dlo_nn.pt")
    args = ap.parse_args()

    d = torch.load(ROOT / args.data, weights_only=False)
    V_in, V_out = d["V_in"], d["V_out"]
    # everything but the trailing length column; k_b now occupies N-2 of them
    action, params = d["action"], d["params"][:, :-1]
    rest = d["rest_angle"]
    n_nodes = V_in.shape[1]
    n = V_in.shape[0]

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(0.1 * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    norm = build_normalizer(d, n_nodes).to(DEV)
    net = DLONetwork(n_nodes=n_nodes, width=args.width, n_rest=n_nodes - 2,
                     n_params=params.shape[1], combine=args.combine)
    model = DLOPredictor(net, norm).to(DEV)
    n_par = sum(x.numel() for x in model.parameters())
    print(f"device={DEV}  train={len(tr_idx)}  val={len(val_idx)}  "
          f"params={n_par/1e6:.2f}M  combine={args.combine}")

    tr = DataLoader(TensorDataset(V_in[tr_idx], V_out[tr_idx], action[tr_idx],
                                  params[tr_idx], rest[tr_idx]),
                    batch_size=args.batch, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    va = DataLoader(TensorDataset(V_in[val_idx], V_out[val_idx], action[val_idx],
                                  params[val_idx], rest[val_idx]),
                    batch_size=4096, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1)

    # A do-nothing predictor is the bar to beat: it is what "V_out = V_in"
    # scores, i.e. the size of the deformation itself. A model that cannot beat
    # it has learned nothing about the dynamics.
    with torch.no_grad():
        base = torch.cat([paper_metric(b[0], b[1]) for b in va]).mean().item()
    print(f"baseline (predict no change): D = {base*1000:.2f} mm "
          f"(sum over {n_nodes} nodes)\n")

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for vi, vo, a, p, r in tr:
            vi, vo = vi.to(DEV, non_blocking=True), vo.to(DEV, non_blocking=True)
            a, p = a.to(DEV, non_blocking=True), p.to(DEV, non_blocking=True)
            r = r.to(DEV, non_blocking=True)
            loss = torch.nn.functional.mse_loss(model(vi, a, p, r), vo)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * vi.shape[0]
            cnt += vi.shape[0]

        model.eval()
        errs, mses = [], []
        with torch.no_grad():
            for vi, vo, a, p, r in va:
                vi, vo = vi.to(DEV), vo.to(DEV)
                pred = model(vi, a.to(DEV), p.to(DEV), r.to(DEV))
                errs.append(paper_metric(pred, vo).cpu())
                mses.append(torch.nn.functional.mse_loss(
                    pred, vo, reduction="none").mean((1, 2)).cpu())
        D = torch.cat(errs).mean().item()
        node_err = torch.cat(errs).mean().item() / n_nodes

        if D < best:
            best = D
            torch.save({"state_dict": model.net.state_dict(),
                        "norm": {k: v for k, v in norm.__dict__.items()},
                        "n_nodes": n_nodes, "width": args.width,
                        "n_rest": n_nodes - 2, "combine": args.combine,
                        "config": d.get("config")}, ROOT / args.out)
        print(f"ep {ep+1:>3}/{args.epochs}  train_mse {tot/cnt:.3e}  "
              f"val D {D*1000:>7.3f} mm  per-node {node_err*1000:>6.3f} mm  "
              f"({100*D/base:>5.1f}% of baseline)  {time.time()-t0:.0f}s")

    print(f"\nbest val D = {best*1000:.3f} mm  ({100*best/base:.1f}% of baseline)")
    print(f"saved {ROOT / args.out}")


if __name__ == "__main__":
    main()
