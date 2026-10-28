"""Benchmark surrogate architectures against the paper's MLP, like for like.

Everything except the architecture is held fixed: the same data, the same
train/val split (seeded identically), the same optimiser, schedule, epoch
count and loss. Only then does a difference in validation error mean anything
about the architecture rather than about the training budget.

Reported per model:
  D            the paper's Sec. II-C metric, summed over nodes
  per-node     D / n_nodes -- comparable to the mm/node figures used elsewhere
  % baseline   against "predict no change", which is the bar any surrogate
               must clear to have learned anything at all
  params       capacity, because a win bought purely with more weights is not
               an architectural win
  train / inf  wall time; the surrogate exists to be fast and differentiable,
               so a model that is better but far slower may still be the wrong
               trade for the estimator loop that calls it hundreds of times.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wire_model.nn_model import DLONetwork, DLOPredictor
from wire_model.nn_models_alt import ARCHITECTURES
from train_nn import build_normalizer, paper_metric

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(name, n_nodes, width, n_rest, n_params=4):
    if name in ("film", "concat"):
        return DLONetwork(n_nodes=n_nodes, width=width, n_rest=n_rest,
                          n_params=n_params,
                          combine=name)
    return ARCHITECTURES[name](n_nodes=n_nodes, width=width, n_rest=n_rest,
                               n_params=n_params)


def run(name, data, args):
    V_in, V_out, action, params, rest, norm, n_nodes = data
    g = torch.Generator().manual_seed(0)          # identical split every model
    perm = torch.randperm(V_in.shape[0], generator=g)
    n_val = int(0.1 * V_in.shape[0])
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    torch.manual_seed(0)
    net = build(name, n_nodes, args.width, n_nodes - 2, params.shape[1])
    model = DLOPredictor(net, norm).to(DEV)
    n_par = sum(x.numel() for x in model.parameters())

    tr = DataLoader(TensorDataset(V_in[tr_idx], V_out[tr_idx], action[tr_idx],
                                  params[tr_idx], rest[tr_idx]),
                    batch_size=args.batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    va = DataLoader(TensorDataset(V_in[val_idx], V_out[val_idx],
                                  action[val_idx], params[val_idx],
                                  rest[val_idx]),
                    batch_size=4096, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1)

    with torch.no_grad():
        base = torch.cat([paper_metric(b[0], b[1]) for b in va]).mean().item()

    best, t_start = float("inf"), time.time()
    for ep in range(args.epochs):
        model.train()
        for vi, vo, a, p, r in tr:
            vi, vo = vi.to(DEV, non_blocking=True), vo.to(DEV, non_blocking=True)
            a, p, r = a.to(DEV), p.to(DEV), r.to(DEV)
            loss = torch.nn.functional.mse_loss(model(vi, a, p, r), vo)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        errs = []
        with torch.no_grad():
            for vi, vo, a, p, r in va:
                pred = model(vi.to(DEV), a.to(DEV), p.to(DEV), r.to(DEV))
                errs.append(paper_metric(pred, vo.to(DEV)).cpu())
        D = torch.cat(errs).mean().item()
        if D < best:
            best = D
            if args.save:
                torch.save({"state_dict": model.net.state_dict(),
                            "norm": {k: v for k, v in norm.__dict__.items()},
                            "n_nodes": n_nodes, "width": args.width,
                            "n_rest": n_nodes - 2, "arch": name,
                            # recorded explicitly: inferring them from weight
                            # shapes needs one of the two to be known, and both
                            # have now changed at least once
                            "n_action": action.shape[1],
                            "n_params": params.shape[1]},
                           ROOT / (args.save_as or f"data/bench_{name}.pt"))
        print(f"    ep {ep+1:>2}/{args.epochs}  val per-node "
              f"{1000*D/n_nodes:7.3f} mm", flush=True)
    train_s = time.time() - t_start

    # inference timing: one estimator step's worth of forward passes
    vi, vo, a, p, r = next(iter(va))
    vi, a, p, r = vi.to(DEV), a.to(DEV), p.to(DEV), r.to(DEV)
    with torch.no_grad():
        for _ in range(3):
            model(vi, a, p, r)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            model(vi, a, p, r)
        if DEV == "cuda":
            torch.cuda.synchronize()
    inf_ms = 1000 * (time.time() - t0) / 20 / vi.shape[0] * 1000   # us per sample

    return {"arch": name, "D_mm": 1000 * best,
            "per_node_mm": 1000 * best / n_nodes,
            "pct_baseline": 100 * best / base, "params": n_par,
            "train_s": train_s, "inf_us_per_sample": inf_ms,
            "baseline_mm": 1000 * base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dlo_dataset_rest.pt")
    ap.add_argument("--models", default="film,bilstm,conv1d,transformer")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save-as", default=None, help="checkpoint path (default data/bench_<arch>.pt, which other rows read)")
    ap.add_argument("--out", default="data/bench_results.json")
    args = ap.parse_args()

    d = torch.load(ROOT / args.data, weights_only=False)
    n_nodes = d["V_in"].shape[1]
    # everything but the trailing length column; k_b is N-2 columns wide now
    data = (d["V_in"], d["V_out"], d["action"], d["params"][:, :-1],
            d["rest_angle"], build_normalizer(d, n_nodes).to(DEV), n_nodes)
    print(f"device={DEV}  n={d['V_in'].shape[0]}  nodes={n_nodes}  "
          f"epochs={args.epochs}  width={args.width}\n")

    results = []
    for name in args.models.split(","):
        print(f"--- {name} ---", flush=True)
        results.append(run(name, data, args))
        r = results[-1]
        print(f"    -> per-node {r['per_node_mm']:.3f} mm  "
              f"({r['pct_baseline']:.1f}% of baseline)  "
              f"{r['params']/1e6:.2f}M params  {r['train_s']:.0f}s\n", flush=True)

    results.sort(key=lambda r: r["per_node_mm"])
    print(f"{'Modell':<13}{'mm/Knoten':>11}{'% Baseline':>12}{'Params':>10}"
          f"{'Training':>10}{'Inferenz':>12}")
    for r in results:
        print(f"{r['arch']:<13}{r['per_node_mm']:>11.3f}"
              f"{r['pct_baseline']:>11.1f}%{r['params']/1e6:>9.2f}M"
              f"{r['train_s']:>9.0f}s{r['inf_us_per_sample']:>10.1f}us")
    (ROOT / args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
