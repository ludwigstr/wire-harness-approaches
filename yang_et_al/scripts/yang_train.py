"""Yang et al. (RAS 2022) step 3: train the IN-biLSTM on the synthetic data ONLY.

Paper Sec. 4.2.2: Adam, fixed learning rate 1e-4, batch 512, 90/10
train/validation split, one-step prediction with an L2 loss on the particle
state, inputs augmented with additive white Gaussian noise; f_e, the state
encoder and the effect predictor are 4-layer MLPs with 150 hidden units,
ReLU except the effect predictor's output; 2-layer bi-LSTM, 150 units.
Not stated in the paper and chosen here: the split is by ROLLOUT (no
transition of a validation rollout is trained on), the noise level (1 mm on
the input positions, the target corrected to the noisy input, so the net
learns to pull a perturbed state back), the number of epochs (the
checkpoint with the lowest validation loss is kept).

One transition = one cable, one video frame (0.25 s):
    inputs  p_t, v_t = (p_t - p_{t-1}) / dt (0 at the rollout's first frame),
            end actions (p_t -> p_{t+1} of the two ends), rest turning angles
    target  dp = p_{t+1} - p_t (all 16 nodes)
Transitions whose connector gap (tip-to-cavity, either frame) exceeds
--max-gap are dropped (over-stretched by the motion scaling).

    CUDA_VISIBLE_DEVICES=1 python scripts/yang_train.py --data data/wm/yang_synth --out data/wm/yang_inbilstm.pt
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.yang.model import INBiLSTM, end_actions, turning  # noqa: E402

DT = 0.25


def load_transitions(data_dir, max_gap, gap_ref_tol=None):
    """gap_ref_tol (user, 2026-10-27): tolerant filter -- a transition of rollout
    <clip>_<NN>.npz is dropped only where the gap (MuJoCo: connector offset
    change; Newton: 16-node arc-length change) exceeds the largest gap of its
    clip's REAL-TIMING rollout <clip>_00.npz plus gap_ref_tol; the threshold is
    never below max_gap"""
    import re
    files = sorted(glob.glob(str(ROOT / data_dir / "*.npz")))
    ref = {}
    if gap_ref_tol is not None:
        for f in files:
            mt = re.match(r"(.*)_(\d{2,3})$", Path(f).stem)
            if mt and int(mt.group(2)) == 0:
                ref[mt.group(1)] = float(np.load(f)["gap"].max())
    rec = []
    n_drop = n_all = 0
    for rid, f in enumerate(files):
        z = np.load(f)
        V, gap, rest = z["V"], z["gap"], z["rest"]                    # (T, 4, 16, 2), (T,), (4, 16, 2)
        T = len(V)
        if T < 2:
            continue
        thr = max_gap
        if gap_ref_tol is not None:
            r = ref.get(re.sub(r"_\d{2,3}$", "", Path(f).stem))
            if r is not None:
                thr = max(max_gap, r + gap_ref_tol)
        ok = (gap[:-1] <= thr) & (gap[1:] <= thr)
        n_all += 4 * (T - 1); n_drop += 4 * int((~ok).sum())
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        prev = np.concatenate([V[:1], V[:-1]])                           # p_{t-1}, = p_0 at t = 0
        for k in range(4):
            rec.append(dict(prev=prev[idx, k], cur=V[idx, k], nxt=V[idx + 1, k],
                            rest=np.repeat(rest[k][None], len(idx), 0), rid=np.full(len(idx), rid)))
    cat = {key: np.concatenate([r[key] for r in rec]) for key in rec[0]}
    print(f"{len(files)} rollouts -> {len(cat['cur'])} transitions ({100 * n_drop / max(n_all, 1):.1f}% dropped, gap > "
          f"{1000 * max_gap:.0f} mm" + ("" if gap_ref_tol is None else
                                       f" or real-timing reference + {1000 * gap_ref_tol:.0f} mm, whichever is larger") + ")",
          flush=True)
    return cat, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wm/yang_synth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.001, help="std of the input-position noise, m")
    ap.add_argument("--max-gap", type=float, default=0.015)
    ap.add_argument("--no-rest", action="store_true", help="ablation: no rest-shape input (the paper's inputs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap-ref-tol", type=float, default=None,
                    help="drop a transition only where its gap exceeds its clip's real-timing rollout (<clip>_00) by "
                         "more than this (m), never below --max-gap; default: the fixed --max-gap cut")
    ap.add_argument("--val-by-clip", action="store_true",
                    help="hold out whole SOURCE clips (all rollouts <clip>_<NN>.npz of a clip on the same side) "
                         "instead of random rollouts; default unchanged")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = a.device
    D, files = load_transitions(a.data, a.max_gap, a.gap_ref_tol)
    n_roll = len(files)
    if a.val_by_clip:
        import re
        src = [re.sub(r"_\d{2,3}$", "", Path(f).stem) for f in files]
        cl = sorted(set(src))
        vcl = {cl[i] for i in rng.permutation(len(cl))[: max(1, len(cl) // 10)]}
        val_r = {i for i, c in enumerate(src) if c in vcl}
        print(f"clip-level split: {len(cl) - len(vcl)} train clips / {len(vcl)} val clips", flush=True)
    else:
        val_r = set(rng.permutation(n_roll)[: max(1, n_roll // 10)].tolist())
    val_files = sorted(Path(files[i]).name for i in val_r)
    is_val = np.array([r in val_r for r in D["rid"]])
    T = {k: torch.tensor(v, dtype=torch.float32) for k, v in D.items() if k != "rid"}
    T["restang"] = torch.zeros(len(T["cur"]), 16)
    T["restang"][:, 1:-1] = turning(T["rest"])
    T["act"] = end_actions(T["cur"], T["nxt"])
    T["vel"] = (T["cur"] - T["prev"]) / DT
    T["dp"] = T["nxt"] - T["cur"]
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    print(f"train {len(tr)}, val {len(va)} transitions ({len(val_r)} of {n_roll} rollouts held out)", flush=True)

    net = INBiLSTM(use_rest=not a.no_rest).to(dev)
    with torch.no_grad():                                               # normalisation from the training split
        sub = torch.tensor(tr[: 200000])
        net.s_pos.fill_(float((T["cur"][sub, 1:] - T["cur"][sub, :-1]).std()))
        net.s_vel.fill_(float(T["vel"][sub].std()) + 1e-6)
        act = T["act"][sub][:, [0, -1]].reshape(-1, 3)
        net.s_act.copy_(act.std(0) + 1e-6)
        net.s_out.fill_(float(T["dp"][sub].std()))
    print(f"scales: pos {net.s_pos.item():.4f} m, vel {net.s_vel.item():.4f} m/s, act {net.s_act.tolist()}, out {net.s_out.item():.4f} m", flush=True)
    for k in T:
        T[k] = T[k].to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def batch_loss(ix, train):
        p, prev, nxt = T["cur"][ix], T["prev"][ix], T["nxt"][ix]
        if train and a.noise > 0:                       # additive white Gaussian noise on the input state
            p = p + a.noise * torch.randn_like(p)
            prev = prev + a.noise * torch.randn_like(prev)
        v = (p - prev) / DT
        act = end_actions(p, nxt)
        pred = net(p, v, act, T["restang"][ix])
        err = (pred - (nxt - p)) / net.s_out
        return (err ** 2).mean(), (pred - (nxt - p)).norm(dim=-1).mean()

    log = []
    best = np.inf
    t0 = time.time()
    tr_t = torch.tensor(tr, device=dev); va_t = torch.tensor(va, device=dev)
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        tl = []
        for b0 in range(0, len(perm), a.batch):
            loss, _ = batch_loss(perm[b0:b0 + a.batch], True)
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item())
        net.eval()
        with torch.no_grad():
            vl, vm = [], []
            for b0 in range(0, len(va_t), 4096):
                l_, m_ = batch_loss(va_t[b0:b0 + 4096], False)
                vl.append(l_.item()); vm.append(m_.item())
        rec = {"epoch": ep, "train": float(np.mean(tl)), "val": float(np.mean(vl)), "val_mm": 1000 * float(np.mean(vm)),
               "sec": time.time() - t0}
        log.append(rec)
        tag = ""
        if rec["val"] < best:
            best = rec["val"]
            torch.save({"state_dict": net.state_dict(), "use_rest": not a.no_rest, "epoch": ep, "val": rec,
                        "args": vars(a), "dt": DT, "val_files": val_files}, out)
            tag = "  *"
        print(f"ep {ep}: train {rec['train']:.4f}  val {rec['val']:.4f}  val one-step {rec['val_mm']:.3f} mm/node  ({rec['sec']:.0f}s){tag}", flush=True)
    out.with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    print(f"best val {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
