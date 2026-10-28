"""Yang et al. IN-biLSTM on the 14 clips -- the world-model rows' protocol.

Per cable: start from the measured frame 0 (velocity 0), roll the net out
open loop with the measured end motions as actions, both ends pinned to the
measured positions after every step, the net's own predictions as the next
input. Score = mean node distance in mm (16 nodes), mean over the 4 cables,
per frame; clip value = mean over frames.
  "mittel"        every predicted frame 1..T-1 (the headline number)
  "mittel_incl0"  frames 0..T-1 with frame 0 = the given state (error 0)
  "mittel_warm5"  AdaptiGraph/RopeDreamer protocol: frames 0-4 given, open
                  loop from frame 4, scored on frames 5..T-1
Optionally (--synth) the same rollout on the held-out synthetic rollouts
(paper Table 6 "source/source"), against the simulator's own trajectories.

    CUDA_VISIBLE_DEVICES=1 python scripts/yang_eval.py --ckpt data/wm/yang_inbilstm.pt --out data/wm/yang_inbilstm_14.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from wire_model.yang.model import INBiLSTM, end_actions, turning  # noqa: E402
from clip_set import CLIP_TRIMS  # noqa: E402

TRACKS = Path("./data/tracks14")


@torch.no_grad()
def rollout(net, P, dt, warm=1, rest_shape=None):
    """P (C, T, 16, 2) measured; frames < warm are given. Returns the predicted (C, T, 16, 2)."""
    C, T = P.shape[:2]
    rest = torch.zeros(C, 16, device=P.device); rest[:, 1:-1] = turning(P[:, 0] if rest_shape is None else rest_shape)
    out = [P[:, t] for t in range(warm)]
    for t in range(warm - 1, T - 1):
        p = out[t]; prev = out[t - 1] if t > 0 else out[0]
        v = (p - prev) / dt
        act = end_actions(P[:, t], P[:, t + 1])           # the MEASURED end motion
        nxt = p + net(p, v, act, rest)
        nxt[:, 0] = P[:, t + 1, 0]; nxt[:, -1] = P[:, t + 1, -1]
        out.append(nxt)
    return torch.stack(out, 1)


def frame_err(pred, P):
    return 1000 * (pred - P).norm(dim=-1).mean(-1).mean(0).cpu().numpy()      # (T,) mean over nodes, cables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth", default="", help="synthetic data dir: also score the held-out synthetic rollouts")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    net = INBiLSTM(use_rest=ck["use_rest"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    dt = ck["dt"]
    res = {}
    for clip in sorted(CLIP_TRIMS):
        z = np.load(TRACKS / f"{clip}.npz")
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        assert abs(1.0 / float(z["fps"]) - dt) < 1e-6
        e = frame_err(rollout(net, P, dt), P)
        ew = frame_err(rollout(net, P, dt, warm=5), P)
        res[clip] = {"start": float(e[1]), "ende": float(e[-1]), "mittel": float(e[1:].mean()),
                     "mittel_incl0": float(e.mean()), "mittel_warm5": float(ew[5:].mean()),
                     "n_frames": int(len(e)), "per_frame": e.round(3).tolist()}
        print(f"{clip}: {res[clip]['mittel']:.2f} mm/node (incl. frame 0 {res[clip]['mittel_incl0']:.2f}, "
              f"5-frame warm-up {res[clip]['mittel_warm5']:.2f}), {len(e)} frames", flush=True)
    summ = {}
    for key in ("mittel", "mittel_incl0", "mittel_warm5"):
        v = np.array([r[key] for r in res.values()])
        summ[key] = {"mean": float(v.mean()), "median": float(np.median(v))}
    report = {"ckpt": a.ckpt, "clips": res, "summary": summ, "n_clips": len(res),
              "protocol": "frame 0 measured, v0 = 0, open loop, measured end motions, ends pinned; "
                          "mittel = mean over predicted frames 1..T-1"}
    if a.synth:
        files = sorted(glob.glob(str(ROOT / a.synth / "*.npz")))
        rng = np.random.default_rng(ck["args"]["seed"])
        if "val_files" in ck:          # the split the training used (clip-level or not)
            names = set(ck["val_files"])
            val = [i for i, f in enumerate(files) if Path(f).name in names]
        else:
            val = sorted(rng.permutation(len(files))[: max(1, len(files) // 10)].tolist())
        sv = []
        for i in val:
            zz = np.load(files[i])
            V = zz["V"]
            thr = ck["args"]["max_gap"]
            if ck["args"].get("gap_ref_tol") is not None:      # the training's tolerant cut
                import re
                r0 = Path(re.sub(r"_\d{2}$", "", files[i][:-4]) + "_00.npz")
                if r0.exists():
                    thr = max(thr, float(np.load(r0)["gap"].max()) + ck["args"]["gap_ref_tol"])
            ok = np.nonzero(zz["gap"] > thr)[0]
            T = int(ok[0]) if len(ok) else len(V)                 # stop at the first over-stretched frame
            if T < 3:
                continue
            Pv = torch.tensor(V[:T].transpose(1, 0, 2, 3), dtype=torch.float32, device=dev)
            R = torch.tensor(zz["rest"], dtype=torch.float32, device=dev)
            sv.append(float(frame_err(rollout(net, Pv, dt, rest_shape=R), Pv)[1:].mean()))
        report["synthetic_val_rollouts"] = {"n": len(sv), "mean": float(np.mean(sv)), "median": float(np.median(sv))}
        print(f"held-out synthetic rollouts (source/source): {len(sv)}, mean {np.mean(sv):.2f}, median {np.median(sv):.2f} mm/node")
    out.write_text(json.dumps(report, indent=1))
    for key, s in summ.items():
        print(f"{key}: mean {s['mean']:.2f}  median {s['median']:.2f} mm/node over {len(res)} clips")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
