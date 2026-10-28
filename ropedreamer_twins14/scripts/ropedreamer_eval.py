"""Evaluate a trained RopeDreamer on a clip set with the paper's protocol
(Sec. V-B): warm up on the first `--warm` frames with ground-truth states and
actions, then predict open loop with the measured end motions only. Reports
per clip the frame-mean node error in mm (the table metric, mean over the 4
cables, frames after the warm-up) in the X-Planar JSON format, plus the
paper's RMSE-vs-step curve and the representation floor.

    python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_S.pt \
        --tracks X-Planar_vids/mujoco_videos/tracks --out X-Planar_vids/ropedreamer_S_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from wire_model.ropedreamer.model import RopeDreamer  # noqa: E402
from wire_model.ropedreamer.data import load_tracks, trajectory_tensors  # noqa: E402
from wire_model.ropedreamer.kinematics import chain_to_positions  # noqa: E402
import wire_model.ropedreamer.data as RD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pin-ends", action="store_true", help="score the raw chain; default overwrites both end nodes with the measurement like every other row")
    a = ap.parse_args()
    ck = torch.load(ROOT / a.ckpt, map_location=a.device, weights_only=False)
    RD.REPR = ck.get("repr", "chain")
    if ck.get("model", "ropedreamer") == "deformnet":
        from wire_model.deformnet import DeformNet
        model = DeformNet(ck["n_nodes"], beta=ck["beta"], pos_weight=ck.get("pos_weight", 0.0), free_nats=ck.get("free_nats", 0.0),
                          rollout_weight=ck.get("rollout_weight", 0.0), warm=ck.get("warm", 5)).to(a.device)
    else:
        model = RopeDreamer(ck["n_nodes"], ck["size"], beta=ck["beta"], pos_weight=ck.get("pos_weight", 0.0), free_nats=ck.get("free_nats", 0.0),
                            rollout_weight=ck.get("rollout_weight", 0.0), warm=ck.get("warm", 5), repr=ck.get("repr", "chain")).to(a.device)
    # load for BOTH branches (2026-10-23: this sat on the RopeDreamer line only, so DeformNet was scored untrained)
    model.load_state_dict(ck["state_dict"]); model.eval()
    off = ck["offset"].to(a.device)
    trajs = load_tracks([ROOT / a.tracks])
    per_clip = {}
    curves, floors = [], []
    by_clip = {}
    for t in trajs:
        by_clip.setdefault(t["clip"], []).append(t)
    for clip, cabs in by_clip.items():
        errs_c, n_frames = [], None
        for t in sorted(cabs, key=lambda x: x["cable"]):
            d = trajectory_tensors(t["P"]); T = d["s"].shape[0]
            s = (d["s"][None].to(a.device) - off); idx = d["idx"][None].to(a.device); dp = d["dp"][None].to(a.device)
            h, z = model.warmup(s[:, :a.warm], idx[:, :a.warm], dp[:, :a.warm])
            pred = model.imagine(h, z, idx[:, a.warm - 1:T - 1], dp[:, a.warm - 1:T - 1]) + off
            truth = d["P"][a.warm:]
            if not a.no_pin_ends and model.repr == "chain":
                # ends driven = measured: hub via p0 BEFORE the forward
                # kinematics, plug overwritten afterwards (every row's protocol)
                pred = pred.clone(); pred[0, :, :2] = truth[:, 0].to(a.device)
            P = model.positions(pred[0], d["ell"].to(a.device))[..., :2].cpu()
            if not a.no_pin_ends:
                P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]
            e = (P - truth).norm(dim=-1).mean(-1).numpy() * 1000          # (T-warm,)
            errs_c.append(e)
            floors.append((model.positions(d["s"].to(a.device), d["ell"].to(a.device))[..., :2].cpu() - d["P"]).norm(dim=-1).mean().item() * 1000)
            n_frames = T
        e = np.mean(errs_c, axis=0)
        curves.append(e[:50] if len(e) >= 50 else np.pad(e, (0, 50 - len(e)), constant_values=np.nan))
        per_clip[clip] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()),
                          "n_frames": int(len(e)), "video": "", "warmup_frames": a.warm}
    v = np.array([x["mittel"] for x in per_clip.values()])
    curve = np.nanmean(np.stack(curves), axis=0)
    out = ROOT / a.out
    out.write_text(json.dumps(per_clip, indent=1))
    summary = {"ckpt": a.ckpt, "tracks": a.tracks, "n_clips": len(v), "mittel": float(v.mean()), "median": float(np.median(v)),
               "warmup_frames": a.warm, "ends_pinned": not a.no_pin_ends, "representation_floor_mm": float(np.mean(floors)),
               "rmse_curve_mm_steps_1_50": [float(x) for x in curve]}
    Path(str(out).replace(".json", "_summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"{a.ckpt} on {a.tracks}: {len(v)} clips, Mittel {v.mean():.2f}, Median {np.median(v):.2f} mm/node "
          f"(frames after a {a.warm}-frame warm-up); representation floor {np.mean(floors):.2f} mm; "
          f"open-loop step 1/10/50: {curve[0]:.1f}/{curve[9]:.1f}/{curve[49]:.1f} mm")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
