"""Measure the per-joint fit in mm/node over the 14-clip set, and render.

Separate from genesis_sysid_validate.py because that one takes scalar
stiffness/damping; here both are 15-element arrays (one value per hinge joint).
The metric is unchanged -- mean node distance to the tracked route -- so the
number stays comparable with every other bar in the project.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))


def _eval_one(job):
    clip, st, dp, ms = job
    import sys as _s
    _s.path.insert(0, str(ROOT / "engines"))
    from clip_set import CLIP_TRIMS as CT
    from mujoco_scene import track_clip as tc, DIR as D
    import genesis_full_scene as GG
    import cv2 as cv
    trim = CT[clip]
    cap = cv.VideoCapture(f"{D}/{clip}.mp4")
    f = (cap.get(cv.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
    cap.release()
    t1 = int(round(trim * f)) if trim else None
    shp, _, _, f = tc(clip, t0=8, t1=t1)
    e = GG.eval_clip(clip, t1_sec=trim, shp_fps=(shp, f),
                     stiffness=np.asarray(st), damping=np.asarray(dp),
                     mass_per_m=ms)
    return clip, float(np.mean(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="data/genesis_perjoint_sysid.json")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--out-dir", default="work/genesis_perjoint_video")
    ap.add_argument("--out", default="data/genesis_perjoint_validation.json")
    args = ap.parse_args()

    from clip_set import CLIP_TRIMS
    d = json.loads((ROOT / args.fit).read_text())
    st, dp, ms = d["stiffness"], d["damping"], d["mass_per_m"]
    print(f"per-gelenk-parameter aus {args.fit}", flush=True)
    print(f"  stiffness {min(st):.4g}-{max(st):.4g}   damping {min(dp):.4g}-"
          f"{max(dp):.4g}   mass {ms:.4g}   (skelettverlust {d['best_loss']:.4f})",
          flush=True)

    clips = list(CLIP_TRIMS)
    per_clip, t0 = {}, time.time()
    from multiprocessing import get_context
    jobs = [(c, st, dp, ms) for c in clips]
    with get_context("spawn").Pool(args.jobs) as pool:
        for clip, v in pool.map(_eval_one, jobs):
            per_clip[clip] = v
            print(f"  {clip:<20}{v:7.2f} mm/Knoten", flush=True)
    print(f"  ({time.time()-t0:.0f}s, {args.jobs} parallel)", flush=True)

    v = np.array(list(per_clip.values()))
    print(f"\n=== GENESIS 1.3.3, per gelenk + mover-kollision, {v.size} clips ===",
          flush=True)
    print(f"  Mittel {v.mean():.2f} mm/Knoten", flush=True)
    print(f"  Median {np.median(v):.2f} mm/Knoten", flush=True)
    print(f"  Spanne {v.min():.2f} - {v.max():.2f}", flush=True)
    (ROOT / args.out).write_text(json.dumps(
        {"stiffness": st, "damping": dp, "mass_per_m": ms,
         "best_loss": d["best_loss"], "n_clips": int(v.size),
         "mean": float(v.mean()), "median": float(np.median(v)),
         "collision": "mover collision ENABLED", "per_clip": per_clip}, indent=1))
    print(f"-> {ROOT / args.out}", flush=True)

    if args.videos:
        import genesis_full_scene as G
        t0 = time.time()
        for i, c in enumerate(clips, 1):
            G.render_full_video(c, t1_sec=CLIP_TRIMS[c], out_dir=args.out_dir,
                                stiffness=np.asarray(st), damping=np.asarray(dp),
                                mass_per_m=ms)
            print(f"  [{i:2d}/{len(clips)}] video {c}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print("PERJOINT_VALIDATE_DONE", flush=True)


if __name__ == "__main__":
    main()
