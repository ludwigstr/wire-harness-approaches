"""Fresh-process validation of a Newton per-cable parameter set (row 12 layout,
newton_fit_everything_cmaes.unpack, mass pinned 0.044) on a stored track set.

Same evaluation as data/newton_fit_everything_pinned_validate.json
(newton_full_scene.eval_clip, solver_iterations=10, scored on the 15 body
origins vs the first 15 tracked nodes), but the routes come from the npz of
--tracks (whole clips, no trims, no live re-tracking). --live14 instead
re-tracks the 14 clips live exactly like the original validation (t0=8, trims
from clip_set), to check that the default path still reproduces 7.395.

    CUDA_VISIBLE_DEVICES=1,2 python3 scripts/newton_validate_tracks.py \
        --params data/newton_fit_everything_pinned.npz --tracks data/tracks107 \
        --out data/newton_fit_everything_pinned_on_t107.json --gpus 0,1 --workers 8
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WIRE_MASS_PIN", "0.044")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
_dev = {}


def _init(q):
    _dev["device"] = f"cuda:{q.get()}"


def one(job):
    c, params, tracks = job
    t = time.time()
    import newton_fit_everything_cmaes as E
    import newton_full_scene as N
    kw = E.unpack(np.load(params, allow_pickle=True)["best"])
    if tracks:
        z = np.load(ROOT / tracks / f"{c}.npz")
        shp, fps = {k: np.asarray(z[f"c{k}"], float) for k in range(4)}, float(z["fps"])
    else:
        from mujoco_scene import track_clip
        shp, _, _, fps = track_clip(c, t0=8, t1=E._clip_frame_t1(c, E.CLIP_TRIMS[c]))
    errs = N.eval_clip(c, shp_fps=(shp, fps), solver_iterations=10, device=_dev.get("device"), **kw)
    return c, float(np.mean(errs)), len(shp[0]), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--tracks", default="")
    ap.add_argument("--live14", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1", help="warp device indices under CUDA_VISIBLE_DEVICES")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert bool(a.tracks) != a.live14, "give exactly one of --tracks / --live14"
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    if a.tracks:
        clips = sorted(q.stem for q in (ROOT / a.tracks).glob("*.npz"))
        clips.sort(key=lambda c: -len(np.load(ROOT / a.tracks / f"{c}.npz")["c0"]))   # longest first
    else:
        from clip_set import CLIP_TRIMS
        clips = sorted(CLIP_TRIMS)
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(a.workers):
        q.put(gpus[i % len(gpus)])
    t0 = time.time()
    res = {}
    with ctx.Pool(a.workers, initializer=_init, initargs=(q,)) as pool:
        for n, (c, e, T, dt) in enumerate(pool.imap_unordered(
                one, [(c, str(ROOT / a.params), a.tracks) for c in clips]), 1):
            res[c] = {"mittel": e, "n_frames": T, "seconds": dt}
            print(f"[{n}/{len(clips)}] {c}: {e:.3f} mm/node, {T} frames, {dt:.0f}s (total {time.time() - t0:.0f}s)",
                  flush=True)
    res = dict(sorted(res.items()))
    v = np.array([r["mittel"] for r in res.values()])
    out.write_text(json.dumps({"params": a.params, "tracks": a.tracks or "live re-track (14 clips, t0=8, trims)",
                               "protocol": "newton_full_scene.eval_clip, 10 VBD iterations, mass 0.044, "
                                           "15 body origins vs first 15 tracked nodes",
                               "mean": float(v.mean()), "median": float(np.median(v)), "n_clips": len(v),
                               "seconds": time.time() - t0, "clips": res}, indent=1))
    print(f"mean {v.mean():.3f}  median {np.median(v):.3f} mm/node over {len(v)} clips, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
