"""Genesis hinge chain with UNCALIBRATED parameters, all 14 clips.

The table's Genesis rows all start from values this project swept for
(stiffness 0.1 / damping 0.5, mass_per_m 0.0368). This runs the configuration
that was in place BEFORE that sweep -- hinge stiffness and damping at 0.02,
the values the chain was first built with -- so the Genesis block gets the
same "engine as handed to us" row that Newton and the DLO-Lab rod have.

Needs the 1.3.3 interpreter:
    ~/venvs/genesis133/bin/python scripts/genesis_uncal_eval.py
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines")); sys.path.insert(0, str(ROOT))


def _one(args):
    clip, kw = args
    sys.path.insert(0, str(ROOT / "engines")); sys.path.insert(0, str(ROOT))
    from clip_set import CLIP_TRIMS
    from genesis_full_scene import eval_clip
    try:
        e = eval_clip(clip, t1_sec=CLIP_TRIMS.get(clip), **kw)
        vals = list(e.values()) if isinstance(e, dict) else e
        return clip, float(np.mean(vals))
    except Exception as exc:
        print(f"  FEHLER {clip}: {type(exc).__name__}: {exc}", flush=True)
        return clip, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stiffness", type=float, default=0.02)
    ap.add_argument("--damping", type=float, default=0.02)
    ap.add_argument("--mass-per-m", type=float, default=None)
    ap.add_argument("--out", default="data/genesis_uncalibrated.json")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    from clip_set import CLIP_TRIMS
    kw = dict(stiffness=a.stiffness, damping=a.damping)
    if a.mass_per_m is not None:
        kw["mass_per_m"] = a.mass_per_m
    clips = list(CLIP_TRIMS.keys())
    print(f"Genesis uncalibrated: {kw} auf {len(clips)} Clips, {a.jobs} Jobs", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.jobs) as pool:
        res = pool.map(_one, [(c, kw) for c in clips])

    per_clip = {c: v for c, v in res if v is not None}
    vals = np.array(list(per_clip.values()))
    out = {**kw, "n_clips": len(per_clip), "per_clip": per_clip,
           "mittel": float(vals.mean()), "median": float(np.median(vals))}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(f"\n{len(per_clip)} Clips, Mittel {vals.mean():.2f}  Median "
          f"{np.median(vals):.2f} mm/Knoten ({time.time()-t0:.0f}s)", flush=True)
    print("GENESIS_UNCAL_FERTIG")


if __name__ == "__main__":
    main()
