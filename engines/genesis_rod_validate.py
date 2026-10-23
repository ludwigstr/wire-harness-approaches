"""mm/node validation of the native rod over the 14 kept clips.

Takes K/E from a gradient fit (genesis_rod_autodiff.py) or from the command
line and rolls every clip out once, reporting the same per-node xy error the
rest of the project reports.

ENGINE CAVEAT: the native rod lives only in the DLO-Lab fork. Its numbers are
NOT comparable to the genesis-world 1.3.3 hinge-chain results (24.4 / 18.4mm) --
different solver, different cable model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
from dlolab_path import DLO_LAB  # noqa: E402
sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(DLO_LAB / "examples"))
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))


def _one(job):
    clip, t1_sec, gpu, kw = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import genesis_rod_scene as R
    t0 = time.time()
    try:
        errs = R.eval_clip(clip, t1_sec=t1_sec, **kw)
        return clip, float(np.mean(errs)), len(errs), time.time() - t0, None
    except Exception as e:
        return clip, None, 0, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="data/genesis_rod_autodiff.json",
                    help="json from the gradient fit; '' = use --K/--E")
    ap.add_argument("--K", type=float, default=1.0e4)
    ap.add_argument("--E", type=float, default=1.0e5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--no-hinge", action="store_true",
                    help="ignore a per-joint profile stored in the fit file")
    ap.add_argument("--out", default="data/genesis_rod_validation.json")
    args = ap.parse_args()

    from clip_set import CLIP_TRIMS
    K, E, src, hinge = args.K, args.E, "kommandozeile", None
    extra = {}
    if args.fit and (ROOT / args.fit).exists():
        d = json.loads((ROOT / args.fit).read_text())
        K, E, src = float(d["K"]), float(d["E"]), args.fit
        hinge = d.get("hinge_scale")
        for key, name in (("G", "G"), ("stretch_scale", "stretch_scale"),
                          ("twist_scale", "twist_scale"), ("slack", "slack"),
                          ("n_nodes", "n_nodes"),
                          ("table_fraction", "table_fraction"),
                          ("plastic_yield", "plastic_yield"),
                          ("plastic_creep", "plastic_creep"),
                          ("kappa", "kappa_rest"),
                          ("damping", "damping"),
                          ("hinge_damping", "hinge_damping"),
                          ("state_a", "state_a"), ("state_b", "state_b"),
                          ("state_a_load", "state_a_load"),
                          ("state_a_unload", "state_a_unload"),
                          ("codebook", "codebook")):
            v = d.get(key)
            if v not in (None, 0, 0.0):
                extra[name] = v
    if args.no_hinge:
        hinge = None
    print(f"parameter aus {src}: K={K:.4g}  E={E:.4g}"
          + (f"  + {len(hinge)} gelenkskalen" if hinge else "  (uniform)")
          + ("".join(f"  {k}={v if not isinstance(v, list) else len(v)}"
                     for k, v in extra.items())),
          flush=True)

    clips = args.clips or sorted(CLIP_TRIMS)
    gpus = args.gpus.split(",")
    kw = dict(K=K, E=E, use_inextensible=False, hinge_scale=hinge, **extra)
    jobs = [(c, CLIP_TRIMS.get(c), gpus[i % len(gpus)], kw)
            for i, c in enumerate(clips)]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")            # Genesis does not survive fork
    per, errs = {}, []
    with ctx.Pool(args.workers) as pool:
        for clip, mm, n, dt, err in pool.imap_unordered(_one, jobs):
            if err:
                print(f"  {clip}  FEHLER: {err}", flush=True)
                per[clip] = None
                continue
            per[clip] = mm
            errs.append(mm)
            print(f"  {clip}  {mm:6.2f} mm/Knoten  ({n} frames, {dt:.0f}s)",
                  flush=True)

    res = {"K": K, "E": E, "quelle": src, "hinge_scale": hinge, "per_clip": per,
           "mittel": float(np.mean(errs)) if errs else None,
           "median": float(np.median(errs)) if errs else None,
           "n_clips": len(errs)}
    (ROOT / args.out).write_text(json.dumps(res, indent=1))
    print(f"\nMittel {res['mittel']:.2f} mm/Knoten   "
          f"Median {res['median']:.2f}   n={res['n_clips']}", flush=True)
    print(f"geschrieben {ROOT / args.out}", flush=True)
    print("VALIDATE_DONE", flush=True)


if __name__ == "__main__":
    main()
