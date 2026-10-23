"""Row 7 of the approaches table: Newton with ENGINE DEFAULTS, no fit.

The stored number (15.05 mm/node on the 14 clips) was produced when the full
scene ran 20 VBD iterations. The iteration study of 2026-10-20 moved every
Newton row to 10, so row 7 has to be re-measured at 10 or it is the only
Newton row left on the old protocol.

Both settings are run here: 20 first, as a reproduction check against the
stored 15.05, then 10 as the new table value. A 20 that does not reproduce
means the code path moved since and the comparison is void.

    WIRE_ITERS=20,10 python3 scripts/newton_default_validate.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
ITERS = [int(x) for x in os.environ.get("WIRE_ITERS", "20,10").split(",")]
OUT = ROOT / os.environ.get("WIRE_OUT", "data/newton_default_iters.json")
GPUS = [int(x) for x in os.environ.get("WIRE_GPUS", "1,2").split(",")]


def _one(job):
    clip, t1, gpu, iters = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from newton_full_scene import eval_clip
    t0 = time.time()
    try:
        errs = eval_clip(clip, t1_sec=t1, solver_iterations=iters)
        return clip, float(np.mean(errs)), time.time() - t0, None
    except Exception as e:
        return clip, None, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    from clip_set import CLIP_TRIMS
    import multiprocessing as mp
    clips = sorted(CLIP_TRIMS)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ctx = mp.get_context("spawn")
    for iters in ITERS:
        if str(iters) in res:
            continue
        print(f"\n=== {iters} Iterationen, Default-Parameter ===", flush=True)
        jobs = [(c, CLIP_TRIMS.get(c), GPUS[i % len(GPUS)], iters)
                for i, c in enumerate(clips)]
        per, vals = {}, []
        with ctx.Pool(len(GPUS)) as pool:
            for clip, mm, dt, err in pool.imap_unordered(_one, jobs):
                if err:
                    print(f"  {clip}  FEHLER: {err}", flush=True)
                else:
                    per[clip] = mm; vals.append(mm)
                    print(f"  {clip}  {mm:7.3f} mm  ({dt:.0f}s)", flush=True)
        res[str(iters)] = {"mittel": float(np.mean(vals)),
                           "median": float(np.median(vals)),
                           "n_clips": len(vals), "per_clip": per}
        print(f"  Mittel {np.mean(vals):.3f}  Median {np.median(vals):.3f} mm"
              f"  n={len(vals)}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    for k in sorted(res, key=int):
        print(f"N={k:>3}  Mittel {res[k]['mittel']:.3f}  "
              f"Median {res[k]['median']:.3f}  n={res[k]['n_clips']}",
              flush=True)
    print("DEFAULT_VALIDATE_FERTIG", flush=True)


if __name__ == "__main__":
    main()
