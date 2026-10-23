"""Fresh-process validation of the row-3c fit (springs + rest angles).

The fit's objective is already the reported metric (rollout_errs mean over
all 14 clips), but the CMA process only stores the scalar. This re-evaluates
the saved best in a new process (reproduction check, same rule as for Newton)
and writes mean, median and per-clip values to JSON for the table.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_jnt_stiffness as F  # noqa: E402
import fit_jnt_spring_rest_cmaes as R  # noqa: E402


def main():
    src = R.ROOT / "data/jnt_spring_rest_fit.npz"
    out = R.ROOT / "data/jnt_spring_rest_validate.json"
    ck = np.load(src, allow_pickle=True)
    x = ck["best_x"]
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    log(f"stored best_f {float(ck['best_f']):.4f}  k {ck['best'].min():.3f}..{ck['best'].max():.3f}  "
        f"rest {ck['rest_deg'].min():+.1f}..{ck['rest_deg'].max():+.1f} deg")
    clips = list(F.CLIP_TRIMS)
    R._CACHE.update(F.load_clips(clips, log))
    pool = mp.get_context("fork").Pool(14)
    t = time.time()
    res = {c: e for _, c, e in pool.map(R._eval_one, [(0, c, x) for c in clips])}
    vals = np.array(list(res.values()))
    log(f"fresh mean {vals.mean():.4f}  median {np.median(vals):.4f}  ({time.time()-t:.0f}s)")
    for c, v in sorted(res.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} {v:6.2f}")
    out.write_text(json.dumps({"mean": float(vals.mean()), "median": float(np.median(vals)),
                               "stored_best_f": float(ck["best_f"]), "per_clip": res}, indent=1))
    pool.close(); pool.join()
    log(f"-> {out}")
    log("=== VALIDATE_FERTIG ===")


if __name__ == "__main__":
    main()
