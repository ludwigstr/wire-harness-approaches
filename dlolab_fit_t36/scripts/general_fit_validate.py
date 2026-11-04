"""Fresh-process validation of a CMA-ES general fit (MuJoCo / Genesis / DLO-Lab):
one new interpreter per clip runs the fit script's own _eval_one((0, clip, best_x)),
exactly the procedure behind data/*_general_fit*_validate.json (2026-10-27), but with a
selectable track dir (2026-11-01, refit on data/tracks107: every *.npz, untrimmed).

    python scripts/general_fit_validate.py mujoco data/mujoco_general_fit3_t107.npz \
        --tracks data/tracks107 --out data/mujoco_general_fit3_t107_validate.json --workers 10

Genesis needs --python ~/venvs/genesis133/bin/python. Without --tracks the 14-clip set
(clip_set.CLIP_TRIMS on the cached tracks14) is used. Output: {clip: mm/node}; per-clip
wall seconds go to stdout only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from clip_set import CLIP_TRIMS  # noqa: E402

MODS = {"mujoco": "mujoco_general_fit_cmaes", "genesis": "genesis_general_fit_cmaes",
        "dlolab": "dlolab_general_fit_cmaes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=sorted(MODS))
    ap.add_argument("params", help=".npz (MuJoCo) or .json (Genesis / DLO-Lab) holding best_x")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", default="1", help="OMP/MKL/BLAS threads per clip process")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    params = (ROOT / a.params).resolve()
    load = (f"np.load({str(params)!r})['best_x']" if params.suffix == ".npz"
            else f"json.load(open({str(params)!r}))['best_x']")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": a.threads,
           "MKL_NUM_THREADS": a.threads, "OPENBLAS_NUM_THREADS": a.threads}
    if a.tracks:
        tracks = (ROOT / a.tracks).resolve()
        env["GENERAL_FIT_TRACKS"] = str(tracks)
        clips = sorted(p.stem for p in tracks.glob("*.npz"))
    else:
        env.pop("GENERAL_FIT_TRACKS", None)
        clips = sorted(CLIP_TRIMS)
    mod = MODS[a.engine]

    def run(c):
        code = [f"import sys, json, numpy as np; sys.path[:0] = ['scripts', '.']; import {mod} as M",
                f"x = np.asarray({load}, float)"]
        if a.engine == "mujoco":   # its _eval_one expects the track cache filled by main()
            code.append(f"z = np.load(M.TRACKS / '{c}.npz'); M._CACHE['{c}'] = ({{k: z[f'c{{k}}'] for k in range(4)}}, float(z['fps']))")
        code.append(f"print('RES', M._eval_one((0, '{c}', x))[2])")
        t = time.time()
        p = subprocess.run([a.python, "-c", "\n".join(code)], capture_output=True, text=True, env=env, cwd=ROOT)
        r = [ln for ln in p.stdout.splitlines() if ln.startswith("RES")]
        res = float(r[-1].split()[1]) if r else f"CRASH rc={p.returncode} {p.stderr[-300:]!r}"
        print(f"{c}: {res}  ({time.time() - t:.0f}s)", flush=True)
        return c, res

    t0 = time.time()
    print(f"{a.engine}: {params} on {len(clips)} clips ({a.tracks or 'tracks14'}), {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(ex.map(run, clips))
    out.write_text(json.dumps(res, indent=1))
    v = np.array([e for e in res.values() if isinstance(e, float)])
    print(f"{a.engine}: mean {v.mean():.3f} median {np.median(v):.3f} n={len(v)}/{len(clips)} "
          f"(failed scored as-is: {sum(e >= 1e3 for e in v)} at 1e3)  wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
