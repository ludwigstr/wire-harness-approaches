"""DLO-Lab (rows 15-18 of the table): the native Cosserat rod of the DLO-Lab fork
of Genesis (UMass-Embodied-AGI/DLO-Lab), one rod per cable in the measured
scene, differentiable; the fits are reverse-mode gradient descent on a
distance-transform loss against the video masks.

  row 15  uncalibrated                    K 1e4, E 1e5 (as shipped)                    33.9 mm/node
  row 16  gradient fit, uniform stiffness K, E                                         18.6
  row 17  + per-joint stiffness           14 hinge scales on top of 16, K/E frozen     17.0
  row 18  + rest curvature                big fit (K, E, G, 15 stretch, 14 hinge, 14 twist
                                          scales) + 28 rest curvatures over all 14 clips  16.56
Note: rows 16 and 17 were fitted on clip 20260602_170108 only (row 18 on all 14).

Setup: ./setup_dlolab.sh clones DLO-Lab at c5026a9 into extern/DLO-Lab-perjoint and
applies patches/dlo-lab-perjoint.patch (per-element stiffness scales and rest
curvature offsets in the rod solver; behaviour-neutral at scale 1).
$WIRE_DLOLAB points elsewhere if you already have a patched checkout.

    python approaches/04_dlolab.py eval --row 18 [--gpus 0,1]
    python approaches/04_dlolab.py eval --row all
    python approaches/04_dlolab.py fit --row 18    # needs videos ($WIRE_VIDEO_DIR) and masks ($WIRE_SEG_DIR)
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

from common import ROOT, CLIP_TRIMS, record, write_results

sys.path.insert(0, str(ROOT / "engines"))
from dlolab_path import DLO_LAB  # noqa: E402
sys.path.insert(0, str(DLO_LAB)); sys.path.insert(0, str(DLO_LAB / "examples")); sys.path.insert(0, str(ROOT / "engines"))

FIT_FILES = {16: "data/genesis_rod_autodiff_dt.json", 17: "data/genesis_rod_autodiff_perjoint_frozen.json",
             18: "data/genesis_rod_allclip_best.json"}
KEYS = {"G": "G", "stretch_scale": "stretch_scale", "twist_scale": "twist_scale", "slack": "slack", "n_nodes": "n_nodes",
        "table_fraction": "table_fraction", "plastic_yield": "plastic_yield", "plastic_creep": "plastic_creep",
        "kappa": "kappa_rest", "damping": "damping", "hinge_damping": "hinge_damping"}


def row_kwargs(row):
    if row == 15:
        return dict(K=1.0e4, E=1.0e5, use_inextensible=False, hinge_scale=None)
    d = json.loads((ROOT / FIT_FILES[row]).read_text())
    extra = {name: d[key] for key, name in KEYS.items() if d.get(key) not in (None, 0, 0.0)}
    return dict(K=float(d["K"]), E=float(d["E"]), use_inextensible=False, hinge_scale=d.get("hinge_scale"), **extra)


def _one(job):
    clip, gpu, kw = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import genesis_rod_scene as R
    return clip, np.asarray(R.eval_clip(clip, t1_sec=CLIP_TRIMS[clip], **kw), float)


def evaluate(row, gpus, workers, out):
    kw = row_kwargs(row)
    jobs = [(c, gpus[i % len(gpus)], kw) for i, c in enumerate(sorted(CLIP_TRIMS))]
    with mp.get_context("spawn").Pool(workers) as pool:          # Genesis does not survive fork
        res = {c: record(e) for c, e in pool.map(_one, jobs)}
    write_results(res, out, f"DLO-Lab row {row}")


def fit(row, gpus):
    eng = ROOT / "engines"; py = sys.executable; g = ",".join(map(str, gpus))
    (ROOT / "work").mkdir(exist_ok=True)
    if row == 16:
        steps = [([py, str(eng / "genesis_rod_autodiff.py"), "--loss", "dt", "--gpus", g, "--out", "work/genesis_rod_autodiff_dt.json"], {})]
    elif row == 17:
        steps = [([py, str(eng / "genesis_rod_autodiff.py"), "--loss", "dt", "--per-joint", "14", "--freeze-ke", "--warm", "data/genesis_rod_autodiff_dt.json", "--n-iters", "50", "--gpus", g,
                   "--out", "work/genesis_rod_autodiff_perjoint_frozen.json"], {})]
    elif row == 18:
        steps = [([py, str(eng / "genesis_rod_bigfit.py")], {}),                                   # -> data/genesis_rod_bigfit_bptt120.json
                 ([py, str(eng / "genesis_rod_kappa_fit.py")], {}),                                # 2-clip warm start
                 ([py, str(eng / "genesis_rod_kappa_fit_all.py")], {"LS_MAXF": "30", "LS_BASE": "data/genesis_rod_bigfit_bptt120.json"})]
        print("row 18 = K, E and scales of genesis_rod_bigfit_bptt120.json + 'kappa' of genesis_rod_kappa_fit_all.json "
              "(merged into data/genesis_rod_allclip_best.json)")
    else:
        raise SystemExit("row 15 has no fit")
    for cmd, env in steps:
        print(" ".join(f"{k}={v}" for k, v in env.items()), " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["eval", "fit"])
    ap.add_argument("--row", default="18"); ap.add_argument("--gpus", default="0"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(",")]
    rows = [15, 16, 17, 18] if a.row == "all" else [int(a.row)]
    for r in rows:
        evaluate(r, gpus, a.workers, f"results/dlolab_row{r}.json") if a.step == "eval" else fit(r, gpus)


if __name__ == "__main__":
    main()
