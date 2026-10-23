"""Genesis (rows 5-7 of the table): each cable as a hinge chain (MJCF loaded into
Genesis 1.3.3, CPU backend) in the measured scene; hub and plugs PD-driven along
their tracks, movers as colliding boxes; error = mean node distance, every frame.

  row 5  uncalibrated               stiffness 0.02, damping 0.02, mass 0.0368 kg/m   20.5 mm/node
  row 6  uniform, no mover contact  stiffness 0.10, damping 0.50 (sweep)              24.4
  row 7  per-joint stiffness + mover contact: 15 stiffnesses + 15 dampings + mass,
         CMA-ES on a skeleton-distance loss against the video masks               18.4

Environment: genesis-world 1.3.3 exactly (newer versions and the DLO-Lab fork
order the links differently and give wrong numbers). Put its interpreter in
$GENESIS_PYTHON (e.g. a venv with genesis-world==1.3.3, mujoco, cma, CPU torch);
this script re-executes itself with it.

    python approaches/02_genesis.py eval --row 7
    python approaches/02_genesis.py eval --row all
    python approaches/02_genesis.py fit  --row 7   # needs the videos ($WIRE_VIDEO_DIR) and masks ($WIRE_SEG_DIR)
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
ROWS = {5: dict(stiffness=0.02, damping=0.02), 6: dict(stiffness=0.1, damping=0.5)}


def _ensure_genesis_133():
    try:
        import genesis
        ok = getattr(genesis, "__version__", "") == "1.3.3"
    except ImportError:
        ok = False
    if not ok:
        py = os.environ.get("GENESIS_PYTHON")
        if not py or os.path.realpath(py) == os.path.realpath(sys.executable):
            raise SystemExit("needs genesis-world 1.3.3: set GENESIS_PYTHON to an interpreter that has it")
        os.execv(py, [py] + sys.argv)


def _one(job):
    clip, kw = job
    sys.path.insert(0, str(ROOT / "engines"))
    import genesis_full_scene as G
    from mujoco_scene import track_clip
    t1 = int(round(CLIP_TRIMS[clip] * 4.0)) if CLIP_TRIMS[clip] else None
    shp, _, _, fps = track_clip(clip, t0=8, t1=t1)
    e = G.eval_clip(clip, t1_sec=CLIP_TRIMS[clip], shp_fps=(shp, fps), **kw)
    return clip, np.asarray(list(e.values()) if isinstance(e, dict) else e, float)


def evaluate(row, jobs, out):
    if row == 7:
        d = json.loads((ROOT / "data/genesis_perjoint_sysid.json").read_text())
        kw = dict(stiffness=np.asarray(d["stiffness"]), damping=np.asarray(d["damping"]), mass_per_m=d["mass_per_m"])
    else:
        kw = ROWS[row]
    os.environ["WIRE_GENESIS_MOVER_CONTACT"] = "0" if row == 6 else "1"
    with mp.get_context("spawn").Pool(jobs) as pool:
        res = {c: record(e) for c, e in pool.map(_one, [(c, kw) for c in CLIP_TRIMS])}
    write_results(res, out, f"Genesis row {row}")


def fit(row):
    if row != 7:
        raise SystemExit("row 6 is a stiffness/damping sweep, row 5 has no fit")
    (ROOT / "work").mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / "engines/genesis_perjoint_sysid.py")]
    print(" ".join(cmd), "(writes data/genesis_perjoint_sysid.json)"); subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["eval", "fit"])
    ap.add_argument("--row", default="7"); ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    _ensure_genesis_133()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Genesis runs on the CPU backend here
    rows = [5, 6, 7] if a.row == "all" else [int(a.row)]
    for r in rows:
        evaluate(r, a.jobs, f"results/genesis_row{r}.json") if a.step == "eval" else fit(r)


if __name__ == "__main__":
    main()
