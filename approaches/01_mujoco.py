"""MuJoCo (rows 1-4 of the table): the four cables as MuJoCo `elasticity.cable`
composites in the measured scene (board, hub, movers), the hub and the four
plugs driven by PD actuators along their tracked poses, 300-step settle, then a
replay of the whole clip; error = mean node distance over every frame.

  row 1  uncalibrated            mass 0.05 kg/m, mu 3.0            20.3 mm/node
  row 2  mass + friction tuned   mass 0.150, mu 1.0 (sweep)         20.3
  row 3  + per-joint rest angles weld constraint per joint pulling to the MEASURED
                                 angle, solref chosen per joint (coordinate descent,
                                 56 = 4 cables x 14 joints)         11.6
  row 4  + per-joint bending stiffness: ball-joint springs per joint (56) plus a
                                 fitted rest angle per joint about the board normal
                                 (56), CMA-ES, 60 generations        8.5 (median 7.9)

The objective of every fit IS the reported metric (mujoco_full_loco.rollout_errs).

    python approaches/01_mujoco.py eval --row 4 [--workers 14]
    python approaches/01_mujoco.py eval --row all
    python approaches/01_mujoco.py fit --row 4     # CMA-ES from the shipped warm start (hours)
    python approaches/01_mujoco.py fit --row 3     # weld coordinate descent from its warm start

Fit chains as run for the table (warm starts shipped in data/):
  row 3: weld_fit_progress_14.npz -> fit_weld_stiffness_par.py --train-all -> weld_fit_meanframe.npz
  row 4: jnt_stiffness_fit_wide.npz -> fit_jnt_stiffness.py (springs only, 56 values)
         -> jnt_stiffness_fit_r3.npz -> fit_jnt_spring_rest_cmaes.py -> jnt_spring_rest_fit.npz
Needs: mujoco >= 3.11, cma, numpy, opencv-python.
"""
import argparse
import multiprocessing as mp
import subprocess
import sys

import numpy as np

from common import ROOT, CLIP_TRIMS, record, write_results

sys.path.insert(0, str(ROOT / "engines"))

ROWS = {1: {"mass_per_m": 0.05, "mu": 3.0}, 2: {}}


def _row12(job):
    from mujoco_full_loco import eval_clip
    clip, kw = job
    return clip, np.asarray(eval_clip(clip, t1_sec=CLIP_TRIMS[clip], cable_kw=kw))


def _row3(job):
    from mujoco_full_loco import eval_clip
    clip, solref = job
    return clip, np.asarray(eval_clip(clip, t1_sec=CLIP_TRIMS[clip], weld_solref=solref))


def _row4(job):
    import mujoco
    import fit_jnt_spring_rest_cmaes as R
    from mujoco_full_loco import rollout_errs
    from mujoco_scene import build_xml, set_per_joint_stiffness, track_clip
    from fit_jnt_stiffness import clip_frame_t1
    clip, x = job
    shp, _, _, fps = track_clip(clip, t0=8, t1=clip_frame_t1(clip, CLIP_TRIMS[clip]))
    m = mujoco.MjModel.from_xml_string(build_xml(shp))
    k, delta = R.unpack(x)
    set_per_joint_stiffness(m, k, rest_quat=R.rest_quats(delta, R.board_normal_axes(m)))
    return clip, rollout_errs(m, shp, fps, per_cable=True).mean(1)


def evaluate(row, workers, out):
    clips = list(CLIP_TRIMS)
    if row in (1, 2):
        fn, jobs = _row12, [(c, ROWS[row]) for c in clips]
    elif row == 3:
        solref = np.load(ROOT / "data/weld_fit_meanframe.npz", allow_pickle=True)["best"]
        fn, jobs = _row3, [(c, solref) for c in clips]
    else:
        x = np.load(ROOT / "data/jnt_spring_rest_fit.npz", allow_pickle=True)["best_x"]
        fn, jobs = _row4, [(c, x) for c in clips]
    with mp.get_context("fork").Pool(workers) as pool:
        res = {c: record(e) for c, e in pool.map(fn, jobs)}
    write_results(res, out, f"MuJoCo row {row}")


def fit(row, workers):
    eng = ROOT / "engines"
    if row == 3:
        cmd = [sys.executable, str(eng / "fit_weld_stiffness_par.py"), "--train-all", "--init", "data/weld_fit_progress_14.npz",
               "--workers", str(workers), "--out", "work/weld_fit_meanframe.npz", "--log", "work/weld_fit_meanframe_log.jsonl"]
    elif row == 4:
        cmd = [sys.executable, str(eng / "fit_jnt_spring_rest_cmaes.py"), "--init", "data/jnt_stiffness_fit_r3.npz",
               "--workers", str(workers), "--out", "work/jnt_spring_rest_fit.npz", "--log", "work/jnt_spring_rest_fit_log.jsonl"]
    else:
        raise SystemExit("rows 1 and 2 have no fit of their own (row 2 is the mass/friction sweep: eval with other values)")
    (ROOT / "work").mkdir(exist_ok=True)
    print(" ".join(cmd)); subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["eval", "fit"])
    ap.add_argument("--row", default="4", help="1-4 or all")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    rows = [1, 2, 3, 4] if a.row == "all" else [int(a.row)]
    for r in rows:
        evaluate(r, a.workers, f"results/mujoco_row{r}.json") if a.step == "eval" else fit(r, a.workers)


if __name__ == "__main__":
    main()
