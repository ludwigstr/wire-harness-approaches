"""Newton (rows 8-14 of the table): NVIDIA Newton's native rod model (VBD solver,
10 iterations, 1 ms steps) in the full scene -- board, hub, movers as colliding
bodies -- with the cable ends kinematically on their tracks; 3-D, and the fits
also match the cable height measured from the video (Depth-Anything targets,
data/depth_targets). Error = xy mean node distance, every frame.

  row  8  uncalibrated                                 14.1 mm/node
  row  9  shared fit + measured cable height (29 values: 14 bend ke, 14 bend kd, mass)   8.6
  row 10  + Dahl hysteresis + 14 rest angles (45 values)                                   8.1
  row 11  + per-joint twist (59; twist was a copy of bend before)                          7.8
  row 12  + sliding friction freed (47, warm start row 10)                                 7.9
  row 13  mass pinned to the measured 0.044 kg/m; K stretch + all four frictions free (63) 8.0
  row 14  per-CABLE fit: ke, kd, mass, rest per cable (174)                                7.7
All fits are CMA-ES on xy + |z_sim - z_depth| over all 14 clips (in-sample).

    python approaches/03_newton.py eval --row 11 [--gpus 0,1]
    python approaches/03_newton.py eval --row all
    python approaches/03_newton.py fit --row 11     # CMA-ES, many GPU hours; writes work/
Needs: newton >= 1.5, warp-lang, cma, CUDA.
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

from common import ROOT, CLIP_TRIMS, record, write_results

sys.path.insert(0, str(ROOT / "engines"))
SHARED = {9: "data/newton_fit_depth.npz", 10: "data/newton_fit_big.npz", 11: "data/newton_fit_twist.npz"}


def row_kwargs(row):
    """the eval_clip keyword set of a table row, from its stored fit"""
    if row == 8:
        return {}
    if row in SHARED:
        from newton_iteration_sweep import unpack
        ke, kd, mass, eps, tau, rest, kt = unpack(np.load(ROOT / SHARED[row], allow_pickle=True)["best"])
        return dict(per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt, mass_per_m=mass, dahl_eps_max=eps, dahl_tau=tau, per_joint_rest=rest)
    if row == 12:
        os.environ.update(WIRE_MU="1", WIRE_REST="1", WIRE_DAHL_MODE="scalar")
        import newton_fit_mu_cmaes as M
        ke, kd, mass, eps, tau, rest, extra, kt, mu = M._params_from_log(M._to_x(np.load(ROOT / "data/newton_fit_mu.npz", allow_pickle=True)["best"]))
        return dict(per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt, mass_per_m=mass, shape_mu=float(mu[0]), cable_mu=float(mu[1]),
                    dahl_eps_max=eps, dahl_tau=tau, per_joint_rest=np.tile(rest, (4, 1)), subdiv=M.SUBDIV)
    if row == 13:
        import newton_fit_physical_cmaes as P
        return P.unpack(np.load(ROOT / "data/newton_fit_physical.npz", allow_pickle=True)["best"])
    if row == 14:
        os.environ.update(WIRE_REST="1", WIRE_DAHL_MODE="scalar")
        import newton_fit_percable_cmaes as M
        ke, kd, mass, eps, tau, rest = M._params_from_log(M._to_x(np.load(ROOT / "data/newton_fit_percable.npz", allow_pickle=True)["best"]))[:6]
        return dict(per_joint_ke=ke, per_joint_kd=kd, mass_per_m=mass, dahl_eps_max=eps, dahl_tau=tau, per_joint_rest=rest, subdiv=M.SUBDIV)
    raise SystemExit(f"no Newton row {row}")


def _one(job):
    clip, row, gpu = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(ROOT / "engines"))
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    t1 = int(round(CLIP_TRIMS[clip] * 4.0)) if CLIP_TRIMS[clip] else None
    shp, _, _, fps = track_clip(clip, t0=8, t1=t1)
    return clip, np.asarray(eval_clip(clip, shp_fps=(shp, fps), solver_iterations=10, **row_kwargs(row)), float)


def evaluate(row, gpus, workers, out):
    clips = sorted(CLIP_TRIMS)
    jobs = [(c, row, gpus[i % len(gpus)]) for i, c in enumerate(clips)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = {c: record(e) for c, e in pool.map(_one, jobs)}
    write_results(res, out, f"Newton row {row} (10 VBD iterations)")


FITS = {   # row -> (script, environment) exactly as run for the table
    9: ("newton_fit_depth_cmaes.py", {}),
    10: ("newton_fit_dahl_cmaes.py", dict(WIRE_REST="1", WIRE_DAHL_MODE="scalar", WIRE_SIGMA0="0.2", WIRE_REST_STD="0.06",
                                          WIRE_WARM="data/newton_fit_dahlC.npz", WIRE_OUT="work/newton_fit_big.npz")),
    11: ("newton_fit_dahl_cmaes.py", dict(WIRE_REST="1", WIRE_TWIST="1", WIRE_DAHL_MODE="scalar", WIRE_SIGMA0="0.2", WIRE_REST_STD="0.06",
                                          WIRE_WARM="data/newton_fit_big.npz", WIRE_OUT="work/newton_fit_twist.npz")),
    12: ("newton_fit_mu_cmaes.py", dict(WIRE_MU="1", WIRE_REST="1", WIRE_DAHL_MODE="scalar", WIRE_WARM="data/newton_fit_big.npz",
                                        WIRE_OUT="work/newton_fit_mu.npz")),
    13: ("newton_fit_physical_cmaes.py", dict(WIRE_MASS="0.044", WIRE_WARM="data/newton_fit_twist.npz", WIRE_GENS="14", WIRE_POPSIZE="12",
                                              WIRE_OUT="work/newton_fit_physical.npz")),
    14: ("newton_fit_percable_cmaes.py", dict(WIRE_REST="1", WIRE_DAHL_MODE="scalar", WIRE_WARM="data/newton_fit_big.npz", WIRE_SIGMA0="0.2",
                                              WIRE_OUT="work/newton_fit_percable.npz")),
}


def fit(row, gpus):
    if row not in FITS:
        raise SystemExit("row 8 has no fit")
    script, env = FITS[row]
    (ROOT / "work").mkdir(exist_ok=True)
    e = {**os.environ, **env, "WIRE_GPUS": ",".join(map(str, gpus)), "WIRE_FITLOG": f"work/newton_fit_row{row}_log.jsonl"}
    print(" ".join(f"{k}={v}" for k, v in env.items()), script)
    subprocess.run([sys.executable, str(ROOT / "engines" / script)], cwd=ROOT, env=e, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["eval", "fit"])
    ap.add_argument("--row", default="11"); ap.add_argument("--gpus", default="0"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    rows = list(range(8, 15)) if a.row == "all" else [int(a.row)]
    for r in rows:
        evaluate(r, gpus, a.workers, f"results/newton_row{r}.json") if a.step == "eval" else fit(r, gpus)


if __name__ == "__main__":
    main()
