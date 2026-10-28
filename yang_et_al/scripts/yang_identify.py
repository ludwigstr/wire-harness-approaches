"""Yang et al. (RAS 2022) step 1: parameter identification by differential
evolution on the ONE-STEP error, Eq. (1), on a few real trajectories.

    Theta* = argmin sum_i || x_{i+1} - f_sim(x_i, u_i, Theta, dt) ||^2

f_sim = the MuJoCo harness scene (wire_model/yang/sim.py, mapping of the
paper's Table 1 there), reset to the OBSERVED state x_i of every transition,
driven one frame (dt = 0.25 s) by the measured mover motion u_i. Objective =
mean squared node distance over all transitions, 4 cables x 16 nodes.

DE: scipy.optimize.differential_evolution (the paper's implementation),
strategy best1bin = Eq. (2) + binomial crossover, SciPy's default population
(15 x n_params), mutation (0.5, 1), recombination 0.7, Latin-hypercube init;
no gradient polish (serial L-BFGS on a 45-s objective).

    python scripts/yang_identify.py --workers 32 --maxiter 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
# the simulator module is chosen by --engine (mujoco: wire_model/yang/sim.py, newton: sim_newton.py)
_pre = argparse.ArgumentParser(add_help=False); _pre.add_argument("--engine", default="mujoco")
_ENGINE = _pre.parse_known_args()[0].engine
if _ENGINE == "newton":
    from wire_model.yang.sim_newton import THETA_BOUNDS, THETA_DEFAULT, THETA_NAMES, theta_dict  # noqa: E402
else:
    from wire_model.yang.sim import OneStep, THETA_BOUNDS, THETA_DEFAULT, THETA_NAMES, theta_dict  # noqa: E402

TRACKS = Path("./data/tracks14")
# One very dynamic clip (the paper picks one trajectory that excites the DLO:
# 102713 moves 30 mm/node/frame, 4x any other clip) + the two clips with the
# cleanest tracking (cable length std 0.4 % / 0.5 % over the clip).
ID_CLIPS = ["20260602_102713", "20260602_170108", "20260602_193329"]
_DATA, _SIM = {}, {}


def _load(clip):
    z = np.load(TRACKS / f"{clip}.npz")
    return {k: z[f"c{k}"] for k in range(4)}, float(z["fps"])


def _task(job):
    """(cand, clip, transition indices, x) -> (cand, [squared err], [err])"""
    cand, clip, idx, x, settle = job
    key = (clip, tuple(np.round(x, 12)))
    if key not in _SIM:
        _SIM.clear()
        shp, fps = _DATA[clip]
        _SIM[key] = OneStep(shp, fps, np.asarray(x), settle=settle)
    o = _SIM[key]
    sq, ab = [], []
    for i in idx:
        try:
            s, a = o.error(i)
        except Exception:
            s, a = np.inf, np.inf
        sq.append(s); ab.append(a)
    return cand, clip, idx, sq, ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join(ID_CLIPS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=int, default=0, help="steps with the movers held at frame i after the reset")
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--out", default="data/wm/yang_ident.json")
    ap.add_argument("--probe", action="store_true", help="evaluate a few hand-picked Theta and exit")
    ap.add_argument("--engine", default="mujoco", choices=["mujoco", "newton"],
                    help="newton: wire_model/yang/sim_newton.py, all (candidate, transition) pairs batched as "
                         "Newton worlds on one GPU (--workers is then unused)")
    ap.add_argument("--batch-cands", type=int, default=20, help="newton: candidates per GPU batch")
    ap.add_argument("--bound", action="append", default=[],
                    help="override a log10 search bound, NAME=LO,HI (e.g. k_stretch=1,8); used when a previous "
                         "run's optimum sat on a bound (user rule: a boundary optimum means the range is too small)")
    a = ap.parse_args()
    out_path = ROOT / a.out
    for b in a.bound:
        name, rng_ = b.split("=")
        lo, hi = map(float, rng_.split(","))
        THETA_BOUNDS[THETA_NAMES.index(name)] = (lo, hi)
        print(f"bound override: {name} log10 in [{lo}, {hi}]", flush=True)
    assert not out_path.exists(), f"{out_path} exists -- never overwrite results"
    from scipy.optimize import differential_evolution
    import multiprocessing as mp

    clips = a.clips.split(",")
    for c in clips:
        _DATA[c] = _load(c)
    trans = [(c, i) for c in clips for i in range(len(_DATA[c][0][0]) - 1)]
    jobs_of = lambda cand, x: [(cand, c, list(range(i0, min(i0 + a.chunk, len(_DATA[c][0][0]) - 1))), tuple(x), a.settle)
                               for c in clips for i0 in range(0, len(_DATA[c][0][0]) - 1, a.chunk)]
    pool = mp.get_context("fork").Pool(a.workers) if a.engine == "mujoco" else None
    log_path = out_path.with_suffix(".log.jsonl")
    hist = []
    t_start = time.time()

    def evaluate(xs):
        if a.engine == "newton":
            return evaluate_newton(xs)
        jobs = [j for ci, x in enumerate(xs) for j in jobs_of(ci, x)]
        sq = [[] for _ in xs]; ab = [[] for _ in xs]; per = [{} for _ in xs]
        for cand, clip, idx, s, b in pool.imap_unordered(_task, jobs):
            sq[cand] += s; ab[cand] += b
            per[cand].setdefault(clip, []).extend(b)
        f = [float(np.mean(v)) if np.isfinite(v).all() else 1.0 for v in sq]
        return f, [float(np.mean(v)) for v in ab], per

    frames = {c: np.stack([_DATA[c][0][k] for k in range(4)], 1).astype(float) for c in clips}
    newton_cache = {}

    def evaluate_newton(xs):
        """every (candidate, transition) pair is one Newton world; batches of
        --batch-cands candidates x all transitions"""
        from wire_model.yang.sim_newton import onestep_errors
        tl = [(frames[c], _DATA[c][1], i) for c, i in trans]
        sq = np.zeros((len(xs), len(tl))); ab = np.zeros((len(xs), len(tl)))
        for c0 in range(0, len(xs), a.batch_cands):
            cs = list(range(c0, min(c0 + a.batch_cands, len(xs))))
            X = np.repeat(np.asarray([xs[c] for c in cs], float), len(tl), axis=0)
            s, b = onestep_errors(newton_cache, tl * len(cs), X, settle=a.settle, device="cuda:0")
            sq[cs] = s.reshape(len(cs), -1); ab[cs] = b.reshape(len(cs), -1)
        f = [float(np.mean(v)) if np.isfinite(v).all() else 1.0 for v in sq]
        per = [{c: [float(ab[ci, j]) for j, (cc, _) in enumerate(trans) if cc == c] for c in clips} for ci in range(len(xs))]
        return f, [float(np.mean(v)) for v in ab], per

    # reference values: scene defaults and the no-change baseline
    f0, a0, per0 = evaluate([THETA_DEFAULT])
    nc = []
    for c in clips:
        S = np.stack([_DATA[c][0][k] for k in range(4)])
        nc += list(np.linalg.norm(np.diff(S, axis=1), axis=3).mean(axis=(0, 2)))
    print(f"{len(trans)} transitions on {clips}; no-change {1000 * np.mean(nc):.2f} mm/node; "
          f"scene defaults {THETA_DEFAULT.round(3).tolist()}: MSE {f0[0]:.3e}, {1000 * a0[0]:.2f} mm/node "
          f"({time.time() - t_start:.0f}s)", flush=True)
    if a.probe:
        if len(THETA_DEFAULT) == 6:
            probes = [THETA_DEFAULT + d for d in (np.array([1, 0, 0, 0, 0, 0.]), np.array([-1, 0, 0, 0, 0, 0.]),
                                                  np.array([0, 0, 1, 1, 0, 0.]), np.array([0, 0, 0, 0, -0.5, 0.]),
                                                  np.array([0, 0, 0, 0, 0, -1.]))]
        else:          # newton: one coordinate at a time, -1 decade (mass: -0.5)
            probes = [THETA_DEFAULT - np.eye(len(THETA_DEFAULT))[i] * (0.5 if THETA_NAMES[i] == "mass_per_m" else 1.0)
                      for i in range(len(THETA_DEFAULT)) if THETA_NAMES[i] != "d_stretch"]
            probes.append(THETA_DEFAULT + np.eye(len(THETA_DEFAULT))[1] * 3)
        f, ab, per = evaluate(probes)
        for x, fi, ai, pi in zip(probes, f, ab, per):
            print(np.round(x, 2).tolist(), f"MSE {fi:.3e}  {1000 * ai:.2f} mm/node  ",
                  {c: round(1000 * float(np.mean(v)), 2) for c, v in pi.items()}, flush=True)
        return

    gen = [0]

    def workers_map(func, xs):        # scipy hands us the whole population at once
        xs = list(xs)
        t = time.time()
        f, ab, _ = evaluate(xs)
        i = int(np.argmin(f))
        gen[0] += 1
        rec = {"gen": gen[0], "best_f": f[i], "best_mm": 1000 * ab[i], "mean_f": float(np.mean(f)),
               "best_x": np.asarray(xs[i]).tolist(), "sec": time.time() - t}
        hist.append(rec)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] eval batch {gen[0]}: {len(xs)} cand, best MSE {f[i]:.4e} "
              f"({1000 * ab[i]:.2f} mm/node), mean {np.mean(f):.4e}  x {np.round(xs[i], 3).tolist()}  ({time.time() - t:.0f}s)",
              flush=True)
        return f

    res = differential_evolution(lambda x: 0.0, THETA_BOUNDS, strategy="best1bin", maxiter=a.maxiter,
                                 popsize=a.popsize, tol=a.tol, seed=a.seed, polish=False, init="latinhypercube",
                                 updating="deferred", workers=workers_map)
    fb, ab, per = evaluate([res.x])
    if pool is not None:
        pool.close(); pool.join()
    out = {"engine": a.engine, "method": "scipy differential_evolution best1bin on Eq. (1), one-step error, observed-state reset",
           "clips": clips, "n_transitions": len(trans), "settle": a.settle, "popsize": a.popsize,
           "maxiter": a.maxiter, "tol": a.tol, "seed": a.seed, "nit": int(res.nit), "nfev": int(res.nfev),
           "message": str(res.message), "theta_log10": res.x.tolist(), "theta": theta_dict(res.x),
           "theta_names": THETA_NAMES, "bounds_log10": THETA_BOUNDS,
           "objective_mse_m2": fb[0], "onestep_mm": 1000 * ab[0],
           "onestep_mm_per_clip": {c: 1000 * float(np.mean(v)) for c, v in per[0].items()},
           "default_theta": theta_dict(THETA_DEFAULT), "default_mse_m2": f0[0], "default_onestep_mm": 1000 * a0[0],
           "default_onestep_mm_per_clip": {c: 1000 * float(np.mean(v)) for c, v in per0[0].items()},
           "nochange_mm": 1000 * float(np.mean(nc)), "history": hist, "seconds": time.time() - t_start}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"identified {theta_dict(res.x)}  one-step {1000 * ab[0]:.2f} mm/node (defaults {1000 * a0[0]:.2f}, "
          f"no-change {1000 * np.mean(nc):.2f}); {res.nit} generations, {res.nfev} evaluations -> {a.out}")


if __name__ == "__main__":
    main()
