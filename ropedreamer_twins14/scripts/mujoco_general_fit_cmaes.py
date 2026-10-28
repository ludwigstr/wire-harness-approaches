"""MuJoCo GENERAL fit: everything MuJoCo exposes for the cable, per cable where it
is per joint, mass free per cable but bounded -- the counterpart of Newton
row 10 (user, 2026-10-25: "so einen General-Fit auch fuer Mujoco").

    spring stiffness   56  (4 cables x 14 ball joints, log)      jnt_stiffness
    rest angle         56  (deg, about the board normal)         qpos_spring
    joint damping      56  (log)                                 dof_damping (3 dofs per ball joint)
    plugin bend         4  (per cable, log)                      elasticity.cable config
    plugin twist        4  (per cable, log)
    mu slide            1  (log, cables + board)
    mass                4  (per cable, log, bounded 0.02 .. 0.2 kg/m)
                      181
The mass is free but may not degenerate (user, 2026-10-25): bounded to a range
that holds both the measured 0.044 and MuJoCo's swept 0.15 kg/m.

Warm start = row 4 (jnt_spring_rest_fit.npz: springs + rest angles) with the
plugin's 1e5 / 1e5, damping 0.15, mu 1.0 and row 4's 0.15 kg/m; generation 0
therefore reproduces row 4 (8.54). Fitness = the table metric itself (rollout_errs, every frame,
all 14 clips, mean over cables and clips). Tracks: the cached 14-clip tracks of
the approaches repository (identical to track_clip's output).

    python scripts/mujoco_general_fit_cmaes.py --gens 40 --popsize 18 --workers 20
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_set import CLIP_TRIMS                                            # noqa: E402
from mujoco_scene import build_xml, cable_ball_joints, set_per_joint_stiffness  # noqa: E402
from fit_jnt_spring_rest_cmaes import board_normal_axes, rest_quats        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRACKS = Path("./data/tracks14")
N_C, N_J = 4, 14
N_PJ = N_C * N_J
MASS0, MASS_LO, MASS_HI = 0.15, 0.02, 0.2
RADIUS = 0.003
I_K, I_REST, I_DAMP = 0, N_PJ, 2 * N_PJ
I_BEND, I_TWIST, I_MU = 3 * N_PJ, 3 * N_PJ + N_C, 3 * N_PJ + 2 * N_C
I_MASS = I_MU + 1
N_PAR = I_MASS + N_C                                                        # 181
_CACHE, _AXIS = {}, {}


def unpack(x):
    x = np.asarray(x, float)
    pj = lambda i: x[i:i + N_PJ].reshape(N_C, N_J)
    return dict(k=np.exp(pj(I_K)), rest_deg=pj(I_REST), damp=np.exp(pj(I_DAMP)),
                bend=np.exp(x[I_BEND:I_BEND + N_C]), twist=np.exp(x[I_TWIST:I_TWIST + N_C]),
                mu=float(np.exp(x[I_MU])), mass=np.exp(x[I_MASS:I_MASS + N_C]))


def make_model(shp, p):
    """compiled model of one clip with the parameter set p (unpack output)"""
    import mujoco
    xml = build_xml(shp, mass_per_m=MASS0, mu=p["mu"], radius=RADIUS)
    # one elasticity.cable plugin and one capsule density per cable, in cable order:
    # give each cable its own bend, twist and mass
    it = iter(p["mass"] / (np.pi * RADIUS ** 2))
    xml = re.sub(r'density="[^"]*"', lambda _m: f'density="{next(it):.0f}"', xml)
    for key in ("twist", "bend"):
        it = iter(p[key])
        xml = re.sub(rf'<config key="{key}" value="[^"]*"/>',
                     lambda _m: f'<config key="{key}" value="{next(it):g}"/>', xml)
    m = mujoco.MjModel.from_xml_string(xml)
    clip_axes = board_normal_axes(m)
    set_per_joint_stiffness(m, p["k"], rest_quat=rest_quats(p["rest_deg"], clip_axes))
    ids = cable_ball_joints(m)
    for c in range(N_C):
        for j in range(N_J):
            a = m.jnt_dofadr[ids[c, j]]
            m.dof_damping[a:a + 3] = p["damp"][c, j]
    return m


def _eval_one(job):
    cand_i, clip, x = job
    from mujoco_full_loco import rollout_errs
    shp, fps = _CACHE[clip]
    try:
        m = make_model(shp, unpack(x))
        e = float(rollout_errs(m, shp, fps, per_cable=True).mean(0).mean())
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:                       # an unstable candidate must not kill the run
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/jnt_spring_rest_fit.npz")
    ap.add_argument("--resume", default="", help="continue from an earlier run's best_x (its .npz)")
    ap.add_argument("--mass-max", type=float, default=MASS_HI,
                    help="upper mass bound, kg/m (user, 2026-10-27: 0.5 after 0.2 was hit)")
    ap.add_argument("--sigma0", type=float, default=1.0, help="CMA step scale (0.5 to refine a resumed run)")
    ap.add_argument("--out", default="data/mujoco_general_fit.npz")
    ap.add_argument("--log", default="data/mujoco_general_fit_log.jsonl")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-deg", type=float, default=40.0)
    ap.add_argument("--smoke", action="store_true", help="evaluate the warm start on 2 clips and exit")
    a = ap.parse_args()
    import cma
    import multiprocessing as mp

    log_f = open(ROOT / a.log, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    clips = sorted(CLIP_TRIMS)
    for c in clips:
        z = np.load(TRACKS / f"{c}.npz")
        _CACHE[c] = ({k: z[f"c{k}"] for k in range(4)}, float(z["fps"]))

    init = np.load(ROOT / a.init, allow_pickle=True)
    x0 = np.concatenate([np.log(np.asarray(init["best"], float)).ravel(),
                         np.asarray(init["rest_deg"], float).ravel(),
                         np.full(N_PJ, np.log(0.15)),
                         np.full(N_C, np.log(1e5)), np.full(N_C, np.log(1e5)), [0.0],
                         np.full(N_C, np.log(MASS0))])
    if a.resume:
        x0 = np.asarray(np.load(ROOT / a.resume, allow_pickle=True)["best_x"], float)
    assert len(x0) == N_PAR
    if a.smoke:
        for c in clips[:2]:
            print(_eval_one((0, c, x0)))
        return

    stds = [0.2] * N_PJ + [3.0] * N_PJ + [0.3] * N_PJ + [0.5] * (2 * N_C) + [0.3] + [0.2] * N_C
    lo = ([-np.inf] * N_PJ + [-a.max_deg] * N_PJ + [np.log(1e-4)] * N_PJ + [-np.inf] * (2 * N_C) + [np.log(0.05)]
          + [np.log(MASS_LO)] * N_C)
    hi = ([np.inf] * N_PJ + [a.max_deg] * N_PJ + [np.log(5.0)] * N_PJ + [np.inf] * (2 * N_C) + [np.log(5.0)]
          + [np.log(a.mass_max)] * N_C)
    log(f"=== MuJoCo GENERAL fit: {N_PAR} params, per cable, mass free per cable in {MASS_LO} .. {a.mass_max} kg/m (CMA-ES) ===")
    log(f"popsize {a.popsize}, gens {a.gens}, workers {a.workers}; " + (f"resumed from {a.resume}, sigma0 {a.sigma0}; " if a.resume else f"warm start {a.init} (row 4) + bend/twist 1e5, damping 0.15, mu 1.0; ")
        + f"fitness = table metric over {len(clips)} clips")

    pool = mp.get_context("fork").Pool(a.workers)

    def evaluate(xs):
        acc = [[] for _ in xs]
        for i, c, e in pool.map(_eval_one, [(i, c, x) for i, x in enumerate(xs) for c in clips]):
            acc[i].append(e)
        return [float(np.mean(v)) for v in acc]

    def save(best_x, best_f, f0, hist):
        p = unpack(best_x)
        np.savez(ROOT / a.out, best_x=best_x, best_f=best_f, base_f=f0, history=json.dumps(hist), **p)

    t0 = time.time()
    f0 = evaluate([x0])[0]
    log(f"gen 0 ({'resumed best' if a.resume else f'row 4 at mass {MASS0}'}): {f0:.3f} mm/node ({time.time() - t0:.0f}s)")
    hist = [{"gen": 0, "best": f0}]
    best_f, best_x = f0, x0.copy()
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, {"popsize": a.popsize, "verbose": -9, "CMA_stds": stds, "bounds": [lo, hi]})
    for gen in range(1, a.gens + 1):
        t_g = time.time()
        xs = es.ask()
        fs = evaluate(xs)
        es.tell(xs, fs)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], float).copy()
        p = unpack(best_x)
        hist.append({"gen": gen, "best": float(fs[i]), "mean": float(np.mean(fs)), "best_ever": best_f})
        log(f"gen {gen}/{a.gens}: best {fs[i]:.3f}  mean {np.mean(fs):.3f}  best_ever {best_f:.3f}  ({time.time() - t_g:.0f}s)  "
            f"k {p['k'].min():.3f}..{p['k'].max():.2f}  rest {p['rest_deg'].min():+.1f}..{p['rest_deg'].max():+.1f}  "
            f"damp {p['damp'].min():.3f}..{p['damp'].max():.2f}  mu {p['mu']:.2f}  mass {np.round(p['mass'], 4).tolist()}")
        save(best_x, best_f, f0, hist)

    log(f"fit done: {best_f:.3f} (gen 0 {f0:.3f}), {time.time() - t0:.0f}s")
    # final validation in the same pool, per clip (fresh models from the stored x)
    per_clip = {c: e for _, c, e in pool.map(_eval_one, [(0, c, best_x) for c in clips])}
    pool.close(); pool.join()
    v = np.array(list(per_clip.values()))
    np.savez(ROOT / a.out, best_x=best_x, best_f=best_f, base_f=f0, history=json.dumps(hist),
             per_clip=json.dumps(per_clip), final_avg=v.mean(), final_median=np.median(v), **unpack(best_x))
    log(f"final result: {v.mean():.3f} mm/node (median {np.median(v):.3f}), 14 clips IN-SAMPLE")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
