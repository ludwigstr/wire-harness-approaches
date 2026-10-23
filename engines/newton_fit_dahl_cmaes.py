"""Newton refit WITH Dahl bending hysteresis -- 31 shared parameters.

Newton implements the Dahl friction model for cable bending hysteresis: a
real harness wire does not spring back after being bent, because its strands
and jacket layers rub internally. That is RATE-INDEPENDENT hysteresis, which
neither an elastic bending term nor a viscous damper can express. It is
enabled per joint via dahl_eps_max / dahl_tau, both of which default to ZERO
-- so every Newton number in this project so far (9.03 mm included) was
produced with hysteresis OFF.

This fit adds Dahl PER JOINT -- 14 eps_max + 14 tau, matching how ke and kd are
already resolved per joint -- to the existing 29 (14 ke + 14 kd + mass),
57 in total, and warm-starts the first 29 from the current best, so the comparison is
"same fit, plus hysteresis".

ORIGINAL DOCSTRING FOLLOWS
"""
"""Shared-parameter full-scene CMA-ES fit with a MEASURED height objective.

Successor of newton_fit_shared_cmaes.py (whose 65 mm cap was an assumed
prior): per explicit user decision (2026-08-23), the fitness now compares
the simulated per-node cable heights against Depth-Anything-V2 height
targets extracted from the raw footage
(scripts/depth_height_targets.py -> data/depth_targets/{clip}.npz,
calibrated per clip via the known 55 mm connector height of the end
nodes).

fitness = xy trajectory mean (mm)
        + Z_WEIGHT * mean over frames/cables/nodes of
          |z_sim - z_target| (mm, finite targets only)

Z_WEIGHT is deliberately below 1: the calibrated targets carry the
scaled-up sensor noise (~5-7 mm raw node noise x ~4-5 calibration factor),
so per-node they are coarse; the systematic signal (lying at ~5 mm vs
flying at ~100 mm) is what the term is meant to transport, and averaging
over ~50 frames x 15 nodes x 4 cables per clip suppresses the noise.

Everything else matches newton_fit_shared_cmaes.py: 29 shared params
(14 log-ke + 14 log-kd + 1 log-mass; the four cables are identical
hardware), full scene in every evaluation, 10 VBD iterations for fit
evals / 20 for the final validation, GPUs 1+2 with 2 workers each,
clean post-hub-fix tracking, resumable checkpoint.
"""
import os
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import cma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from newton_fit_stiffness import CLIP_TRIMS

ROOT = Path(__file__).resolve().parents[1]
# Cable discretisation: SUBDIV rod segments per tracked node pair. The
# fitted per-joint profile always lives on the 14-joint tracked grid; the
# simulation resamples it to its own (15*SUBDIV - 1) joints, which keeps the
# search space independent of resolution.
SUBDIV = int(os.environ.get("WIRE_SUBDIV", "1"))
N_JOINTS = 14
# GPU0 only (2026-08-24, user instruction: GPUs 1+2 are needed by others;
# GPU0 is ours despite the co-resident vLLM server -- if that server
# serves actively, expect the measured ~12x contention slowdown).
GPU_IDS = [int(x) for x in os.environ.get("WIRE_GPUS", "0,1").split(",")]
N_WORKERS = int(os.environ.get("WIRE_WORKERS", "4"))
POPSIZE = int(os.environ.get("WIRE_POPSIZE", "12"))
FIT_SOLVER_ITERATIONS = 10
MAX_GENERATIONS = int(os.environ.get("WIRE_GENS", "12"))
# sigma0 is in LOG space: 1.0 means one standard step multiplies every
# parameter by e = 2.72, i.e. a global re-search. That silently discards a warm
# start -- measured three times: every Dahl run warm-started at a proven
# optimum and ended WORSE than where it began. For refining a known-good point
# use ~0.2 (a factor of 1.22 per step).
SIGMA0 = float(os.environ.get("WIRE_SIGMA0", "1.0"))
X0_KE = 0.5
X0_KD = 0.05
X0_MASS = 0.15
Z_WEIGHT = 0.5
# "perjoint" = 14 eps + 14 tau (57 params), "scalar" = one eps + one tau (31).
# The probe found the SCALAR value beat every profile it tried, and 57
# dimensions searched with the budget of 29 ended worse than its own warm
# start -- so the slim variant is not a fallback, it is the better-posed fit.
DAHL_MODE = os.environ.get("WIRE_DAHL_MODE", "perjoint")
# Per-joint REST ANGLES appended to the vector. They are the one family here
# that must NOT live in log space: a rest angle is signed -- a cable can be set
# to either side -- while every other parameter is a positive stiffness, mass or
# time constant. So the vector is mixed, and CMA-ES gets per-coordinate step
# sizes instead of one sigma for all (CMA_stds), otherwise a step that is
# sensible for a stiffness is absurd for an angle.
FIT_REST = os.environ.get("WIRE_REST", "0") == "1"
# Three globals the pipeline never fitted: axial stiffness, torsional
# stiffness and the cable-table friction. They are strictly positive, so they
# join the log-space core rather than the linear rest-angle block.
# Per-joint TORSIONAL stiffness as a parameter of its own. Up to the
# 45-parameter fit, Newton's twist was never free: add_rod defaults twist to
# bend, and the per-joint loop wrote the fitted bend value into the bend AND
# the twist slot of each joint. Measured on the rod, twist barely moves the
# 2-D metric -- but "coupled to bend by construction" is not a finding, it is
# an untested assumption, so it gets its own 14 values here.
FIT_TWIST = os.environ.get("WIRE_TWIST", "0") == "1"
FIT_EXTRA = os.environ.get("WIRE_EXTRA", "0") == "1"
N_EXTRA = 3 if FIT_EXTRA else 0
N_TWIST = 14 if FIT_TWIST else 0
X0_EXTRA = [1.0e5, 1.0e3, 1.0]        # stretch, twist, mu
REST_STD = float(os.environ.get("WIRE_REST_STD", "0.06"))   # rad
N_DAHL = N_JOINTS if DAHL_MODE == "perjoint" else 1
WARM_FILE = os.environ.get("WIRE_WARM", "data/newton_fit_depth.npz")
X0_DAHL_EPS = float(os.environ.get("WIRE_DAHL_EPS", "0.3"))
X0_DAHL_TAU = float(os.environ.get("WIRE_DAHL_TAU", "1.0"))

TRAIN_CLIPS = list(CLIP_TRIMS.keys())

# /home sits under a user disk quota that can fill up mid-run (it killed a
# 5-generation run on 2026-08-24 and truncated the checkpoint to 0 bytes).
# Checkpoints are therefore written atomically (tmp + os.replace) and fall
# back to FALLBACK_DIR (outside the /home quota) when the write fails.
FALLBACK_DIR = Path(__file__).resolve().parents[1] / "work" / "fallback"


def safe_savez(path, log=None, **arrays):
    # np.savez appends ".npz" to names lacking it, so the tmp name must
    # keep the suffix for os.replace to find the actual file written
    tmp = path.with_name("tmp_" + path.name)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        fb = FALLBACK_DIR / path.name
        np.savez(fb, **arrays)
        if log:
            log(f"WARNING: could not write {path} ({e}) -- checkpoint saved to {fb}")
        return fb

_worker_cache = {}


def _from_x(x):
    """optimiser coordinates -> stored parameter row.

    The core is exponentiated, the rest angles are NOT -- exponentiating a
    signed angle would silently turn every set to one side.
    """
    x = np.asarray(x, float)
    n_core = 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA
    row = list(np.exp(x[:n_core]))
    if FIT_REST:
        row += list(x[n_core:n_core + N_JOINTS])
    if FIT_TWIST:                       # positive stiffness -> log space again
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        row += list(np.exp(x[t0:t0 + N_TWIST]))
    return np.asarray(row, float)


def _pad_extra(row):
    """A warm start from a 45-value run has no extra block -- append X0."""
    row = np.asarray(row, float)
    want = 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA + (N_JOINTS if FIT_REST else 0)
    if N_EXTRA and len(row) == want - N_EXTRA:
        core = 2 * N_JOINTS + 1 + 2 * N_DAHL
        row = np.concatenate([row[:core], X0_EXTRA, row[core:]])
    return row


def _to_x(row):
    """stored parameter row -> optimiser coordinates (log core + linear rest)."""
    row = _pad_extra(row)
    n_core = 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA
    x = list(np.log(row[:n_core]))
    if FIT_REST:
        r = row[n_core:n_core + N_JOINTS]
        x += list(r if len(r) == N_JOINTS else np.zeros(N_JOINTS))
    if FIT_TWIST:
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        t = row[t0:t0 + N_TWIST]
        # a warm row without a twist block starts coupled, i.e. twist := bend
        x += list(np.log(t if len(t) == N_TWIST else row[:N_JOINTS]))
    return x


def _init_worker(gpu_id):
    # Pass the device EXPLICITLY instead of juggling CUDA_VISIBLE_DEVICES.
    # Setting that variable inside the worker is a timing trap: CUDA picks its
    # device when the context is created, and if anything touched CUDA first
    # the override is silently ignored -- measured, 12 workers all landed on
    # GPU 0 despite WIRE_GPUS=1,2. The shell restricts WHICH GPUs are visible;
    # this picks one of them by warp device index.
    _worker_cache["device"] = f"cuda:{gpu_id}"


def _clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    from mujoco_scene import DIR
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fps = (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def _params_from_log(log_row):
    v = np.asarray(log_row, float)
    n_core = 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA
    rest = v[n_core:n_core + N_JOINTS] if FIT_REST else None
    row = np.exp(v[:n_core])
    ke = np.tile(row[:N_JOINTS], (4, 1))
    kd = np.tile(row[N_JOINTS:2 * N_JOINTS], (4, 1))
    mass = np.full(4, row[2 * N_JOINTS])
    # Dahl PER JOINT, like ke/kd -- 14 eps + 14 tau, shared across the four
    # (physically identical) cables.
    e0 = 2 * N_JOINTS + 1
    if DAHL_MODE == "perjoint":
        eps = np.tile(row[e0:e0 + N_JOINTS], (4, 1))
        tau = np.tile(row[e0 + N_JOINTS:e0 + 2 * N_JOINTS], (4, 1))
    else:
        eps = np.full((4, N_JOINTS), float(row[e0]))
        tau = np.full((4, N_JOINTS), float(row[e0 + 1]))
    extra = row[n_core - N_EXTRA:n_core] if N_EXTRA else None
    kt = None
    if FIT_TWIST:
        t0 = n_core + (N_JOINTS if FIT_REST else 0)
        kt = np.tile(np.exp(v[t0:t0 + N_TWIST]), (4, 1))
    return ke, kd, mass, eps, tau, rest, extra, kt


def _z_loss_mm(z_nodes, targets_mm):
    """Mean |z_sim - z_target| in mm over finite targets.

    z_nodes: list of (4, n_bodies) arrays in meters, one per frame.
    targets_mm: (T, 4, 16) calibrated depth targets; only the first
    n_bodies node columns are used (matches the xy metric's convention of
    scoring bodies against shp[k][t][:len(bodies)]).
    """
    zs = np.stack(z_nodes) * 1000.0                    # (T, 4, n_bodies)
    tg = targets_mm[:zs.shape[0], :, :zs.shape[2]]
    m = np.isfinite(tg)
    if not m.any():
        return 0.0
    return float(np.abs(zs[m] - tg[m]).mean())


def _eval_job(args):
    (log_row,) = args
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    if "clips" not in _worker_cache:
        _worker_cache["clips"] = {}
        _worker_cache["targets"] = {}
    clips = _worker_cache["clips"]
    targets = _worker_cache["targets"]
    for c in TRAIN_CLIPS:
        if c not in clips:
            t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
            shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
            clips[c] = (shp, fps)
            targets[c] = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
    ke, kd, mass, eps, tau, rest, extra, kt = _params_from_log(log_row)
    xy_errs, z_errs = [], []
    for c in TRAIN_CLIPS:
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt,
                         mass_per_m=mass,
                         **({} if extra is None else dict(
                             stretch_stiffness=float(extra[0]),
                             twist_stiffness=float(extra[1]),
                             cable_mu=float(extra[2]))),
                         shp_fps=clips[c], solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, subdiv=SUBDIV,
                         dahl_eps_max=eps, dahl_tau=tau,
                         per_joint_rest=(None if rest is None
                                         else np.tile(rest, (4, 1))),
                         device=_worker_cache.get("device"))
        xy_errs.append(float(np.mean(errs)))
        z_errs.append(_z_loss_mm(z_out, targets[c]))
    return float(np.mean(xy_errs)), float(np.mean(z_errs))


def main():
    log_path = ROOT / os.environ.get("WIRE_FITLOG", "data/newton_fit_dahl_log.jsonl")
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            log_f.write(line + "\n")
            log_f.flush()
        except OSError:
            pass  # quota full: stdout (captured by the watchdog log) survives

    # fail fast if the depth targets are incomplete
    missing = [c for c in TRAIN_CLIPS if not (ROOT / f"data/depth_targets/{c}.npz").exists()]
    if missing:
        log(f"ERROR: depth targets missing for {missing} -- run depth_height_targets.py first")
        sys.exit(1)

    log("=== Newton shared-parameter full-scene CMA-ES fit (Depth-Anything z objective) started ===")
    log(f"popsize={POPSIZE}, max_generations={MAX_GENERATIONS}, workers={N_WORKERS} on GPUs {GPU_IDS}")
    n_par = (29 + 2 * N_DAHL + N_EXTRA + (N_JOINTS if FIT_REST else 0) + N_TWIST)
    what = (f"29 + Dahl {DAHL_MODE}"
            + (" + stretch/twist/mu" if N_EXTRA else "")
            + (f" + {N_JOINTS} Ruhewinkel" if FIT_REST else "")
            + (f" + {N_TWIST} Torsionssteifigkeiten je Gelenk" if N_TWIST else ""))
    log(f"{n_par} shared params ({what}); x0: ke={X0_KE}, kd={X0_KD}, mass={X0_MASS} kg/m; "
        f"fitness = xy mean + {Z_WEIGHT} * |z_sim - z_depth| (mm)")

    ctx = mp.get_context("spawn")
    pools = [ctx.Pool(N_WORKERS // len(GPU_IDS), initializer=_init_worker, initargs=(g,))
            for g in GPU_IDS]

    def eval_population(log_rows):
        async_results = []
        for i, row in enumerate(log_rows):
            pool = pools[i % len(pools)]
            async_results.append(pool.apply_async(_eval_job, ((row,),)))
        return [r.get() for r in async_results]

    out_path = ROOT / os.environ.get("WIRE_OUT", "data/newton_fit_dahl.npz")
    # warm start: the 29 fitted parameters from the current best, plus the
    # two Dahl parameters at the value the probe liked
    warm = ROOT / WARM_FILE
    if warm.exists():
        prev = np.asarray(np.load(warm, allow_pickle=True)["best"], float)
        core = prev[:2 * N_JOINTS + 1]
        e0 = 2 * N_JOINTS + 1
        if len(prev) >= e0 + 2 * N_JOINTS:          # warm file carries per-joint Dahl
            pe, pt = prev[e0:e0 + N_JOINTS], prev[e0 + N_JOINTS:e0 + 2 * N_JOINTS]
        elif len(prev) >= e0 + 2:                    # ... or scalar Dahl
            pe = np.full(N_JOINTS, prev[e0])
            pt = np.full(N_JOINTS, prev[e0 + 1])
        else:                                        # ... or none at all
            pe = np.full(N_JOINTS, X0_DAHL_EPS)
            pt = np.full(N_JOINTS, X0_DAHL_TAU)
        if N_DAHL == 1:                              # collapse a profile to one value
            pe, pt = np.array([float(np.exp(np.mean(np.log(pe))))]), \
                     np.array([float(np.exp(np.mean(np.log(pt))))])
        x0 = np.log(np.concatenate([core, pe, pt])).tolist()
        if N_EXTRA:
            # a 45-value warm file has no extra block: start it at X0_EXTRA
            px = (prev[2 * N_JOINTS + 1 + 2 * N_DAHL:][:N_EXTRA]
                  if len(prev) >= 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA
                     + (N_JOINTS if FIT_REST else 0) else np.array(X0_EXTRA))
            x0 += list(np.log(px if len(px) == N_EXTRA else X0_EXTRA))
        if FIT_REST:
            tail = prev[2 * N_JOINTS + 1 + 2 * N_DAHL:]
            if N_EXTRA and len(tail) >= N_EXTRA + N_JOINTS:
                tail = tail[N_EXTRA:]               # skip the extra block
            prev_rest = tail if len(tail) >= N_JOINTS else np.zeros(N_JOINTS)
            x0 += list(prev_rest[:N_JOINTS]) + [0.0] * max(
                0, N_JOINTS - len(prev_rest))
        if FIT_TWIST:
            tail = prev[2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA:]
            prev_t = tail[N_JOINTS:N_JOINTS + N_TWIST] if FIT_REST else tail[:N_TWIST]
            # no twist block in the warm file -> start exactly where row 11
            # sits: twist = bend, which makes this fit a strict superset
            x0 += list(np.log(prev_t if len(prev_t) == N_TWIST else core[:N_JOINTS]))
        print(f"warmstart aus {warm.name} ({len(prev)} param) -> "
              f"{len(x0)} param, Dahl-modus {DAHL_MODE}", flush=True)
    else:
        x0 = ([np.log(X0_KE)] * N_JOINTS + [np.log(X0_KD)] * N_JOINTS
              + [np.log(X0_MASS)] + [np.log(X0_DAHL_EPS)] * N_DAHL
              + [np.log(X0_DAHL_TAU)] * N_DAHL)
    opts = {"popsize": POPSIZE, "verbose": -9}
    if FIT_REST:
        # one step size per coordinate: log-space for the stiffnesses, radians
        # for the angles. The rest block is NOT the tail any more once twist
        # is fitted, so the three blocks are spelled out instead of counted
        # from the end.
        n_core = 2 * N_JOINTS + 1 + 2 * N_DAHL + N_EXTRA
        opts["CMA_stds"] = ([SIGMA0] * n_core + [REST_STD] * N_JOINTS
                            + [SIGMA0] * N_TWIST)
        assert len(opts["CMA_stds"]) == len(x0), (len(opts["CMA_stds"]), len(x0))
        es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
    else:
        es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
    gen = 0
    gen_history = []
    t_start = time.time()
    while not es.stop() and gen < MAX_GENERATIONS:
        t_g = time.time()
        solutions = es.ask()
        results = eval_population(solutions)
        fitnesses = [xy + Z_WEIGHT * ze for xy, ze in results]
        es.tell(solutions, fitnesses)
        gen += 1
        i_best = int(np.argmin(fitnesses))
        gen_history.append({"gen": gen, "best": float(fitnesses[i_best]),
                            "best_xy": results[i_best][0], "best_z": results[i_best][1],
                            "mean_fit": float(np.mean(fitnesses))})
        log(f"gen {gen}/{MAX_GENERATIONS}: best={fitnesses[i_best]:.2f} "
            f"(xy={results[i_best][0]:.2f}, z-err={results[i_best][1]:.2f}mm) "
            f"mean={np.mean(fitnesses):.2f} ({time.time()-t_g:.0f}s)")
        best_row = _from_x(es.result.xbest)
        safe_savez(out_path, log=log, best=best_row, history=json.dumps(gen_history))

    best_row = _from_x(es.result.xbest)
    log(f"fit done: {gen} generations, {time.time()-t_start:.0f}s, "
        f"ke={np.round(best_row[:N_JOINTS],3).tolist()}, "
        f"kd={np.round(best_row[N_JOINTS:2*N_JOINTS],4).tolist()}, "
        f"mass={best_row[2*N_JOINTS]:.4f} kg/m")
    for p in pools:
        p.close(); p.join()

    log("=== final validation: fitted shared params, full scene, 10 iterations (IN-SAMPLE) ===")
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    ke, kd, mass, eps, tau, rest, extra, kt = _params_from_log(_to_x(best_row))
    per_clip, per_clip_z = {}, {}
    for c in TRAIN_CLIPS:
        t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
        shp, t0_, t1_, fps = track_clip(c, t0=8, t1=t1)
        tg = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt,
                         mass_per_m=mass,
                         **({} if extra is None else dict(
                             stretch_stiffness=float(extra[0]),
                             twist_stiffness=float(extra[1]),
                             cable_mu=float(extra[2]))),
                         shp_fps=(shp, fps), z_nodes_out=z_out, subdiv=SUBDIV,
                         # explicit: a long-lived process keeps the scene module it
                         # imported at launch, defaults included (2026-10-21)
                         solver_iterations=FIT_SOLVER_ITERATIONS,
                         dahl_eps_max=eps, dahl_tau=tau,
                         per_joint_rest=(None if rest is None
                                         else np.tile(rest, (4, 1))))
        per_clip[c] = float(np.mean(errs))
        per_clip_z[c] = _z_loss_mm(z_out, tg)
    final_avg = float(np.mean(list(per_clip.values())))
    final_z = float(np.mean(list(per_clip_z.values())))
    safe_savez(out_path, log=log, best=best_row, history=json.dumps(gen_history),
              final_avg=final_avg, final_z_err=final_z,
              per_clip=json.dumps(per_clip), per_clip_z=json.dumps(per_clip_z))
    log(f"final result: xy {final_avg:.2f} mm/node, z-err vs depth {final_z:.2f} mm (IN-SAMPLE)")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} xy {v:6.1f}  z-err {per_clip_z[c]:6.1f}")
    log(f"-> {out_path}")
    log("=== DONE ===")
    print("NEWTON_FIT_DEPTH_DONE")


if __name__ == "__main__":
    main()
