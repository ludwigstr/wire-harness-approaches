"""Newton full-scene CMA-ES fit with the MASS PINNED and the three never-fitted
globals freed: axial stiffness, torsional friction, rolling friction.

Why this fit exists (user decision 2026-10-20):
  * The fitted linear density grew 1.46 -> 2.03 -> 2.70 kg/m over rows 8, 9,
    9a, for a cable that weighs ~0.044 kg/m (32 g over 0.729 m). Mass and the
    stiffness scale are redundant to the optimiser, so it used mass as a free
    knob. Here it is PINNED to the measured value and the stiffnesses have to
    carry the dynamics on their own.
  * K stretch sat at 1e5 in every Newton row and was never fitted. The
    iteration study showed the error rising monotonically with the VBD
    iteration count, i.e. residual axial compliance of the under-converged
    solver was doing real work. If that softness is physical, this is the
    parameter that should absorb it.
  * mu_torsional (0.005) and mu_rolling (0.0001) are Newton's library defaults
    on every shape and were never moved.

Warm start: the 59-parameter twist set (row 9a). Because mass and stiffness
are redundant, ke, kd, kt and K stretch are RESCALED by mass_pin / mass_warm
so the warm point keeps every stiffness-to-inertia ratio -- what changes is
gravity's share, which is exactly what the fit then has to resolve. x0 is
evaluated explicitly as "gen 0" so the cost of pinning the mass is on record
before any coordinate moves.

Vector layout (CMA-ES coordinates):
  log core : ke x14, kd x14, dahl eps, dahl tau, K stretch,
             shape_mu, cable_mu, mu_tors, mu_roll              (35)
  linear   : rest angle x14                                      (14)
  log      : twist stiffness x14                                 (14)
Stored rows carry the same order with the core exponentiated. Mass is not in
the vector; the pinned value is stored alongside.

Fitness: xy mean over all frames of all 14 clips + 0.5 * |z_sim - z_depth|,
as in every other Newton fit here. 10 VBD iterations, the table's protocol.

  WIRE_MASS=0.044 WIRE_WARM=data/newton_fit_twist.npz WIRE_GENS=14 \\
  WIRE_POPSIZE=12 WIRE_WORKERS=4 WIRE_GPUS=0,1,2 \\
  WIRE_OUT=data/newton_fit_physical.npz WIRE_FITLOG=data/newton_fit_physical_log.jsonl \\
  python3 -u scripts/newton_fit_physical_cmaes.py
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cma
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_set import CLIP_TRIMS   # noqa: E402  -- the 14, nothing else

ROOT = Path(__file__).resolve().parents[1]
N_JOINTS = 14
GPU_IDS = [int(x) for x in os.environ.get("WIRE_GPUS", "0,1,2").split(",")]
N_WORKERS = int(os.environ.get("WIRE_WORKERS", "4"))
POPSIZE = int(os.environ.get("WIRE_POPSIZE", "12"))
MAX_GENERATIONS = int(os.environ.get("WIRE_GENS", "14"))
FIT_SOLVER_ITERATIONS = 10
Z_WEIGHT = 0.5
MASS_PIN = float(os.environ.get("WIRE_MASS", "0.044"))     # kg/m, measured
WARM_FILE = os.environ.get("WIRE_WARM", "data/newton_fit_twist.npz")
SIGMA0 = float(os.environ.get("WIRE_SIGMA0", "0.2"))       # log, refine
GLOB_STD = float(os.environ.get("WIRE_GLOB_STD", "0.5"))   # log, the 5 globals
REST_STD = float(os.environ.get("WIRE_REST_STD", "0.06"))  # rad
X0_STRETCH = 1.0e5
X0_GLOB = [1.0, 1.0, 0.005, 0.0001]        # shape_mu, cable_mu, mu_tors, mu_roll
N_CORE = 2 * N_JOINTS + 2 + 1 + 4          # 35
TRAIN_CLIPS = list(CLIP_TRIMS.keys())
FALLBACK_DIR = Path(__file__).resolve().parents[1] / "work" / "fallback"

_worker_cache = {}


def safe_savez(path, log=None, **arrays):
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
            log(f"WARNING: could not write {path} ({e}) -- saved to {fb}")
        return fb


def from_x(x):
    """optimiser coordinates -> stored row (core exponentiated, rest linear)."""
    x = np.asarray(x, float)
    return np.concatenate([np.exp(x[:N_CORE]),
                           x[N_CORE:N_CORE + N_JOINTS],
                           np.exp(x[N_CORE + N_JOINTS:])])


def to_x(row):
    row = np.asarray(row, float)
    return np.concatenate([np.log(row[:N_CORE]),
                           row[N_CORE:N_CORE + N_JOINTS],
                           np.log(row[N_CORE + N_JOINTS:])])


def unpack(row):
    """stored row -> the eval_clip keyword set."""
    row = np.asarray(row, float)
    ke = np.tile(row[:N_JOINTS], (4, 1))
    kd = np.tile(row[N_JOINTS:2 * N_JOINTS], (4, 1))
    e0 = 2 * N_JOINTS
    eps = np.full((4, N_JOINTS), row[e0])
    tau = np.full((4, N_JOINTS), row[e0 + 1])
    stretch, shape_mu, cable_mu, mu_tors, mu_roll = row[e0 + 2:e0 + 7]
    rest = np.tile(row[N_CORE:N_CORE + N_JOINTS], (4, 1))
    kt = np.tile(row[N_CORE + N_JOINTS:N_CORE + 2 * N_JOINTS], (4, 1))
    return dict(per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt,
                mass_per_m=np.full(4, MASS_PIN),
                dahl_eps_max=eps, dahl_tau=tau, per_joint_rest=rest,
                stretch_stiffness=float(stretch), shape_mu=float(shape_mu),
                cable_mu=float(cable_mu), mu_tors=float(mu_tors),
                mu_roll=float(mu_roll))


def warm_row(path):
    """Build x0's stored row from the 59-value twist set, mass pinned.

    Twist layout: ke14, kd14, mass, eps, tau, rest14, twist14.
    """
    prev = np.asarray(np.load(path, allow_pickle=True)["best"], float)
    assert len(prev) == 59, f"expected the 59-value twist set, got {len(prev)}"
    ke, kd, mass = prev[:14], prev[14:28], prev[28]
    eps, tau = prev[29], prev[30]
    rest, kt = prev[31:45], prev[45:59]
    s = MASS_PIN / mass                      # stiffness-to-inertia ratio kept
    row = np.concatenate([ke * s, kd * s, [eps, tau, X0_STRETCH * s], X0_GLOB,
                          rest, kt * s])
    return row, mass, s


def _init_worker(gpu_id):
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


def _z_loss_mm(z_nodes, targets_mm):
    zs = np.stack(z_nodes) * 1000.0
    tg = targets_mm[:zs.shape[0], :, :zs.shape[2]]
    m = np.isfinite(tg)
    return float(np.abs(zs[m] - tg[m]).mean()) if m.any() else 0.0


def _eval_job(args):
    (x,) = args
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    if "clips" not in _worker_cache:
        _worker_cache["clips"], _worker_cache["targets"] = {}, {}
    clips, targets = _worker_cache["clips"], _worker_cache["targets"]
    for c in TRAIN_CLIPS:
        if c not in clips:
            shp, t0, t1_, fps = track_clip(c, t0=8, t1=_clip_frame_t1(c, CLIP_TRIMS[c]))
            clips[c] = (shp, fps)
            targets[c] = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
    kw = unpack(from_x(x))
    xy_errs, z_errs = [], []
    for c in TRAIN_CLIPS:
        z_out = []
        errs = eval_clip(c, shp_fps=clips[c], solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, device=_worker_cache.get("device"), **kw)
        xy_errs.append(float(np.mean(errs)))
        z_errs.append(_z_loss_mm(z_out, targets[c]))
    return float(np.mean(xy_errs)), float(np.mean(z_errs))


def main():
    log_path = ROOT / os.environ.get("WIRE_FITLOG", "data/newton_fit_physical_log.jsonl")
    out_path = ROOT / os.environ.get("WIRE_OUT", "data/newton_fit_physical.npz")
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            log_f.write(line + "\n"); log_f.flush()
        except OSError:
            pass

    missing = [c for c in TRAIN_CLIPS if not (ROOT / f"data/depth_targets/{c}.npz").exists()]
    if missing:
        log(f"ERROR: depth targets missing for {missing}"); sys.exit(1)

    row0, mass_warm, s = warm_row(ROOT / WARM_FILE)
    x0 = to_x(row0)
    log("=== Newton PHYSICAL fit: mass pinned, K stretch + 4 frictions freed (CMA-ES) ===")
    log(f"{len(x0)} params on {len(TRAIN_CLIPS)} clips; mass pinned {MASS_PIN} kg/m "
        f"(warm set had {mass_warm:.3f}, stiffnesses rescaled x{s:.4f}); "
        f"K stretch x0 {X0_STRETCH * s:.0f}; frictions x0 {X0_GLOB}")
    log(f"popsize={POPSIZE}, gens={MAX_GENERATIONS}, workers={N_WORKERS} on GPUs {GPU_IDS}, "
        f"stds: core {SIGMA0}, globals {GLOB_STD}, rest {REST_STD} rad, twist {SIGMA0}")

    ctx = mp.get_context("spawn")
    pools = [ctx.Pool(max(1, N_WORKERS // len(GPU_IDS)), initializer=_init_worker,
                      initargs=(g,)) for g in GPU_IDS]

    def eval_population(xs):
        res = [pools[i % len(pools)].apply_async(_eval_job, ((x,),)) for i, x in enumerate(xs)]
        return [r.get() for r in res]

    # gen 0: the warm point itself, so the cost of pinning the mass is on record
    t_g = time.time()
    xy0, z0 = eval_population([x0])[0]
    log(f"gen 0/{MAX_GENERATIONS} (x0, nothing moved): fit={xy0 + Z_WEIGHT * z0:.2f} "
        f"(xy={xy0:.2f}, z-err={z0:.2f}mm) ({time.time() - t_g:.0f}s)")
    gen_history = [{"gen": 0, "best": xy0 + Z_WEIGHT * z0, "best_xy": xy0, "best_z": z0}]
    safe_savez(out_path, log=log, best=row0, mass_pin=MASS_PIN,
               history=json.dumps(gen_history))

    stds = ([SIGMA0] * (2 * N_JOINTS + 2) + [GLOB_STD] * 5
            + [REST_STD] * N_JOINTS + [SIGMA0] * N_JOINTS)
    assert len(stds) == len(x0)
    es = cma.CMAEvolutionStrategy(x0, 1.0, {"popsize": POPSIZE, "verbose": -9,
                                            "CMA_stds": stds})
    gen, t_start = 0, time.time()
    best_fit, best_row = xy0 + Z_WEIGHT * z0, row0
    while not es.stop() and gen < MAX_GENERATIONS:
        t_g = time.time()
        sols = es.ask()
        results = eval_population(sols)
        fits = [xy + Z_WEIGHT * ze for xy, ze in results]
        es.tell(sols, fits)
        gen += 1
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit, best_row = fits[i], from_x(sols[i])
        gen_history.append({"gen": gen, "best": float(fits[i]), "best_xy": results[i][0],
                            "best_z": results[i][1], "mean_fit": float(np.mean(fits))})
        r = from_x(sols[i])
        log(f"gen {gen}/{MAX_GENERATIONS}: best={fits[i]:.2f} (xy={results[i][0]:.2f}, "
            f"z-err={results[i][1]:.2f}mm) mean={np.mean(fits):.2f} ({time.time() - t_g:.0f}s)"
            f"  K_stretch={r[30]:.0f} mu(shape,cable,tors,roll)="
            f"{np.round(r[31:35], 4).tolist()}")
        safe_savez(out_path, log=log, best=best_row, mass_pin=MASS_PIN,
                   history=json.dumps(gen_history))

    for p in pools:
        p.close(); p.join()
    log(f"fit done: {gen} generations, {time.time() - t_start:.0f}s; best fit {best_fit:.2f} "
        f"(gen 0 was {xy0 + Z_WEIGHT * z0:.2f})")

    log("=== final validation: 10 iterations, all 14 clips (IN-SAMPLE) ===")
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    kw = unpack(best_row)
    per_clip, per_clip_z = {}, {}
    for c in TRAIN_CLIPS:
        shp, t0_, t1_, fps = track_clip(c, t0=8, t1=_clip_frame_t1(c, CLIP_TRIMS[c]))
        tg = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
        z_out = []
        errs = eval_clip(c, shp_fps=(shp, fps), solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, **kw)
        per_clip[c] = float(np.mean(errs))
        per_clip_z[c] = _z_loss_mm(z_out, tg)
    final_avg = float(np.mean(list(per_clip.values())))
    final_z = float(np.mean(list(per_clip_z.values())))
    safe_savez(out_path, log=log, best=best_row, mass_pin=MASS_PIN,
               history=json.dumps(gen_history), final_avg=final_avg,
               final_z_err=final_z, per_clip=json.dumps(per_clip),
               per_clip_z=json.dumps(per_clip_z))
    r = best_row
    log(f"final result: xy {final_avg:.2f} mm/node (median {np.median(list(per_clip.values())):.2f}), "
        f"z-err {final_z:.2f} mm (IN-SAMPLE); mass {MASS_PIN} kg/m pinned; "
        f"K_stretch {r[30]:.0f}; mu shape {r[31]:.3f} cable {r[32]:.3f} "
        f"tors {r[33]:.4f} roll {r[34]:.5f}")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} xy {v:6.1f}  z-err {per_clip_z[c]:6.1f}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
