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
GPU_IDS = [0]
N_WORKERS = 2
POPSIZE = 8
FIT_SOLVER_ITERATIONS = 10
MAX_GENERATIONS = 8
SIGMA0 = 1.0
X0_KE = 0.5
X0_KD = 0.05
X0_MASS = 0.15
Z_WEIGHT = 0.5

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


def _init_worker(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


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
    row = np.exp(np.asarray(log_row))
    ke = np.tile(row[:N_JOINTS], (4, 1))
    kd = np.tile(row[N_JOINTS:2 * N_JOINTS], (4, 1))
    mass = np.full(4, row[2 * N_JOINTS])
    return ke, kd, mass


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
    ke, kd, mass = _params_from_log(log_row)
    xy_errs, z_errs = [], []
    for c in TRAIN_CLIPS:
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, mass_per_m=mass,
                         shp_fps=clips[c], solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, subdiv=SUBDIV)
        xy_errs.append(float(np.mean(errs)))
        z_errs.append(_z_loss_mm(z_out, targets[c]))
    return float(np.mean(xy_errs)), float(np.mean(z_errs))


def main():
    log_path = ROOT / "data/newton_fit_depth_log.jsonl"
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
    log(f"29 shared params; x0: ke={X0_KE}, kd={X0_KD}, mass={X0_MASS} kg/m; "
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

    out_path = ROOT / "data/newton_fit_depth.npz"
    x0 = [np.log(X0_KE)] * N_JOINTS + [np.log(X0_KD)] * N_JOINTS + [np.log(X0_MASS)]
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, {"popsize": POPSIZE, "verbose": -9})
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
        best_row = np.exp(es.result.xbest)
        safe_savez(out_path, log=log, best=best_row, history=json.dumps(gen_history))

    best_row = np.exp(es.result.xbest)
    log(f"fit done: {gen} generations, {time.time()-t_start:.0f}s, "
        f"ke={np.round(best_row[:N_JOINTS],3).tolist()}, "
        f"kd={np.round(best_row[N_JOINTS:2*N_JOINTS],4).tolist()}, "
        f"mass={best_row[2*N_JOINTS]:.4f} kg/m")
    for p in pools:
        p.close(); p.join()

    log("=== final validation: fitted shared params, full scene, 10 iterations (IN-SAMPLE) ===")
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    ke, kd, mass = _params_from_log(np.log(best_row))
    per_clip, per_clip_z = {}, {}
    for c in TRAIN_CLIPS:
        t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
        shp, t0_, t1_, fps = track_clip(c, t0=8, t1=t1)
        tg = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
        z_out = []
        errs = eval_clip(c, per_joint_ke=ke, per_joint_kd=kd, mass_per_m=mass,
                         shp_fps=(shp, fps), z_nodes_out=z_out, subdiv=SUBDIV)
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
