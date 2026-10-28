"""Newton: EVERYTHING free, per cable where the quantity is per joint.

User request 2026-10-21 ("die ganz große Variante"): one Newton row in which
no parameter is shared or left at a default:

    E bend          56  (4 cables x 14 joints, log)
    G twist         56  (own slot, NOT a copy of bend, log)
    damp bend rate  56  (log)
    rest shape      56  (rad, linear, signed)
    rho             4   (per cable, log)
    K stretch       1   (log)
    mu slide        2   (shape_mu, cable_mu, log)
    mu tors, roll   2   (log)
    hysteresis      2   (Dahl eps, tau, scalar, log)
                   235

Warm start = row 15 (newton_fit_percable.npz: ke, kd, mass, eps, tau, rest
per cable) with twist := bend (row 15 ran coupled), K stretch 1e5 and the
library frictions -- so generation 0 reproduces row 15 and every later
generation is a measured gain over it, not an assumption. Fitness is the
same xy + 0.5 * depth-z objective as rows 10-15: with the mass free per cable
the z term is the guard against the "near-massless cables fly over the
movers" solution rejected in August.

    WIRE_GPUS=0,1,2 WIRE_WORKERS=6 WIRE_POPSIZE=18 WIRE_GENS=30 \
    python3 scripts/newton_fit_everything_cmaes.py

Variant with the MASS PINNED at the measured 0.044 kg/m (user, 2026-10-21):
WIRE_MASS_PIN=0.044 drops the 4 mass coordinates (231 params) and warm-starts
from row 14 (newton_fit_physical.npz: mass 0.044, K stretch 364, frictions and
twist fitted, shared across cables) copied onto each cable, so generation 0
reproduces row 14's 8.03.

    WIRE_MASS_PIN=0.044 WIRE_WARM=data/newton_fit_physical.npz \
    WIRE_OUT=data/newton_fit_everything_pinned.npz \
    WIRE_FITLOG=data/newton_fit_everything_pinned_log.jsonl \
    WIRE_GPUS=1,2 WIRE_WORKERS=4 python3 scripts/newton_fit_everything_cmaes.py
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

import newton_fit_percable_cmaes as P  # noqa: E402  (helpers + worker cache)
from newton_fit_percable_cmaes import CLIP_TRIMS, _clip_frame_t1, _init_worker, _z_loss_mm, safe_savez  # noqa: E402

ROOT = P.ROOT
N_C, N_J = 4, 14
N_PJ = N_C * N_J                      # 56
GPU_IDS = [int(x) for x in os.environ.get("WIRE_GPUS", "0,1,2").split(",")]
N_WORKERS = int(os.environ.get("WIRE_WORKERS", "6"))
POPSIZE = int(os.environ.get("WIRE_POPSIZE", "18"))
MAX_GENERATIONS = int(os.environ.get("WIRE_GENS", "30"))
FIT_SOLVER_ITERATIONS = 10
Z_WEIGHT = 0.5
WARM_FILE = os.environ.get("WIRE_WARM", "data/newton_fit_percable.npz")
OUT = ROOT / os.environ.get("WIRE_OUT", "data/newton_fit_everything.npz")
LOG = ROOT / os.environ.get("WIRE_FITLOG", "data/newton_fit_everything_log.jsonl")
MASS_PIN = float(os.environ["WIRE_MASS_PIN"]) if os.environ.get("WIRE_MASS_PIN") else None
N_MASS = 0 if MASS_PIN else N_C
X0_STRETCH = 1.0e5
X0_GLOB = [1.0, 1.0, 0.005, 0.0001]   # shape_mu, cable_mu, mu_tors, mu_roll
# per-coordinate CMA steps: refine the warm start (0.2 in log space = x1.22),
# search wider for the five globals nobody fitted per cable yet, 0.06 rad
# for the signed rest angles (same as rows 11-15)
STD_PJ, STD_MASS, STD_DAHL, STD_GLOB, STD_REST = 0.2, 0.2, 0.2, 0.5, 0.06

# stored row layout (all positive except rest): ke56 kd56 kt56 | mass4 | eps tau | K | mu4 | rest56
I_KE, I_KD, I_KT = 0, N_PJ, 2 * N_PJ
I_MASS = 3 * N_PJ                     # 168
I_DAHL = I_MASS + N_MASS              # 172 (168 when the mass is pinned)
I_K = I_DAHL + 2                      # 174
I_MU = I_K + 1                        # 175
N_LOG = I_MU + 4                      # 179
I_REST = N_LOG
N_PAR = N_LOG + N_PJ                  # 235
TRAIN_CLIPS = list(CLIP_TRIMS.keys())


def from_x(x):
    x = np.asarray(x, float)
    return np.concatenate([np.exp(x[:N_LOG]), x[N_LOG:]])


def to_x(row):
    row = np.asarray(row, float)
    return np.concatenate([np.log(row[:N_LOG]), row[N_LOG:]])


def unpack(row):
    row = np.asarray(row, float)
    pj = lambda i: row[i:i + N_PJ].reshape(N_C, N_J)
    eps, tau = row[I_DAHL], row[I_DAHL + 1]
    shape_mu, cable_mu, mu_tors, mu_roll = row[I_MU:I_MU + 4]
    return dict(per_joint_ke=pj(I_KE), per_joint_kd=pj(I_KD), per_joint_kt=pj(I_KT),
                mass_per_m=(np.full(N_C, MASS_PIN) if MASS_PIN else row[I_MASS:I_MASS + N_C].copy()),
                dahl_eps_max=np.full((N_C, N_J), eps), dahl_tau=np.full((N_C, N_J), tau),
                stretch_stiffness=float(row[I_K]), shape_mu=float(shape_mu),
                cable_mu=float(cable_mu), mu_tors=float(mu_tors), mu_roll=float(mu_roll),
                per_joint_rest=pj(I_REST))


def warm_row():
    prev = np.asarray(np.load(ROOT / WARM_FILE, allow_pickle=True)["best"], float)
    if MASS_PIN:
        # row 14's 63-value set: ke14 kd14 eps tau K shape_mu cable_mu tors roll | rest14 | kt14
        assert len(prev) == 63, f"expected row 14's 63-value set, got {len(prev)}"
        ke, kd = prev[:14], prev[14:28]
        eps, tau, K = prev[28], prev[29], prev[30]
        mu = prev[31:35]
        rest, kt = prev[35:49], prev[49:63]
        t4 = lambda v: np.tile(v, N_C)
        return np.concatenate([t4(ke), t4(kd), t4(kt), [eps, tau, K], mu, t4(rest)])
    assert len(prev) == 174, f"expected row 15's 174-value set, got {len(prev)}"
    ke, kd, mass = prev[:56], prev[56:112], prev[112:116]
    eps, tau = prev[116], prev[117]
    rest = prev[118:174]
    return np.concatenate([ke, kd, ke.copy(), mass, [eps, tau, X0_STRETCH], X0_GLOB, rest])


def _eval_job(args):
    (x,) = args
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    cache = P._worker_cache
    if "clips" not in cache:
        cache["clips"], cache["targets"] = {}, {}
    for c in TRAIN_CLIPS:
        if c not in cache["clips"]:
            t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
            shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
            cache["clips"][c] = (shp, fps)
            cache["targets"][c] = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
    kw = unpack(from_x(x))
    xy, zs = [], []
    for c in TRAIN_CLIPS:
        z_out = []
        errs = eval_clip(c, shp_fps=cache["clips"][c], solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, device=cache.get("device"), **kw)
        xy.append(float(np.mean(errs)))
        zs.append(_z_loss_mm(z_out, cache["targets"][c]))
    return float(np.mean(xy)), float(np.mean(zs))


def main():
    log_f = open(LOG, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            log_f.write(line + "\n"); log_f.flush()
        except OSError:
            pass

    log(f"=== Newton EVERYTHING fit: {N_PAR} params, all per cable, nothing shared, nothing default (CMA-ES)"
        + (f"; MASS PINNED {MASS_PIN} kg/m" if MASS_PIN else "") + " ===")
    log(f"popsize={POPSIZE}, gens={MAX_GENERATIONS}, workers={N_WORKERS} on GPUs {GPU_IDS}; warm start {WARM_FILE} "
        + ("(row 14 copied onto each cable)" if MASS_PIN
           else f"(row 15) with twist := bend, K stretch {X0_STRETCH:g}, mu {X0_GLOB}"))
    row0 = warm_row()
    assert len(row0) == N_PAR
    x0 = to_x(row0)
    ctx = mp.get_context("spawn")
    pools = [ctx.Pool(N_WORKERS // len(GPU_IDS), initializer=_init_worker, initargs=(g,))
             for g in GPU_IDS]

    def eval_population(xs):
        res = [pools[i % len(pools)].apply_async(_eval_job, ((x,),)) for i, x in enumerate(xs)]
        return [r.get() for r in res]

    t0 = time.time()
    (xy0, z0), = eval_population([x0])
    log(f"gen 0/{MAX_GENERATIONS} (x0 = row {'14' if MASS_PIN else '15'}, nothing moved): fit={xy0 + Z_WEIGHT * z0:.2f} "
        f"(xy={xy0:.2f}, z-err={z0:.2f}mm) ({time.time()-t0:.0f}s)")
    stds = ([STD_PJ] * (3 * N_PJ) + [STD_MASS] * N_MASS + [STD_DAHL] * 2 + [STD_GLOB] * 5
            + [STD_REST] * N_PJ)
    assert len(stds) == N_PAR
    es = cma.CMAEvolutionStrategy(x0, 1.0, {"popsize": POPSIZE, "verbose": -9, "CMA_stds": stds})
    hist = [{"gen": 0, "best": xy0 + Z_WEIGHT * z0, "best_xy": xy0, "best_z": z0}]
    best_fit, best_row = xy0 + Z_WEIGHT * z0, row0
    gen = 0
    while not es.stop() and gen < MAX_GENERATIONS:
        t_g = time.time()
        xs = es.ask()
        res = eval_population(xs)
        fits = [xy + Z_WEIGHT * ze for xy, ze in res]
        es.tell(xs, fits)
        gen += 1
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit, best_row = fits[i], from_x(xs[i])
        hist.append({"gen": gen, "best": float(fits[i]), "best_xy": res[i][0], "best_z": res[i][1],
                     "mean_fit": float(np.mean(fits)), "best_ever": float(best_fit)})
        r = best_row
        log(f"gen {gen}/{MAX_GENERATIONS}: best={fits[i]:.2f} (xy={res[i][0]:.2f}, z-err={res[i][1]:.2f}mm) "
            f"mean={np.mean(fits):.2f} best_ever={best_fit:.2f} ({time.time()-t_g:.0f}s)  "
            f"mass={'pinned' if MASS_PIN else np.round(r[I_MASS:I_MASS+4], 3).tolist()} K={r[I_K]:.0f} "
            f"mu={np.round(r[I_MU:I_MU+4], 4).tolist()} eps/tau={r[I_DAHL]:.3f}/{r[I_DAHL+1]:.3f}")
        safe_savez(OUT, log=log, best=best_row, history=json.dumps(hist))
    log(f"fit done: {gen} generations, {time.time()-t0:.0f}s; best fit {best_fit:.2f} (gen 0 was {hist[0]['best']:.2f})")
    for p in pools:
        p.close(); p.join()

    log(f"=== final validation: {FIT_SOLVER_ITERATIONS} iterations, all 14 clips (IN-SAMPLE) ===")
    from mujoco_scene import track_clip
    from newton_full_scene import eval_clip
    kw = unpack(best_row)
    per_clip, per_clip_z = {}, {}
    for c in TRAIN_CLIPS:
        t1 = _clip_frame_t1(c, CLIP_TRIMS[c])
        shp, _, _, fps = track_clip(c, t0=8, t1=t1)
        tg = np.load(ROOT / f"data/depth_targets/{c}.npz")["calibrated_mm"]
        z_out = []
        errs = eval_clip(c, shp_fps=(shp, fps), solver_iterations=FIT_SOLVER_ITERATIONS,
                         z_nodes_out=z_out, **kw)
        per_clip[c] = float(np.mean(errs))
        per_clip_z[c] = _z_loss_mm(z_out, tg)
    vals = np.array(list(per_clip.values()))
    final_avg, final_med, final_z = float(vals.mean()), float(np.median(vals)), float(np.mean(list(per_clip_z.values())))
    safe_savez(OUT, log=log, best=best_row, history=json.dumps(hist), final_avg=final_avg,
               final_median=final_med, final_z_err=final_z,
               per_clip=json.dumps(per_clip), per_clip_z=json.dumps(per_clip_z))
    r = best_row
    log(f"final result: xy {final_avg:.2f} mm/node (median {final_med:.2f}), z-err {final_z:.2f} mm (IN-SAMPLE); "
        f"mass {MASS_PIN or np.round(r[I_MASS:I_MASS+4], 3).tolist()}; K_stretch {r[I_K]:.0f}; "
        f"mu shape {r[I_MU]:.3f} cable {r[I_MU+1]:.3f} tors {r[I_MU+2]:.4f} roll {r[I_MU+3]:.5f}; "
        f"dahl eps {r[I_DAHL]:.3f} tau {r[I_DAHL+1]:.3f}")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} xy {v:6.1f}  z-err {per_clip_z[c]:6.1f}")
    log(f"-> {OUT}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
