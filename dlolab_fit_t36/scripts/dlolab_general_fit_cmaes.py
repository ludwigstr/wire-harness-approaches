"""DLO-Lab GENERAL fit: everything the native rod exposes, per cable where it is per
element, mass free per cable but bounded -- the counterpart of Newton row 10 and of
the MuJoCo / Genesis general fits (user, 2026-10-25).

    hinge (bending) scale   56  (4 cables x 14 hinges, log)
    stretch scale           60  (4 x 15 edges, log)
    twist scale             56  (4 x 14 hinges, log)
    rest curvature         112  (4 x 14 hinges x 2 components, 1/m, linear)
    K, E                     2  (log)
    G                        1  (G = 1e5 * g^2, g linear: row 14 has G = 0, so g starts at 0)
    mu (cable-board)         1  (log, coup_friction of the rigid material, 0.02 .. 2)
    velocity damping         1  (log, the rod's global damping, 4 .. 400; rows used 40)
    mass                     4  (per cable, log, 0.02 .. 0.2 kg/m)
                           293

Mass: the rod takes a per-VERTEX mass; the rows' 0.002 kg per vertex is 0.045 kg/m
at the mean measured cable length (16 x 0.002 / 0.729 m), so cable k gets
segment_mass = 0.002 * m_k / 0.045 and m_k is reported in kg/m on that scale. The
bound 0.02 .. 0.2 kg/m keeps it from degenerating while still containing the
0.15 that MuJoCo's sweep found (user, 2026-10-25).

Warm start = row 14 (genesis_rod_allclip_best.json: K, E, shared hinge / stretch /
twist scales and rest curvatures) copied onto every cable, mu 0.1 (the
coup_friction every DLO-Lab row was measured with, see genesis_rod_scene.py),
damping 40, mass 0.045: generation 0 reproduces row 14 (16.56).

OPTIMISER: CMA-ES on the table metric, like the MuJoCo and Genesis general fits.
The rod is differentiable, but only partly for this objective: mass, mu and the
damping carry no gradient (rigid coupling, material constants), the earlier
gradient fits needed one rod per scene to keep the adjoint reproducible (the
4-rod scene scored here is not that scene), and they optimised a mask loss.
CMA-ES scores exactly the evaluated 4-rod scene with exactly the reported number.

Rod scenes run on Genesis' CPU backend (genesis_rod_scene.build_rod_scene).

    python scripts/dlolab_general_fit_cmaes.py --gens 30 --popsize 20 --workers 14
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from clip_set import CLIP_TRIMS   # noqa: E402

# default: the cached 14-clip tracks. --tracks DIR (e.g. data/tracks107, 2026-11-01) switches to
# every *.npz in DIR, each clip whole (no trims); the env var carries it into spawned workers.
TRACKS = Path(os.environ.get("GENERAL_FIT_TRACKS") or "./data/tracks14")
CUSTOM_TRACKS = bool(os.environ.get("GENERAL_FIT_TRACKS"))
N_C, N_H, N_E = 4, 14, 15
MASS0, MASS_LO, MASS_HI = 0.045, 0.02, 0.2           # kg/m
SEG_MASS0 = 0.002                                    # kg per vertex at MASS0
I_HS = 0
I_SS = I_HS + N_C * N_H
I_TS = I_SS + N_C * N_E
I_KR = I_TS + N_C * N_H
I_K = I_KR + N_C * N_H * 2
I_E, I_G, I_MU, I_DAMP, I_MASS = I_K + 1, I_K + 2, I_K + 3, I_K + 4, I_K + 5
N_PAR = I_MASS + N_C                                 # 293
_CACHE = {}


def unpack(x):
    x = np.asarray(x, float)
    mass = np.exp(x[I_MASS:I_MASS + N_C])
    return dict(hinge_scale=np.exp(x[I_HS:I_SS]).reshape(N_C, N_H),
                stretch_scale=np.exp(x[I_SS:I_TS]).reshape(N_C, N_E),
                twist_scale=np.exp(x[I_TS:I_KR]).reshape(N_C, N_H),
                kappa_rest=x[I_KR:I_K].reshape(N_C, N_H, 2),
                K=float(np.exp(x[I_K])), E=float(np.exp(x[I_E])), G=float(1e5 * x[I_G] ** 2),
                mu=float(np.exp(x[I_MU])), damping=float(np.exp(x[I_DAMP])),
                segment_mass=SEG_MASS0 * mass / MASS0), mass


def _eval_one(job):
    cand_i, clip, x = job
    import genesis_rod_scene as R
    if clip not in _CACHE:
        z = np.load(TRACKS / f"{clip}.npz")
        _CACHE[clip] = ({k: z[f"c{k}"] for k in range(4)}, float(z["fps"]))
    kw, _ = unpack(x)
    try:
        e = float(np.mean(R.eval_clip(clip, shp_fps=_CACHE[clip], use_inextensible=False, **kw)))
        return cand_i, clip, e if np.isfinite(e) else 1e3
    except Exception as exc:                     # an unstable candidate must not kill the run
        print(f"  {clip}: {type(exc).__name__}: {exc}", flush=True)
        return cand_i, clip, 1e3


def warm_x(path):
    d = json.loads((ROOT / path).read_text())
    t4 = lambda v: np.tile(np.asarray(v, float).ravel(), N_C)
    return np.concatenate([np.log(t4(d["hinge_scale"])), np.log(t4(d["stretch_scale"])),
                           np.log(t4(d["twist_scale"])), t4(d["kappa"]),
                           [np.log(d["K"]), np.log(d["E"]), np.sqrt(float(d.get("G") or 0.0) / 1e5),
                            np.log(0.1), np.log(40.0)], np.full(N_C, np.log(MASS0))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/genesis_rod_allclip_best.json")
    ap.add_argument("--tracks", default="", help="track dir (all *.npz, untrimmed) instead of the 14-clip set")
    ap.add_argument("--resume", default="", help="start from an earlier run's best_x (its .json)")
    ap.add_argument("--sigma0", type=float, default=1.0, help="CMA step scale (0.5 to refine a resumed run)")
    ap.add_argument("--out", default="data/dlolab_general_fit.json")
    ap.add_argument("--log", default="data/dlolab_general_fit_log.jsonl")
    ap.add_argument("--mass-max", type=float, default=MASS_HI, help="upper mass bound, kg/m (t107 refit: 0.5 like fit3)")
    ap.add_argument("--stop-by", default="", help="wall-clock cap, local time 'YYYY-MM-DD HH:MM': no new generation is started that would end after it (best-so-far is kept, final evaluation still runs)")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--smoke", default="", help="clip[,clip]: score the warm start (and one perturbed candidate) and exit")
    a = ap.parse_args()
    if a.tracks:                                   # new clip set: all npz in the dir, no trims
        global TRACKS, CUSTOM_TRACKS
        TRACKS = (ROOT / a.tracks).resolve(); CUSTOM_TRACKS = True
        os.environ["GENERAL_FIT_TRACKS"] = str(TRACKS)
    import cma
    import multiprocessing as mp

    x0 = warm_x(a.init)
    if a.resume:
        x0 = np.asarray(json.loads((ROOT / a.resume).read_text())["best_x"], float)
    assert len(x0) == N_PAR, len(x0)
    if a.smoke:
        for c in a.smoke.split(","):
            t = time.time()
            print(c, "warm start", _eval_one((0, c, x0))[2], f"({time.time() - t:.0f}s)", flush=True)
            x = x0.copy(); x[I_MASS:] = np.log([0.02, 0.2, 0.045, 0.1]); x[I_MU] = np.log(0.5)
            print(c, "mass 0.02/0.2/0.045/0.1, mu 0.5", _eval_one((0, c, x))[2], flush=True)
        return

    log_f = open(ROOT / a.log, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    clips = sorted(p.stem for p in TRACKS.glob("*.npz")) if CUSTOM_TRACKS else sorted(CLIP_TRIMS)
    n_sc = N_C * (2 * N_H + N_E)
    stds = [0.2] * n_sc + [0.01] * (N_C * N_H * 2) + [0.3, 0.3, 0.3, 0.3, 0.3] + [0.2] * N_C
    lo = ([np.log(0.05)] * n_sc + [-0.5] * (N_C * N_H * 2)
          + [np.log(1e2), np.log(1e3), -3.0, np.log(0.02), np.log(4.0)] + [np.log(MASS_LO)] * N_C)
    hi = ([np.log(20.0)] * n_sc + [0.5] * (N_C * N_H * 2)
          + [np.log(1e7), np.log(1e9), 3.0, np.log(2.0), np.log(400.0)] + [np.log(a.mass_max)] * N_C)
    log(f"=== DLO-Lab GENERAL fit: {N_PAR} params, per cable, mass free per cable in {MASS_LO} .. {a.mass_max} kg/m (CMA-ES) ===")
    log(f"popsize {a.popsize}, gens {a.gens}, workers {a.workers}; " + (f"resumed from {a.resume}, sigma0 {a.sigma0}; " if a.resume else
        f"warm start {a.init} (row 14) on every cable, mu 0.1, damping 40, mass {MASS0}; ") + f"fitness = table metric over {len(clips)} clips (CPU backend)")

    pool = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=28)

    def run_jobs(jobs, stall=1800):
        """pool.map that survives a dead worker: the compiled rod kernel can SEGFAULT on a
        NaN state (no Python exception), the pool replaces the worker but its job never
        returns and pool.map waits forever (2026-10-25: 20 h lost). A job still missing
        after `stall` s without any other job finishing is scored 1e3."""
        rs = [pool.apply_async(_eval_one, (j,)) for j in jobs]
        n_done, t_last = -1, time.time()
        while True:
            done = sum(r.ready() for r in rs)
            if done == len(rs):
                break
            if done != n_done:
                n_done, t_last = done, time.time()
            elif time.time() - t_last > stall:
                log(f"  {len(rs) - done} job(s) lost (worker died), scored 1e3")
                break
            time.sleep(5)
        return [r.get() if r.ready() else (j[0], j[1], 1e3) for r, j in zip(rs, jobs)]

    def evaluate(xs):
        acc = [[] for _ in xs]
        for i, c, e in run_jobs([(i, c, x) for i, x in enumerate(xs) for c in clips]):
            acc[i].append(e)
        return [float(np.mean(v)) for v in acc]

    def save(best_x, best_f, f0, hist, **extra):
        kw, mass = unpack(best_x)
        (ROOT / a.out).write_text(json.dumps({
            "best_x": list(map(float, best_x)), "best_f": best_f, "base_f": f0, "mass_per_m": mass.tolist(),
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}, "hist": hist, **extra}, indent=1))

    t0 = time.time()
    f0 = evaluate([x0])[0]
    gen0_secs, gen_secs = time.time() - t0, []
    stop_ts = time.mktime(time.strptime(a.stop_by, '%Y-%m-%d %H:%M')) if a.stop_by else 0.0
    log(f"gen 0 ({'resumed best' if a.resume else f'row 14 on every cable, mu 0.1, mass {MASS0}'}): {f0:.3f} mm/node ({time.time() - t0:.0f}s)")
    hist = [{"gen": 0, "best": f0}]
    best_f, best_x = f0, x0.copy()
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, {"popsize": a.popsize, "verbose": -9, "CMA_stds": stds, "bounds": [lo, hi]})
    for gen in range(1, a.gens + 1):
        if stop_ts and time.time() + 1.15 * (max(gen_secs) if gen_secs else a.popsize * gen0_secs) > stop_ts:
            log(f"wall-clock cap {a.stop_by} reached before gen {gen}: stopping, best_ever {best_f:.3f} kept")
            break
        t_g = time.time()
        xs = es.ask()
        fs = evaluate(xs)
        es.tell(xs, fs)
        gen_secs.append(time.time() - t_g)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], float).copy()
        kw, mass = unpack(best_x)
        hist.append({"gen": gen, "best": float(fs[i]), "mean": float(np.mean(fs)), "best_ever": best_f})
        log(f"gen {gen}/{a.gens}: best {fs[i]:.3f}  mean {np.mean(fs):.3f}  best_ever {best_f:.3f}  ({time.time() - t_g:.0f}s)  "
            f"K {kw['K']:.3g} E {kw['E']:.3g} G {kw['G']:.3g}  mu {kw['mu']:.3f}  damping {kw['damping']:.1f}  "
            f"hinge {kw['hinge_scale'].min():.2f}..{kw['hinge_scale'].max():.2f}  "
            f"kappa {kw['kappa_rest'].min():+.3f}..{kw['kappa_rest'].max():+.3f}  mass {np.round(mass, 4).tolist()}")
        save(best_x, best_f, f0, hist)

    log(f"fit done: {best_f:.3f} (gen 0 {f0:.3f}), {time.time() - t0:.0f}s")
    per_clip = {c: e for _, c, e in run_jobs([(0, c, best_x) for c in clips])}
    pool.close(); pool.join()
    v = np.array(list(per_clip.values()))
    save(best_x, best_f, f0, hist, per_clip=per_clip, final_avg=float(v.mean()), final_median=float(np.median(v)))
    log(f"final result: {v.mean():.3f} mm/node (median {np.median(v):.3f}), {len(clips)} clips IN-SAMPLE")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
