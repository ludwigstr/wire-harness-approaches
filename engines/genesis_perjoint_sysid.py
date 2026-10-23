"""Per-joint cable parameters for Genesis, fitted with the DLO-Lab loss.

Two changes over genesis_dlolab_sysid.py, both requested by the user:

  * MOVER COLLISION is on (genesis_full_scene.py's mover geoms no longer carry
    contype/conaffinity 0). Verified: without it the cables passed through the
    movers in 29/29 frames up to 11.55 mm deep; with it, never. The uniform
    parameters were fitted WITHOUT collision, so they have to be refitted --
    the old best point already degrades 10.18 -> 11.34 mm on clip 170108.

  * PER-JOINT stiffness and damping instead of one value for the whole cable:
    15 hinge joints, so 15 + 15 + 1 (mass) = 31 parameters. That is the same
    degree of freedom Newton's per-joint weld fit uses -- a soft spot where the
    real cable bends, a stiff one where it stays straight.

OPTIMIZER: CMA-ES, not the finite differences the uniform fit used. At 31
parameters an FD gradient costs 63 simulations per step; CMA-ES needs only its
population (~14), so it is ~4x cheaper per step here and does not need the
parameter space to be smooth.

LOSS: unchanged -- mean distance from the simulated centreline to the skeleton
(see genesis_dlolab_sysid.py), deliberately independent of the tracked routes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))

import genesis_dlolab_sysid as base   # noqa: E402  (loss, GT cache, splat)

N_JOINTS = 15          # hinge joints per cable: bodies seg1..seg15
_L: dict = {}


def unpack(x):
    """log-space vector -> (stiffness[15], damping[15], mass)."""
    v = np.exp(np.asarray(x, float))
    return v[:N_JOINTS], v[N_JOINTS:2 * N_JOINTS], float(v[2 * N_JOINTS])


def _init_worker(clips, sigma):
    _L.update(gt=base.gt_masks(list(clips), write=False), sigma=float(sigma))
    base._L.update(_L)


def _clip_loss(job):
    """Skeleton loss for ONE clip at ONE parameter vector."""
    x, clip = job
    import genesis_full_scene as G
    st, dp, mass = unpack(x)
    d = _L["gt"][clip]
    nodes = []
    try:
        G.eval_clip(clip, t1_sec=d["trim"], shp_fps=(d["shp"], d["fps"]),
                    nodes_out=nodes, stiffness=st, damping=dp, mass_per_m=mass)
    except Exception as e:                                    # noqa: BLE001
        print(f"   SIM-FEHLER {clip}: {type(e).__name__}: {e}", flush=True)
        return 1.0, 1
    tf = d["tf"]
    tot, cnt = 0.0, 0
    for i in d["kf"]:
        if i >= len(nodes):
            tot += 1.0
            cnt += 1
            continue
        dt = d["dists"][i]
        H, W = dt.shape
        sd, sn = 0.0, 0
        for k in range(4):
            P = np.array([tf.to_px(q * 1000.0) for q in nodes[i][k]],
                         float) / base.MASK_DOWN
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            if arc[-1] <= 0:
                continue
            fine = np.arange(0.0, arc[-1], 3.0)
            pts = np.stack([np.interp(fine, arc, P[:, c])
                            for c in (0, 1)], 1).astype(int)
            m = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
                 (pts[:, 1] >= 0) & (pts[:, 1] < H))
            if m.any():
                sd += float(np.minimum(dt[pts[m, 1], pts[m, 0]],
                                       base.SKEL_CAP).sum())
                sn += int(m.sum())
        tot += (sd / max(sn, 1)) / base.SKEL_CAP
        cnt += 1
    return tot, cnt


def main():
    from clip_set import CLIP_TRIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=14)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--sigma0", type=float, default=0.6, help="CMA step, log space")
    # start from the uniform fit; per-joint search then breaks the symmetry
    ap.add_argument("--init-stiffness", type=float, default=0.3391)
    ap.add_argument("--init-damping", type=float, default=0.0978)
    ap.add_argument("--init-mass", type=float, default=0.1818)
    ap.add_argument("--warm", default=None,
                    help="npy mit log-parametervektor: warmstart statt uniform")
    ap.add_argument("--out", default="data/genesis_perjoint_sysid.json")
    args = ap.parse_args()
    clips = args.clips if args.clips else list(CLIP_TRIMS)

    gt = base.gt_masks(clips)
    _L.update(gt=gt, sigma=args.sigma)
    base._L.update(_L)
    nk = sum(len(d["kf"]) for d in gt.values())
    print(f"training: {len(clips)} clips, {nk} keyframes", flush=True)
    print(f"parameter: {N_JOINTS} stiffness + {N_JOINTS} damping + 1 mass "
          f"= {2 * N_JOINTS + 1}", flush=True)
    print(f"start: stiffness={args.init_stiffness:g} damping={args.init_damping:g} "
          f"mass={args.init_mass:g} (uniform)", flush=True)

    import cma
    if args.warm:
        # Continuation run: 31 parameters need far more than 30 CMA
        # generations -- the population mean was still falling three times
        # faster than the best value at generation 10 -- so phase 2 restarts
        # from phase 1's best point with a smaller step size.
        x0 = np.load(args.warm)
        assert x0.size == 2 * N_JOINTS + 1, x0.size
        print(f"warmstart aus {args.warm}", flush=True)
    else:
        x0 = np.concatenate([np.full(N_JOINTS, math.log(args.init_stiffness)),
                             np.full(N_JOINTS, math.log(args.init_damping)),
                             [math.log(args.init_mass)]])
    lo = np.concatenate([np.full(N_JOINTS, math.log(1e-3)),
                         np.full(N_JOINTS, math.log(1e-3)), [math.log(5e-3)]])
    hi = np.concatenate([np.full(N_JOINTS, math.log(1e2)),
                         np.full(N_JOINTS, math.log(1e2)), [math.log(5.0)]])
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, {
        "popsize": args.popsize, "bounds": [list(lo), list(hi)],
        "verbose": -9, "seed": 1})

    from multiprocessing import get_context
    ctx = get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_init_worker,
                    initargs=(clips, args.sigma))
    hist, best = [], None
    try:
        for gen in range(args.generations):
            X = es.ask()
            t0 = time.time()
            jobs = [(x, c) for x in X for c in clips]
            res = pool.map(_clip_loss, jobs)
            n = len(clips)
            F = []
            for i in range(len(X)):
                tot = sum(r[0] for r in res[i * n:(i + 1) * n])
                cnt = sum(r[1] for r in res[i * n:(i + 1) * n])
                F.append(tot / max(cnt, 1))
            es.tell(X, F)
            i_best = int(np.argmin(F))
            if best is None or F[i_best] < best[0]:
                best = (float(F[i_best]), np.array(X[i_best], float))
            st, dp, ms = unpack(best[1])
            hist.append({"gen": gen, "best": best[0], "gen_best": float(F[i_best]),
                         "gen_mean": float(np.mean(F)), "mass": ms})
            print(f"[gen {gen:03d}] best={best[0]:.4f}  gen_best={F[i_best]:.4f}  "
                  f"mean={np.mean(F):.4f}  mass={ms:.4g}  "
                  f"stiff {st.min():.3g}-{st.max():.3g}  "
                  f"damp {dp.min():.3g}-{dp.max():.3g}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            st, dp, ms = unpack(best[1])
            (ROOT / args.out).write_text(json.dumps(
                {"hist": hist, "best_loss": best[0], "stiffness": st.tolist(),
                 "damping": dp.tolist(), "mass_per_m": ms, "clips": clips,
                 "n_joints": N_JOINTS, "loss": "skeleton_only",
                 "collision": "mover collision ENABLED"}, indent=1))
    finally:
        pool.close()
        pool.join()

    st, dp, ms = unpack(best[1])
    print(f"\nfinal loss {best[0]:.4f}  mass={ms:.4g}", flush=True)
    print(f"  stiffness je gelenk: {np.round(st, 4).tolist()}", flush=True)
    print(f"  damping   je gelenk: {np.round(dp, 4).tolist()}", flush=True)
    print("PERJOINT_DONE", flush=True)


if __name__ == "__main__":
    main()
