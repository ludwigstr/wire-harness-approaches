"""DLO-Lab SysID (arXiv:2606.04206) applied to the Genesis full-scene sim.

Follows extern/DLO-Lab/examples/system_id.py, in the finite-difference mode
already used for MuJoCo in Wire_parameter/scripts/dlolab_sysid.py -- so no
differentiability through Genesis is required:

  * loss: soft-IoU between a Gaussian-splatted SOFT MASK of the simulated
    cables and the real wire mask at keyframes -- a pixel loss, not a node
    distance, so it does not inherit the tracked route's errors
  * parameters (log10): bending stiffness and damping, the direct analogue of
    DLO-Lab's E and its damping term (their K = stretching is dropped; the
    Genesis cable is a hinge chain and inextensible by construction)
  * gradient: batched central differences (5 sims per iteration: base,
    stiffness +/-, damping +/-), run in parallel -- Genesis uses the CPU
    backend here, so processes scale
  * optimizer: vector-normalized momentum (beta=0.5) with cosine LR decay

WHY a mask loss and not the node error: the tracked routes are themselves
imperfect (two clips had to be excluded outright), and a node loss would fit
the simulation to those errors. The wire mask is the raw image evidence.

genesis resolves to DLO-Lab's fork (genesis-world 1.0.0); site-packages holds
only a shadowing stub, so PYTHONPATH must point at extern/DLO-Lab.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import importlib.util

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from dlolab_path import DLO_LAB
# Only prepend DLO-Lab's fork when no genesis is importable yet -- under the
# genesis-world 1.3.3 venv the installed package must win.
_sp = importlib.util.find_spec("genesis")
if _sp is None or _sp.origin is None:
    sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))

MASK_DOWN = 2          # masks at half resolution, as in the MuJoCo version
SKEL_CAP = 150.0       # px (half-res). MEASURED 2026-08-25: at the old 40 px
                       # value, 32% of the sampled points sat AT the cap and so
                       # contributed no gradient -- the loss barely moved when the
                       # node error halved (108.6 -> 53.8 mm shifted it only
                       # 0.597 -> 0.590). Real distances reach 146 px (90th pct 69),
                       # so 150 clips nothing real but still bounds a stray cable.
# The mask/soft-IoU term is dropped on user instruction: skeleton only.
_L: dict = {}


def keyframes_for(n_frames, k=6):
    """k evenly spaced frames, skipping the settling transient at the start."""
    lo = max(1, n_frames // 8)
    return sorted(set(np.linspace(lo, n_frames - 1, k).round().astype(int).tolist()))


def soft_maske(P_px, H, W, sigma, schritt_px=3.0):
    """DLO-Lab splat: dense points along the polyline as additive Gaussian
    density, soft = 1 - exp(-sum)."""
    d = np.linalg.norm(np.diff(P_px, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.zeros((H, W), np.float32)
    fein = np.arange(0.0, s[-1], schritt_px)
    pts = np.stack([np.interp(fein, s, P_px[:, c]) for c in (0, 1)], 1)
    win = int(max(3, math.ceil(3.0 * sigma)))
    off = np.arange(-win, win + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for u, v in pts:
        ui, vi = int(round(u)), int(round(v))
        px, py = ui + ox, vi + oy
        g = np.exp(-((px - u) ** 2 + (py - v) ** 2) * inv2s2)
        m = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.add.at(acc, (py[m], px[m]), g[m])
    return 1.0 - np.exp(-acc)


def soft_iou_verlust(soft, gt):
    inter = float((soft * gt).sum())
    union = float((soft + gt - soft * gt).sum())
    return 1.0 - inter / (union + 1e-6)


def gt_masks(clips, write=True):
    """Real wire mask + skeleton distance field at each clip's keyframes.

    `write=False` for pool workers: several processes calling np.savez on the
    same file corrupt it mid-read (seen as EOFError out of zipfile). The parent
    builds the cache once before the pool starts; workers only read it."""
    import cv2
    from clip_set import CLIP_TRIMS
    from mujoco_scene import track_clip, DIR
    from wire_model.perception.rectify import find_board
    from wire_model.perception.harness import wire_mask
    from wire_model.perception import seg_masks as _sm
    from fit_clip import read

    cache = ROOT / "data/genesis_sysid_gt.npz"
    trk = ROOT / "data/genesis_sysid_tracks.npz"
    tracks = dict(np.load(trk)) if trk.exists() else {}
    store, meta = {}, {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        store = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(str(z["__meta__"]))
    out = {}
    for clip in clips:
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps = (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
        trim = CLIP_TRIMS[clip]
        t1 = int(round(trim * fps)) if trim else None
        # Tracking is the expensive part (~30 s/clip) and is deterministic, so
        # it is cached: spawn-mode workers must not redo it every launch.
        ck = f"{clip}:shp"
        if ck in tracks:
            arr = tracks[ck]
            shp = {k: arr[k] for k in range(4)}
            t0 = int(tracks[f"{clip}:t0"])
        else:
            shp, t0, _, fps = track_clip(clip, t0=8, t1=t1)
            tracks[ck] = np.stack([shp[k] for k in range(4)])
            tracks[f"{clip}:t0"] = np.array(t0)
        n = len(shp[0])
        kf = keyframes_for(n)
        tf = find_board(read(cap, 0))
        masks, dists = {}, {}
        for i in kf:
            key = f"{clip}:{i}"
            if key in store and key + ":dt" in store:
                masks[i] = store[key]
                dists[i] = store[key + ":dt"]
                continue
            fr = read(cap, t0 + i)
            w, _ = wire_mask(fr, tf, clip=clip, orig_idx=t0 + i)
            m = w.astype(np.float32)[::MASK_DOWN, ::MASK_DOWN]
            # Skeleton distance field: the thinned strand IS the observed
            # centreline, and the simulation produces a centreline too, so this
            # compares like with like -- the mask term alone only says the tube
            # overlaps somewhere, not that the centre sits right.
            sk = _sm.skeleton_frame(clip, t0 + i, fr)
            if sk is None:
                dt = np.full_like(m, SKEL_CAP, dtype=np.float32)
            else:
                skd = sk.astype(np.uint8)[::MASK_DOWN, ::MASK_DOWN]
                dt = cv2.distanceTransform((skd == 0).astype(np.uint8),
                                           cv2.DIST_L2, 3).astype(np.float32)
            masks[i] = m
            dists[i] = dt
            store[key] = m
            store[key + ":dt"] = dt
        cap.release()
        out[clip] = {"kf": kf, "masks": masks, "dists": dists, "tf": tf, "shp": shp,
                     "fps": fps, "trim": trim, "n": n}
        meta[clip] = kf
    if write:
        np.savez_compressed(cache, __meta__=json.dumps(meta), **store)
        np.savez_compressed(trk, **tracks)
    return out


def _clip_loss(args):
    """Skeleton-distance loss for ONE clip at ONE parameter triple.

    Three parameters, not two: a grid probe on 2026-08-25 (with the link-order
    bug fixed) showed mass_per_m is the STRONGEST lever of the three -- at
    stiffness 1.0 / damping 0.1 it moves the clip error 53.9 -> 46.2 -> 41.9 mm
    across 0.0368 -> 0.15 -> 0.5 kg/m -- while a stiffness+damping-only fit
    crawled (loss 0.2099 -> 0.2109 over three iterations) because the dominant
    parameter was pinned at its default.

    Parallelised per (candidate, clip) rather than per candidate: with 14 clips
    a whole candidate is ~10 min of simulation, so splitting only 5 ways leaves
    most of the machine idle.
    """
    theta, clip = args
    import genesis_full_scene as G
    d = _L["gt"][clip]
    stiff, damp, mass = (10.0 ** theta[0], 10.0 ** theta[1], 10.0 ** theta[2])
    nodes = []
    try:
        G.eval_clip(clip, t1_sec=d["trim"], shp_fps=(d["shp"], d["fps"]),
                    nodes_out=nodes, stiffness=stiff, damping=damp,
                    mass_per_m=mass)
    except Exception as e:                                   # noqa: BLE001
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
        skel_d, skel_n = 0.0, 0
        for k in range(4):
            P_px = np.array([tf.to_px(q * 1000.0) for q in nodes[i][k]],
                            float) / MASK_DOWN
            # dense samples along the simulated centreline, read off the
            # skeleton distance field
            seg = np.linalg.norm(np.diff(P_px, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            if arc[-1] > 0:
                fine = np.arange(0.0, arc[-1], 3.0)
                pts = np.stack([np.interp(fine, arc, P_px[:, c])
                                for c in (0, 1)], 1).astype(int)
                m = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
                     (pts[:, 1] >= 0) & (pts[:, 1] < H))
                if m.any():
                    skel_d += float(np.minimum(dt[pts[m, 1], pts[m, 0]],
                                               SKEL_CAP).sum())
                    skel_n += int(m.sum())
        # Skeleton ONLY (user instruction): the thinned strand is the
        # observed centreline and the simulation produces a centreline, so
        # this compares like with like. The soft-IoU mask term is dropped --
        # it only says the tube overlaps somewhere, and its Gaussian splat
        # dominated the runtime.
        tot += (skel_d / max(skel_n, 1)) / SKEL_CAP
        cnt += 1
    return tot, cnt


def _init_worker(clips, sigma):
    """Rebuild the per-worker state once. Genesis does not survive a fork, so
    the pool runs in spawn mode and each worker builds its own state from the
    on-disk caches -- cheap, because tracking and masks are both cached."""
    _L.update(gt=gt_masks(list(clips), write=False), sigma=float(sigma))





def main():
    from clip_set import CLIP_TRIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="default: the full 14-clip evaluation set")
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--lr-min", type=float, default=0.03)
    ap.add_argument("--fd-eps", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=3.0)
    # Warm start at the best point of the grid probe, which sat at the EDGE
    # in both stiffness and mass -- so the optimum is likely beyond it and the
    # bounds below are opened up accordingly.
    ap.add_argument("--init-stiffness", type=float, default=1.0)
    ap.add_argument("--init-damping", type=float, default=0.1)
    ap.add_argument("--init-mass", type=float, default=0.5)
    ap.add_argument("--out", default="data/genesis_dlolab_sysid.json")
    args = ap.parse_args()
    clips = args.clips if args.clips else list(CLIP_TRIMS)

    gt = gt_masks(clips)
    H, W = next(iter(next(iter(gt.values()))["masks"].values())).shape
    _L.update(gt=gt, sigma=args.sigma)
    nk = sum(len(d["kf"]) for d in gt.values())
    print(f"training: {len(gt)} clips, {nk} keyframes, masks {W}x{H}", flush=True)
    print(f"loss = mean skeleton distance / {SKEL_CAP:.0f}px (skeleton only)",
          flush=True)

    logS = math.log10(args.init_stiffness)
    logD = math.log10(args.init_damping)
    logM = math.log10(args.init_mass)
    eps, mom, beta = args.fd_eps, np.zeros(3), 0.5
    hist = []
    from multiprocessing import get_context
    ctx = get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_init_worker,
                    initargs=(clips, args.sigma))
    try:
        for it in range(args.n_iters):
            kand = [(logS, logD, logM),
                    (logS + eps, logD, logM), (logS - eps, logD, logM),
                    (logS, logD + eps, logM), (logS, logD - eps, logM),
                    (logS, logD, logM + eps), (logS, logD, logM - eps)]
            tasks = [(k, c) for k in kand for c in clips]
            t0 = time.time()
            res = pool.map(_clip_loss, tasks)
            # regroup: candidate i owns the len(clips) entries starting at i*n
            n = len(clips)
            L = []
            for i in range(7):
                tot = sum(r[0] for r in res[i * n:(i + 1) * n])
                cnt = sum(r[1] for r in res[i * n:(i + 1) * n])
                L.append(tot / max(cnt, 1))
            g = np.array([(L[1] - L[2]) / (2 * eps),
                          (L[3] - L[4]) / (2 * eps),
                          (L[5] - L[6]) / (2 * eps)])
            g = g / (np.linalg.norm(g) + 1e-12)
            mom = beta * mom + (1 - beta) * g
            lr_t = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
                1 + math.cos(math.pi * it / max(1, args.n_iters - 1)))
            logS = float(np.clip(logS - lr_t * mom[0],
                                 math.log10(1e-3), math.log10(1e3)))
            logD = float(np.clip(logD - lr_t * mom[1],
                                 math.log10(1e-3), math.log10(1e3)))
            logM = float(np.clip(logM - lr_t * mom[2],
                                 math.log10(5e-3), math.log10(5.0)))
            hist.append({"it": it, "loss": L[0], "stiffness": 10 ** logS,
                         "damping": 10 ** logD, "mass_per_m": 10 ** logM})
            print(f"[it {it:03d}] loss={L[0]:.4f}  stiffness={10**logS:.4g}  "
                  f"damping={10**logD:.4g}  mass={10**logM:.4g}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            (ROOT / args.out).write_text(json.dumps(
                {"hist": hist, "stiffness": 10 ** logS, "damping": 10 ** logD,
                 "mass_per_m": 10 ** logM, "clips": clips,
                 "loss": "skeleton_only"}, indent=1))
    finally:
        pool.close()
        pool.join()

    print(f"\nSysID final: stiffness={10**logS:.4g}  damping={10**logD:.4g}  "
          f"mass_per_m={10**logM:.4g}  loss={hist[-1]['loss']:.4f}", flush=True)
    print(f"-> {ROOT / args.out}", flush=True)
    print("SYSID_DONE", flush=True)


if __name__ == "__main__":
    main()
