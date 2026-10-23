"""Per-hinge rest curvature, fitted over ALL 14 clips -- the same footing
Newton's 9.0 mm was calibrated on.

Newton's headline number is IN-SAMPLE: 29 parameters fitted against every clip
(plus a depth-based z objective), and its cable ends are kinematically pinned so
2 of 16 nodes carry zero error by construction. Every rod fit so far used one or
two clips and was measured on fourteen, i.e. mostly out-of-sample. This puts the
rod on Newton's footing so the comparison means something.

Cost: 14 clips x 4 cables = 56 rollouts per iteration. Rather than 56 resident
processes on two GPUs, eight workers each own a fixed slice of the jobs and
cache their clips' tracking, so the expensive part (track_clip + ground-truth
distance transforms) is paid once per worker, not once per iteration.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
from dlolab_path import DLO_LAB  # noqa: E402
sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(ROOT))

BPTT = int(os.environ.get("LS_BPTT", "120"))
N_ITERS = int(os.environ.get("LS_ITERS", "20"))
LR = float(os.environ.get("LS_LR", "0.04"))
LR_MIN = float(os.environ.get("LS_LRMIN", "0.005"))
KEYFRAMES = int(os.environ.get("LS_KF", "4"))
# Cap the rollout length. All 14 clips at full length costs ~3000 frames per
# iteration, measured at over two hours per iteration -- 40+ hours for the run.
# The point of using every clip is that every clip is REPRESENTED (Newton's
# calibration is in-sample on all of them), not that every frame is. A capped
# window keeps all 14 in the fit at a tenth of the cost.
MAX_FRAMES = int(os.environ.get("LS_MAXF", "30"))
WARM = os.environ.get("LS_WARM", "data/genesis_rod_kappa_fit.json")
BASE = os.environ.get("LS_BASE", "data/genesis_rod_bigfit_bptt120.json")
OUT = os.environ.get("LS_OUT", "data/genesis_rod_kappa_fit_all.json")


def worker(jobs, gpu, conn):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    import genesis as gs
    import genesis_rod_scene as R
    from genesis_rod_autodiff import LOSS_DOWN, gt_dt, dt_loss
    from mujoco_scene import track_clip, DIR
    from wire_model.perception.rectify import find_board
    from fit_clip import read
    import cv2
    from clip_set import CLIP_TRIMS

    H, W = 2128 // LOSS_DOWN, 1600 // LOSS_DOWN
    names = ["hub"] + [f"plug{k}" for k in range(4)]
    d = json.loads((ROOT / BASE).read_text())
    K, E = float(d["K"]), float(d["E"])
    hs = np.asarray(d["hinge_scale"], float)
    ss = np.asarray(d["stretch_scale"], float)

    cache = {}
    for clip, _ in jobs:
        if clip in cache:
            continue
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fp = (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
        t1s = CLIP_TRIMS.get(clip)
        shp, t0, _, fps = track_clip(
            clip, t0=8, t1=None if t1s is None else int(round(t1s * fp)))
        tf = find_board(read(cap, 0))
        cap.release()
        nf = min(len(shp[0]), MAX_FRAMES) if MAX_FRAMES else len(shp[0])
        kf = sorted(set(np.linspace(max(1, nf // 8), nf - 1, KEYFRAMES)
                        .round().astype(int).tolist()))
        cache[clip] = dict(shp=shp, tf=tf, nf=nf,
                           spf=max(1, round((1.0 / fps) / 1e-3)),
                           gts={i: gt_dt(clip, t0, i, tf, H, W) for i in kf})
    conn.send(("ready", len(jobs), sum(cache[c]["nf"] for c, _ in jobs)))

    def once(clip, cable, kap):
        c = cache[clip]
        shp, tf, nf, spf, gts = (c["shp"], c["tf"], c["nf"], c["spf"], c["gts"])
        scene, ents, rods, _ = R.build_rod_scene(
            shp, 0, K=K, E=E, requires_grad=True, bptt_window=BPTT,
            grad_clip=1.0, use_inextensible=False, only_cable=cable)
        scene.build(n_envs=1)
        rod = rods[0]
        n_v = rod.n_vertices
        rod.attach_to_rigid_link(R._mod_link(ents, "hub"),
                                 list(range(R.N_BIND)))
        rod.attach_to_rigid_link(R._mod_link(ents, f"plug{cable}"),
                                 list(range(n_v - R.N_BIND, n_v)))
        rod.set_stretching_scale(torch.as_tensor(ss.astype(np.float32)))
        rod.set_bending_scale(torch.as_tensor(hs.astype(np.float32)))
        rod.set_kappa_rest_offset(torch.as_tensor(kap.astype(np.float32)))
        dofs, dtop = {}, {}
        for n in names:
            e = ents[n]
            dofs[n] = (e.get_joint(f"{n}_x").dofs_idx_local
                       + e.get_joint(f"{n}_y").dofs_idx_local)
            e.set_dofs_kp(np.full(2, 4000.0), dofs[n])
            e.set_dofs_kv(np.full(2, 120.0), dofs[n])
            et = ents[f"{n}_top"]
            xy = (et.get_joint(f"{n}_x").dofs_idx_local
                  + et.get_joint(f"{n}_y").dofs_idx_local)
            tu = et.get_joint(f"{n}_turn").dofs_idx_local
            dtop[n] = (xy, tu)
            et.set_dofs_kp(np.full(2, 4000.0), xy)
            et.set_dofs_kv(np.full(2, 120.0), xy)
            et.set_dofs_kp(np.full(1, 60.0), tu)
            et.set_dofs_kv(np.full(1, 4.0), tu)
        p0 = R.MJ_poses(shp, 0)
        base = {n: np.asarray(p0[n][0], float) for n in names}
        by = {n: float(p0[n][1]) for n in names}
        prev = dict(by)
        loss = torch.zeros((), dtype=torch.float32)
        for t in range(nf):
            p = R.MJ_poses(shp, t)
            for n in names:
                pos, yaw = p[n]
                prev[n] += np.arctan2(np.sin(yaw - prev[n]),
                                      np.cos(yaw - prev[n]))
                rel = np.asarray(pos, float) - base[n]
                ents[n].control_dofs_position(rel, dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel, dtop[n][0])
                ents[f"{n}_top"].control_dofs_position(
                    np.array([prev[n] - by[n]]), dtop[n][1])
            for _ in range(spf):
                scene.step()
            if t in gts:
                loss = loss + dt_loss(rod, tf, gts[t], H, W)
        loss = loss / max(len(gts), 1)
        loss.backward()
        out = (float(loss), rod.get_kappa_rest_grad()[:, :, 0].numpy())
        gs.destroy()
        return out

    while True:
        m = conn.recv()
        if m is None:
            break
        kap = np.asarray(m, float)
        L, g, n_ok = 0.0, np.zeros((14, 2)), 0
        for clip, cable in jobs:
            try:
                l, gg = once(clip, cable, kap)
                L += l
                g += gg
                n_ok += 1
            except Exception as e:
                conn.send(("err", f"{clip}/{cable}: {type(e).__name__}: {e}"))
                return
        conn.send(("ok", L, g.tolist(), n_ok))
    conn.close()


def main():
    from clip_set import CLIP_TRIMS
    import multiprocessing as mp

    clips = sorted(CLIP_TRIMS)
    jobs = [(c, k) for c in clips for k in range(4)]
    n_w = int(os.environ.get("LS_WORKERS", "8"))
    gpus = os.environ.get("LS_GPUS", "0,1").split(",")
    slices = [jobs[i::n_w] for i in range(n_w)]

    ctx = mp.get_context("spawn")
    procs, conns = [], []
    for i, sl in enumerate(slices):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=worker,
                         args=(sl, gpus[i % len(gpus)], b), daemon=True)
        pr.start()
        procs.append(pr)
        conns.append(a)
    total = 0
    for c in conns:
        tag, n_jobs, n_frames = c.recv()
        total += n_frames
    print(f"{n_w} arbeiter, {len(jobs)} rollouts je iteration ueber "
          f"{len(clips)} clips ({total} frames gesamt)", flush=True)

    kap = np.zeros((14, 2))
    if WARM and (ROOT / WARM).exists():
        kap = np.asarray(json.loads((ROOT / WARM).read_text())["kappa"], float)
        print(f"warmstart aus {WARM}", flush=True)
    mom, hist, best = np.zeros((14, 2)), [], None
    for it in range(N_ITERS):
        t0 = time.time()
        for c in conns:
            c.send(kap.tolist())
        L, g, n_ok = 0.0, np.zeros((14, 2)), 0
        for c in conns:
            r = c.recv()
            if r[0] != "ok":
                print(f"  FEHLER {r[1]}", flush=True)
                continue
            L += r[1]
            g += np.asarray(r[2], float)
            n_ok += r[3]
        L /= max(n_ok, 1)
        nrm = float(np.linalg.norm(g))
        gn = g / nrm if nrm > 1e-14 else g
        mom = 0.3 * mom + 0.7 * gn
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (
            1 + math.cos(math.pi * it / max(1, N_ITERS - 1)))
        rec = {"it": it, "loss": L, "gnorm": nrm, "n_ok": n_ok,
               "kappa": kap.tolist()}
        hist.append(rec)
        if best is None or L < best["loss"]:
            best = dict(rec)
        print(f"[it {it:03d}] loss={L:.5f}  |grad|={nrm:.3e}  lr={lr:.4f}  "
              f"rollouts={n_ok}  ({time.time()-t0:.0f}s)", flush=True)
        kap = kap - lr * mom
        (ROOT / OUT).write_text(json.dumps(
            {"hist": hist, "best": best, "clips": clips,
             "kappa": best["kappa"]}, indent=1))
    for c in conns:
        c.send(None)
    print(f"\nbester stand: loss={best['loss']:.5f} it {best['it']}",
          flush=True)
    print("KAPPA_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
