"""The big fit: every local stiffness the patched solver can express,
by gradient, across SEVERAL clips at once.

Parameter vector (46 for a 16-node cable), all shared by the four physically
identical cables:

  logK, logE, logG        global stretching / bending / twisting stiffness
  15 stretching scales    per EDGE
  14 bending scales       per HINGE  (= per joint)
  14 twisting scales      per HINGE

Everything is fitted by reverse-mode autodiff -- the scales multiply into the
forward energy kernels, whose adjoints this fork generates automatically.

WHY MULTIPLE CLIPS: every rod fit so far was fitted on ONE clip (170108) and
measured on 14. That leaves generalisation on the table -- it is the single
biggest lever left, bigger than adding more parameters to a one-clip fit. One
worker process per (clip, cable), losses and gradients summed.

WHY ONE ROD PER PROCESS: with several rods in one scene this solver's adjoint
is not reproducible (identical runs gave gradients 22 orders of magnitude
apart). See genesis_rod_autodiff.py.

NOT FITTED HERE: friction. mu_s/mu_k are per-vertex, but vertices_param is
declared needs_grad=False and the friction adjoint is hand-written, so no
gradient reaches them. They need a search, not a gradient.
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
from dlolab_path import DLO_LAB  # noqa: E402
sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(DLO_LAB / "examples"))
sys.path.insert(0, str(ROOT))

from genesis_rod_autodiff import LOSS_DOWN, gt_dt, dt_loss  # noqa: E402


def unpack(x, n_e, n_h):
    """flat vector -> (logK, logE, logG, stretch, bend, twist)"""
    return (float(x[0]), float(x[1]), float(x[2]),
            x[3:3 + n_e], x[3 + n_e:3 + n_e + n_h],
            x[3 + n_e + n_h:3 + n_e + 2 * n_h])


def _worker(job, args, conn):
    clip, cable = job
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[
        hash((clip, cable)) % len(args.gpus.split(","))]
    import torch
    import genesis as gs
    import genesis_rod_scene as R
    from mujoco_scene import track_clip, DIR
    from wire_model.perception.rectify import find_board
    from fit_clip import read
    import cv2
    from clip_set import CLIP_TRIMS

    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fps_p = (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
    t1s = CLIP_TRIMS.get(clip)
    shp, t0, _, fps = track_clip(
        clip, t0=8, t1=None if t1s is None else int(round(t1s * fps_p)))
    tf = find_board(read(cap, 0))
    cap.release()
    n_frames = len(shp[0])
    steps_per_frame = max(1, round((1.0 / fps) / 1e-3))
    H, W = 2128 // LOSS_DOWN, 1600 // LOSS_DOWN
    kf = sorted(set(np.linspace(max(1, n_frames // 8), n_frames - 1,
                                args.keyframes).round().astype(int).tolist()))
    gts = {i: gt_dt(clip, t0, i, tf, H, W) for i in kf}
    names = ["hub"] + [f"plug{k}" for k in range(4)]
    conn.send(("ready", clip, cable, n_frames))

    def once(x):
        logK, logE, logG, ls, lb, lt = unpack(x, args.n_edges, args.n_hinges)
        G_use = 0.0 if args.no_twist else 10.0 ** logG
        scene, ents, rods, _ = R.build_rod_scene(
            shp, 0, K=10.0 ** logK, E=10.0 ** logE, G=G_use,
            requires_grad=True, bptt_window=args.bptt,
            grad_clip=args.grad_clip, use_inextensible=False,
            only_cable=cable, n_nodes=args.n_nodes or None,
            table_fraction=args.table_fraction or None)
        scene.build(n_envs=1)
        rod = rods[0]
        n_v = rod.n_vertices
        rod.attach_to_rigid_link(R._mod_link(ents, "hub"),
                                 list(range(R.N_BIND)))
        rod.attach_to_rigid_link(R._mod_link(ents, f"plug{cable}"),
                                 list(range(n_v - R.N_BIND, n_v)))
        rod.set_stretching_scale(torch.as_tensor(
            (10.0 ** np.asarray(ls)).astype(np.float32)))
        rod.set_bending_scale(torch.as_tensor(
            (10.0 ** np.asarray(lb)).astype(np.float32)))
        if not args.no_twist:
            rod.set_twisting_scale(torch.as_tensor(
                (10.0 ** np.asarray(lt)).astype(np.float32)))
        if args.slack != 1.0:
            rod.set_rest_length_scale(args.slack)

        dofs, dofs_top = {}, {}
        for n in names:
            e = ents[n]
            dofs[n] = (e.get_joint(f"{n}_x").dofs_idx_local
                       + e.get_joint(f"{n}_y").dofs_idx_local)
            e.set_dofs_kp(np.full(2, 4000.0), dofs[n])
            e.set_dofs_kv(np.full(2, 120.0), dofs[n])
            et = ents[f"{n}_top"]
            xy = (et.get_joint(f"{n}_x").dofs_idx_local
                  + et.get_joint(f"{n}_y").dofs_idx_local)
            turn = et.get_joint(f"{n}_turn").dofs_idx_local
            dofs_top[n] = (xy, turn)
            et.set_dofs_kp(np.full(2, 4000.0), xy)
            et.set_dofs_kv(np.full(2, 120.0), xy)
            et.set_dofs_kp(np.full(1, 60.0), turn)
            et.set_dofs_kv(np.full(1, 4.0), turn)
        p0 = R.MJ_poses(shp, 0)
        base = {n: np.asarray(p0[n][0], float) for n in names}
        base_yaw = {n: float(p0[n][1]) for n in names}
        prev = dict(base_yaw)

        loss = torch.zeros((), dtype=torch.float32)
        for t in range(n_frames):
            p = R.MJ_poses(shp, t)
            for n in names:
                pos, yaw = p[n]
                prev[n] = prev[n] + np.arctan2(np.sin(yaw - prev[n]),
                                               np.cos(yaw - prev[n]))
                rel = np.asarray(pos, float) - base[n]
                ents[n].control_dofs_position(rel, dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel, dofs_top[n][0])
                ents[f"{n}_top"].control_dofs_position(
                    np.array([prev[n] - base_yaw[n]]), dofs_top[n][1])
            for _ in range(steps_per_frame):
                scene.step()
            if t in gts:
                loss = loss + dt_loss(rod, tf, gts[t], H, W)
        loss = loss / max(len(gts), 1)
        loss.backward()

        ln10 = math.log(10.0)
        g = np.zeros(len(x))
        g[0] = float(rod.get_stretching_stiffness_grad()[0]) * 10.0 ** logK * ln10
        g[1] = float(rod.get_bending_stiffness_grad()[0]) * 10.0 ** logE * ln10
        g[2] = (0.0 if args.no_twist else
                float(rod.get_twisting_stiffness_grad()[0]) * 10.0 ** logG * ln10)
        n_e, n_h = args.n_edges, args.n_hinges
        g[3:3 + n_e] = (rod.get_stretching_scale_grad()[:, 0].numpy()
                        * 10.0 ** np.asarray(ls) * ln10)
        g[3 + n_e:3 + n_e + n_h] = (rod.get_bending_scale_grad()[:, 0].numpy()
                                    * 10.0 ** np.asarray(lb) * ln10)
        if not args.no_twist:
            g[3 + n_e + n_h:] = (rod.get_twisting_scale_grad()[:, 0].numpy()
                                 * 10.0 ** np.asarray(lt) * ln10)
        out = (float(loss), g.tolist())
        gs.destroy()
        return out

    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            conn.send(("ok",) + once(np.asarray(msg, float)))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}", None))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*",
                    default=["20260602_170108", "20260602_114019"])
    ap.add_argument("--n-iters", type=int, default=45)
    ap.add_argument("--keyframes", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--lr-min", type=float, default=0.01)
    ap.add_argument("--bptt", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--slack", type=float, default=1.0)
    ap.add_argument("--n-nodes", type=int, default=0)
    ap.add_argument("--table-fraction", type=float, default=0.0)
    ap.add_argument("--warm", default="data/genesis_rod_autodiff_perjoint_frozen.json")
    ap.add_argument("--init-G", type=float, default=1.0e6)
    ap.add_argument("--no-twist", action="store_true",
                    help="keep G=0 (twisting off, as in the validated 16.95mm "
                         "configuration). Measured: switching twisting ON "
                         "moves the optimum away from the warm-started bending "
                         "profile and the fit then only climbs, while with "
                         "G=0 a line search along the gradient improves at "
                         "every step size tried.")
    ap.add_argument("--freeze-globals", action="store_true",
                    help="hold K/E/G at their warm-started values and spend "
                         "the whole step budget on the local profiles")
    ap.add_argument("--momentum", type=float, default=0.3)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out", default="data/genesis_rod_bigfit.json")
    args = ap.parse_args()

    n_nodes = args.n_nodes or 16
    args.n_edges = n_nodes - 1
    args.n_hinges = n_nodes - 2

    w = json.loads((ROOT / args.warm).read_text())
    x = np.zeros(3 + args.n_edges + 2 * args.n_hinges)
    x[0], x[1] = math.log10(float(w["K"])), math.log10(float(w["E"]))
    x[2] = math.log10(args.init_G)
    hs = w.get("hinge_scale")
    if hs:
        hs = np.asarray(hs, float)
        if len(hs) != args.n_hinges:
            u = np.linspace(0, len(hs) - 1.0, args.n_hinges)
            hs = np.interp(u, np.arange(len(hs)), hs)
        x[3 + args.n_edges:3 + args.n_edges + args.n_hinges] = np.log10(hs)
    print(f"start: K={10**x[0]:.4g} E={10**x[1]:.4g} G={10**x[2]:.4g}, "
          f"{len(x)} parameter, clips {args.clips}", flush=True)

    jobs = [(c, k) for c in args.clips for k in range(4)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs, conns = [], []
    for j in jobs:
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=_worker, args=(j, args, child), daemon=True)
        pr.start()
        procs.append(pr)
        conns.append(parent)
    for c in conns:
        tag, clip, cable, nf = c.recv()
        print(f"  {clip} kabel {cable} bereit ({nf} frames)", flush=True)

    mom, hist, best = np.zeros(len(x)), [], None
    for it in range(args.n_iters):
        t0i = time.time()
        for c in conns:
            c.send(x.tolist())
        loss, g, n_ok = 0.0, np.zeros(len(x)), 0
        for i, c in enumerate(conns):
            r = c.recv()
            if r[0] != "ok":
                print(f"  job {jobs[i]} FEHLER: {r[1]}", flush=True)
                continue
            loss += r[1]
            g += np.asarray(r[2], float)
            n_ok += 1
        loss /= max(n_ok, 1)
        if args.freeze_globals:
            # Measured twice now: left free, the three global stiffnesses
            # dominate the vector-normalised gradient and walk straight off
            # the warm-started optimum (2.8354 -> 2.9269 over 30 iterations
            # while K tripled). Zeroing them spends every step on the local
            # profiles, which is what a per-element fit is for.
            g[:3] = 0.0
        nrm = float(np.linalg.norm(g))
        gn = g / nrm if nrm > 1e-14 else g
        mom = args.momentum * mom + (1.0 - args.momentum) * gn
        lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1 + math.cos(math.pi * it / max(1, args.n_iters - 1)))
        rec = {"it": it, "loss": loss, "gnorm": nrm, "n_ok": n_ok,
               "K": 10 ** x[0], "E": 10 ** x[1], "G": 10 ** x[2],
               "stretch_scale": (10 ** x[3:3 + args.n_edges]).tolist(),
               "hinge_scale": (10 ** x[3 + args.n_edges:
                                       3 + args.n_edges + args.n_hinges]).tolist(),
               "twist_scale": (10 ** x[3 + args.n_edges + args.n_hinges:]).tolist()}
        hist.append(rec)
        if best is None or loss < best["loss"]:
            best = dict(rec)
        print(f"[it {it:03d}] loss={loss:.4f}  K={10**x[0]:.4g}  "
              f"E={10**x[1]:.4g}  G={10**x[2]:.4g}  |grad|={nrm:.3e}  "
              f"({time.time()-t0i:.0f}s)", flush=True)
        x = x - lr * mom
        x[:3] = np.clip(x[:3], 1.0, 8.0)
        x[3:] = np.clip(x[3:], -1.0, 1.0)
        (ROOT / args.out).write_text(json.dumps(
            {"hist": hist, "best": best, "clips": args.clips,
             "K": best["K"], "E": best["E"],
             "G": (0.0 if args.no_twist else best["G"]),
             "stretch_scale": best["stretch_scale"],
             "hinge_scale": best["hinge_scale"],
             "twist_scale": best["twist_scale"],
             "slack": args.slack, "n_nodes": args.n_nodes,
             "table_fraction": args.table_fraction}, indent=1))
    for c in conns:
        c.send(None)
    for pr in procs:
        pr.join(timeout=30)
    print(f"\nbester stand: loss={best['loss']:.4f}  it {best['it']}", flush=True)
    print("BIGFIT_DONE", flush=True)


if __name__ == "__main__":
    main()
