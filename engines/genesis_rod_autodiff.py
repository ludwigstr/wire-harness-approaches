"""Gradient-based SysID for the native Genesis rod -- DLO-Lab's actual method.

Everything before this fitted through finite differences or CMA-ES because the
MJCF hinge chain lives in Genesis's rigid solver, which has no backward pass,
and because the skeleton loss read a distance transform by INTEGER pixel index
(piecewise constant -> zero gradient). Both blockers are gone here:

  * the cable is a native RODEntity, whose solver DOES carry gradients
    (requires_grad on SimOptions + requires_grad_K/E on the material), and
  * the loss is a SOFT mask -- the rod's tube surface is sampled, projected
    through the board transform (which is affine, hence differentiable) and
    splatted with a Gaussian, then compared by soft-IoU. Every step is torch.

So dL/dK and dL/dE come straight out of `loss.backward()` +
`rope.get_stretching_stiffness_grad()`, and the optimiser is the one DLO-Lab
uses: momentum on the VECTOR-NORMALISED gradient with cosine decay, in log10
space.

ENGINE CAVEAT: the native rod exists only in the DLO-Lab fork; numbers here are
not comparable to the 1.3.3 hinge-chain pipeline.
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
sys.path.insert(0, str(ROOT / "engines"))
sys.path.insert(0, str(ROOT))


def soft_iou(soft, gt):
    inter = (soft * gt).sum()
    union = (soft + gt - soft * gt).sum()
    return 1.0 - inter / (union + 1e-6)


def board_project(pts_m, tf, down):
    """rod points (N,3) in metres -> pixel coords, differentiably.

    BoardTransform.to_px is affine (origin + x*e_x + y*e_y, all constants), so
    it carries straight over to torch and the gradient flows through it.
    """
    import torch
    ex = torch.as_tensor(np.asarray(tf.e_x, np.float32))
    ey = torch.as_tensor(np.asarray(tf.e_y, np.float32))
    o = torch.as_tensor(np.asarray(tf.origin_px, np.float32))
    s = float(1.0 / tf.mm_per_px) * 1000.0        # metres -> mm -> px units
    uv = o + pts_m[:, :1] * s * ex + pts_m[:, 1:2] * s * ey
    return uv / down


def splat(uv, H, W, sigma):
    """Differentiable Gaussian splat of points into an (H, W) soft mask."""
    import torch
    ys = torch.arange(H, dtype=torch.float32).view(H, 1, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, W, 1)
    u = uv[:, 0].view(1, 1, -1)
    v = uv[:, 1].view(1, 1, -1)
    d2 = (xs - u) ** 2 + (ys - v) ** 2
    acc = torch.exp(-d2 / (2.0 * sigma * sigma)).sum(-1)
    return 1.0 - torch.exp(-acc)


# The soft mask is built at a COARSER resolution than the stored GT (down=8 vs
# the GT's down=2) and from the rod CENTRELINE rather than its full tube
# surface. Both are memory decisions: the splat materialises an (H, W, N)
# tensor, and at full resolution with surface samples that is tens of GB. A
# gradient does not need the fine grid -- it needs a smooth one.
LOSS_DOWN = 8
SIGMA_PX = 3.0


def rod_soft_mask(rods, tf, H, W, sigma=SIGMA_PX, down=LOSS_DOWN):
    """One soft mask for all four cables, differentiable w.r.t. the rod state."""
    import torch
    acc = None
    for r in rods:
        p = r.get_state().pos            # (B, n_v, 3), torch, grad-carrying
        p = p.reshape(-1, 3)
        uv = board_project(p, tf, down)
        m = splat(uv, H, W, sigma)
        acc = m if acc is None else torch.maximum(acc, m)
    return acc


def gt_soft(clip, t0, frame, tf, H, W, down=LOSS_DOWN):
    """Real wire mask at this frame, resampled to the loss grid."""
    import cv2
    import torch
    from wire_model.perception.harness import wire_mask
    from fit_clip import read
    from mujoco_scene import DIR
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fr = read(cap, t0 + frame)
    cap.release()
    w, _ = wire_mask(fr, tf, clip=clip, orig_idx=t0 + frame)
    m = cv2.resize(w.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    return torch.as_tensor((m > 0.05).astype(np.float32))


def gt_dt(clip, t0, frame, tf, H, W, down=LOSS_DOWN):
    """Distance transform of the real wire mask, on the loss grid, in px.

    The project's skeleton loss was ruled out for gradients because it read the
    distance transform by INTEGER pixel index -- piecewise constant, so the
    derivative is zero almost everywhere. That objection is about the READ, not
    about the transform: the DT itself is a fixed image (it depends only on the
    ground truth), and sampling it BILINEARLY at the rod's projected position is
    differentiable w.r.t. that position. So the same quantity the project
    actually reports -- distance to the wire -- becomes a usable objective, and
    unlike soft-IoU it does not go flat when the overlap is small, which for a
    3mm cable on a 2128px frame is always.
    """
    import cv2
    import torch
    from wire_model.perception.harness import wire_mask
    from fit_clip import read
    from mujoco_scene import DIR
    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    fr = read(cap, t0 + frame)
    cap.release()
    w, _ = wire_mask(fr, tf, clip=clip, orig_idx=t0 + frame)
    m = cv2.resize(w.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    dt = cv2.distanceTransform((m <= 0.05).astype(np.uint8), cv2.DIST_L2, 5)
    return torch.as_tensor(dt.astype(np.float32))


def dt_loss(rod, tf, dt_img, H, W, down=LOSS_DOWN):
    """Mean bilinearly-sampled distance from the rod's vertices to the wire.

    The interpolation is written out by hand rather than left to
    F.grid_sample: grid_sample's backward never reaches the rod solver's
    adjoint -- measured dL/dK and dL/dE come back as EXACTLY 0.0 where the
    soft-IoU path returns 1e-12, and scaling the loss by 1e6 leaves them at
    0.0, so it is not underflow but a broken chain. Written as four weighted
    lookups the gradient is ordinary arithmetic on uv and flows the same way
    the Gaussian splat's does.
    """
    import torch
    p = rod.get_state().pos.reshape(-1, 3)
    uv = board_project(p, tf, down)
    x = uv[:, 0].clamp(0.0, W - 1.001)
    y = uv[:, 1].clamp(0.0, H - 1.001)
    x0 = x.detach().floor()
    y0 = y.detach().floor()
    fx = x - x0
    fy = y - y0
    ix = x0.long()
    iy = y0.long()
    i00 = dt_img[iy, ix]
    i10 = dt_img[iy, ix + 1]
    i01 = dt_img[iy + 1, ix]
    i11 = dt_img[iy + 1, ix + 1]
    top = i00 + (i10 - i00) * fx
    bot = i01 + (i11 - i01) * fx
    return (top + (bot - top) * fy).mean()


def _cable_worker(cable, args, conn):
    """One process, one rod, one scene, driven by (logK, logE) over a pipe.

    Why one rod per process: with four rods in a single scene the solver's
    adjoint is not reproducible -- identical runs returned gradients 22 orders
    of magnitude apart on rods 1 and 2 while rods 0 and 3 stayed put, and that
    survived dropping inextensibility, single binding vertices, single-threaded
    execution and reset_grad(). Built one at a time the same gradient comes
    back bit-identical (measured spread 1.00 over three repeats). The four
    cables are physically identical, so the parent simply SUMS their losses and
    gradients -- the gradient of a sum is the sum of the gradients, which is
    exactly the shared fit we want.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(",")[cable % len(
        args.gpus.split(","))]
    import torch
    import genesis as gs
    import genesis_rod_scene as R
    from mujoco_scene import track_clip, DIR
    from wire_model.perception.rectify import find_board
    from fit_clip import read
    import cv2

    cap = cv2.VideoCapture(f"{DIR}/{args.clip}.mp4")
    fps = (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0) or 4.0
    shp, t0, _, fps = track_clip(args.clip, t0=8,
                                 t1=int(round(args.t1_sec * fps)))
    tf = find_board(read(cap, 0))
    cap.release()
    n_frames = min(len(shp[0]), args.max_frames) if args.max_frames else len(shp[0])
    steps_per_frame = max(1, round((1.0 / fps) / 1e-3))
    H, W = 2128 // LOSS_DOWN, 1600 // LOSS_DOWN
    kf = sorted(set(np.linspace(max(1, n_frames // 8), n_frames - 1,
                                args.keyframes).round().astype(int).tolist()))
    use_dt = getattr(args, "loss", "iou") == "dt"
    gts = {i: (gt_dt if use_dt else gt_soft)(args.clip, t0, i, tf, H, W)
           for i in kf}
    names = ["hub"] + [f"plug{k}" for k in range(4)]
    conn.send(("ready", cable, n_frames, kf))

    def once(logK, logE, logs):
        """logs: per-hinge log10 bending scales (None = uniform)."""
        scene, ents, rods, _ = R.build_rod_scene(
            shp, 0, K=10.0 ** logK, E=10.0 ** logE, requires_grad=True,
            bptt_window=args.bptt, grad_clip=args.grad_clip,
            use_inextensible=False, only_cable=cable)
        scene.build(n_envs=1)
        rod = rods[0]
        n_v = rod.n_vertices
        rod.attach_to_rigid_link(R._mod_link(ents, "hub"),
                                 list(range(R.N_BIND)))
        rod.attach_to_rigid_link(R._mod_link(ents, f"plug{cable}"),
                                 list(range(n_v - R.N_BIND, n_v)))
        if logs is not None:
            # Per-JOINT bending stiffness: the patched solver multiplies the
            # rod's single bending stiffness by one scale per hinge. Upstream
            # stores only (n_rods, B), so a cable could only ever be uniform.
            rod.set_bending_scale(
                torch.as_tensor((10.0 ** np.asarray(logs)).astype(np.float32)))
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
                loss = loss + (dt_loss(rod, tf, gts[t], H, W) if use_dt
                               else soft_iou(rod_soft_mask([rod], tf, H, W),
                                             gts[t]))
        loss = loss / max(len(gts), 1)
        loss.backward()
        ln10 = math.log(10.0)
        gs_hinge = None
        if logs is not None:
            sc = 10.0 ** np.asarray(logs)
            gs_hinge = (rod.get_bending_scale_grad()[:, 0].numpy()
                        * sc * ln10).tolist()
        out = (float(loss),
               float(rod.get_stretching_stiffness_grad()[0]) * 10.0 ** logK * ln10,
               float(rod.get_bending_stiffness_grad()[0]) * 10.0 ** logE * ln10,
               gs_hinge)
        gs.destroy()
        return out

    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            conn.send(("ok",) + once(*msg))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}", 0.0, 0.0, None))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20260602_170108")
    ap.add_argument("--t1-sec", type=float, default=9)
    ap.add_argument("--n-iters", type=int, default=40)
    ap.add_argument("--keyframes", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = whole clip")
    ap.add_argument("--lr", type=float, default=0.20)
    ap.add_argument("--lr-min", type=float, default=0.03)
    ap.add_argument("--init-K", type=float, default=1.0e4)
    ap.add_argument("--init-E", type=float, default=1.0e5)
    ap.add_argument("--bptt", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--per-joint", type=int, default=0,
                    help="number of hinges to fit a bending-stiffness scale "
                         "for (14 for a 16-node cable); 0 = uniform")
    ap.add_argument("--freeze-ke", action="store_true",
                    help="hold K and E at their warm-started values and fit "
                         "ONLY the per-joint scales")
    ap.add_argument("--warm", default="",
                    help="json from an earlier fit to start K/E from")
    ap.add_argument("--loss", choices=["iou", "dt"], default="iou",
                    help="soft-IoU on the mask, or bilinear distance transform "
                         "(loss reads in loss-grid px)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--out", default="data/genesis_rod_autodiff.json")
    args = ap.parse_args()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")            # Genesis does not survive fork
    procs, conns = [], []
    for k in range(4):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=_cable_worker, args=(k, args, child),
                         daemon=True)
        pr.start()
        procs.append(pr)
        conns.append(parent)
    for c in conns:
        tag, cab, n_frames, kf = c.recv()
        print(f"kabel {cab} bereit: {n_frames} frames, keyframes {kf}",
              flush=True)

    logK, logE = math.log10(args.init_K), math.log10(args.init_E)
    if args.warm:
        w = json.loads((ROOT / args.warm).read_text())
        logK, logE = math.log10(float(w["K"])), math.log10(float(w["E"]))
        print(f"warmstart aus {args.warm}: K={w['K']:.4g}  E={w['E']:.4g}",
              flush=True)
    nj = int(args.per_joint)
    # log10 scales, 0 = scale 1.0 = exactly the uniform cable, so a per-joint
    # run starts from the uniform optimum rather than somewhere else.
    logs = np.zeros(nj) if nj else None
    mom, hist, best = np.zeros(2 + nj), [], None
    for it in range(args.n_iters):
        t0i = time.time()
        for c in conns:
            c.send((logK, logE, None if logs is None else logs.tolist()))
        loss, gK, gE, per = 0.0, 0.0, 0.0, []
        gj = np.zeros(nj)
        for k, c in enumerate(conns):
            r = c.recv()
            if r[0] != "ok":
                print(f"  kabel {k} FEHLER: {r[1]}", flush=True)
                continue
            loss += r[1]; gK += r[2]; gE += r[3]; per.append(r[1])
            if nj and r[4] is not None:
                gj += np.asarray(r[4], float)
        g = np.concatenate([[gK, gE], gj])
        if args.freeze_ke:
            # The warm-started K/E are already the uniform optimum. Left free
            # they dominate the normalised gradient and walk straight off it
            # (measured: 2.6554 -> 2.7704 over 20 iterations while K tripled),
            # so a per-joint run zeroes them and spends the whole step budget
            # on the profile -- which is the thing being tested.
            g[:2] = 0.0
        nrm = float(np.linalg.norm(g))
        if nrm > 1e-14:
            g = g / nrm
        mom = 0.5 * mom + 0.5 * g
        lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1 + math.cos(math.pi * it / max(1, args.n_iters - 1)))
        rec = {"it": it, "loss": loss / max(len(per), 1), "K": 10 ** logK,
               "E": 10 ** logE, "gK": gK, "gE": gE, "gnorm": nrm,
               "per_cable": per}
        if nj:
            rec["hinge_scale"] = (10.0 ** logs).tolist()
        hist.append(rec)
        if best is None or rec["loss"] < best["loss"]:
            best = dict(rec)
        print(f"[it {it:03d}] loss={rec['loss']:.4f}  K={10**logK:.4g}  "
              f"E={10**logE:.4g}  |grad|={nrm:.3e}  lr={lr:.3f}  "
              f"({time.time()-t0i:.0f}s)", flush=True)
        logK = float(np.clip(logK - lr * mom[0], 1.0, 8.0))
        logE = float(np.clip(logE - lr * mom[1], 1.0, 8.0))
        if nj:
            # +/-1 decade around the shared stiffness is plenty of profile;
            # wider just lets single hinges run away and stall the rollout.
            logs = np.clip(logs - lr * mom[2:], -1.0, 1.0)
        (ROOT / args.out).write_text(json.dumps(
            {"hist": hist, "best": best, "clip": args.clip, "loss": args.loss,
             "K": best["K"], "E": best["E"],
             "hinge_scale": best.get("hinge_scale")}, indent=1))
    for c in conns:
        c.send(None)
    for pr in procs:
        pr.join(timeout=30)
    print(f"\nbester stand: K={best['K']:.4g}  E={best['E']:.4g}  "
          f"loss={best['loss']:.4f}", flush=True)
    print("AUTODIFF_DONE", flush=True)


if __name__ == "__main__":
    main()
