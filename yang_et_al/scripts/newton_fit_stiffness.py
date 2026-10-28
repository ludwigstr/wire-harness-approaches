"""Coordinate-descent fit of a per-cable, per-joint bend/twist stiffness for
Newton's rod, mirroring scripts/fit_weld_stiffness.py's method exactly (same
CLIP_TRIMS, same 5-clip fast training subset, same 4-candidates-per-
coordinate greedy search, same checkpoint/resume pattern) -- 4 cables x 14
joints = 56 independent target_ke values (twist coupled equal to bend, same
as add_rod's own default when twist isn't set separately -- matches the
weld-fit's parameter count).

One optimisation the MuJoCo version couldn't make: MuJoCo's 4 cables all
live in ONE compiled model, so every eval steps all 4 whether or not their
joints changed. Newton's rods are fully independent scenes -- tuning cable
c's joints only ever affects cable c's own physics, so each coordinate's
candidate evaluations build and step a SINGLE-cable scene, not all 4. That's
roughly a 4x speedup per evaluation on top of matching the method.

Damping is NOT fit here (kept at the already-tuned global bend_damping=0.5
from genesis... no, from Newton's own default -- see below); only per-joint
target_ke (bend, twist coupled) varies, same scope as the MuJoCo weld fit.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, CAV_Z, DIR as CLIP_DIR
from newton_full_scene import make_kinematic

ROOT = Path(__file__).resolve().parents[1]
N_JOINTS = 14
BEND_DAMPING = 0.05  # global, matches newton_full_scene.py's own established default; not fit here

# The clip set now lives in clip_set.py, which imports no simulator -- so
# Genesis-only environments (no warp/newton) can use it too. Re-exported
# here unchanged so every existing caller keeps working.
from clip_set import (CLIP_TRIMS, EXCLUDED_TRACKING,  # noqa: F401
                      ALL_CLIP_TRIMS as _ALL_CLIP_TRIMS)
TRAIN_CLIPS_FAST = ["20260602_170108", "20260602_121006", "20260602_162650",
                    "20260602_193953", "20260602_165113"]
CANDIDATES = [0.02, 0.1, 0.5, 2.0]


def clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    cap = cv2.VideoCapture(f"{CLIP_DIR}/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def load_clips(clip_names, log):
    cache = {}
    for c in clip_names:
        t1 = clip_frame_t1(c, CLIP_TRIMS[c])
        t_start = time.time()
        shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
        cache[c] = (shp, fps)
        log(f"  getrackt: {c} ({len(shp[0])} Frames, {time.time()-t_start:.0f}s)")
    return cache


def eval_single_cable(shp, fps, cable, per_joint_ke, per_joint_kd=None, radius=0.003,
                      stretch_stiffness=1.0e5, dt=1e-3):
    # Density/mass intentionally left at Newton's own default (unchanged from
    # newton_scene.py/newton_full_scene.py) -- this fit only varies per-joint
    # bend/twist stiffness (and, if per_joint_kd is given, damping), not
    # mass, to isolate those as the levers.
    import newton
    from newton.utils import create_parallel_transport_cable_quaternions

    n_frames = len(shp[0])
    steps_per_frame = max(1, round((1.0 / fps) / dt))

    def route3(t):
        P = shp[cable][t].astype(float)
        z = np.full((len(P), 1), CAV_Z)
        return np.hstack([P, z])

    r0 = route3(0)
    builder = newton.ModelBuilder()
    # bend_damping=0.0 here (not BEND_DAMPING) whenever per_joint_kd is
    # given -- same reasoning as bend_stiffness=0.0 below: add_rod's own
    # material bend/twist spring-damper is a SEPARATE mechanism from the
    # joint_target_ke/kd PD-control one this fit tunes; leaving material
    # damping non-zero would mix an unfit, fixed damping source in with the
    # per-joint one, muddying what's actually being fit.
    bodies, joints = builder.add_rod(
        positions=[wp.vec3(*p) for p in r0], radius=radius,
        stretch_stiffness=stretch_stiffness, bend_stiffness=0.0,
        bend_damping=(0.0 if per_joint_kd is not None else BEND_DAMPING),
        body_frame_origin="start", label="cable")
    assert len(joints) == N_JOINTS, f"expected {N_JOINTS} joints, got {len(joints)}"
    for j in range(N_JOINTS):
        builder.joint_target_ke[4 * j + 2] = float(per_joint_ke[j])  # bend
        builder.joint_target_ke[4 * j + 3] = float(per_joint_ke[j])  # twist, coupled
        if per_joint_kd is not None:
            builder.joint_target_kd[4 * j + 2] = float(per_joint_kd[j])
            builder.joint_target_kd[4 * j + 3] = float(per_joint_kd[j])
    make_kinematic(builder, bodies[0])
    make_kinematic(builder, bodies[-1])
    builder.color()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    collision_pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverVBD(model, iterations=5)
    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    def pin_ends(t):
        bq = state0.body_q.numpy()
        r = route3(t)
        pts = [wp.vec3(*p) for p in r]
        edge_q = create_parallel_transport_cable_quaternions(pts)
        bq[bodies[0]] = np.concatenate([r[0], np.array(list(edge_q[0]))])
        bq[bodies[-1]] = np.concatenate([r[-2], np.array(list(edge_q[-1]))])
        state0.body_q.assign(bq)
        state1.body_q.assign(bq)

    def step_physics():
        nonlocal state0, state1
        state0.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    pin_ends(0)
    for _ in range(300):
        step_physics()
    for t in range(n_frames):
        pin_ends(t)
        for _ in range(steps_per_frame):
            step_physics()

    bq = state0.body_q.numpy()
    sim_pts = bq[bodies][:, :2]
    real_pts = shp[cable][-1][:len(bodies)]
    return float(np.linalg.norm(sim_pts - real_pts, axis=1).mean()) * 1000


def eval_full_clip(shp, fps, best_4x14, best_kd_4x14=None):
    """All 4 cables (each independent), for the full-16-clip validation
    pass -- returns (4,) final-frame errors."""
    if best_kd_4x14 is None:
        return np.array([eval_single_cable(shp, fps, c, best_4x14[c]) for c in range(4)])
    return np.array([eval_single_cable(shp, fps, c, best_4x14[c], best_kd_4x14[c]) for c in range(4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/newton_stiffness_fit_progress.npz")
    ap.add_argument("--log", default="data/newton_stiffness_fit_log.jsonl")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    all_clip_names = list(CLIP_TRIMS.keys())
    train_clips = TRAIN_CLIPS_FAST

    log_path = ROOT / args.log
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    out_path = ROOT / args.out
    best = np.full((4, N_JOINTS), 0.5)  # start at newton_full_scene.py's established default (0.5) -- NOT a tuned optimum, that was never swept for Newton specifically (0.1 is Genesis's tuned value, a different engine, mistakenly assumed equivalent in an earlier draft of this script)
    history = []
    base_avg = None
    base_per_clip = None
    if args.resume and out_path.exists():
        ckpt = np.load(out_path, allow_pickle=True)
        best = ckpt["best"]
        history = json.loads(str(ckpt["history"]))
        base_avg = float(ckpt["base_avg"])
        base_per_clip = json.loads(str(ckpt["base_per_clip"]))
        log(f"=== Newton Stiffness Fit FORTGESETZT ({len(history)}/{4*N_JOINTS} Koordinaten schon fertig) ===")
    else:
        log("=== Newton Stiffness Fit gestartet ===")
    log(f"Trainings-Clips ({len(train_clips)}): {train_clips}")
    log(f"Kandidaten pro Koordinate: {CANDIDATES}")
    log("Tracke alle 16 Clips einmal (Cache fuer den ganzen Lauf)...")
    cache = load_clips(all_clip_names, log)

    if base_avg is None:
        log("Baseline (uniform 0.1, kein per-joint Fit) auf allen 16 Clips...")
        base = {}
        for c in all_clip_names:
            shp, fps = cache[c]
            base[c] = eval_full_clip(shp, fps, best)
        base_avg = float(np.mean([e.mean() for e in base.values()]))
        base_per_clip = {c: float(e.mean()) for c, e in base.items()}
        log(f"Baseline Mittel (Endfehler, 16 Clips): {base_avg:.2f} mm/Knoten")
        np.savez(out_path, best=best, history=json.dumps(history),
                base_avg=base_avg, base_per_clip=json.dumps(base_per_clip))
    else:
        log(f"Baseline aus Checkpoint: {base_avg:.2f} mm/Knoten")

    done_coords = {(h["cable"], h["joint"]) for h in history}
    n_done_before = len(history)
    n_done_session = 0
    t_run_start = time.time()
    total_coords = 4 * N_JOINTS
    for c in range(4):
        for j in range(N_JOINTS):
            if (c, j) in done_coords:
                continue
            t_c = time.time()
            scores = []
            for cand in CANDIDATES:
                trial_row = best[c].copy()
                trial_row[j] = cand
                errs = []
                for clip in train_clips:
                    shp, fps = cache[clip]
                    errs.append(eval_single_cable(shp, fps, c, trial_row))
                scores.append(float(np.mean(errs)))
            best_idx = int(np.argmin(scores))
            best[c, j] = CANDIDATES[best_idx]
            history.append({"cable": c, "joint": j, "scores": scores,
                            "chosen": CANDIDATES[best_idx]})
            np.savez(out_path, best=best, history=json.dumps(history),
                    base_avg=base_avg, base_per_clip=json.dumps(base_per_clip))
            n_done_session += 1
            coord_i = n_done_before + n_done_session
            elapsed = time.time() - t_run_start
            eta = elapsed / n_done_session * (total_coords - coord_i)
            log(f"[{coord_i}/{total_coords}] M{c} Gelenk{j}: "
                f"scores={[round(s,1) for s in scores]} -> {CANDIDATES[best_idx]:g} "
                f"({time.time()-t_c:.0f}s, ETA {eta/60:.0f} min)")

    log("=== Pass 1 fertig, validiere auf allen 16 Clips ===")
    final = {}
    for c in all_clip_names:
        shp, fps = cache[c]
        final[c] = eval_full_clip(shp, fps, best)
    final_avg = float(np.mean([e.mean() for e in final.values()]))
    per_clip = {c: float(e.mean()) for c, e in final.items()}
    np.savez(out_path, best=best, history=json.dumps(history),
            base_avg=base_avg, base_per_clip=json.dumps(base_per_clip),
            final_avg=final_avg, per_clip=json.dumps(per_clip))
    log(f"Baseline (uniform 0.1):        {base_avg:.2f} mm/Knoten")
    log(f"Gefittete Pro-Gelenk-Steifigkeit: {final_avg:.2f} mm/Knoten")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} {v:6.1f}  (Baseline {base_per_clip[c]:6.1f})")
    log(f"-> {out_path}")
    log("=== FERTIG ===")
    print("NEWTON_STIFFNESS_FIT_FERTIG")


if __name__ == "__main__":
    main()
