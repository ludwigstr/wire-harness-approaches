"""Yang et al. (RAS 2022) step 2: a large synthetic dataset from the
IDENTIFIED simulator (MuJoCo scene with Theta* from yang_identify.py).

Anchored in real states, as scripts/make_dataset_real.py does for the
mass-spring generator (lesson of 2026-10-23: never invent cable shapes):
  * every rollout starts from a REAL tracked scene: frame 0 of one of the 103
    X-Planar clips (never the 14 evaluation clips); the cables' rest shape is
    that frame, exactly as in the evaluation;
  * the movers follow that clip's REAL measured pose sequence, transformed per
    rollout (the paper's "random manipulation" in place of a robot):
      - displacement scale s:  pose(t) = pose(0) + s (pose(t) - pose(0)),
        s = 1 for 30 % of the rollouts, else U(0.6, 2.0);
      - time warp w: the sequence replayed at speed w, w = 1 for 50 %, else
        U(0.5, 2.0) (length capped at the clip's own length);
      - random smooth perturbation for 50 % of the rollouts: per mover a sum of
        3 random sinusoids (period 4-40 s), amplitude U(0, 40) mm in x/y and
        U(0, 0.2) rad in yaw, ramped in from zero at t = 0.
Recorded at the video cadence (one frame = 0.25 s, the step of the network):
the 4 x 16 cable vertices (x, y) after the 300-step settle and after every
frame, plus the connector offsets (tip-to-cavity vectors; "gap" = largest
change of any of them since frame 0) to drop transitions in which the scaled
motion over-stretched the inextensible cable.

    python scripts/yang_make_dataset.py --per-clip 8 --workers 32 --out data/wm/yang_synth
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from wire_model.yang.sim import Drive, NAMES, make_model, unstable, vertices  # noqa: E402
from clip_set import ALL_CLIP_TRIMS  # noqa: E402


# Domain randomization (user, 2026-10-27): the synthetic data must not use
# exactly the fitted values. Per rollout one log-uniform factor per parameter
# GROUP, applied to all its per-cable / per-joint values (the fitted spatial
# pattern is kept), plus an independent per-cable jitter on the stiffnesses.
# Drawn from a separate rng stream (seed, DR_STREAM), so the end-motion draws
# of transform_seq are the same as without randomization.
DR_RANGES = {"bend": (0.5, 2.0), "twist": (0.5, 2.0), "stretch": (0.5, 2.0), "damping": (1 / 3, 3.0),
             "mass": (0.7, 1.4), "mu": (0.5, 2.0), "rest": (0.8, 1.2), "jitter": (0.85, 1.18)}
MASS_BOUNDS = (0.01, 0.5)          # kg/m, never outside (user rule)
DR_STREAM = 20261027


def sample_dr(seed):
    """the randomization factors of one rollout (json-serialisable)"""
    r = np.random.default_rng([seed, DR_STREAM])
    lu = lambda g, n=None: np.exp(r.uniform(np.log(DR_RANGES[g][0]), np.log(DR_RANGES[g][1]), n))
    f = {g: float(lu(g)) for g in ("bend", "twist", "stretch", "damping", "mass", "mu", "rest")}
    for g in ("bend", "twist"):
        f[f"jitter_{g}"] = lu("jitter", 4).tolist()          # per cable
    return f


def dr_mujoco_general(p, f):
    """randomized copy of a mujoco_general_fit_cmaes.unpack() dict.
    bend group = per-joint spring stiffness k (4 x 15) and the plugin bend
    modulus (per cable); twist = plugin twist modulus (per cable); damping =
    per-joint dof damping; mass per cable (clipped to MASS_BOUNDS); mu; rest angles."""
    jb, jt = np.asarray(f["jitter_bend"]), np.asarray(f["jitter_twist"])
    q = dict(p)
    q["k"] = p["k"] * f["bend"] * jb[:, None]
    q["bend"] = p["bend"] * f["bend"] * jb
    q["twist"] = p["twist"] * f["twist"] * jt
    q["damp"] = p["damp"] * f["damping"]
    q["mass"] = np.clip(p["mass"] * f["mass"], *MASS_BOUNDS)
    q["mu"] = float(p["mu"] * f["mu"])
    q["rest_deg"] = p["rest_deg"] * f["rest"]
    return q


def dr_newton_general(kw, f):
    """randomized copy of the row-12 eval_clip kwargs (newton_fit_everything_cmaes.unpack).
    bend = per_joint_ke, twist = per_joint_kt (both with per-cable jitter), stretch =
    the scalar K stretch (one value per scene in build_scene, so no per-cable jitter),
    damping = per_joint_kd (bend-rate damping), mass per cable (clipped), mu = shape_mu
    and cable_mu (same factor), rest = per_joint_rest. Torsional / rolling friction
    and the Dahl hysteresis parameters stay at the fit."""
    jb, jt = np.asarray(f["jitter_bend"]), np.asarray(f["jitter_twist"])
    q = dict(kw)
    q["per_joint_ke"] = np.asarray(kw["per_joint_ke"]) * f["bend"] * jb[:, None]
    q["per_joint_kt"] = np.asarray(kw["per_joint_kt"]) * f["twist"] * jt[:, None]
    q["stretch_stiffness"] = float(kw["stretch_stiffness"] * f["stretch"])
    q["per_joint_kd"] = np.asarray(kw["per_joint_kd"]) * f["damping"]
    q["mass_per_m"] = np.clip(np.asarray(kw["mass_per_m"], float) * f["mass"], *MASS_BOUNDS)
    q["shape_mu"] = float(kw["shape_mu"] * f["mu"]); q["cable_mu"] = float(kw["cable_mu"] * f["mu"])
    q["per_joint_rest"] = np.asarray(kw["per_joint_rest"]) * f["rest"]
    return q


def transform_seq(seq, rng, T_cap):
    """one random manipulation of a measured pose sequence; returns (new seq, meta)"""
    T = len(seq)
    s = 1.0 if rng.random() < 0.3 else float(rng.uniform(0.6, 2.0))
    w = 1.0 if rng.random() < 0.5 else float(rng.uniform(0.5, 2.0))
    pert = rng.random() < 0.5
    t_new = np.arange(0, T - 1 + 1e-9, w)[:T_cap]
    arr = {n: np.array([[*seq[t][n][0], seq[t][n][1]] for t in range(T)]) for n in NAMES}   # (T, 3)
    out = {}
    for n in NAMES:
        A = arr[n]
        A = A[0] + s * (A - A[0])
        B = np.stack([np.interp(t_new, np.arange(T), A[:, c]) for c in range(3)], 1)
        if pert:
            tt = t_new * 0.25                                        # seconds
            amp = np.r_[rng.uniform(0, 0.040, 2), rng.uniform(0, 0.2)]
            P = np.zeros_like(B)
            for c in range(3):
                for _ in range(3):
                    per = rng.uniform(4.0, 40.0)
                    P[:, c] += np.sin(2 * np.pi * tt / per + rng.uniform(0, 2 * np.pi)) / 3
                P[:, c] -= P[0, c]
                P[:, c] *= amp[c]
            ramp = np.clip(tt / 2.0, 0, 1)
            B = B + P * ramp[:, None]
        out[n] = B
    new = [{n: (out[n][t, :2], out[n][t, 2]) for n in NAMES} for t in range(len(t_new))]
    return new, {"scale": s, "warp": w, "perturbed": bool(pert)}


# ---------------------------------------------------------------- speed-only motion (user, 2026-10-27)
C_TRIES = 4            # warps tried per rollout against the in-simulation contact constraint (c)
CONTACT_TOL = 0.001    # m: every generated contact (cable geoms carry a 1 mm margin) counts
OWN_SEGS = 2           # cable segments at each end that may touch their OWN mover (connector region)


def mujoco_contact_tables(m):
    """per geom: cable index / segment index (0..14) / mover index (0 hub, 1+k plug k), -1 if none"""
    import re
    import mujoco
    cab = -np.ones(m.ngeom, int); seg = -np.ones(m.ngeom, int); mov = -np.ones(m.ngeom, int)
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        mt = re.match(r"Wire(\d)B_(first|last|\d+)$", name)
        if mt:
            cab[g] = int(mt.group(1))
            seg[g] = {"first": 0, "last": 14}.get(mt.group(2), None) if mt.group(2) in ("first", "last") else int(mt.group(2))
        elif name in ("hub", "hub_mod"):
            mov[g] = 0
        else:
            mp_ = re.match(r"plug(\d)(_mod)?$", name)
            if mp_:
                mov[g] = 1 + int(mp_.group(1))
    return cab, seg, mov


def mujoco_contact_pairs(d, tabs):
    """(c) at the current substep: (4, 5) counts of cable k - mover m contacts
    outside the cable's own connector region (a cable may touch only its OWN hub /
    plug with its first / last OWN_SEGS segments), number of mover-mover contacts,
    deepest penetration (m). Every contact MuJoCo generates counts (distance below
    CONTACT_TOL = the cable geoms' 1 mm margin, i.e. touching or within 1 mm)."""
    pairs = np.zeros((4, 5), int)
    n = d.ncon
    if n == 0:
        return pairs, 0, 0.0
    cab, seg, mov = tabs
    g = d.contact.geom[:n]; dist = d.contact.dist[:n]
    hit = dist < CONTACT_TOL
    if not hit.any():
        return pairs, 0, 0.0
    g = g[hit]; dist = dist[hit]
    c0, c1, m0, m1 = cab[g[:, 0]], cab[g[:, 1]], mov[g[:, 0]], mov[g[:, 1]]
    mm = (m0 >= 0) & (m1 >= 0) & (m0 != m1)
    cm = ((c0 >= 0) & (m1 >= 0)) | ((c1 >= 0) & (m0 >= 0))
    k = np.where(c0 >= 0, c0, c1); sg = np.where(c0 >= 0, seg[g[:, 0]], seg[g[:, 1]]); mv = np.where(m0 >= 0, m0, m1)
    own = ((mv == 0) & (sg < OWN_SEGS)) | ((mv == 1 + k) & (sg > 14 - OWN_SEGS))
    bad = cm & ~own
    if bad.any():
        np.add.at(pairs, (k[bad], mv[bad]), 1)
    nm = int(mm.sum())
    pen = float(max(0.0, -dist[bad | mm].min())) if (bad.any() or nm) else 0.0
    return pairs, nm, pen


def _mujoco_speed_run(m, seq, sub, tabs, cav):
    """one rollout along pose sequence seq with the per-substep contact scan;
    returns Vs, gaps, contact stats"""
    import mujoco
    d = mujoco.MjData(m)
    drv = Drive(m)

    def gap_of(V):
        return np.array([[V[k, 0] - d.xpos[h], V[k, -1] - d.xpos[q]] for k, (h, q) in enumerate(cav)])
    drv.apply(d, seq[0])
    for _ in range(300):
        mujoco.mj_step(m, d)
    V = vertices(m, d); Vs, gaps = [V[:, :, :2]], [gap_of(V)]
    pairs = np.zeros((4, 5), int); nm = 0; pen = 0.0
    for t in range(len(seq) - 1):
        a0, a1 = seq[t], seq[t + 1]
        for s_ in range(sub):
            f = (s_ + 1) / sub
            drv.apply(d, {n: (a0[n][0] + f * (a1[n][0] - a0[n][0]), a0[n][1] + f * (a1[n][1] - a0[n][1]))
                          for n in NAMES})
            mujoco.mj_step(m, d)
            pr, mm_, pe = mujoco_contact_pairs(d, tabs)
            pairs += pr; nm += mm_; pen = max(pen, pe)
        V = vertices(m, d)
        if not np.isfinite(V).all() or unstable(d):
            break
        Vs.append(V[:, :, :2]); gaps.append(gap_of(V))
    return Vs, gaps, {"pairs": pairs, "mover_mover": nm, "max_penetration_mm": round(1000 * pen, 3), "frames": len(Vs)}


def _job_speed(job):
    """the speed-only rollout on the MuJoCo general env: variant 0 = the real
    motion, variants > 0 = a shared speed warp (x0.8-1.2) checked against (a)/(b)
    (speed_warp.sample_valid), simulated with the contact scan (c) on every
    substep. Contacts are tolerated as far as the REAL timing with the SAME
    physical parameters shows them (user decision): if the warped rollout has any
    forbidden contact, that real-timing reference is simulated (once) and the
    warped rollout is rejected and resampled (up to C_TRIES) only if it exceeds
    it (speed_warp.contact_excess). Variant 0 is always written (its contacts =
    the real recording's own, reported)."""
    import mujoco
    from wire_model.yang import speed_warp as SW
    npz, j, seed, x, out_dir, env, randomize, motion = job[:8]
    opts = job[8] if len(job) > 8 else {}
    assert env == "general", "speed motion with MuJoCo is implemented for --env general"
    clip = Path(npz).stem
    path = Path(out_dir) / (f"{clip}_{j:03d}.npz" if opts.get("twins") else f"{clip}_{j:02d}.npz")
    if opts.get("twins") and j == 0:
        randomize = False          # rollout 0 = the EXACT twin (general fit, real timing)
    if path.exists():
        return str(path), "exists", 0.0
    t0 = time.time()
    z = np.load(npz)
    F = np.stack([z[f"c{k}"] for k in range(4)], 1).astype(float)
    shp = {k: F[:, k] for k in range(4)}
    fps = float(z["fps"])
    T = len(F)
    rng = np.random.default_rng([seed, 4242])                # own stream for the warps
    import mujoco_general_fit_cmaes as G
    p = G.unpack(x)
    meta = {"motion": "speed"}
    if randomize:
        dr = sample_dr(seed)
        p = dr_mujoco_general(p, dr)
        meta.update({"dr": dr, "dr_mass": np.asarray(p["mass"]).tolist()})
    m = G.make_model(shp, p)
    tabs = mujoco_contact_tables(m)
    lim = SW.real_limits(F, "cavity")
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    cav = [(bid(f"hub_cav{k}"), bid(f"plug{k}_cav0")) for k in range(4)]
    E_real = SW.end_frames(F, np.repeat(np.arange(T, dtype=float)[:, None], 5, 1))
    seq_real = Drive.pose_seq([[E_real[t][k] for k in range(4)] for t in range(T)])
    ref = None
    attempts = []
    accepted = False
    for attempt in range(1 if j == 0 else C_TRIES):
        if j == 0:
            seq, wmeta = seq_real, {"real": True, "n_frames": T}
        else:
            E, _, wmeta = SW.sample_valid(F, rng, "cavity", lim)
            seq = Drive.pose_seq([[E[t][k] for k in range(4)] for t in range(len(E))])
        Vs, gaps, cs = _mujoco_speed_run(m, seq, sub, tabs, cav)
        rec = {**{k: wmeta[k] for k in ("tries", "rejected_a", "rejected_b") if k in wmeta},
               "pairs": cs["pairs"].tolist(), "mover_mover": cs["mover_mover"],
               "max_penetration_mm": cs["max_penetration_mm"]}
        reason = None
        if j > 0 and (cs["pairs"].any() or cs["mover_mover"]):
            if ref is None and opts.get("ref") is not None:
                # --twin-reference: tolerate what the clip's exact twin (rollout 0) shows
                ref = {"pairs": np.asarray(opts["ref"]["pairs"]), "frames": opts["ref"]["frames"], "mover_mover": 0}
            if ref is None:
                _, _, ref = _mujoco_speed_run(m, seq_real, sub, tabs, cav)
            reason = SW.contact_excess(cs["pairs"], cs["frames"], ref["pairs"], ref["frames"],
                                       cs["mover_mover"], ref["mover_mover"])
        rec["rejected_reason"] = reason
        attempts.append(rec)
        if reason is None:
            accepted = True
            break
    meta.update({"warp": wmeta, "attempts": attempts, "rejected_c": len(attempts) - (1 if accepted else 0),
                 "accepted": accepted,
                 "real_reference": None if ref is None else {"pairs": ref["pairs"].tolist(), "frames": ref["frames"],
                                                             "mover_mover": ref["mover_mover"]},
                 "real_limits": {"ratio_max": lim["real_ratio_max"].tolist(),
                                 "min_gap_mm": 1000 * lim["real_min_gap"]}})
    if not accepted:
        return str(path), "REJECTED " + json.dumps({"attempts": attempts}), time.time() - t0
    rest = np.stack([shp[k][0] for k in range(4)])
    offs = np.stack(gaps)
    gap = np.linalg.norm(offs - offs[:1], axis=-1).max(axis=(1, 2))
    np.savez(path, V=np.stack(Vs).astype(np.float32), gap=gap.astype(np.float32), offs=offs.astype(np.float32),
             rest=rest.astype(np.float32),
             meta=json.dumps({"clip": clip, "variant": j, "seed": seed, **meta, "n_frames_seq": len(seq)}))
    return str(path), (f"{len(Vs)}/{len(seq)} frames, max gap {1000 * gap.max():.1f} mm, tries "
                       f"{[a.get('tries', 0) for a in attempts]}, contacts/attempt "
                       f"{[int(np.sum(a['pairs'])) for a in attempts]}, rejected {[a['rejected_reason'] for a in attempts]}"
                       + ("" if ref is None else f", real-timing ref {int(ref['pairs'].sum())}")), time.time() - t0


def _job(job):
    import mujoco
    if len(job) > 7 and job[7] == "speed":
        return _job_speed(job)
    npz, j, seed, x, out_dir, env, randomize = job[:7]
    clip = Path(npz).stem
    path = Path(out_dir) / f"{clip}_{j:02d}.npz"
    if path.exists():
        return str(path), "exists", 0.0
    t0 = time.time()
    z = np.load(npz)
    shp = {k: z[f"c{k}"].astype(float) for k in range(4)}
    fps = float(z["fps"])
    T = len(shp[0])
    rng = np.random.default_rng(seed)
    seq0 = Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(T)])
    seq, meta = (seq0, {"scale": 1.0, "warp": 1.0, "perturbed": False}) if j == 0 else transform_seq(seq0, rng, T)
    if env == "general":       # the MuJoCo general fit (table row 5 / summary row 7) as the environment
        import mujoco_general_fit_cmaes as G
        p = G.unpack(x)
        if randomize:
            dr = sample_dr(seed)
            p = dr_mujoco_general(p, dr)
            meta = {**meta, "dr": dr, "dr_mass": np.asarray(p["mass"]).tolist()}
        m = G.make_model(shp, p)
    else:
        m = make_model(shp, x)
    d = mujoco.MjData(m)
    drv = Drive(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    cav = [(bid(f"hub_cav{k}"), bid(f"plug{k}_cav0")) for k in range(4)]

    def gap(V):        # connector offset vectors (4 cables x hub/plug x 3); a static ~1-2 cm sag is normal
        return np.array([[V[k, 0] - d.xpos[h], V[k, -1] - d.xpos[p]] for k, (h, p) in enumerate(cav)])
    drv.apply(d, seq[0])
    for _ in range(300):
        mujoco.mj_step(m, d)
    V = vertices(m, d); Vs, gaps = [V[:, :, :2]], [gap(V)]
    for t in range(len(seq) - 1):
        drv.step_frame(d, seq[t], seq[t + 1], sub)
        V = vertices(m, d)
        if not np.isfinite(V).all() or unstable(d):
            break
        Vs.append(V[:, :, :2]); gaps.append(gap(V))
    rest = np.stack([shp[k][0] for k in range(4)])
    offs = np.stack(gaps)                                              # (T, 4, 2, 3)
    gap = np.linalg.norm(offs - offs[:1], axis=-1).max(axis=(1, 2))    # stretch: change of any offset since frame 0
    gaps = gap
    np.savez(path, V=np.stack(Vs).astype(np.float32), gap=gap.astype(np.float32), offs=offs.astype(np.float32),
             rest=rest.astype(np.float32),
             meta=json.dumps({"clip": clip, "variant": j, "seed": seed, **meta, "n_frames_seq": len(seq)}))
    return str(path), f"{len(Vs)} frames, max gap {1000 * max(gaps):.1f} mm, {meta}", time.time() - t0


def dr_theta_newton(x, f):
    """randomized copy of the 8 global log10 Newton parameters of the identification
    (sim_newton.THETA_NAMES); one group factor each, no per-cable jitter (the
    parameters are global, as in the paper); mass kept inside MASS_BOUNDS"""
    from wire_model.yang import sim_newton as SN
    g = {"k_stretch": "stretch", "d_stretch": "damping", "k_bend": "bend", "d_bend": "damping",
         "k_twist": "twist", "d_twist": "damping", "mass_per_m": "mass", "mu": "mu"}
    y = np.array(x, float)
    for i, name in enumerate(SN.THETA_NAMES):
        y[i] += np.log10(f[g[name]])
    i = SN.THETA_NAMES.index("mass_per_m")
    y[i] = np.clip(y[i], *np.log10(MASS_BOUNDS))
    return y


def run_newton(jobs, x, out_dir, batch, randomize=False):
    """--engine newton: the same rollouts (same transform_seq draws, same seeds)
    on wire_model/yang/sim_newton.py, `batch` rollouts per GPU call (one Newton
    world each). Cable ends ride rigidly on their movers (offsets from the
    tracked frame 0, sim_newton.schedule_from_poses). Newton rods have no
    connector gap (the ends are kinematic) but they are extensible, so "gap" is
    the largest change of any cable's 16-node arc length since frame 0
    (over-stretch by the scaled motion), "offs" the per-cable change (T, 4)."""
    from wire_model.yang import sim_newton as SN
    todo = []
    for npz, j, seed, *_ in jobs:
        clip = Path(npz).stem
        path = Path(out_dir) / f"{clip}_{j:02d}.npz"
        if path.exists():
            continue
        z = np.load(npz)
        shp = {k: z[f"c{k}"].astype(float) for k in range(4)}
        T = len(shp[0])
        rng = np.random.default_rng(seed)
        seq0 = Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(T)])
        seq, meta = (seq0, {"scale": 1.0, "warp": 1.0, "perturbed": False}) if j == 0 else transform_seq(seq0, rng, T)
        rest = np.stack([shp[k][0] for k in range(4)])
        todo.append((path, clip, j, seed, float(z["fps"]), rest, SN.schedule_from_poses(rest, seq), meta, len(seq)))
    todo.sort(key=lambda r: r[8])
    print(f"newton: {len(todo)} rollouts to run in batches of {batch}", flush=True)
    t0, n = time.time(), 0
    for b0 in range(0, len(todo), batch):
        B = todo[b0:b0 + batch]
        assert len({r[4] for r in B}) == 1
        tb = time.time()
        drs = [sample_dr(r[3]) if randomize else None for r in B]
        X = np.stack([dr_theta_newton(x, f) if f else np.asarray(x, float) for f in drs])
        Vs = SN.rollouts([r[5] for r in B], [r[6] for r in B], B[0][4], X, device="cuda:0")
        for (path, clip, j, seed, fps, rest, _, meta, Tseq), V, f in zip(B, Vs, drs):
            L = np.linalg.norm(np.diff(V, axis=2), axis=-1).sum(-1)            # (T, 4) arc length
            offs = L - L[:1]
            gap = np.abs(offs).max(1)
            np.savez(path, V=V[..., :2].astype(np.float32), gap=gap.astype(np.float32), offs=offs.astype(np.float32),
                     rest=rest.astype(np.float32),
                     meta=json.dumps({"clip": clip, "variant": j, "seed": seed, **meta, "n_frames_seq": Tseq,
                                      "engine": "newton", "dr": f}))
            n += 1
            print(f"[{n}/{len(todo)}] {path.name}: {len(V)}/{Tseq} frames, max stretch {1000 * gap.max():.1f} mm, "
                  f"{meta}", flush=True)
        print(f"batch of {len(B)} ({max(r[8] for r in B)} frames): {time.time() - tb:.0f}s; total {time.time() - t0:.0f}s",
              flush=True)


def run_newton_general(jobs, out_dir, batch, randomize):
    """--env newton_general: same jobs (same X-Planar starts, same transform_seq
    draws and seeds) on the row-12 Newton model. Variant 0 drives the cable
    ends with the clip's REAL tracked frames, exactly as the fit's eval_clip;
    variants > 0 with 4-node drive frames from the transformed mover poses
    (newton_general.drive_from_poses). "gap" = largest change of any cable's
    16-node arc length since frame 0, "offs" the per-cable change (T, 4)."""
    from wire_model.yang import newton_general as NG
    kw = NG.fit_kwargs()
    todo = []
    for npz, j, seed, *_ in jobs:
        clip = Path(npz).stem
        path = Path(out_dir) / f"{clip}_{j:02d}.npz"
        if path.exists():
            continue
        z = np.load(npz)
        shp = {k: z[f"c{k}"].astype(float) for k in range(4)}
        T = len(shp[0])
        rng = np.random.default_rng(seed)
        seq0 = Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(T)])
        rest = np.stack([shp[k][0] for k in range(4)])
        if j == 0:
            seq, meta, drive = seq0, {"scale": 1.0, "warp": 1.0, "perturbed": False}, shp
        else:
            seq, meta = transform_seq(seq0, rng, T)
            drive = NG.drive_from_poses(rest, seq)
        kw_r = kw
        if randomize:
            dr = sample_dr(seed)
            kw_r = dr_newton_general(kw, dr)
            meta = {**meta, "dr": dr, "dr_mass": np.asarray(kw_r["mass_per_m"]).tolist()}
        todo.append((path, clip, j, seed, float(z["fps"]), {k: shp[k][:1] for k in range(4)}, drive, meta,
                     len(seq), rest, kw_r))
    todo.sort(key=lambda r: r[8])
    print(f"newton_general: {len(todo)} rollouts to run in batches of {batch}", flush=True)
    t0, n = time.time(), 0
    for b0 in range(0, len(todo), batch):
        B = todo[b0:b0 + batch]
        assert len({r[4] for r in B}) == 1
        tb = time.time()
        Vs = NG.rollouts([r[5] for r in B], [r[6] for r in B], kw=[r[10] for r in B], device="cuda:0", fps=B[0][4])
        for (path, clip, j, seed, fps, _, _, meta, Tseq, rest, _), V in zip(B, Vs):
            L = np.linalg.norm(np.diff(V, axis=2), axis=-1).sum(-1)            # (T, 4) arc length
            offs = L - L[:1]
            gap = np.abs(offs).max(1)
            np.savez(path, V=V[..., :2].astype(np.float32), gap=gap.astype(np.float32), offs=offs.astype(np.float32),
                     rest=rest.astype(np.float32),
                     meta=json.dumps({"clip": clip, "variant": j, "seed": seed, **meta, "n_frames_seq": Tseq,
                                      "engine": "newton_general_row12"}))
            n += 1
            print(f"[{n}/{len(todo)}] {path.name}: {len(V)}/{Tseq} frames, max stretch {1000 * gap.max():.1f} mm, "
                  f"{meta}", flush=True)
        print(f"batch of {len(B)} ({max(r[8] for r in B)} frames): {time.time() - tb:.0f}s; total {time.time() - t0:.0f}s",
              flush=True)


def run_newton_speed(jobs, out_dir, batch, randomize, env, x=None):
    """--motion speed on the Newton environments (env "newton_general" = row 12,
    env "paper" = sim_newton with the identified Theta): variant 0 = the real
    motion, variants > 0 = a shared speed warp (x0.8-1.2) checked against (a)/(b)
    on the kinematic end nodes (speed_warp.sample_valid, kind "nodes"); the
    movers ride along the warped end nodes exactly as in the real-motion protocol.
    Constraint (c) every 25 substeps (newton_general.contact_violations). As for
    MuJoCo, contacts are tolerated as far as the REAL timing with the SAME
    physical parameters shows them: a warped rollout with forbidden contacts gets
    that real-timing reference simulated (once, in a later round) and is rejected
    and resampled (up to C_TRIES) only if it exceeds it (speed_warp.contact_excess)."""
    from wire_model.yang import speed_warp as SW
    from wire_model.yang import newton_general as NG
    if env == "newton_general":
        kw = NG.fit_kwargs()
    else:
        from wire_model.yang import sim_newton as SN
    items = []
    for npz, j, seed, *_ in jobs:
        clip = Path(npz).stem
        path = Path(out_dir) / f"{clip}_{j:02d}.npz"
        if path.exists():
            continue
        z = np.load(npz)
        F = np.stack([z[f"c{k}"] for k in range(4)], 1).astype(float)
        it = {"path": path, "clip": clip, "j": j, "seed": seed, "F": F, "fps": float(z["fps"]),
              "rng": np.random.default_rng([seed, 4242]), "lim": SW.real_limits(F, "nodes"), "attempts": [],
              "meta": {"motion": "speed"}, "ref": None, "pending": None}
        if randomize:
            dr = sample_dr(seed)
            it["meta"]["dr"] = dr
            if env == "newton_general":
                it["kw"] = dr_newton_general(kw, dr); it["meta"]["dr_mass"] = np.asarray(it["kw"]["mass_per_m"]).tolist()
            else:
                it["X"] = dr_theta_newton(x, dr)
        else:
            if env == "newton_general":
                it["kw"] = kw
            else:
                it["X"] = np.asarray(x, float)
        items.append(it)
    print(f"newton speed ({env}): {len(items)} rollouts, batches of {batch}", flush=True)
    t0, n_done, n_rej, rnd = time.time(), 0, 0, 0
    real_E = lambda it: SW.end_frames(it["F"], np.repeat(np.arange(len(it["F"]), dtype=float)[:, None], 5, 1))

    def save(it, V, E, w):
        nonlocal n_done
        L = np.linalg.norm(np.diff(V, axis=2), axis=-1).sum(-1)
        offs = L - L[:1]
        gap = np.abs(offs).max(1)
        meta = {**it["meta"], "warp": w, "attempts": it["attempts"], "rejected_c": len(it["attempts"]) - 1,
                "accepted": True, "real_reference": it["ref"],
                "real_limits": {"ratio_max": it["lim"]["real_ratio_max"].tolist(),
                                "min_gap_mm": 1000 * it["lim"]["real_min_gap"]}}
        np.savez(it["path"], V=V[..., :2].astype(np.float32), gap=gap.astype(np.float32),
                 offs=offs.astype(np.float32), rest=it["F"][0].astype(np.float32),
                 meta=json.dumps({"clip": it["clip"], "variant": it["j"], "seed": it["seed"], **meta,
                                  "n_frames_seq": len(E), "engine": env}))
        n_done += 1
        print(f"[{n_done}/{len(items)}] {it['path'].name}: {len(V)}/{len(E)} frames, max stretch {1000 * gap.max():.1f} mm, "
              f"tries {[a.get('tries', 0) for a in it['attempts']]}, contacts {[a['n'] for a in it['attempts']]}, "
              f"rejected {[a['rejected_reason'] for a in it['attempts']]}"
              + ("" if it["ref"] is None else f", real-timing ref {int(np.sum(it['ref']['pairs']))}"), flush=True)

    tasks = [(it, "new") for it in items]
    while tasks:
        rnd += 1
        sims = []                       # (item, kind, E, warp meta)
        for it, kind in tasks:
            if kind == "ref":
                sims.append((it, "ref", real_E(it), {"real": True}))
            elif it["j"] == 0:
                sims.append((it, "warp", real_E(it), {"real": True, "n_frames": len(it["F"])}))
            else:
                E, _, w = SW.sample_valid(it["F"], it["rng"], "nodes", it["lim"])
                sims.append((it, "warp", E, w))
        sims.sort(key=lambda r: len(r[2]))
        nxt = []
        for b0 in range(0, len(sims), batch):
            B = sims[b0:b0 + batch]
            tb = time.time()
            st = []
            if env == "newton_general":
                Vs = NG.rollouts([{k: r[0]["F"][:1, k] for k in range(4)} for r in B],
                                 [{k: r[2][:, k] for k in range(4)} for r in B], kw=[r[0]["kw"] for r in B],
                                 device="cuda:0", fps=B[0][0]["fps"], check_every=25, stats=st)
            else:
                scheds = []
                for r in B:
                    F16 = np.zeros((len(r[2]), 4, 16, 2))
                    F16[:, :, [0, 1, 14, 15]] = r[2]
                    scheds.append(SN.schedule_tracked(F16))
                Vs = SN.rollouts([r[0]["F"][0] for r in B], scheds, B[0][0]["fps"], np.stack([r[0]["X"] for r in B]),
                                 device="cuda:0", check_every=25, stats=st)
            for (it, kind, E, w), V, (nv, dm, pr) in zip(B, Vs, st):
                if kind == "ref":
                    it["ref"] = {"pairs": np.asarray(pr).tolist(), "frames": len(V)}
                    Vw, Ew, ww, prw = it["pending"]
                    reason = SW.contact_excess(prw, len(Vw), pr, len(V))
                    it["attempts"][-1]["rejected_reason"] = reason
                    if reason is None:
                        save(it, Vw, Ew, ww)
                    elif len(it["attempts"]) < C_TRIES:
                        nxt.append((it, "new"))
                    else:
                        n_rej += 1
                        print(f"REJECTED {it['path'].name}: {json.dumps(it['attempts'])}", flush=True)
                    continue
                it["attempts"].append({**{k: w[k] for k in ("tries", "rejected_a", "rejected_b") if k in w},
                                       "n": int(nv), "pairs": np.asarray(pr).tolist(),
                                       "min_box_dist_mm": None if not np.isfinite(dm) else round(1000 * dm, 2),
                                       "rejected_reason": None})
                if it["j"] == 0 or nv == 0:
                    save(it, V, E, w)
                    continue
                if it["ref"] is None:
                    it["pending"] = (V, E, w, np.asarray(pr))
                    nxt.append((it, "ref"))
                    continue
                reason = SW.contact_excess(pr, len(V), it["ref"]["pairs"], it["ref"]["frames"])
                it["attempts"][-1]["rejected_reason"] = reason
                if reason is None:
                    save(it, V, E, w)
                elif len(it["attempts"]) < C_TRIES:
                    nxt.append((it, "new"))
                else:
                    n_rej += 1
                    print(f"REJECTED {it['path'].name}: {json.dumps(it['attempts'])}", flush=True)
            print(f"round {rnd} batch of {len(B)} ({max(len(r[2]) for r in B)} frames): {time.time() - tb:.0f}s; "
                  f"total {time.time() - t0:.0f}s", flush=True)
        tasks = nxt
        if nxt:
            print(f"round {rnd}: {sum(k == 'ref' for _, k in nxt)} real-timing references, "
                  f"{sum(k == 'new' for _, k in nxt)} resampled warps next", flush=True)
    print(f"newton speed done: {n_done} written, {n_rej} rejected after {C_TRIES} tries", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", default="data/wm/yang_ident.json")
    ap.add_argument("--tracks", default="X-Planar_vids/mujoco_videos/tracks")
    ap.add_argument("--per-clip", type=int, default=8, help="rollouts per X-Planar clip (variant 0 = the real motion)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wm/yang_synth")
    ap.add_argument("--env", default="yang", choices=["yang", "general", "newton_general"],
                    help="yang: Theta* of yang_identify.py; general: the MuJoCo general fit (--general-fit), "
                         "used for RopeDreamer's simulation-only training (user, 2026-10-27)")
    ap.add_argument("--general-fit", default="data/mujoco_general_fit3.npz")
    ap.add_argument("--engine", default="mujoco", choices=["mujoco", "newton"],
                    help="newton: wire_model/yang/sim_newton.py with --ident from the Newton identification "
                         "(only with --env yang); batched on the GPU, --workers unused")
    ap.add_argument("--batch", type=int, default=128, help="newton: rollouts per GPU batch")
    ap.add_argument("--randomize", dest="randomize", action="store_true", default=True,
                    help="domain randomization of the physical parameters per rollout (default; only "
                         "--env general / newton_general), factors in each npz's meta['dr']")
    ap.add_argument("--no-randomize", dest="randomize", action="store_false",
                    help="the exact fitted / identified parameters (the old behaviour)")
    ap.add_argument("--allow-eval-clips", action="store_true",
                    help="do NOT exclude the evaluation clips (clip_set.ALL_CLIP_TRIMS) -- explicit user decision for "
                         "the twins14 RopeDreamer set (2026-10-28); logged in config.json")
    ap.add_argument("--frames-per-clip", type=int, default=0,
                    help="instead of --per-clip: ceil(N / clip length) rollouts per clip, file names <clip>_<NNN>; "
                         "rollout 000 = the exact fit (no randomization), real timing")
    ap.add_argument("--twin-reference", action="store_true",
                    help="with --frames-per-clip: run every clip's rollout 000 first and tolerate in the variations the "
                         "contacts that twin shows (instead of a real-timing re-run with each rollout's parameters)")
    ap.add_argument("--motion", default="scale", choices=["scale", "speed"],
                    help="scale: transform_seq (spatial scale + perturbation, the old runs); speed: speed-only "
                         "time warps of the exact recorded paths with hard constraints (wire_model/yang/speed_warp.py; "
                         "user 2026-10-27, use for every new run)")
    a = ap.parse_args()
    assert not (a.engine == "newton" and a.env != "yang"), "--engine newton needs --env yang"
    # newton_general: the Newton general fit (table row 12, newton_fit_everything_pinned.npz) as the
    # environment, built by the fit's own scene code (wire_model/yang/newton_general.py); user, 2026-10-27
    import multiprocessing as mp
    if a.env == "newton_general":
        x = np.zeros(0)
    elif a.env == "general":
        x = np.asarray(np.load(ROOT / a.general_fit)["best_x"], float)
    else:
        x = np.array(json.loads((ROOT / a.ident).read_text())["theta_log10"])
    files = sorted(glob.glob(str(ROOT / a.tracks / "*.npz")))
    bad = set(ALL_CLIP_TRIMS)
    if a.allow_eval_clips:
        # EXPLICIT user decision 2026-10-28 (RopeDreamer twins14 set): the 14 evaluation clips'
        # mover paths go into training. Never a default; logged in config.json.
        print("WARNING --allow-eval-clips: evaluation clips are NOT excluded", flush=True)
    else:
        files = [f for f in files if Path(f).stem not in bad]
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.env == "yang" and a.engine == "mujoco":
        a.randomize = False            # DR is defined for the general-fit environments and the Newton identification
    (out_dir / "config.json").write_text(json.dumps({**vars(a), "x": x.tolist(), "n_clips": len(files),
                                                     "dr_ranges": DR_RANGES if a.randomize else None,
                                                     "mass_bounds": MASS_BOUNDS,
                                                     "speed_range": (__import__("wire_model.yang.speed_warp", fromlist=["x"]).SPEED_RANGE
                                                                     if a.motion == "speed" else None)}, indent=1))
    print(f"domain randomization: {'ON ' + json.dumps(DR_RANGES) if a.randomize else 'off'}", flush=True)
    if a.frames_per_clip:
        # rollouts per clip scaled by clip length so every clip contributes ~frames_per_clip frames;
        # seeds spaced so that up to 100002 rollouts per clip never collide
        n_of = {f: int(np.ceil(a.frames_per_clip / len(np.load(f)["c0"]))) for f in files}
        jobs = [(f, j, a.seed * 10000019 + i * 100003 + j, x, str(out_dir), a.env, a.randomize, a.motion,
                 {"twins": True}) for i, f in enumerate(files) for j in range(n_of[f])]
        (out_dir / "rollouts_per_clip.json").write_text(json.dumps({Path(f).stem: n for f, n in n_of.items()}, indent=1))
        print(f"rollouts per clip: {json.dumps({Path(f).stem: n for f, n in n_of.items()})}", flush=True)
    else:
        jobs = [(f, j, a.seed * 100003 + i * 101 + j, x, str(out_dir), a.env, a.randomize, a.motion)
                for i, f in enumerate(files) for j in range(a.per_clip)]
    print(f"{len(files)} clips, {len(jobs)} jobs, env {a.env}, {len(x)} params", flush=True)
    t0 = time.time()
    if a.motion == "speed" and (a.env == "newton_general" or a.engine == "newton"):
        run_newton_speed(jobs, out_dir, a.batch, a.randomize, "newton_general" if a.env == "newton_general" else "paper", x)
        print("YANG_DATASET_DONE")
        return
    if a.env == "newton_general":
        run_newton_general(jobs, out_dir, a.batch, a.randomize)
        print("YANG_DATASET_DONE")
        return
    if a.engine == "newton":
        if (ROOT / a.ident).exists():
            assert json.loads((ROOT / a.ident).read_text()).get("engine") == "newton", "--ident is not a Newton fit"
        run_newton(jobs, x, out_dir, a.batch, a.randomize)
        print("YANG_DATASET_DONE")
        return
    if a.twin_reference:
        # phase 1: every clip's rollout 0 (the exact twin); phase 2: the variations, whose
        # contacts are tolerated as far as their clip's twin shows them
        assert a.frames_per_clip and a.motion == "speed" and a.env == "general"
        first = [jb for jb in jobs if jb[1] == 0]
        with mp.get_context("fork").Pool(a.workers) as pool:
            for n, (p, msg, dt) in enumerate(pool.imap_unordered(_job, first), 1):
                print(f"[twin {n}/{len(first)}] {Path(p).name}: {msg} ({dt:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
        refs = {}
        for jb in first:
            z = np.load(Path(out_dir) / f"{Path(jb[0]).stem}_000.npz")
            mt = json.loads(str(z["meta"]))
            refs[Path(jb[0]).stem] = {"pairs": mt["attempts"][0]["pairs"], "frames": len(z["V"])}
        (out_dir / "twin_contacts.json").write_text(json.dumps(refs, indent=1))
        jobs = [jb[:8] + ({**jb[8], "ref": refs[Path(jb[0]).stem]},) for jb in jobs if jb[1] > 0]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, (p, msg, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{n}/{len(jobs)}] {Path(p).name}: {msg} ({dt:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
    print("YANG_DATASET_DONE")


if __name__ == "__main__":
    main()
