"""Full 4-cable Newton evaluation -- extends newton_scene.py's validated
single-cable smoke test to all 4 cables in one scene, matching
genesis_full_scene.py's eval_clip() metric exactly (per-frame error averaged
over all 4 cables, then averaged over the clip) for a direct comparison.

No mover geometry here: the error metric only needs the cavity TARGET
points, which are the raw tracked hub/plug endpoints (route[0]/route[-1]) --
same quantity genesis_full_scene.py drives its cable ends to. Mover boxes
only matter for rendering, not for this numeric evaluation.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, CAV_Z, DIR

ROOT = Path(__file__).resolve().parents[1]


def subdivide_path(P, subdiv):
    """Split every segment of a polyline into `subdiv` equal pieces.

    The original vertices survive at indices 0, subdiv, 2*subdiv, ... so a
    simulation built on the refined path can still be scored against the
    tracked nodes by taking every subdiv-th body -- the accuracy metric
    stays numerically comparable across resolutions."""
    P = np.asarray(P, float)
    if subdiv <= 1:
        return P
    out = []
    for a, b in zip(P[:-1], P[1:]):
        for j in range(subdiv):
            out.append(a + (b - a) * (j / subdiv))
    out.append(P[-1])
    return np.array(out)


def resample_per_joint(vals, n_joints, subdiv=1, scale=None):
    """Map a per-joint parameter profile onto a finer/coarser joint grid.

    Bending gains are resampled along the cable's normalised arc position
    and then scaled by `subdiv`: a discrete elastic rod's per-joint bend
    stiffness goes like EI/l, so halving the segment length l doubles the
    per-joint gain needed to represent the same physical rod."""
    vals = np.asarray(vals, float)
    if scale is None:
        scale = float(subdiv)
    if vals.shape[-1] == n_joints and scale == 1.0:
        return vals
    src = np.linspace(0.0, 1.0, vals.shape[-1])
    dst = np.linspace(0.0, 1.0, n_joints)
    out = np.stack([np.interp(dst, src, row) for row in np.atleast_2d(vals)])
    out = out * float(scale)
    return out if vals.ndim > 1 else out[0]


def initial_route3(shp, k, table_fraction=0.6, z_table=0.003, z_conn=None,
                   subdiv=1):
    """Initial 3D node layout for cable k: the tracked 2D path with a
    z-profile laid out so `table_fraction` of the cable's arc length RESTS
    ON THE TABLE at start (explicit user requirement, 2026-08-24) instead
    of the old hard initial condition where the entire cable floated at
    the 55 mm connector height. Ends stay at connector height; symmetric
    linear ramps (each (1-f)/2 of the arc length) connect down to table
    level. The rod's rest length is derived from these positions, so this
    also makes the built-in length slightly longer (ramp slope), closer to
    the real cable's slack."""
    from mujoco_scene import CAV_Z as _CAV_Z
    if z_conn is None:
        z_conn = _CAV_Z
    P = subdivide_path(shp[k][0].astype(float), subdiv)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    ramp = max(1e-6, (1.0 - table_fraction) / 2.0 * L)
    z = np.empty(len(P))
    for i, si in enumerate(s):
        if si < ramp:
            f = si / ramp
        elif si > L - ramp:
            f = (L - si) / ramp
        else:
            f = 1.0
        z[i] = z_conn + f * (z_table - z_conn)
    return np.hstack([P, z[:, None]])


def make_kinematic(builder, body_idx):
    builder.body_mass[body_idx] = 0.0
    builder.body_inv_mass[body_idx] = 0.0
    builder.body_inertia[body_idx] = wp.mat33(0.0)
    builder.body_inv_inertia[body_idx] = wp.mat33(0.0)


def eval_clip(clip, t1_sec=None, radius=0.003,
             stretch_stiffness=1.0e5, bend_stiffness=0.5, bend_damping=0.05,
             twist_stiffness=None, twist_damping=None,
             per_joint_ke=None, per_joint_kd=None, per_joint_kt=None,
             mass_per_m=None,
             dt=1e-3, steps_per_frame=None, device=None,
             scene_collision=True, solver_iterations=None,
             return_per_cable=False, shp_fps=None, z_max_out=None,
             z_nodes_out=None, table_fraction=0.6, subdiv=1,
             contact_buffer=64, joint_scale=None, mass_exact=False,
             nodes_xyz_out=None, dahl_eps_max=None, dahl_tau=None,
             per_joint_rest=None, cable_mu=None, shape_mu=None,
             mu_tors=None, mu_roll=None):
    """Full-scene evaluation (cables + board + movers, always -- explicit
    user requirement 2026-08-22: 'In der Simulation muss es IMMER Kabel,
    Tisch und Mover geben').

    per_joint_ke/per_joint_kd: optional (4, N_JOINTS) arrays of per-joint
    bend PD gains. A Newton cable joint carries four material slots in the
    order stretch / shear / BEND / TWIST, so the fitted value goes into slot
    2 -- and, unless per_joint_kt says otherwise, into slot 3 as well, which
    is the historical "twist coupled equal" behaviour every row up to the
    45-parameter fit ran with. per_joint_kt: optional (4, N_JOINTS) array
    that frees slot 3, i.e. a torsional stiffness of its own. They replace
    the uniform
    bend_stiffness/bend_damping material terms (which are then set to 0 so
    the two mechanisms don't mix -- same reasoning as
    newton_fit_stiffness.eval_single_cable). mass_per_m: optional per-cable
    (4,) or scalar linear density in kg/m, applied via the rod ShapeConfig's
    volumetric density (Newton's default worked out to a too-light
    0.028 kg/m, which let long spans hold aerial arches that glide OVER the
    movers instead of draping onto the board like the real cable).
    shp_fps: optional pre-tracked (shp, fps) to skip re-tracking in fit
    workers. return_per_cable: per-frame (n_frames, 4) errors instead of
    the per-frame scene mean.

    subdiv: cable discretisation multiplier. 1 keeps one rod segment per
    tracked node pair (15 bodies / 14 joints); 2 splits every segment in
    half (30 bodies / 29 joints). Scoring always falls back onto the
    tracked node grid by taking every subdiv-th body, so error numbers
    stay comparable across resolutions. per_joint_ke/kd given on a
    coarser joint grid are resampled and rescaled automatically."""
    import newton
    import cv2
    from newton.utils import create_parallel_transport_cable_quaternions

    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        t1 = None
        if t1_sec is not None:
            cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
            fps_probe = (cap_probe.get(cv2.CAP_PROP_FPS) if cap_probe.isOpened() else 0) or 4.0
            cap_probe.release()
            t1 = int(round(t1_sec * fps_probe))
        shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))
    # More iterations whenever contact is in play -- 5 leaves visibly bumpy
    # contact response (measured in the render pipeline, 2026-08-18).
    if solver_iterations is None:
        # 10, not 20 (changed 2026-10-20). Neither is converged: a probe
        # against an N=320 reference found the trajectory still 5.1 mm/node
        # away at N=160, so the iteration count is a model parameter here, not
        # a numerical detail, and 20 was never anything but another arbitrary
        # point. 10 is the one the evidence supports -- an interior optimum
        # against the video (5: 7.840, 10: 7.761, 15: 7.999, 20: 8.214,
        # 30: 8.500 mm/node) and the value every fit already optimises at, so
        # fitting and validation finally agree.
        solver_iterations = 10 if scene_collision else 5

    def route3(k, t):
        P = shp[k][t].astype(float)
        z = np.full((len(P), 1), CAV_Z)
        return np.hstack([P, z])

    builder = newton.ModelBuilder()
    # Contact friction of EVERY shape in the scene -- board, movers, hub. It sat
    # at this hardcoded 1.0 from the day the scene was written and was never
    # swept or fitted, unlike the cable's own rod_cfg.mu below.
    builder.default_shape_cfg.mu = 1.0 if shape_mu is None else float(shape_mu)
    # Torsional and rolling friction, Newton's two other contact coefficients.
    # They sat at the library defaults (0.005 / 0.0001) on every shape in every
    # row and were never moved; the physical fit (newton_fit_physical_cmaes.py)
    # is the first to treat them as parameters. One value each for the whole
    # scene -- shapes AND cable -- since a contact pair takes the combination.
    if mu_tors is not None:
        builder.default_shape_cfg.mu_torsional = float(mu_tors)
    if mu_roll is not None:
        builder.default_shape_cfg.mu_rolling = float(mu_roll)
    # DAHL CABLE FRICTION (bending hysteresis). A real harness wire does not
    # spring back to its original shape after being bent: the strands and
    # jacket layers rub against each other internally, which is RATE-
    # INDEPENDENT hysteresis, not plasticity and not viscous damping. Newton
    # implements the Dahl friction model for exactly this, per joint, but BOTH
    # dahl_eps_max and dahl_tau default to ZERO -- i.e. it is off unless
    # explicitly authored, and this project never authored it. The attributes
    # have to be registered on the builder BEFORE finalize.
    _use_dahl = dahl_eps_max is not None and dahl_tau is not None
    if _use_dahl:
        newton.solvers.SolverVBD.register_custom_attributes(builder)

    if mass_per_m is not None:
        mpm = np.broadcast_to(np.asarray(mass_per_m, float).ravel(), (4,))
    use_per_joint = per_joint_ke is not None
    cable_joints = []

    n_joints_sim = (len(shp[0][0]) - 1) * subdiv - 1
    if per_joint_ke is not None:
        per_joint_ke = resample_per_joint(per_joint_ke, n_joints_sim, subdiv,
                                          scale=joint_scale)
    if per_joint_kd is not None:
        per_joint_kd = resample_per_joint(per_joint_kd, n_joints_sim, subdiv,
                                          scale=joint_scale)
    if per_joint_kt is not None:
        per_joint_kt = resample_per_joint(per_joint_kt, n_joints_sim, subdiv,
                                          scale=joint_scale)

    cable_bodies = []
    for k in range(4):
        r0 = initial_route3(shp, k, table_fraction=table_fraction,
                            subdiv=subdiv)
        positions = [wp.vec3(*p) for p in r0]
        rod_cfg = None
        if mass_per_m is not None:
            rod_cfg = builder.ShapeConfig()
            rod_cfg.mu = 1.0 if cable_mu is None else float(cable_mu)
            if mu_tors is not None:
                rod_cfg.mu_torsional = float(mu_tors)
            if mu_roll is not None:
                rod_cfg.mu_rolling = float(mu_roll)
            # A rod segment is a CAPSULE, so each one carries a full sphere's
            # worth of end caps on top of its cylinder. Deriving the density
            # from the cylinder alone therefore overshoots the intended linear
            # density by (4/3)r/l per segment -- 12% at subdiv=1, and it grows
            # with resolution (24% at subdiv=2), which would silently make a
            # finer cable heavier. mass_exact solves for the density that hits
            # mass_per_m * L regardless of segment count; it is off by default
            # because the fitted mass absorbed the subdiv=1 overshoot.
            seg_len = np.linalg.norm(np.diff(r0[:, :3], axis=0), axis=1)
            if mass_exact:
                vol = (np.pi * radius ** 2 * seg_len.sum()
                       + len(seg_len) * (4.0 / 3.0) * np.pi * radius ** 3)
                rod_cfg.density = float(mpm[k] * seg_len.sum() / vol)
            else:
                rod_cfg.density = float(mpm[k] / (np.pi * radius ** 2))
        bodies, joints = builder.add_rod(
            positions=positions, radius=radius, cfg=rod_cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=(0.0 if use_per_joint else bend_stiffness),
            bend_damping=(0.0 if per_joint_kd is not None else bend_damping),
            twist_stiffness=twist_stiffness, twist_damping=twist_damping,
            body_frame_origin="start", label=f"cable{k}")
        if per_joint_rest is not None:
            # A Newton rod carries its rest shape in the JOINT FRAMES, not in a
            # target angle: joint_X_p/X_c hold the segment offset with identity
            # rotation for a straight cable. joint_target_q is the setpoint of a
            # drive controller and the VBD rod solver does not derive its rest
            # configuration from it -- measured, a rest angle of 8.6 deg written
            # there changed the result by 0.005 mm, i.e. nothing. Rotating the
            # child frame about the bend axis is what actually pre-bends the
            # cable, which is what a permanently set harness wire is.
            rk = np.asarray(per_joint_rest, float)
            rk = rk[k] if rk.ndim == 2 else rk
            jb = len(builder.joint_X_c) - len(joints)
            for j in range(len(joints)):
                ang = float(rk[j])
                q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ang)
                cur = builder.joint_X_c[jb + j]
                builder.joint_X_c[jb + j] = wp.transform(
                    wp.transform_get_translation(cur),
                    wp.mul(q, wp.transform_get_rotation(cur)))
        if use_per_joint:
            base = len(builder.joint_target_ke) - 4 * len(joints)
            for j in range(len(joints)):
                builder.joint_target_ke[base + 4 * j + 2] = float(per_joint_ke[k][j])
                builder.joint_target_ke[base + 4 * j + 3] = float(
                    per_joint_ke[k][j] if per_joint_kt is None
                    else per_joint_kt[k][j])
                if per_joint_kd is not None:
                    builder.joint_target_kd[base + 4 * j + 2] = float(per_joint_kd[k][j])
                    builder.joint_target_kd[base + 4 * j + 3] = float(per_joint_kd[k][j])
        make_kinematic(builder, bodies[0])
        make_kinematic(builder, bodies[-1])
        cable_bodies.append(bodies)
        cable_joints.append(list(joints))

    # Proper collision environment (2026-08-22, explicit user requirement:
    # "Ich will auch eine ordentliche Kollisionsberechnung!"): the original
    # eval simulated ONLY the 4 cables -- no table for them to drape on
    # (they could sag below z=0 into the void) and no mover bodies to plow
    # them (measured on 172926: the real footage shows a mover pushing a
    # cable, which the cables-only sim cannot reproduce). Board + movers are
    # kinematic, driven the same way the render pipeline drives them.
    platform_bodies, module_bodies = {}, {}
    if scene_collision:
        from mujoco_scene import (BOARD_W, BOARD_H, PLATFORM_HALF, MODULE_HALF,
                                  PLATFORM_Z, MODULE_Z, PLATE_HALF, PLATE_Z)
        from newton_pseudo_render import mover_poses as _mover_poses
        board_body = builder.add_body(
            xform=wp.transform(wp.vec3(BOARD_W / 2, BOARD_H / 2, -0.005), wp.quat_identity()),
            is_kinematic=True, label="board")
        builder.add_shape_box(board_body, hx=BOARD_W / 2, hy=BOARD_H / 2, hz=0.005,
                              label="board_geo")
        for name in ["hub", "plug0", "plug1", "plug2", "plug3"]:
            pb = builder.add_body(is_kinematic=True, label=f"{name}_platform")
            builder.add_shape_box(pb, hx=PLATFORM_HALF[0], hy=PLATFORM_HALF[1],
                                  hz=PLATFORM_HALF[2], label=f"{name}_platform_geo")
            builder.add_shape_box(pb, xform=wp.transform(wp.vec3(0.0, 0.0, PLATE_Z), wp.quat_identity()),
                                  hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2],
                                  label=f"{name}_plate_geo")
            platform_bodies[name] = pb
            mb = builder.add_body(is_kinematic=True, label=f"{name}_module")
            builder.add_shape_box(mb, hx=MODULE_HALF, hy=MODULE_HALF, hz=MODULE_HALF,
                                  label=f"{name}_module_geo")
            module_bodies[name] = mb

    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    if _use_dahl:
        def _per_joint(v, name):
            """scalar | (4,) per cable | (4, n_joints) per cable and joint."""
            a = np.asarray(v, float)
            n_j = len(cable_joints[0]) if cable_joints else 0
            if a.ndim == 0:
                return np.full((4, n_j), float(a))
            if a.ndim == 1 and a.size == 4:
                return np.repeat(a[:, None], n_j, axis=1)
            if a.ndim == 1 and a.size == n_j:
                return np.repeat(a[None, :], 4, axis=0)
            if a.shape == (4, n_j):
                return a
            raise ValueError(f"{name}: shape {a.shape} fits neither a scalar, "
                             f"4 cables nor (4, {n_j}) joints")

        eps_pc = _per_joint(dahl_eps_max, "dahl_eps_max")
        tau_pc = _per_joint(dahl_tau, "dahl_tau")
        eps = np.zeros(model.joint_count, dtype=np.float32)
        tau = np.zeros(model.joint_count, dtype=np.float32)
        for k, jts in enumerate(cable_joints):
            for j, jid in enumerate(jts):
                eps[jid] = eps_pc[k, j]
                tau[jid] = tau_pc[k, j]
        model.vbd.dahl_eps_max.assign(eps)
        model.vbd.dahl_tau.assign(tau)

    collision_pipeline = newton.CollisionPipeline(model)
    # VBD's per-body contact list holds 64 entries by default, which this
    # scene overflows (measured 65-66) -- surplus contacts are dropped. A
    # finer cable puts more bodies into each contact zone, so leaving the
    # default would penalise higher subdiv for a buffer reason rather than
    # a physical one.
    solver = newton.solvers.SolverVBD(
        model, iterations=solver_iterations,
        rigid_body_contact_buffer_size=contact_buffer)
    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    def end_pose(t):
        """Position+quaternion for both ends of all 4 cables at frame t."""
        out = []
        for k in range(4):
            r = route3(k, t)
            pts = [wp.vec3(*p) for p in r]
            edge_q = create_parallel_transport_cable_quaternions(pts)
            out.append((r[0], np.array(list(edge_q[0])), r[-2], np.array(list(edge_q[-1]))))
        movers = _mover_poses(shp, t) if scene_collision else None
        return out, movers

    def nlerp_quat(q0, q1, f):
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)

    def pin_ends_interp(pose_a, pose_b, f):
        """Set both cable-end kinematic bodies to a linear/nlerp blend
        between frame-t and frame-(t+1) targets -- called every substep with
        an increasing f, NOT once per frame with a hard jump. Matches
        mujoco_video.py's actuator-target interpolation (f=(s+1)/sub) instead
        of the old teleport-then-settle approach, which produced a visible
        shock/oscillation each frame under the rod's low bend_damping (see
        chat history, 2026-08-18: user flagged extreme jumping in the
        render, traced to this being the only one of the 3 engines that
        hard-resets its driven endpoints instead of interpolating them like
        MuJoCo's PD actuators do)."""
        cable_a, movers_a = pose_a
        cable_b, movers_b = pose_b
        bq = state0.body_q.numpy()
        for k in range(4):
            p0a, q0a, p0b, q0b = cable_a[k]
            p1a, q1a, p1b, q1b = cable_b[k]
            bodies = cable_bodies[k]
            bq[bodies[0]] = np.concatenate([p0a + f * (p1a - p0a), nlerp_quat(q0a, q1a, f)])
            bq[bodies[-1]] = np.concatenate([p0b + f * (p1b - p0b), nlerp_quat(q0b, q1b, f)])
        if scene_collision:
            from mujoco_scene import PLATFORM_Z, MODULE_Z
            for name in movers_a:
                pos_a, yaw_a = movers_a[name]
                pos_b, yaw_b = movers_b[name]
                p = pos_a + f * (pos_b - pos_a)
                bq[platform_bodies[name]] = [p[0], p[1], PLATFORM_Z, 0.0, 0.0, 0.0, 1.0]
                qa = np.array(list(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw_a))))
                qb = np.array(list(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw_b))))
                bq[module_bodies[name]] = np.concatenate(
                    [[p[0], p[1], PLATFORM_Z + MODULE_Z], nlerp_quat(qa, qb, f)])
        state0.body_q.assign(bq)
        state1.body_q.assign(bq)

    def step_physics():
        nonlocal state0, state1
        state0.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    pose0 = end_pose(0)
    pin_ends_interp(pose0, pose0, 0.0)
    for _ in range(300):
        step_physics()

    errs = []
    pose_prev = pose0
    for t in range(n_frames):
        pose_next = end_pose(t)
        for s in range(steps_per_frame):
            f = (s + 1) / steps_per_frame
            pin_ends_interp(pose_prev, pose_next, f)
            step_physics()
        pose_prev = pose_next
        bq = state0.body_q.numpy()
        e_all = []
        for k in range(4):
            bodies = cable_bodies[k][::subdiv]
            sim_pts = bq[bodies][:, :2]
            real_pts = shp[k][t][:len(bodies)]
            e_all.append(float(np.linalg.norm(sim_pts - real_pts, axis=1).mean()) * 1000)
        errs.append(e_all if return_per_cable else float(np.mean(e_all)))
        if z_nodes_out is not None:
            # Full per-node heights (m), shape (4, n_bodies) per frame --
            # compared against the Depth-Anything height targets in the
            # measurement-based fit objective.
            z_nodes_out.append(np.stack([bq[cable_bodies[k][::subdiv]][:, 2]
                                         for k in range(4)]))
        if nodes_xyz_out is not None:
            # EVERY body's full xyz (not the scored subset) -- the collision
            # probe has to see the nodes between the scored ones, since those
            # are exactly where a too-coarse cable can slip through an
            # obstacle unnoticed.
            nodes_xyz_out.append(np.stack([bq[cable_bodies[k]][:, :3]
                                           for k in range(4)]))
        if z_max_out is not None:
            # Per-frame max node height per cable (m). Used by the fit's
            # height prior: the metric itself is 2D (xy only), which let a
            # CMA-ES fit exploit the unmeasured z axis -- light cables flew
            # ABOVE the movers in xy-convenient straight lines (measured up
            # to 122 mm, above the 80 mm module tops) instead of draping
            # and colliding like the real cable, which never rises above
            # its 55 mm connector height.
            z_max_out.append(np.array([bq[cable_bodies[k]][:, 2].max()
                                       for k in range(4)]))

    return np.array(errs) if return_per_cable else errs


if __name__ == "__main__":
    errs = eval_clip("20260602_170108", t1_sec=9)
    print(f"Start {errs[0]:.1f}  Ende {errs[-1]:.1f}  Mittel {np.mean(errs):.1f} mm/Knoten")
