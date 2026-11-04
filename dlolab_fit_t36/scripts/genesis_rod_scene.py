"""Genesis scene with NATIVE Cosserat rods instead of an MJCF hinge chain.

Why this exists (user instruction, 2026-08-25): the previous cable was a chain
of hinge-jointed capsules written as MJCF. That model
  * cannot be differentiated (Genesis ships backward passes for FEM/MPM/PBD/
    SPH/SF/Tool/Kinematic -- not for the rigid solver), so every fit had to go
    through finite differences or CMA-ES;
  * has no plasticity, which this project's own notes call the dominant shape
    parameter;
  * could only be pinned by POSITION, so the cables did not leave the module
    holders perpendicular until a weld hack was added.

DLO-Lab's genesis fork ships a real rod: RODSolver + gs.materials.ROD.Base with
K (stretch), E (bend), G (twist) and plastic_yield/plastic_creep, and
`attach_to_rigid_link`, which binds rod vertices to a rigid link's FULL pose --
position and orientation together, which is exactly the boundary condition the
hinge chain lacked.

NOTE ON THE ENGINE: the native rod exists ONLY in the DLO-Lab fork
(genesis-world 1.0.0, extern/DLO-Lab).
genesis-world 1.3.3 -- the engine every previous project number comes from --
has no rod_solver at all. Results here are therefore NOT directly comparable to
the 18.45 mm of the hinge-chain pipeline: different engine AND different cable
model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dlolab_path import DLO_LAB  # noqa: E402
sys.path.insert(0, str(DLO_LAB))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from mujoco_scene import (track_clip, _rotz, DIR, CAV_Z, HUB_CAVS, PLUG_CAV,  # noqa: E402
                          COLORS,
                          BOARD_W, BOARD_H, PLATFORM_HALF, PLATE_HALF, PLATE_Z,
                          MODULE_HALF, MODULE_Z, PLATFORM_Z)
import genesis_full_scene as MJ                                    # noqa: E402

# rod vertices bound to each holder: TWO, not one -- a single vertex fixes only
# the position, two fix the direction as well, which is what makes the cable
# leave the holder perpendicular.
N_BIND = 2


def resample_route(P, n):
    """Resample an (m, d) polyline to n points along its INDEX parameter.

    Index parameter, not arc length, on purpose: the tracked route's nodes are
    the reference the error is measured against, so node i must keep sitting at
    parameter i/(m-1) no matter how finely the rod is discretised. Arc-length
    resampling would slide the correspondence as soon as the spacing is uneven.
    """
    P = np.asarray(P, float)
    m = len(P)
    u = np.linspace(0.0, m - 1.0, n)
    i0 = np.clip(np.floor(u).astype(int), 0, m - 2)
    w = (u - i0)[:, None]
    return P[i0] * (1.0 - w) + P[i0 + 1] * w


def sim_at_tracked(sim, n_track):
    """Simulated polyline -> the n_track positions the tracked nodes sit at."""
    return resample_route(sim, n_track)


def table_route(route2, table_fraction, z_conn, z_table=0.003):
    """Tracked xy path + a z-profile that rests `table_fraction` of the arc
    length on the table, ends at connector height, symmetric linear ramps --
    the project's established initial condition (newton_full_scene.
    initial_route3, user requirement 2026-08-24)."""
    P = np.asarray(route2, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = s_arc[-1]
    ramp = max(1e-6, (1.0 - table_fraction) / 2.0 * L)
    f = np.ones(len(P))
    f = np.where(s_arc < ramp, s_arc / ramp, f)
    f = np.where(s_arc > L - ramp, (L - s_arc) / ramp, f)
    z = z_conn + f * (z_table - z_conn)
    return np.hstack([P, z[:, None]])


def rod_file(route_m, path):
    """Write a rod's vertex polyline as the .npy that gs.morphs.Rod loads."""
    v = np.asarray(route_m, np.float32)
    np.save(path, v)
    return str(path)


def build_rod_scene(shp, frame_idx=0, K=1.0e4, E=1.0e5, G=0.0,
                    plastic_yield=10.0, plastic_creep=1.0,
                    segment_radius=0.003, segment_mass=0.002,
                    static_friction=0.3, kinetic_friction=0.25,
                    restitution=0.0,
                    use_inextensible=True, mu=1.0, requires_grad=False,
                    scratch=None, bptt_window=20, grad_clip=1.0,
                    only_cable=None, n_nodes=None, table_fraction=None,
                    damping=40.0, rigid_material=True):
    """Board + 5 movers (rigid, from the MJCF builders) + 4 native rods."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    # The gradient flags live on RODOptions, not on the material. And the rigid
    # solver gets skip_backward: it has no backward pass at all (Genesis ships
    # one for FEM/MPM/PBD/SPH/SF/Tool/Kinematic only), so without this the
    # backward call would fail on the board and movers.
    scene = gs.Scene(
        show_viewer=False,
        # bptt_window TRUNCATES backprop through time. Without it the chain
        # runs over every simulated step -- 29 frames x 250 substeps -- and the
        # gradient blew up to 1e32 and then NaN on the first iteration.
        # grad_clip bounds what survives per step. Both are what DLO-Lab's own
        # system_id.py uses ("truncated BPTT window = 20").
        sim_options=gs.options.SimOptions(
            dt=1e-3, requires_grad=requires_grad,
            bptt_window=(bptt_window if requires_grad else 0)),
        rod_options=gs.options.RODOptions(
            damping=damping, angular_damping=min(damping, 10.0),
            n_pbd_iters=10,
            grad_clip=(grad_clip if requires_grad else 0.0),
            requires_grad_K=requires_grad, requires_grad_E=requires_grad,
            requires_grad_G=requires_grad),
        rigid_options=gs.options.RigidOptions(skip_backward=True),
    )

    scratch = Path(scratch or (ROOT / "data" / "genesis_rod" / f"pid{os.getpid()}"))
    scratch.mkdir(parents=True, exist_ok=True)
    MJ._sweep_stale_pid_dirs(scratch.parent)

    def write(name, xml):
        p = scratch / f"{name}.xml"
        p.write_text(xml)
        return str(p)

    # CABLE-TO-BOARD FRICTION (2026-10-19). The MJCF `friction` attribute only
    # ever reached rigid-vs-rigid contacts: the rod coupler reads
    # geoms_info.coup_friction, which Genesis takes from the rigid MATERIAL and
    # defaults to 0.1. So every mu passed here was silently ignored -- measured,
    # mu = 0.2 / 0.5 / 2.0 / 4.0 gave BIT-IDENTICAL errors on three clips, which
    # is what finally gave it away. Handing the material in makes mu real.
    # rigid_material=False restores the pre-2026-10-19 scene, in which every
    # table row 11-14 (old 15-18) was measured: no material, i.e. the library's
    # coup_friction 0.1 for the cable-board contact, whatever mu says.
    rigid_mat = (gs.materials.Rigid(friction=float(np.clip(mu, 0.01, 5.0)),
                                    coup_friction=float(mu))
                 if rigid_material else None)

    ents = {}
    ents["board"] = scene.add_entity(
        gs.morphs.MJCF(file=write("board", MJ.board_xml(mu=mu))),
        **({} if rigid_mat is None else {"material": rigid_mat}))

    poses0 = MJ_poses(shp, frame_idx)
    for name in ["hub"] + [f"plug{k}" for k in range(4)]:
        pos, yaw = poses0[name]
        # The MJCF bodies sit at the origin and are CARRIED by their slide
        # joints, so the entity itself must be placed at the mover's world
        # position and the joints then driven RELATIVE to it (same convention
        # as genesis_full_scene.add_mover). Leaving this out puts every mover
        # at (0,0) and the cables chase them across the board.
        xy = (float(pos[0]), float(pos[1]), 0.0)
        ents[name] = scene.add_entity(
            gs.morphs.MJCF(file=write(name, MJ.platform_xml(name)), pos=xy),
            **({} if rigid_mat is None else {"material": rigid_mat}))
        ents[f"{name}_top"] = scene.add_entity(
            gs.morphs.MJCF(file=write(f"{name}_top", MJ.topper_xml(name, yaw)),
                           pos=xy),
            **({} if rigid_mat is None else {"material": rigid_mat}))

    # only_cable: build a SINGLE rod instead of all four. The rod solver's
    # adjoint proved non-reproducible with four rods in one scene -- identical
    # runs gave gradients 22 orders of magnitude apart on rods 1 and 2 while 0
    # and 3 stayed put -- so fitting one cable per scene is the way to keep the
    # gradient usable.
    rods = []
    cable_ids = range(4) if only_cable is None else [int(only_cable)]
    for k in cable_ids:
        r2 = shp[k][frame_idx]
        if table_fraction is None:
            route = np.hstack([r2, np.full((len(r2), 1), CAV_Z)])
        else:
            route = table_route(r2, table_fraction, CAV_Z)
        if n_nodes:
            route = resample_route(route, int(n_nodes))
        f = rod_file(route, scratch / f"cable{k}.npy")
        rods.append(scene.add_entity(
            material=gs.materials.ROD.Base(
                K=K, E=E, G=G, plastic_yield=plastic_yield,
                plastic_creep=plastic_creep, segment_radius=segment_radius,
                # segment_mass: one value for all rods, or one per cable (general fit)
                segment_mass=(float(segment_mass[k]) if np.ndim(segment_mass) else segment_mass),
                use_inextensible=use_inextensible,
                # The rod carries its OWN friction; the `mu` above only sets the
                # board's MJCF friction. That claim held only because the
                # rigid MATERIAL kept its 0.1 coup_friction default: with the
                # material wired in (2026-10-19) sweeping mu 0.1 -> 4.0 spans
                # 7.1mm over the 14 clips. The two below are a SECOND, separate
                # friction -- per-vertex mu_s/mu_k inside the rod solver, still
                # at library defaults and never fitted.
                static_friction=static_friction,
                kinetic_friction=kinetic_friction,
                restitution=restitution),
            morph=gs.morphs.Rod(file=f, rest_state="default", fixed=False),
            # Colour per cable, matching the tracked-route overlay -- without
            # it every rod renders white and the four cables are impossible to
            # tell apart in the video.
            surface=gs.surfaces.Default(color=(*COLORS[k], 1.0))))
    return scene, ents, rods, poses0


def _per_cable(v, k, shared_ndim):
    """cable k's profile: an array with one more axis than the shared profile
    ((4, n) scales, (4, n, 2) rest curvatures) gives every cable its own row
    (general fit); a shared profile applies to all four cables, as before"""
    a = np.asarray(v, float)
    return a[k] if a.ndim > shared_ndim else a


def MJ_poses(shp, t):
    """(pos_xy, yaw) for hub + 4 plugs -- same formula as the MJCF scene."""
    r2 = [shp[k][t] for k in range(4)]
    out = np.mean([P[1] - P[0] for P in r2], axis=0)
    hy = np.arctan2(out[1], out[0]) + np.pi / 2
    hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
    res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
    for k, P in enumerate(r2):
        v = P[-1] - P[-2]
        y = np.arctan2(v[1], v[0]) + np.pi / 2
        o = _rotz(np.array(PLUG_CAV), y)
        res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
    return res


def _mod_link(ents, name):
    """The ROTATING module link of a mover -- the part the cable plugs into."""
    for l in ents[f"{name}_top"].links:
        if l.name == f"{name}_mod":
            return l
    raise KeyError(f"{name}_mod not found")


def eval_clip(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
              kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
              shp_fps=None, verts_out=None, settle=300, hinge_scale=None,
              slack=1.0, stretch_scale=None, twist_scale=None,
              kappa_rest=None, pin_ends=False, hinge_damping=None,
              damping=None, state_a=0.0, state_b=0.0,
              state_a_load=None, state_a_unload=None,
              codebook=None, **rod_kw):
    """Full-scene rollout with native rods; returns per-frame mm/node error.

    The cable ends are ATTACHED to the module links, so they inherit the
    holder's full pose. That is the boundary condition the hinge chain could
    not express: the cable leaves the holder perpendicular by construction
    rather than by a driven rotation dof.
    """
    import genesis as gs
    import cv2

    t1 = None
    if t1_sec is not None:
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_p = cap.get(cv2.CAP_PROP_FPS) or 4.0
        cap.release()
        t1 = int(round(t1_sec * fps_p))
    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        shp, _t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    if damping is not None:
        rod_kw["damping"] = damping
    scene, ents, rods, _ = build_rod_scene(shp, 0, **rod_kw)
    # n_envs=1 rather than an unbatched build: attach_to_rigid_link expects
    # BATCHED link poses ((B,3) / (B,4)) and fails with an einsum shape error
    # on an unbatched scene.
    scene.build(n_envs=1)

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        dofs[n] = (e.get_joint(f"{n}_x").dofs_idx_local
                   + e.get_joint(f"{n}_y").dofs_idx_local)
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])
        et = ents[f"{n}_top"]
        xy = (et.get_joint(f"{n}_x").dofs_idx_local
              + et.get_joint(f"{n}_y").dofs_idx_local)
        turn = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy, turn)
        et.set_dofs_kp(np.full(2, kp_mover), xy)
        et.set_dofs_kv(np.full(2, kv_mover), xy)
        et.set_dofs_kp(np.full(1, kp_turn), turn)
        et.set_dofs_kv(np.full(1, kv_turn), turn)

    # Bind BOTH ends of every cable to its holder: N_BIND vertices each, so the
    # direction is fixed too, not only the point.
    n_v = rods[0].n_vertices
    for k, rod in enumerate(rods):
        rod.attach_to_rigid_link(_mod_link(ents, "hub"), list(range(N_BIND)))
        rod.attach_to_rigid_link(_mod_link(ents, f"plug{k}"),
                                 list(range(n_v - N_BIND, n_v)))
        if hinge_scale is not None:
            # Per-JOINT bending stiffness (patched solver, see
            # extern/DLO-Lab-perjoint). All four cables are the same cable, so
            # one profile applies to all of them.
            import torch
            hs = _per_cable(hinge_scale, k, 1)
            if len(hs) != rod.n_internal_vertices:
                # The profile was fitted at one discretisation; it describes
                # WHERE along the cable it is stiff, so it is resampled along
                # the same index parameter rather than truncated or padded.
                hs = resample_route(hs[:, None],
                                    rod.n_internal_vertices)[:, 0]
            rod.set_bending_scale(torch.as_tensor(hs.astype(np.float32)))
        if stretch_scale is not None:
            import torch
            ss = _per_cable(stretch_scale, k, 1)
            if len(ss) != rod.n_edges:
                ss = resample_route(ss[:, None], rod.n_edges)[:, 0]
            rod.set_stretching_scale(torch.as_tensor(ss.astype(np.float32)))
        if twist_scale is not None:
            import torch
            ts = _per_cable(twist_scale, k, 1)
            if len(ts) != rod.n_internal_vertices:
                ts = resample_route(ts[:, None], rod.n_internal_vertices)[:, 0]
            rod.set_twisting_scale(torch.as_tensor(ts.astype(np.float32)))
        if state_a or state_b:
            # State-conditioned stiffness (EgoPhys eq. 4-6): the stiffness
            # offset depends on the CURRENT bending and bending rate, so one
            # pair of numbers expresses what no set of constants can -- that a
            # cable resists gentle and hard bending differently.
            rod.set_state_coeffs(state_a, state_b)
        if state_a_load is not None or state_a_unload is not None:
            import torch
            n_h = rod.n_internal_vertices
            lo = np.full(n_h, 0.0) if state_a_load is None else np.asarray(state_a_load, float)
            un = np.full(n_h, 0.0) if state_a_unload is None else np.asarray(state_a_unload, float)
            if lo.size == 1: lo = np.full(n_h, float(lo))
            if un.size == 1: un = np.full(n_h, float(un))
            # THERMODYNAMIC CONSTRAINT, not a numerical guard: stiffening on
            # UNLOAD means the hinge returns more energy than it took, which is
            # an amplifier, not friction. Measured -- (0, +15) blew the solver
            # up, while (+15, 0) damped the swing as intended. Real hysteresis
            # loses energy every cycle, so loading must be at least as stiff as
            # unloading. Left to itself an optimiser WILL walk into the
            # unstable branch, because it lowers the loss right up until the
            # kernel dies.
            un = np.minimum(un, lo)
            rod.set_hinge_state_a(torch.as_tensor(lo.astype(np.float32)),
                                  torch.as_tensor(un.astype(np.float32)))
        if codebook is not None:
            # EgoPhys eq. 6: K prototypes per sign bank, mixed by a softmax
            # over features. `codebook` is a flat vector of 2K + K*F numbers.
            import torch
            cb = np.asarray(codebook, float)
            K, F = 4, 4
            rod.set_codebook(cb[:K], cb[K:2 * K],
                             cb[2 * K:2 * K + K * F].reshape(K, F),
                             tau=0.7, on=True)
        if hinge_damping is not None:
            # `is not None`, not truthiness: a kd of exactly 0 must still take
            # this call. The 1e-12 control run came out 3.8mm away from the
            # no-call baseline, i.e. the SETTER moves the scene and the damping
            # value barely does -- so "no damping" and "damping set to zero"
            # have to be measurable separately.
            import torch
            rod.set_bending_damping(torch.as_tensor(
                np.full(rod.n_internal_vertices, hinge_damping, np.float32)))
        if kappa_rest is not None:
            # Per-hinge REST CURVATURE -- the rod's equivalent of the per-joint
            # rest angle Newton fits, and the strongest single lever measured
            # here (16x more loss reduction per step than the stiffness scales).
            import torch
            kr = _per_cable(kappa_rest, k, 2)
            if len(kr) != rod.n_internal_vertices:
                kr = np.stack([resample_route(kr[:, i:i + 1],
                                              rod.n_internal_vertices)[:, 0]
                               for i in range(kr.shape[1])], axis=1)
            rod.set_kappa_rest_offset(torch.as_tensor(kr.astype(np.float32)))
        if slack != 1.0:
            rod.set_rest_length_scale(slack)

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    p0 = MJ_poses(shp, 0)
    base = {n: np.asarray(p0[n][0], float) for n in names}
    base_yaw = {n: float(p0[n][1]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply_movers(t, f=1.0):
        """Drive the holders to frame t's measured pose.

        `pin_ends` switches from PD control to a direct state overwrite, which
        is Newton's protocol: there the two end bodies are made kinematic and
        their poses are written every substep (interpolated, not teleported
        once per frame). Since the cable ends are ATTACHED to the holder links,
        writing the holder pose exactly puts the cable ends exactly on the
        tracked endpoints -- so 2 of the 16 scored nodes carry zero error by
        construction, exactly as they do for Newton. Only fair to switch on
        when comparing against Newton; it is a protocol change, not a better
        simulation.
        """
        p = MJ_poses(shp, t)
        for n in names:
            pos, yaw = p[n]
            prev_yaw[n] = unwrap(prev_yaw[n], yaw)
            rel_xy = np.asarray(pos, float) - base[n]
            rel_yaw = np.array([prev_yaw[n] - base_yaw[n]])
            if pin_ends:
                cur = ents[n].get_dofs_position(dofs[n]).cpu().numpy()
                tgt = cur + f * (rel_xy - cur)
                ents[n].set_dofs_position(tgt, dofs[n])
                ents[n].set_dofs_velocity(np.zeros(2), dofs[n])
                et = ents[f"{n}_top"]
                cur_t = et.get_dofs_position(dofs_top[n][0]).cpu().numpy()
                et.set_dofs_position(cur_t + f * (rel_xy - cur_t),
                                     dofs_top[n][0])
                cur_y = et.get_dofs_position(dofs_top[n][1]).cpu().numpy()
                et.set_dofs_position(cur_y + f * (rel_yaw - cur_y),
                                     dofs_top[n][1])
                et.set_dofs_velocity(np.zeros(2), dofs_top[n][0])
                et.set_dofs_velocity(np.zeros(1), dofs_top[n][1])
            else:
                ents[n].control_dofs_position(rel_xy, dofs[n])
                ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
                ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])

    apply_movers(0)
    for _ in range(settle):
        scene.step()

    errs = []
    for t in range(n_frames):
        if pin_ends:
            # interpolate the target across the substeps instead of jumping to
            # it once per frame -- a hard reset shocks the rod visibly
            for s_i in range(steps_per_frame):
                apply_movers(t, (s_i + 1) / steps_per_frame)
                scene.step()
        else:
            apply_movers(t)
            for _ in range(steps_per_frame):
                scene.step()
        sim = [r.get_state().pos.cpu().numpy().reshape(-1, 3) for r in rods]
        # A finer rod has more vertices than the 16 tracked nodes; score the
        # positions the tracked nodes sit at, so the number stays comparable
        # across discretisations.
        e_all = [float(np.linalg.norm(
            sim_at_tracked(sim[k], len(shp[k][t]))[:, :2] - shp[k][t],
            axis=1).mean()) * 1000 for k in range(4)]
        errs.append(float(np.mean(e_all)))
        if verts_out is not None:
            verts_out.append(np.stack(sim))
    return errs
