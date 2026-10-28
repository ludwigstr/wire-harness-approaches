"""The black-box simulator of Yang, Stork, Stoyanov (RAS 158, 2022) -- Newton here.

Same role as wire_model/yang/sim.py (MuJoCo), with Newton's VBD rod in place
of AGX Dynamics. The scene is the project's Newton full scene
(scripts/newton_full_scene.py): four `add_rod` cables (capsule segments joined
by CABLE joints, 15 bodies per cable on the 16 tracked nodes), a board, and the
five movers (platform + module boxes) as kinematic colliders. One global value
per parameter, as in the paper.

Mapping of the paper's Table 1 (THETA_NAMES, all searched in log10). A Newton
CABLE joint has four material slots, stretch / shear / bend / twist, each a
stiffness (joint_target_ke) and a damping (joint_target_kd) used directly by
the VBD solver (force = ke * strain + kd * strain rate):
    k_stretch  stretch_stiffness, N/m per joint          <- E_stretch (and nu_stretch:
               the shear slot is set equal to stretch, Newton's isotropic default,
               i.e. the Poisson coupling AGX derives from E and nu is not separate)
    d_stretch  stretch_damping, N s/m (shear damping = it)  <- zeta_stretch
    k_bend     bend_stiffness, N m/rad per joint          <- E_bend (EI/l of the segment)
    d_bend     bend_damping, N m s/rad                    <- zeta_bend
    k_twist    twist_stiffness, N m/rad, passed EXPLICITLY <- E_twist / nu_twist (GJ/l)
               (add_rod copies bend into the twist slot when twist_stiffness AND
               twist_damping are both None -- the "twist was a copy" bug of the
               45-parameter fit. Here both are always given, so twist is free.)
    d_twist    twist_damping, N m s/rad                   <- zeta_twist
    mass_per_m linear density, kg/m, as the capsules' volumetric density solved so
               that the rod (cylinders + end caps) weighs exactly mass_per_m * L
               (newton_full_scene's mass_exact) <- mass; BOUNDED to 0.01-0.5 kg/m
               (measured 0.044) -- never an unbounded mass
    mu         contact friction of every shape in the scene (cables, board,
               movers); VBD mixes a pair as sqrt(mu_a mu_b), so one value for all
               shapes makes mu the friction of every contact. Not in Table 1, the
               cable lies on a board here (as mu in sim.py).
Unlike MuJoCo's inextensible capsule chain, a Newton rod is extensible: stretch
stiffness and damping are identifiable, so 8 of AGX's 10 Table-1 quantities
have a counterpart (the two Poisson ratios are folded into the stiffnesses).

Rest shape: Newton's VBD takes the rest pose from model.body_q at solver
creation, i.e. from the build positions. Every scene is built on the measured
frame 0 of its clip (lifted by newton_full_scene.initial_route3: 60 % of the
arc on the table, ends at connector height) -- the rest shape is frame 0, as in
sim.py and every Newton row.

Driving: the cable END segments are kinematic (as in newton_full_scene) and
follow the tracked end nodes -- position of node 0 / node 14, heading of the
first / last segment. One deviation from newton_full_scene.eval_clip: there,
every frame the ends get a fresh parallel-transport quaternion of a FLAT route,
which (a) tilts them away from the rest pose's sloped end segments and (b)
rolls them about their own axis whenever the end heading turns. With twist
stiffness a free parameter that would inject spurious torsion, so here an end
moves RIGIDLY: q(t) = Rz(yaw(t) - yaw(0)) q_rest. The movers' boxes are
kinematic colliders placed with the same pose formula as sim.Drive.poses.

Batched: every rollout / one-step transition is one Newton WORLD of a single
model (worlds do not collide with each other), so a whole DE population or a
dataset chunk is integrated in one GPU call per substep. dt = 1e-3, 250
substeps per 0.25 s frame, 10 VBD iterations (the Newton rows' protocol).

Metric: 16 nodes per cable, exactly as sim.py / mujoco_replay.cable_vertices:
the 15 segment-body origins (body_frame_origin="start", body j sits on node j)
plus the TIP of the last segment, body 14's origin + its local +z times the
segment's rest length. (newton_full_scene.eval_clip scores only the 15 body
origins against shp[:15], i.e. drops node 15; here node 15 is kept.)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

THETA_NAMES = ["k_stretch", "d_stretch", "k_bend", "d_bend", "k_twist", "d_twist", "mass_per_m", "mu"]
THETA_BOUNDS = [(1.0, 6.0),             # k_stretch, N/m   (newton_full_scene 1e5, physical fit 364)
                (-5.0, 0.0),            # d_stretch, N s/m (Newton default 0)
                (-4.0, 1.0),            # k_bend, N m/rad  (newton_full_scene 0.5)
                (-6.0, 0.0),            # d_bend, N m s/rad (newton_full_scene 0.05)
                (-4.0, 1.0),            # k_twist          (default = copy of bend, 0.5)
                (-6.0, 0.0),            # d_twist          (default = copy of bend, 0.05)
                (np.log10(0.01), np.log10(0.5)),   # kg/m (measured 0.044)
                (np.log10(0.05), np.log10(5.0))]   # mu   (scene 1.0)
# newton_full_scene defaults; stretch damping is 0 there (log10 needs a value:
# 1e-5 N s/m, the lower bound, is numerically nothing); mass = measured 0.044
THETA_DEFAULT = np.log10([1e5, 1e-5, 0.5, 0.05, 0.5, 0.05, 0.044, 1.0])
RADIUS = 0.003
N_NODES = 16
DT = 1e-3
ITERATIONS = 10
TABLE_FRACTION = 0.6
NAMES = ["hub"] + [f"plug{k}" for k in range(4)]


def theta_dict(x):
    v = 10.0 ** np.asarray(x, float)
    return dict(zip(THETA_NAMES, v.tolist()))


# --------------------------------------------------------------- quaternions (x, y, z, w)
def _qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def _qrot(q, v):
    u = np.asarray(q[:3]); w = q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _qz(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def _qbetween(u, v):
    """minimal rotation taking unit u onto unit v (x, y, z, w)"""
    c = float(np.dot(u, v))
    if c < -0.999999:
        ax = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return np.array([*ax, 0.0])
    q = np.array([*np.cross(u, v), 1.0 + c])
    return q / np.linalg.norm(q)


def _rotz2(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def lift(P2):
    """tracked (16, 2) route -> (16, 3) with newton_full_scene.initial_route3's z profile"""
    from newton_full_scene import initial_route3
    return initial_route3({0: [np.asarray(P2, float)]}, 0, table_fraction=TABLE_FRACTION)


def _yaw(v):
    return float(np.arctan2(v[1], v[0]))


def _unwrap(a):
    return np.unwrap(np.asarray(a, float), axis=0)


# --------------------------------------------------------------- kinematic schedules
# Per world 18 kinematic bodies in this order:
#   0-3  hub end segment of cable k   (body 0 of the rod)
#   4-7  plug end segment of cable k  (body 14 of the rod)
#   8-12 platform of hub, plug0..3    (no rotation)
#   13-17 module of hub, plug0..3     (yaw)
# A schedule gives per frame (T, 18, 2) xy and (T, 18) yaw; z and the base
# quaternion are per-world constants (rest pose).
N_KIN = 18


def mover_pose_seq(frames):
    """sim.Drive.pose_seq without importing MuJoCo; frames = (T, 4, 16, 2)"""
    from wire_model.yang.sim import Drive
    return Drive.pose_seq([[frames[t][k] for k in range(4)] for t in range(len(frames))])


def schedule_tracked(frames):
    """kinematic schedule from tracked frames (T, 4, 16, 2): cable ends follow
    the tracked end nodes / end-segment headings, movers the Drive pose formula"""
    F = np.asarray(frames, float)
    T = len(F)
    xy = np.zeros((T, N_KIN, 2)); yaw = np.zeros((T, N_KIN))
    xy[:, 0:4] = F[:, :, 0]
    xy[:, 4:8] = F[:, :, 14]
    yaw[:, 0:4] = _unwrap(np.arctan2(F[:, :, 1, 1] - F[:, :, 0, 1], F[:, :, 1, 0] - F[:, :, 0, 0]))
    yaw[:, 4:8] = _unwrap(np.arctan2(F[:, :, 15, 1] - F[:, :, 14, 1], F[:, :, 15, 0] - F[:, :, 14, 0]))
    seq = mover_pose_seq(F)
    for t, p in enumerate(seq):
        for m, n in enumerate(NAMES):
            xy[t, 8 + m] = p[n][0]; xy[t, 13 + m] = p[n][0]; yaw[t, 13 + m] = p[n][1]
    return xy, yaw


def schedule_from_poses(rest_frame, seq):
    """kinematic schedule from a (possibly transformed) mover pose sequence
    (list of {name: (xy, yaw)}, sim.Drive.pose_seq format), for synthetic data.
    Each cable end is RIGIDLY attached to its mover: its offset and heading in
    the mover frame are taken from the tracked frame 0 (rest_frame (4, 16, 2))."""
    R = np.asarray(rest_frame, float)
    p0 = seq[0]
    ends = []          # (mover, local offset xy, local heading)
    for k in range(4):
        hp, hy = p0["hub"]
        ends.append(("hub", _rotz2(R[k, 0] - hp, -hy), _yaw(R[k, 1] - R[k, 0]) - hy))
    for k in range(4):
        pp, py = p0[f"plug{k}"]
        ends.append((f"plug{k}", _rotz2(R[k, 14] - pp, -py), _yaw(R[k, 15] - R[k, 14]) - py))
    T = len(seq)
    xy = np.zeros((T, N_KIN, 2)); yaw = np.zeros((T, N_KIN))
    for t, p in enumerate(seq):
        for e, (n, off, hd) in enumerate(ends):
            mp, my = p[n]
            xy[t, e] = np.asarray(mp) + _rotz2(off, my); yaw[t, e] = my + hd
        for m, n in enumerate(NAMES):
            xy[t, 8 + m] = p[n][0]; xy[t, 13 + m] = p[n][0]; yaw[t, 13 + m] = p[n][1]
    return xy, yaw


# --------------------------------------------------------------- the batched engine
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def set_kinematic(idx: wp.array(dtype=wp.int32), xy_a: wp.array(dtype=wp.vec2),
                          xy_b: wp.array(dtype=wp.vec2), yaw_a: wp.array(dtype=float),
                          yaw_b: wp.array(dtype=float), yaw_ref: wp.array(dtype=float),
                          z: wp.array(dtype=float), qb: wp.array(dtype=wp.quat), f: float,
                          q0: wp.array(dtype=wp.transform), q1: wp.array(dtype=wp.transform)):
            i = wp.tid()
            p = xy_a[i] + f * (xy_b[i] - xy_a[i])
            a = yaw_a[i] + f * (yaw_b[i] - yaw_a[i]) - yaw_ref[i]
            q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a), qb[i])
            X = wp.transform(wp.vec3(p[0], p[1], z[i]), q)
            q0[idx[i]] = X
            q1[idx[i]] = X
        _KERNEL = set_kinematic
    return _KERNEL


class Engine:
    """W independent copies of the harness scene, one Newton world each.

    rests: list of W rest frames (4, 16, 2) (measured frame 0 of each world's
    clip). Parameters are set per world with set_theta (no rebuild)."""

    def __init__(self, rests, device=None, iterations=ITERATIONS, dt=DT, theta_build=THETA_DEFAULT):
        import newton
        import warp as wp
        from newton_full_scene import make_kinematic
        from mujoco_scene import (BOARD_W, BOARD_H, PLATFORM_HALF, MODULE_HALF, PLATFORM_Z, MODULE_Z,
                                  PLATE_HALF, PLATE_Z)
        wp.config.quiet = True
        self.W = len(rests)
        self.dt, self.iterations, self.device = dt, iterations, device
        self.rests = [np.asarray(r, float) for r in rests]
        p = theta_dict(theta_build)
        self.m_build = p["mass_per_m"]
        main = newton.ModelBuilder()
        self.cab = np.zeros((self.W, 4, 15), np.int64)       # cable body ids
        self.jnt = []                                        # cable joint ids per world
        self.kin = np.zeros((self.W, N_KIN), np.int64)
        self.kz = np.zeros((self.W, N_KIN)); self.kq = np.zeros((self.W, N_KIN, 4)); self.kyaw = np.zeros((self.W, N_KIN))
        self.L_last = np.zeros((self.W, 4))
        self.L_rest = np.zeros((self.W, 4))
        self.rest3 = []
        for w, R in enumerate(self.rests):
            b = newton.ModelBuilder()
            b.default_shape_cfg.mu = p["mu"]
            r3 = [lift(R[k]) for k in range(4)]
            self.rest3.append(r3)
            jw = []
            for k in range(4):
                cfg = b.ShapeConfig()
                cfg.mu = p["mu"]
                seg = np.linalg.norm(np.diff(r3[k], axis=0), axis=1)
                vol = np.pi * RADIUS ** 2 * seg.sum() + len(seg) * (4.0 / 3.0) * np.pi * RADIUS ** 3
                cfg.density = float(self.m_build * seg.sum() / vol)
                bodies, joints = b.add_rod(positions=[wp.vec3(*q) for q in r3[k]], radius=RADIUS, cfg=cfg,
                                           stretch_stiffness=p["k_stretch"], stretch_damping=p["d_stretch"],
                                           bend_stiffness=p["k_bend"], bend_damping=p["d_bend"],
                                           twist_stiffness=p["k_twist"], twist_damping=p["d_twist"],
                                           body_frame_origin="start", label=f"cable{k}")
                assert len(bodies) == 15, len(bodies)
                make_kinematic(b, bodies[0]); make_kinematic(b, bodies[-1])
                self.cab[w, k] = bodies
                jw.append(list(joints))
                self.L_last[w, k] = seg[-1]; self.L_rest[w, k] = seg.sum()
            board = b.add_body(xform=wp.transform(wp.vec3(BOARD_W / 2, BOARD_H / 2, -0.005), wp.quat_identity()),
                               is_kinematic=True, label="board")
            b.add_shape_box(board, hx=BOARD_W / 2, hy=BOARD_H / 2, hz=0.005)
            plat, mod = [], []
            for n in NAMES:
                pb = b.add_body(is_kinematic=True, label=f"{n}_platform")
                b.add_shape_box(pb, hx=PLATFORM_HALF[0], hy=PLATFORM_HALF[1], hz=PLATFORM_HALF[2])
                b.add_shape_box(pb, xform=wp.transform(wp.vec3(0.0, 0.0, PLATE_Z), wp.quat_identity()),
                                hx=PLATE_HALF[0], hy=PLATE_HALF[1], hz=PLATE_HALF[2])
                mb = b.add_body(is_kinematic=True, label=f"{n}_module")
                b.add_shape_box(mb, hx=MODULE_HALF, hy=MODULE_HALF, hz=MODULE_HALF)
                plat.append(pb); mod.append(mb)
            off_b, off_j = main.body_count, main.joint_count
            main.add_world(b)
            self.cab[w] += off_b
            self.jnt.append([[j + off_j for j in js] for js in jw])
            self.kin[w] = np.r_[self.cab[w, :, 0], self.cab[w, :, -1], plat, mod] + np.r_[np.zeros(8, int), np.full(10, off_b)]
            # constants: z and base quaternion (rest pose) of every kinematic body
            bq = np.array([list(main.body_q[i]) for i in self.kin[w][:8]])
            self.kz[w, :8] = bq[:, 2]; self.kq[w, :8] = bq[:, 3:7]
            self.kyaw[w, 0:4] = [_yaw(R[k, 1] - R[k, 0]) for k in range(4)]
            self.kyaw[w, 4:8] = [_yaw(R[k, 15] - R[k, 14]) for k in range(4)]
            self.kz[w, 8:13] = PLATFORM_Z; self.kz[w, 13:18] = PLATFORM_Z + MODULE_Z
            self.kq[w, 8:] = [0, 0, 0, 1.0]
        main.color()
        self.model = model = main.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        self._mass0 = model.body_mass.numpy().copy(); self._imass0 = model.body_inv_mass.numpy().copy()
        self._I0 = model.body_inertia.numpy().copy(); self._iI0 = model.body_inv_inertia.numpy().copy()
        self._mu0 = model.shape_material_mu.numpy().copy()
        self.shape_world = model.shape_world.numpy()
        self.qd_start = model.joint_qd_start.numpy()
        self.pipe = newton.CollisionPipeline(model, broad_phase="sap")
        self.contacts = self.pipe.contacts()
        self.kin_flat = wp.array(self.kin.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self.kz_d = wp.array(self.kz.ravel(), dtype=float, device=device)
        self.kq_d = wp.array(self.kq.reshape(-1, 4), dtype=wp.quat, device=device)
        self.kyaw_d = wp.array(self.kyaw.ravel(), dtype=float, device=device)
        self.solver = None
        self.theta = None

    # ------------------------------------------------------------ parameters
    def set_theta(self, X):
        """X: (W, 8) log10 parameters, or one (8,) vector for all worlds"""
        import newton
        X = np.broadcast_to(np.asarray(X, float), (self.W, len(THETA_NAMES)))
        self.theta = X.copy()
        m = self.model
        ke = m.joint_target_ke.numpy(); kd = m.joint_target_kd.numpy()
        mass, imass = self._mass0.copy(), self._imass0.copy()
        I, iI = self._I0.copy(), self._iI0.copy()
        mu = self._mu0.copy()
        if not hasattr(self, "_jd"):        # index tables, built once
            self._jd = np.stack([self.qd_start[np.concatenate([np.asarray(js) for js in self.jnt[w]])]
                                 for w in range(self.W)])                         # (W, n_joints)
            self._dyn = self.cab[:, :, 1:-1].reshape(self.W, -1)                  # (W, 52)
            self._sw = self.shape_world.copy()
        V = 10.0 ** X                                                             # (W, 8)
        col = lambda name: V[:, THETA_NAMES.index(name)][:, None]
        for off, name in ((0, "k_stretch"), (1, "k_stretch"), (2, "k_bend"), (3, "k_twist")):
            ke[self._jd + off] = col(name)
        for off, name in ((0, "d_stretch"), (1, "d_stretch"), (2, "d_bend"), (3, "d_twist")):
            kd[self._jd + off] = col(name)
        sc = col("mass_per_m") / self.m_build
        mass[self._dyn] *= sc; imass[self._dyn] /= sc
        I[self._dyn] *= sc[:, :, None, None]; iI[self._dyn] /= sc[:, :, None, None]
        ok = self._sw >= 0
        mu[ok] = V[self._sw[ok], THETA_NAMES.index("mu")]
        m.joint_target_ke.assign(ke); m.joint_target_kd.assign(kd)
        m.body_mass.assign(mass); m.body_inv_mass.assign(imass)
        m.body_inertia.assign(I); m.body_inv_inertia.assign(iI)
        m.shape_material_mu.assign(mu)
        # the VBD solver reads stiffness, damping and masses at construction
        self.solver = newton.solvers.SolverVBD(m, iterations=self.iterations, rigid_body_contact_buffer_size=64)

    # ------------------------------------------------------------ state
    def reset_state(self, frames_i=None, sched_xy=None, sched_yaw=None):
        """fresh states. frames_i None: the rest pose (frame 0). Otherwise
        frames_i (W, 4, 16, 2): every cable laid through the observed nodes of
        its world's frame i (lifted by the same z profile as the rest), segment
        frames by parallel transport from the rigidly moved hub-end frame; the
        kinematic bodies at the schedule's pose (sched_xy (W, 18, 2), sched_yaw (W, 18))."""
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.ctrl = self.model.control()
        if frames_i is None:
            return
        key = getattr(self, "_reset_key", None)
        if key is not None and key == getattr(self, "_reset_key_req", None):
            self.s0.body_q.assign(self._reset_bq); self.s1.body_q.assign(self._reset_bq)
            self._set_kin(sched_xy, sched_xy, sched_yaw, sched_yaw, 0.0)
            return
        bq = self.s0.body_q.numpy()
        for w in range(self.W):
            for k in range(4):
                P = lift(frames_i[w][k])
                q = _qmul(_qz(sched_yaw[w, k] - self.kyaw[w, k]), self.kq[w, k])
                d_prev = _qrot(q, np.array([0, 0, 1.0]))
                ids = self.cab[w, k]
                for j in range(15):
                    d = P[j + 1] - P[j]; d /= np.linalg.norm(d)
                    if j > 0:
                        q = _qmul(_qbetween(d_prev, d), q)
                    q /= np.linalg.norm(q)
                    bq[ids[j]] = np.r_[P[j], q]
                    d_prev = d
        self.s0.body_q.assign(bq); self.s1.body_q.assign(bq)
        self._reset_bq, self._reset_key = bq.copy(), getattr(self, "_reset_key_req", None)
        if sched_xy is not None:
            self._set_kin(sched_xy, sched_xy, sched_yaw, sched_yaw, 0.0)

    def _set_kin(self, xya, xyb, ya, yb, f):
        import warp as wp
        dev = self.device
        wp.launch(_kernel(), dim=self.W * N_KIN,
                  inputs=[self.kin_flat, wp.array(np.ascontiguousarray(xya.reshape(-1, 2)), dtype=wp.vec2, device=dev),
                          wp.array(np.ascontiguousarray(xyb.reshape(-1, 2)), dtype=wp.vec2, device=dev),
                          wp.array(ya.ravel(), dtype=float, device=dev), wp.array(yb.ravel(), dtype=float, device=dev),
                          self.kyaw_d, self.kz_d, self.kq_d, float(f), self.s0.body_q, self.s1.body_q], device=dev)

    def _step(self):
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def hold(self, xy, yaw, n):
        """n substeps with the kinematic bodies held at (xy (W,18,2), yaw (W,18))"""
        import warp as wp
        dev = self.device
        xa = wp.array(np.ascontiguousarray(xy.reshape(-1, 2)), dtype=wp.vec2, device=dev)
        ya = wp.array(yaw.ravel(), dtype=float, device=dev)
        for _ in range(n):
            wp.launch(_kernel(), dim=self.W * N_KIN, inputs=[self.kin_flat, xa, xa, ya, ya, self.kyaw_d, self.kz_d,
                                                              self.kq_d, 0.0, self.s0.body_q, self.s1.body_q], device=dev)
            self._step()

    def frame(self, xya, ya, xyb, yb, sub, check_every=0, active=None):
        """one video frame: `sub` substeps, kinematic bodies interpolated a -> b;
        check_every > 0: contact constraint (c) (newton_general.contact_violations)
        every check_every substeps, accumulated in self.contact_n / contact_dmin"""
        import warp as wp
        dev = self.device
        arr = lambda v, t: wp.array(np.ascontiguousarray(v), dtype=t, device=dev)
        xa, xb = arr(xya.reshape(-1, 2), wp.vec2), arr(xyb.reshape(-1, 2), wp.vec2)
        a, b = arr(ya.ravel(), float), arr(yb.ravel(), float)
        for s_ in range(sub):
            wp.launch(_kernel(), dim=self.W * N_KIN, inputs=[self.kin_flat, xa, xb, a, b, self.kyaw_d, self.kz_d,
                                                              self.kq_d, (s_ + 1) / sub, self.s0.body_q, self.s1.body_q],
                      device=dev)
            self._step()
            if check_every and (s_ + 1) % check_every == 0:
                from wire_model.yang.newton_general import contact_violations
                if not hasattr(self, "seglen"):
                    self.seglen = np.stack([[np.linalg.norm(np.diff(r[k], axis=0), axis=1) for k in range(4)]
                                            for r in self.rest3])
                if not hasattr(self, "contact_n"):
                    self.contact_n = np.zeros(self.W, int); self.contact_dmin = np.full(self.W, np.inf)
                    self.contact_pairs = np.zeros((self.W, 4, 5), int)
                n, dm, pr = contact_violations(self.s0.body_q.numpy(), self.cab, self.seglen, self.kin[:, 8:13],
                                               self.kin[:, 13:18], active=active)
                self.contact_n += n; self.contact_dmin = np.minimum(self.contact_dmin, dm); self.contact_pairs += pr

    def vertices(self):
        """(W, 4, 16, 3): 15 body origins + the tip of the last segment"""
        bq = self.s0.body_q.numpy()
        B = bq[self.cab]                                     # (W, 4, 15, 7)
        V = np.empty((self.W, 4, 16, 3))
        V[:, :, :15] = B[..., :3]
        q = B[:, :, 14, 3:7]
        u = q[..., :3]; w_ = q[..., 3:4]
        v = np.zeros_like(u); v[..., 2] = self.L_last
        tip = v + 2.0 * np.cross(u, np.cross(u, v) + w_ * v)
        V[:, :, 15] = B[:, :, 14, :3] + tip
        return V


def _sub(fps, dt=DT):
    return max(1, int(round((1.0 / fps) / dt)))


# --------------------------------------------------------------- one-step error (Eq. 1)
def onestep_errors(engine_cache, transitions, X, settle=0, device=None):
    """Eq. (1) for every (candidate, transition) pair in one batch.

    transitions: list of (frames (T, 4, 16, 2), fps, i); X: (W, 8) log10 per world
    (len(X) == len(transitions)). Each world: reset to the OBSERVED frame i,
    `settle` substeps with the kinematic bodies held at frame i, one frame
    (dt = 1/fps) driven i -> i+1, compare the 16 nodes with frame i+1.
    Returns (sq, ab): (W,) mean squared and plain node distance (m^2, m); inf for
    non-finite worlds. engine_cache: dict reused across calls (key = rests)."""
    W = len(transitions)
    key = tuple((id(fr), i) for fr, _, i in transitions)
    eng = engine_cache.get(key)
    if eng is None:
        if len(engine_cache) >= 3:
            engine_cache.clear()
        eng = Engine([fr[0] for fr, _, _ in transitions], device=device)
        eng._sched = [schedule_tracked(fr) for fr, _, _ in transitions]
        engine_cache[key] = eng
    eng.set_theta(X)
    xa = np.stack([s[0][i] for s, (_, _, i) in zip(eng._sched, transitions)])
    ya = np.stack([s[1][i] for s, (_, _, i) in zip(eng._sched, transitions)])
    xb = np.stack([s[0][i + 1] for s, (_, _, i) in zip(eng._sched, transitions)])
    yb = np.stack([s[1][i + 1] for s, (_, _, i) in zip(eng._sched, transitions)])
    eng._reset_key_req = key          # the reset pose does not depend on theta: computed once per engine
    eng.reset_state(np.stack([fr[i] for fr, _, i in transitions]), xa, ya)
    if settle:
        eng.hold(xa, ya, settle)
    subs = {_sub(f) for _, f, _ in transitions}
    assert len(subs) == 1, f"mixed frame rates {subs}"
    eng.frame(xa, ya, xb, yb, subs.pop())
    V = eng.vertices()[..., :2]
    tgt = np.stack([fr[i + 1] for fr, _, i in transitions])
    dist = np.linalg.norm(V - tgt, axis=-1).reshape(W, -1)
    ok = np.isfinite(dist).all(1)
    sq = np.where(ok, (np.nan_to_num(dist) ** 2).mean(1), np.inf)
    ab = np.where(ok, np.nan_to_num(dist).mean(1), np.inf)
    return sq, ab


# --------------------------------------------------------------- open-loop rollouts
def rollouts(rests, schedules, fps, X, settle=300, device=None, engine=None, check_every=0, stats=None):
    """open-loop rollouts of W scenes (rest = frame 0, `rests` (W, 4, 16, 2))
    along kinematic schedules [(xy (T_w, 18, 2), yaw (T_w, 18))]: 300-substep
    settle at the schedule's frame 0, record, then every frame (mujoco_full_loco
    .rollout_errs protocol). Worlds with shorter schedules hold their last pose.
    Returns a list of (T_w, 4, 16, 3) vertex arrays, cut at the first
    non-finite frame."""
    eng = engine if engine is not None else Engine(rests, device=device)
    eng.set_theta(X)
    eng.reset_state()
    T = max(len(s[0]) for s in schedules)
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], T - len(a), 0)])
    XY = np.stack([pad(s[0]) for s in schedules], 1)       # (T, W, 18, 2)
    YW = np.stack([pad(s[1]) for s in schedules], 1)
    sub = _sub(fps)
    eng.hold(XY[0], YW[0], settle)
    out = [eng.vertices()]
    Ls = np.array([len(s[0]) for s in schedules])
    eng.contact_n = np.zeros(eng.W, int); eng.contact_dmin = np.full(eng.W, np.inf)
    eng.contact_pairs = np.zeros((eng.W, 4, 5), int)
    for t in range(T - 1):
        eng.frame(XY[t], YW[t], XY[t + 1], YW[t + 1], sub, check_every=check_every, active=(t + 1) < Ls)
        out.append(eng.vertices())
    if stats is not None:
        stats.extend(zip(eng.contact_n.tolist(), eng.contact_dmin.tolist(), list(eng.contact_pairs)))
    V = np.stack(out, 1)                                    # (W, T, 4, 16, 3)
    res = []
    for w, s in enumerate(schedules):
        v = V[w, :len(s[0])]
        bad = np.nonzero(~np.isfinite(v).reshape(len(v), -1).all(1))[0]
        res.append(v[:bad[0]] if len(bad) else v)
    return res


def rollout_errs(frames, fps, X, device=None):
    """per-frame mm/node of the simulator alone on tracked frames (T, 4, 16, 2):
    one world, measured end motion, every frame incl. frame 0 after the settle"""
    V = rollouts([frames[0]], [schedule_tracked(frames)], fps, np.asarray(X)[None], device=device)[0]
    return 1000 * np.linalg.norm(V[..., :2] - frames[:len(V)], axis=-1).mean(axis=(1, 2))
