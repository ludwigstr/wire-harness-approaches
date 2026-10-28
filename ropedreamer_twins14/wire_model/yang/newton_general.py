"""Newton "general" environment for the Yang et al. synthetic dataset: the
Newton general fit (approaches_table.py main row 12, "Per-cable fit, mass
pinned 0.044", 7.4 mm/node on the 14 clips) as the black-box simulator, in
place of a DE identification (user decision 2026-10-27).

Fit == data generator: the parameters are loaded with the fit's own
`newton_fit_everything_cmaes.unpack` (WIRE_MASS_PIN=0.044 layout, 231 values:
per-joint bend ke / twist kt / bend-rate kd [4 x 14], per-joint rest angles in
joint_X_c [4 x 14], K stretch, shape/cable/torsional/rolling friction, Dahl
eps/tau, mass 0.044 kg/m), and every world is built by
`newton_full_scene.build_scene` -- the scene-building code of `eval_clip`,
factored out verbatim -- with eval_clip's defaults for everything the fit does
not set. The Dahl values go onto the finalized model exactly as `set_dahl`
does, per world. Solver: VBD, 10 iterations, dt 1e-3, 250 substeps per 0.25 s
frame, 300-step settle, contact buffer 64 -- eval_clip's protocol.

Driving: eval_clip's own `end_pose_of(shp, t)` (kinematic cable-end segments
from the flat route's parallel-transport frames, mover boxes from
newton_pseudo_render.mover_poses), linearly / nlerp interpolated over the
substeps exactly like eval_clip's pin_ends_interp, but in one warp kernel for
all worlds. Only nodes 0, 1, 14, 15 of a drive frame enter end_pose_of (the
route is planar, so the last parallel-transport frame depends only on the
first and last segment directions), so a synthetic end motion is given as
4-node drive frames [n0, n1, n14, n15] per cable.

Batched: one Newton world per rollout, all integrated together.
"""
import inspect
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FIT_FILE = ROOT / "data/newton_fit_everything_pinned.npz"
MASS_PIN = "0.044"
N_KIN = 18          # 4 hub ends, 4 plug ends, 5 platforms, 5 modules
NAMES = ["hub"] + [f"plug{k}" for k in range(4)]


def fit_kwargs(path=FIT_FILE):
    """eval_clip keyword arguments of row 12, via the fit's own unpack()"""
    if os.environ.get("WIRE_MASS_PIN") not in (None, MASS_PIN):
        raise RuntimeError(f"WIRE_MASS_PIN={os.environ['WIRE_MASS_PIN']}, row 12 needs {MASS_PIN}")
    os.environ["WIRE_MASS_PIN"] = MASS_PIN          # read at import by the fit module
    import newton_fit_everything_cmaes as E
    assert E.MASS_PIN == float(MASS_PIN) and E.N_PAR == 231, (E.MASS_PIN, E.N_PAR)
    best = np.asarray(np.load(path, allow_pickle=True)["best"], float)
    assert len(best) == 231, len(best)
    return E.unpack(best)


def _scene_kwargs(kw):
    """build_scene arguments: eval_clip's defaults, overridden by the fit's values"""
    from newton_full_scene import build_scene, eval_clip
    defaults = {n: p.default for n, p in inspect.signature(eval_clip).parameters.items()
                if p.default is not inspect.Parameter.empty}
    names = [n for n, p in inspect.signature(build_scene).parameters.items()
             if p.kind == inspect.Parameter.KEYWORD_ONLY]
    out = {n: defaults[n] for n in names}
    for n, v in kw.items():
        assert n in out, f"fit parameter {n} is not a build_scene argument"
        out[n] = v
    return out


def _quat_yaw(y):
    return np.array([0.0, 0.0, np.sin(y / 2), np.cos(y / 2)])


def kin_poses(drive, t):
    """(18, 7) poses (p, q xyzw) of the kinematic bodies at frame t, from
    eval_clip's end_pose_of on the drive frames (dict k -> (T, n, 2))"""
    from newton_full_scene import end_pose_of
    from mujoco_scene import PLATFORM_Z, MODULE_Z
    cab, movers = end_pose_of(drive, t, True)
    P = np.zeros((N_KIN, 7))
    for k, (p0, q0, p1, q1) in enumerate(cab):
        P[k] = np.r_[p0, q0]
        P[4 + k] = np.r_[p1, q1]
    for m, n in enumerate(NAMES):
        pos, yaw = movers[n]
        P[8 + m] = [pos[0], pos[1], PLATFORM_Z, 0.0, 0.0, 0.0, 1.0]
        P[13 + m] = np.r_[[pos[0], pos[1], PLATFORM_Z + MODULE_Z], _quat_yaw(float(yaw))]
    return P


def contact_violations(bq, cab, seglen, plat, mod, radius=0.003, own=2, nsamp=9, active=None):
    """Constraint (c) for Newton scenes (user, 2026-10-27), geometric: every cable
    capsule (body origin + local +z x segment length, `nsamp` points, radius) against
    every mover box of its own world (platform + plate, axis-aligned; module, yawed).
    A cable may touch only its OWN hub (first `own` segments) / plug k (last `own`).
    Mover-mover contact cannot occur: the movers are kinematic and constraint (b)
    keeps their footprints apart. bq (Nb, 7) body poses; cab (W, 4, 15); seglen
    (W, 4, 15); plat, mod (W, 5) body ids [hub, plug0..3]. Returns (W,) number of
    violating sample points, (W,) smallest box distance (m) among violations and
    (W, 4, 5) violating samples per (cable, mover) pair."""
    from mujoco_scene import PLATFORM_HALF, PLATE_HALF, PLATE_Z, MODULE_HALF
    B = bq[cab]
    p, q = B[..., :3], B[..., 3:7]
    u, w = q[..., :3], q[..., 3:4]
    ez = np.zeros_like(p); ez[..., 2] = 1.0
    ax = ez + 2.0 * np.cross(u, np.cross(u, ez) + w * ez)
    sgrid = np.linspace(0.0, 1.0, nsamp)
    P = p[..., None, :] + ax[..., None, :] * (seglen[..., None, None] * sgrid[:, None])      # (W, 4, 15, ns, 3)
    W = len(cab)
    seg = np.arange(15)[None, None, :, None]
    kk = np.arange(4)[None, :, None, None]
    nviol = np.zeros(W, int); dmin = np.full(W, np.inf); pairs = np.zeros((W, 4, 5), int)
    for m in range(5):
        own_mask = (seg < own) if m == 0 else ((kk == m - 1) & (seg > 14 - own))
        own_mask = np.broadcast_to(own_mask, P.shape[:-1])
        boxes = []
        Pp = bq[plat[:, m]]
        boxes.append((Pp[:, :3], None, np.array(PLATFORM_HALF)))
        boxes.append((Pp[:, :3] + np.array([0, 0, PLATE_Z]), None, np.array(PLATE_HALF)))
        Pm = bq[mod[:, m]]
        boxes.append((Pm[:, :3], Pm[:, 3:7], np.full(3, MODULE_HALF)))
        for c, bqq, half in boxes:
            L = P - c[:, None, None, None, :]
            if bqq is not None:            # rotate into the box frame: q^-1 * v * q
                bu = -bqq[:, None, None, None, :3]; bw = bqq[:, None, None, None, 3:4]
                L = L + 2.0 * np.cross(bu, np.cross(bu, L) + bw * L)
            ex = np.maximum(np.abs(L) - half, 0.0)
            dist = np.linalg.norm(ex, axis=-1)
            v = (dist < radius) & ~own_mask
            if active is not None:
                v &= active[:, None, None, None]
            nviol += v.reshape(W, -1).sum(1)
            pairs[:, :, m] += v.reshape(W, 4, -1).sum(-1)
            dm = np.where(v, dist, np.inf).reshape(W, -1).min(1)
            dmin = np.minimum(dmin, dm)
    return nviol, dmin, pairs


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def pin(idx: wp.array(dtype=wp.int32), pa: wp.array(dtype=wp.transform), pb: wp.array(dtype=wp.transform),
                f: float, q0: wp.array(dtype=wp.transform), q1: wp.array(dtype=wp.transform)):
            # eval_clip.pin_ends_interp: linear position, nlerp rotation (shortest arc)
            i = wp.tid()
            a = pa[i]
            b = pb[i]
            ta = wp.transform_get_translation(a)
            tb = wp.transform_get_translation(b)
            ra = wp.transform_get_rotation(a)
            rb = wp.transform_get_rotation(b)
            if wp.dot(ra, rb) < 0.0:
                rb = -rb
            r = wp.normalize(ra + f * (rb - ra))
            X = wp.transform(ta + f * (tb - ta), r)
            q0[idx[i]] = X
            q1[idx[i]] = X
        _KERNEL = pin
    return _KERNEL


class Batch:
    """W copies of the row-12 scene, one Newton world each; world w is built on
    rests[w] (dict k -> (>=1, 16, 2), frame 0 = rest shape). kw: one eval_clip
    parameter dict for all worlds (default: the row-12 fit) or a list with one
    dict per world (domain randomization)."""

    def __init__(self, rests, kw=None, device=None):
        import newton
        import warp as wp
        from newton_full_scene import build_scene
        wp.config.quiet = True
        kw = fit_kwargs() if kw is None else kw
        kws = list(kw) if isinstance(kw, (list, tuple)) else [kw] * len(rests)
        assert len(kws) == len(rests)
        sks = [_scene_kwargs(k) for k in kws]
        sk = sks[0]
        self.device = device
        self.W = len(rests)
        use_dahl = sk["dahl_eps_max"] is not None and sk["dahl_tau"] is not None
        assert all((q["dahl_eps_max"] is not None and q["dahl_tau"] is not None) == use_dahl for q in sks)
        main = newton.ModelBuilder()
        if use_dahl:
            newton.solvers.SolverVBD.register_custom_attributes(main)
        self.cab = np.zeros((self.W, 4, 15), np.int64)
        self.kin = np.zeros((self.W, N_KIN), np.int64)
        self.L_last = np.zeros((self.W, 4))
        self.seglen = np.zeros((self.W, 4, 15))
        jids = []
        from newton_full_scene import initial_route3
        for w, R in enumerate(rests):
            b = newton.ModelBuilder()
            if use_dahl:
                newton.solvers.SolverVBD.register_custom_attributes(b)
            cb, cj, pb, mb = build_scene(b, R, **sks[w])
            off_b, off_j = main.body_count, main.joint_count
            main.add_world(b)
            self.cab[w] = np.asarray(cb) + off_b
            jids.append([[j + off_j for j in js] for js in cj])
            self.kin[w] = np.r_[[c[0] for c in cb], [c[-1] for c in cb], [pb[n] for n in NAMES],
                                [mb[n] for n in NAMES]] + off_b
            for k in range(4):
                r0 = initial_route3(R, k, table_fraction=sks[w]["table_fraction"], subdiv=sks[w]["subdiv"])
                self.L_last[w, k] = np.linalg.norm(r0[-1] - r0[-2])
                self.seglen[w, k] = np.linalg.norm(np.diff(r0, axis=0), axis=1)
        main.color()
        self.model = model = main.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        if use_dahl:              # set_dahl, for all worlds at once
            eps = np.zeros(model.joint_count, dtype=np.float32)
            tau = np.zeros(model.joint_count, dtype=np.float32)
            for w, cj in enumerate(jids):
                e_pc = np.broadcast_to(np.asarray(sks[w]["dahl_eps_max"], float), (4, 14))
                t_pc = np.broadcast_to(np.asarray(sks[w]["dahl_tau"], float), (4, 14))
                for k, jts in enumerate(cj):
                    for j, jid in enumerate(jts):
                        eps[jid] = e_pc[k, j]; tau[jid] = t_pc[k, j]
            model.vbd.dahl_eps_max.assign(eps)
            model.vbd.dahl_tau.assign(tau)
        self.pipe = newton.CollisionPipeline(model, broad_phase="sap")
        self.contacts = self.pipe.contacts()
        self.solver = newton.solvers.SolverVBD(model, iterations=10, rigid_body_contact_buffer_size=64)
        self.s0, self.s1 = model.state(), model.state()
        self.ctrl = model.control()
        self.idx = wp.array(self.kin.ravel().astype(np.int32), dtype=wp.int32, device=device)
        self.dt = 1e-3

    def _pin(self, A, B, f):
        import warp as wp
        dev = self.device
        wp.launch(_kernel(), dim=self.W * N_KIN,
                  inputs=[self.idx, wp.array(np.ascontiguousarray(A.reshape(-1, 7)), dtype=wp.transform, device=dev),
                          wp.array(np.ascontiguousarray(B.reshape(-1, 7)), dtype=wp.transform, device=dev), float(f),
                          self.s0.body_q, self.s1.body_q], device=dev)

    def _pin_arrays(self, A, B):
        import warp as wp
        dev = self.device
        return (wp.array(np.ascontiguousarray(A.reshape(-1, 7)), dtype=wp.transform, device=dev),
                wp.array(np.ascontiguousarray(B.reshape(-1, 7)), dtype=wp.transform, device=dev))

    def _step(self):
        self.s0.clear_forces()
        self.pipe.collide(self.s0, self.contacts)
        self.solver.step(self.s0, self.s1, self.ctrl, self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0

    def nodes(self):
        """(W, 4, 16, 3): 15 body origins + the tip of the last (kinematic) segment"""
        bq = self.s0.body_q.numpy()
        B = bq[self.cab]
        V = np.empty((self.W, 4, 16, 3))
        V[:, :, :15] = B[..., :3]
        u = B[:, :, 14, 3:6]; w_ = B[:, :, 14, 6:7]
        v = np.zeros_like(u); v[..., 2] = self.L_last
        V[:, :, 15] = B[:, :, 14, :3] + v + 2.0 * np.cross(u, np.cross(u, v) + w_ * v)
        return V

    def run(self, poses, sub=250, settle=300, check_every=0, lengths=None):
        """poses: (T, W, 18, 7) kinematic poses per frame. eval_clip's loop:
        pin frame 0 once, `settle` steps, then for every frame t (t = 0 too)
        `sub` substeps interpolating frame t-1 -> t, record. Returns (W, T, 4, 16, 3).
        check_every > 0: constraint (c) every check_every substeps (worlds only
        while t < lengths[w]); counts in self.contact_n / self.contact_dmin."""
        import warp as wp
        T = len(poses)
        self.contact_n = np.zeros(self.W, int); self.contact_dmin = np.full(self.W, np.inf)
        self.contact_pairs = np.zeros((self.W, 4, 5), int)
        Ls = np.full(self.W, T) if lengths is None else np.asarray(lengths)
        a0, _ = self._pin_arrays(poses[0], poses[0])
        wp.launch(_kernel(), dim=self.W * N_KIN, inputs=[self.idx, a0, a0, 0.0, self.s0.body_q, self.s1.body_q],
                  device=self.device)
        for _ in range(settle):
            self._step()
        out = []
        prev = poses[0]
        for t in range(T):
            A, B = self._pin_arrays(prev, poses[t])
            for s_ in range(sub):
                wp.launch(_kernel(), dim=self.W * N_KIN,
                          inputs=[self.idx, A, B, (s_ + 1) / sub, self.s0.body_q, self.s1.body_q], device=self.device)
                self._step()
                if check_every and (s_ + 1) % check_every == 0:
                    n, dm, pr = contact_violations(self.s0.body_q.numpy(), self.cab, self.seglen, self.kin[:, 8:13],
                                                   self.kin[:, 13:18], active=t < Ls)
                    self.contact_n += n; self.contact_dmin = np.minimum(self.contact_dmin, dm); self.contact_pairs += pr
            prev = poses[t]
            out.append(self.nodes())
        return np.stack(out, 1)


def rollouts(rests, drives, kw=None, device=None, fps=4.0, check_every=0, stats=None):
    """W rollouts: world w built on rests[w], driven by drives[w] (dict k ->
    (T_w, n, 2) drive frames). Shorter drives hold their last pose. Returns a
    list of (T_w, 4, 16, 3), each cut at its first non-finite frame.
    check_every > 0: the contact constraint (c); per-world results are appended
    to the list `stats` as (n violating samples, min distance)."""
    b = Batch(rests, kw=kw, device=device)
    Ts = [len(d[0]) for d in drives]
    T = max(Ts)
    poses = np.zeros((T, len(drives), N_KIN, 7))
    for w, d in enumerate(drives):
        for t in range(T):
            poses[t, w] = kin_poses(d, min(t, Ts[w] - 1)) if t < Ts[w] else poses[t - 1, w]
    V = b.run(poses, sub=max(1, round((1.0 / fps) / b.dt)), check_every=check_every, lengths=Ts)
    if stats is not None:
        stats.extend(zip(b.contact_n.tolist(), b.contact_dmin.tolist(), list(b.contact_pairs)))
    res = []
    for w, Tw in enumerate(Ts):
        v = V[w, :Tw]
        bad = np.nonzero(~np.isfinite(v).reshape(Tw, -1).all(1))[0]
        res.append(v[:bad[0]] if len(bad) else v)
    return res


def drive_from_poses(rest_frame, seq):
    """4-node drive frames [n0, n1, n14, n15] per cable for a (transformed)
    mover pose sequence (sim.Drive.pose_seq format): the end nodes ride
    rigidly on their mover, offsets in the mover frame from the tracked frame 0
    (rest_frame (4, 16, 2)). end_pose_of / mover_poses read only these nodes."""
    R = np.asarray(rest_frame, float)
    rot = lambda v, a: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
    hp0, hy0 = seq[0]["hub"]
    off = []
    for k in range(4):
        pp0, py0 = seq[0][f"plug{k}"]
        off.append(([rot(R[k, i] - hp0, -hy0) for i in (0, 1)], [rot(R[k, i] - pp0, -py0) for i in (14, 15)]))
    out = {k: np.zeros((len(seq), 4, 2)) for k in range(4)}
    for t, p in enumerate(seq):
        hp, hy = p["hub"]
        for k in range(4):
            pp, py = p[f"plug{k}"]
            out[k][t, 0:2] = [np.asarray(hp) + rot(o, hy) for o in off[k][0]]
            out[k][t, 2:4] = [np.asarray(pp) + rot(o, py) for o in off[k][1]]
    return out
