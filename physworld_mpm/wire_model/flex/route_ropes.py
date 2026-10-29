"""AdaptiGraph's FleX rope material on the measured harness routes.

One FleX scene per harness frame: the 4 cables are soft bodies built exactly
like AdaptiGraph's rope (sim/sim_env/scenes.py rope_scene + by_SoftRope:
NvFlexExtCreateSoftFromMesh, particle radius 0.03, cluster stiffness 0.55,
volume sampling 4, dynamic friction 0.1, particle friction 0.25, 4 solver
iterations, 2 substeps, radius x1.5 after creation), but the mesh is a closed
tube that follows the tracked route instead of their straight rope.obj. The
shape-matching rest pose is therefore the measured route (its plastic set).

Stiffness: AdaptiGraph's single scalar s ~ U[0, 1] (phi = s) with their
piecewise mapping to FleX global stiffness and cluster spacing (see material()).

Units. AdaptiGraph's rope is 1.5 * U[2.5, 3.0] = 3.75-4.5 FleX units long
(their table: 1 unit = 0.1 m, so a 0.4 m rope). Our cables are 0.60-0.89 m.
UNIT = 0.2 m per FleX unit puts them at 3.0-4.5 units, i.e. the same length
in particles and clusters as the paper's ropes, so the material constants keep
their meaning. Gravity is scaled consistently: -9.81 m/s^2 / 0.2 m = -49.05
units/s^2 (an option; AdaptiGraph uses FleX's default -9.8 units/s^2 =
0.98 m/s^2 at their 0.1 m/unit). Time is not rescaled: one pyflex.step = 1/60 s
of real time, so one 0.25 s video frame = 15 steps (x 2 substeps).
The table is FleX's ground plane y = 0; FleX (x, z) <-> our top view (x, y) in
metres via X = (x - cx) / UNIT, Z = (y - cy) / UNIT (a mirror, which a planar
problem with vertical gravity does not notice).

Cable ends: all particles within `pin_frac` of the first / last edge (arc
length along the route) get inverse mass 0 and follow the rigid 2-D motion of
the measured end edge (midpoint shift + rotation of nodes 0-1 / 14-15),
interpolated linearly over the 15 steps of a frame; their velocity is set to
the kinematic velocity so friction coupling is consistent.

Cables do not collide with each other (they all share one phase group without
self collision; their hub ends start inside each other) -- like the analytical
mass-spring generator.

Read-out: 16 nodes per cable by arc length. Node k is carried by the particles
of the slice |s - s_k| < slice_h of the creation-time route parametrisation:
the slice's 2-D rigid (Kabsch) transform since frame 0 is applied to the
node's frame-0 position, so pinned end nodes reproduce the driven end motion
exactly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

UNIT = 0.2                   # metres per FleX length unit
FPS_STEPS = 15               # pyflex steps per 0.25 s frame (dt = 1/60 s)
SCENE_IDX = 40               # wm_RouteRopes
N_NODES = 16

# AdaptiGraph rope constants (src/sim/sim_env/scenes.py rope_scene)
RADIUS = 0.03
CLUSTER_RADIUS = 0.0
CLUSTER_STIFFNESS = 0.55
LINK_RADIUS = 0.0
LINK_STIFFNESS = 1.0
SURFACE_SAMPLING = 0.0
VOLUME_SAMPLING = 4.0
PLASTIC_THRESHOLD = 0.0
PLASTIC_CREEP = 0.0
DYNAMIC_FRICTION = 0.1
PARTICLE_FRICTION = 0.25
RELAXATION = 1.0
COLLISION_DISTANCE = RADIUS * 0.5
N_ITER = 4
N_SUBSTEPS = 2
RADIUS_EXPAND = 1.5
# their rope's cross-section: ~0.12 units across (measured on rope_scene, thickness 3.0)
TUBE_RADIUS = 0.06


def material(s):
    """AdaptiGraph's piecewise stiffness mapping, s in [0, 1] -> (global_stiffness, cluster_spacing)"""
    s = float(s)
    if s < 0.5:
        return s * 1e-4 / 0.5, 2.0 + 8.0 * s
    return (s - 0.5) * 4e-4 + 1e-4, 6.0 + 4.0 * (s - 0.5)


def import_pyflex(root=os.path.expanduser("~/src/adaptigraph")):
    sys.path.insert(0, os.path.join(root, "PyFleX/bindings/build"))
    os.environ.setdefault("PYFLEXROOT", os.path.join(root, "PyFleX"))
    import pyflex
    return pyflex


# ----------------------------------------------------------------------------- geometry
def catmull_rom(P, sub=4):
    """(n, d) polyline -> centripetal-free uniform Catmull-Rom through all points, `sub` pieces per segment"""
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.arange(sub) / sub:
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return np.array(out)


def tube_mesh(C, r, n_seg=10):
    """closed tube around a centreline C (n, 3) lying in a horizontal plane -> vertices (V, 3), triangles (F, 3)"""
    n = len(C)
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(T, up); side /= np.linalg.norm(side, axis=1, keepdims=True)
    phi = 2 * np.pi * np.arange(n_seg) / n_seg
    ring = C[:, None, :] + r * (np.cos(phi)[None, :, None] * up + np.sin(phi)[None, :, None] * side[:, None, :])
    V = np.vstack([ring.reshape(-1, 3), C[0], C[-1]])
    F = []
    for i in range(n - 1):
        for k in range(n_seg):
            a, b = i * n_seg + k, i * n_seg + (k + 1) % n_seg
            c, d = a + n_seg, b + n_seg
            F += [(a, b, d), (a, d, c)]
    c0, c1 = n * n_seg, n * n_seg + 1
    for k in range(n_seg):
        F.append((c0, (k + 1) % n_seg, k))
        F.append((c1, (n - 1) * n_seg + k, (n - 1) * n_seg + (k + 1) % n_seg))
    return V.astype(np.float32), np.array(F, dtype=np.int32)


def edge_pose(P, plug):
    """(..., 16, 2) -> midpoint (..., 2), angle (...) of the hub edge (nodes 0-1) or plug edge (14-15)"""
    a, b = (P[..., -2, :], P[..., -1, :]) if plug else (P[..., 0, :], P[..., 1, :])
    d = b - a
    return 0.5 * (a + b), np.arctan2(d[..., 1], d[..., 0])


def arc_param(route, pts):
    """arc-length coordinate of the projection of pts (m, 2) onto polyline route (n, 2)"""
    seg = np.diff(route, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    rel = pts[:, None, :] - route[None, :-1, :]
    t = np.clip((rel * seg[None]).sum(-1) / (L ** 2)[None], 0, 1)
    proj = route[None, :-1, :] + t[..., None] * seg[None]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=-1)
    j = d.argmin(1)
    return cum[j] + t[np.arange(len(pts)), j] * L[j], cum[-1]


def kabsch2d(A, B):
    """best rotation angle mapping centred A onto centred B (both (m, 2))"""
    H = A.T @ B
    return np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------- simulation
class RouteRopes:
    """one FleX scene of n cables built on `routes0` (n, 16, 2) metres with stiffness s per cable"""

    def __init__(self, pyflex, routes0, s, gravity=-9.81 / UNIT, pin_frac=0.5, slice_h=0.04,
                 settle_steps=60, damping=0.0, self_collide=False, tube_radius=TUBE_RADIUS):
        self.pf = pyflex
        routes0 = np.asarray(routes0, np.float64)
        self.n = len(routes0)
        self.center = routes0.reshape(-1, 2).mean(0)
        R = (routes0 - self.center) / UNIT                                 # FleX units, (n, 16, 2) = (X, Z)
        self.R0 = R
        Vs, Fs, per = [], [], []
        y_c = tube_radius + RADIUS                                          # tube resting just above the plane
        for c in range(self.n):
            C2 = catmull_rom(R[c], 4)
            C = np.stack([C2[:, 0], np.full(len(C2), y_c), C2[:, 1]], 1)
            V, F = tube_mesh(C, tube_radius)
            gs, cs = material(s[c])
            Vs.append(V); Fs.append(F); per += [len(V), len(F), cs, gs]
        self.s = np.asarray(s, float)
        params = np.array([RADIUS, DYNAMIC_FRICTION, PARTICLE_FRICTION, COLLISION_DISTANCE, RELAXATION, N_ITER,
                           N_SUBSTEPS, gravity, self.n, VOLUME_SAMPLING, SURFACE_SAMPLING, CLUSTER_RADIUS,
                           CLUSTER_STIFFNESS, LINK_RADIUS, LINK_STIFFNESS, PLASTIC_THRESHOLD, PLASTIC_CREEP,
                           RADIUS_EXPAND, damping, 1.0 if self_collide else 0.0, *per], dtype=np.float32)
        z = np.array([0], dtype=np.int32)
        pyflex.set_scene(SCENE_IDX, params, np.concatenate(Vs).reshape(-1), z, z, z, np.concatenate(Fs).reshape(-1), 0)
        self.off = np.asarray(pyflex.wm_cable_offsets())
        pos = pyflex.get_positions().reshape(-1, 4).copy()
        self.N = len(pos)
        # per particle: cable, arc coordinate along the creation route, pin flags
        self.cable = np.zeros(self.N, int); self.sarc = np.zeros(self.N)
        self.Lc = np.zeros(self.n)
        self.pin_hub = np.zeros(self.N, bool); self.pin_plug = np.zeros(self.N, bool)
        self.slices = []
        for c in range(self.n):
            a, b = self.off[c], self.off[c + 1]
            sa, L = arc_param(R[c], pos[a:b, [0, 2]])
            self.cable[a:b] = c; self.sarc[a:b] = sa; self.Lc[c] = L
            e = L / (N_NODES - 1)
            self.pin_hub[a:b] = sa <= pin_frac * e
            self.pin_plug[a:b] = sa >= L - pin_frac * e
            sk = np.linspace(0, L, N_NODES)
            sl = []
            for k in range(N_NODES):
                idx = np.where(np.abs(sa - sk[k]) < slice_h)[0]
                if len(idx) < 3:
                    idx = np.argsort(np.abs(sa - sk[k]))[:6]
                sl.append(a + idx)
            self.slices.append(sl)
        self.pinned = self.pin_hub | self.pin_plug
        pos[self.pinned, 3] = 0.0
        pyflex.set_positions(pos.reshape(-1))
        # (X, Z) creation positions = the measured route: reference of the read-out and of the pin motion,
        # so any drift while settling shows up as error
        self.ref = pos[:, [0, 2]].copy()
        self.pin_ref = self.ref.copy()
        self.dt = 1.0 / 60.0
        for _ in range(settle_steps):                     # settle onto the plane with the ends held
            self._step()
        # frame-0 end-edge poses (FleX units) per cable, for the rigid pin motion
        self.hub0 = [edge_pose(R[c], False) for c in range(self.n)]
        self.plug0 = [edge_pose(R[c], True) for c in range(self.n)]
        self.y_pin = pyflex.get_positions().reshape(-1, 4)[:, 1].copy()

    def _step(self):
        self.pf.step(draw_planes=0, draw_objects=0, draw_shadow=0)

    def nodes(self):
        """current 16 nodes per cable, metres (n, 16, 2)"""
        X = self.pf.get_positions().reshape(-1, 4)[:, [0, 2]]
        out = np.zeros((self.n, N_NODES, 2))
        for c in range(self.n):
            for k, idx in enumerate(self.slices[c]):
                A, B = self.ref[idx], X[idx]
                ca, cb = A.mean(0), B.mean(0)
                th = kabsch2d(A - ca, B - cb)
                out[c, k] = cb + rot(th) @ (self.R0[c, k] - ca)
        return out * UNIT + self.center

    def pin_targets(self, hub, plug):
        """hub / plug: per cable (midpoint (2,), angle) in FleX units -> target (X, Z) of all pinned particles"""
        tgt = np.zeros((self.N, 2))
        for c in range(self.n):
            for mask, (m0, a0), (m1, a1) in ((self.pin_hub, self.hub0[c], hub[c]), (self.pin_plug, self.plug0[c], plug[c])):
                idx = np.where(mask & (self.cable == c))[0]
                tgt[idx] = (self.pin_ref[idx] - m0) @ rot(a1 - a0).T + m1
        return tgt

    def advance(self, hub_a, plug_a, hub_b, plug_b, n_steps=FPS_STEPS):
        """one frame: end edges move linearly from poses a to b (lists per cable of (mid_m, angle), METRES) over n_steps"""
        to_u = lambda P: [((m - self.center) / UNIT, a) for m, a in P]
        hub_a, plug_a, hub_b, plug_b = map(to_u, (hub_a, plug_a, hub_b, plug_b))
        lerp = lambda A, B, t: [((1 - t) * ma + t * mb, aa + t * np.arctan2(np.sin(ab - aa), np.cos(ab - aa))) for (ma, aa), (mb, ab) in zip(A, B)]
        pinned = self.pinned
        for k in range(n_steps):
            t = (k + 1) / n_steps
            tgt = self.pin_targets(lerp(hub_a, hub_b, t), lerp(plug_a, plug_b, t))
            pos = self.pf.get_positions().reshape(-1, 4).copy()
            vel = self.pf.get_velocities().reshape(-1, 3).copy()
            vel[pinned, 0] = (tgt[pinned, 0] - pos[pinned, 0]) / self.dt
            vel[pinned, 2] = (tgt[pinned, 1] - pos[pinned, 2]) / self.dt
            vel[pinned, 1] = 0.0
            pos[pinned, 0] = tgt[pinned, 0]; pos[pinned, 2] = tgt[pinned, 1]; pos[pinned, 1] = self.y_pin[pinned]
            self.pf.set_positions(pos.reshape(-1)); self.pf.set_velocities(vel.reshape(-1))
            self._step()
        return np.isfinite(self.pf.get_positions()).all()


def end_tracks(P):
    """measured (T, 16, 2) track -> per frame hub / plug edge poses (mid (T, 2), angle (T,))"""
    return edge_pose(P, False), edge_pose(P, True)


def rollout(pyflex, tracks, s, start=0, **kw):
    """simulate a scene built on frame `start` of tracks (n, T, 16, 2) metres, ends driven along the
    tracks' end edges; returns nodes (T - start, n, 16, 2) (row 0 = after settling) and an ok flag"""
    tracks = np.asarray(tracks, float)
    sim = RouteRopes(pyflex, tracks[:, start], s, **kw)
    hubs = [edge_pose(tracks[c], False) for c in range(len(tracks))]
    plugs = [edge_pose(tracks[c], True) for c in range(len(tracks))]
    pose = lambda E, t: [(E[c][0][t], E[c][1][t]) for c in range(len(tracks))]
    out = [sim.nodes()]; ok = True
    for t in range(start, tracks.shape[1] - 1):
        ok = sim.advance(pose(hubs, t), pose(plugs, t), pose(hubs, t + 1), pose(plugs, t + 1))
        if not ok:
            break
        out.append(sim.nodes())
    return np.stack(out), ok, sim
