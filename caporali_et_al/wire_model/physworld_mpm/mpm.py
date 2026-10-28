"""Differentiable 3-D MLS-MPM in NVIDIA Warp (written for the PhysWorld rebuild, 2026-10-28).

Batched: G independent background grids ("scenes", e.g. one per cable or per demo
cable); every particle carries its grid id. The grid is sparse in blocks of 4x4x4
nodes: per frame the blocks touched by particles (dilated by the frame's maximum
drive displacement) are activated, so a 0.8 m x 6 mm cable costs ~1e4-1e5 grid
nodes instead of ~1e6 for the dense box.

Constitutive model (PhysWorld App. B.1 library, implemented in one code path):
  * elastic part: compressible Neo-Hookean in the decoupled form
      tau = mu dev(b_bar) + kappa/2 (J^2 - 1) I            (Kirchhoff stress)
  * plastic part: von Mises (J2) with the finite-strain radial return of Simo (1988)
    on b_bar_e (no SVD needed, so it is differentiable without the degenerate-
    singular-value problem of SVD adjoints). yield stress sy = +inf -> purely elastic.
  Per particle state: x, v, C (APIC affine), b = elastic left Cauchy-Green tensor.
Optional quasi-static MASS SCALING (a deviation from the paper, see twin.py): the
inertial density rhoi may exceed the true density rho; gravity (and through the normal
force the friction) uses rho, inertia and the CFL limit use rhoi.
Boundary: board plane z = 0 (grid nodes with k <= floor_k) with Coulomb friction
(per-node friction coefficient = mass-weighted P2G average of the particles' mu).
Grasping primitive (PhysWorld Sec. 3.1): grid nodes within r_grasp of a grasp
point get the grasp's velocity assigned directly; the grasp is a rigid clamp
(translation + yaw rate about the board normal: v = v_c + omega x (x - c))
because the cable ends sit in hub/plug modules that rotate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

BS = wp.constant(4)
BS3 = wp.constant(64)


@wp.func
def nidx(bmap: wp.array4d(dtype=wp.int32), g: int, i: int, j: int, k: int) -> int:
    if i < 0 or j < 0 or k < 0:
        return -1
    bi = i // BS
    bj = j // BS
    bk = k // BS
    if bi >= bmap.shape[1] or bj >= bmap.shape[2] or bk >= bmap.shape[3]:
        return -1
    b = bmap[g, bi, bj, bk]
    if b < 0:
        return -1
    return b * BS3 + (i - bi * BS) * 16 + (j - bj * BS) * 4 + (k - bk * BS)


@wp.func
def qweights(fx: wp.vec3) -> wp.mat33:
    """rows: quadratic B-spline weights of stencil offsets 0, 1, 2 (per axis)"""
    a = wp.vec3(1.5) - fx
    b = fx - wp.vec3(1.0)
    c = fx - wp.vec3(0.5)
    w0 = 0.5 * wp.cw_mul(a, a)
    w1 = wp.vec3(0.75) - wp.cw_mul(b, b)
    w2 = 0.5 * wp.cw_mul(c, c)
    return wp.mat33(w0[0], w0[1], w0[2], w1[0], w1[1], w1[2], w2[0], w2[1], w2[2])


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
        b: wp.array(dtype=wp.mat33), gid: wp.array(dtype=wp.int32),
        E: wp.array(dtype=float), nu: wp.array(dtype=float), rho: wp.array(dtype=float),
        rhoi: wp.array(dtype=float), muf: wp.array(dtype=float), sy: wp.array(dtype=float),
        vol: wp.array(dtype=float), grav: float,
        origin: wp.vec3, inv_dx: float, dt: float, bmap: wp.array4d(dtype=wp.int32), floor_k: int,
        gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gmu: wp.array(dtype=float),
        b_out: wp.array(dtype=wp.mat33), missed: wp.array(dtype=wp.int32)):
    p = wp.tid()
    g = gid[p]
    Cp = C[p]
    I = wp.identity(n=3, dtype=float)
    f = I + dt * Cp
    bt = f * b[p] * wp.transpose(f)
    J2 = wp.determinant(bt)
    J = wp.sqrt(J2)
    jm = wp.pow(J, -2.0 / 3.0)
    bb = bt * jm
    Ie = (bb[0, 0] + bb[1, 1] + bb[2, 2]) / 3.0
    Ee = E[p]
    n = nu[p]
    mu_l = Ee / (2.0 * (1.0 + n))
    kap = Ee / (3.0 * (1.0 - 2.0 * n))
    s = mu_l * (bb - Ie * I)
    ns = wp.sqrt(wp.ddot(s, s) + 1.0e-20)
    r = 0.816496580927726 * sy[p]
    if ns > r:
        s = s * (r / ns)
        bb = s / mu_l + Ie * I
    b_out[p] = bb / jm
    tau = s + (0.5 * kap * (J2 - 1.0)) * I
    m = rhoi[p] * vol[p]                       # inertial mass (= rho unless mass-scaled)
    mg = rho[p] * vol[p]                       # gravitational mass
    dx = 1.0 / inv_dx
    affine = (-dt * vol[p] * 4.0 * inv_dx * inv_dx) * tau + m * Cp
    xp = (x[p] - origin) * inv_dx
    bi = int(xp[0] - 0.5)
    bj = int(xp[1] - 0.5)
    bk = int(xp[2] - 0.5)
    fx = xp - wp.vec3(float(bi), float(bj), float(bk))
    W = qweights(fx)
    mv = m * v[p] + wp.vec3(0.0, 0.0, -dt * grav * mg)     # momentum + gravity impulse
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = nidx(bmap, g, bi + i, bj + j, bk + k)
                if idx < 0:
                    wp.atomic_add(missed, 0, 1)
                else:
                    dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                    w = W[i, 0] * W[j, 1] * W[k, 2]
                    wp.atomic_add(gmv, idx, w * (mv + affine * dpos))
                    wp.atomic_add(gm, idx, w * m)
                    if bk + k <= floor_k:
                        wp.atomic_add(gmu, idx, w * m * muf[p])      # friction only acts on floor nodes


@wp.kernel
def grid_op(gm: wp.array(dtype=float), gmv: wp.array(dtype=wp.vec3), gmu: wp.array(dtype=float),
            bcoord: wp.array(dtype=wp.vec4i), origin: wp.vec3, dx: float, dt: float, grav: float,
            floor_k: int, gp0: wp.array(dtype=wp.vec3), gvel: wp.array(dtype=wp.vec3),
            gom: wp.array(dtype=float), n_grasp: int, t_arr: wp.array(dtype=float), r_grasp: float, damp: float, gv: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    t_in = t_arr[0]
    m = gm[t]
    if m <= 1.0e-12:
        gv[t] = wp.vec3(0.0)
        return
    bl = t // BS3
    l = t - bl * BS3
    bc = bcoord[bl]
    g = bc[0]
    i = bc[1] * BS + l // 16
    j = bc[2] * BS + (l // 4) % 4
    k = bc[3] * BS + l % 4
    vv = gmv[t] / m                                       # gravity is applied in P2G
    vv = vv * damp
    if k <= floor_k:
        vn = vv[2]
        if vn < 0.0:
            mu = gmu[t] / m
            nt = wp.sqrt(vv[0] * vv[0] + vv[1] * vv[1] + 1.0e-20)
            fac = wp.max(0.0, 1.0 + mu * vn / nt)
            vv = wp.vec3(vv[0] * fac, vv[1] * fac, 0.0)
    pos = origin + wp.vec3(float(i), float(j), float(k)) * dx
    for e in range(n_grasp):
        q = gp0[g * n_grasp + e] + gvel[g * n_grasp + e] * t_in
        rel = pos - q
        if wp.length(rel) < r_grasp:
            om = gom[g * n_grasp + e]
            vv = gvel[g * n_grasp + e] + wp.vec3(-om * rel[1], om * rel[0], 0.0)
    gv[t] = vv


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec3), gid: wp.array(dtype=wp.int32), gv: wp.array(dtype=wp.vec3),
        origin: wp.vec3, inv_dx: float, dt: float, bmap: wp.array4d(dtype=wp.int32),
        x_out: wp.array(dtype=wp.vec3), v_out: wp.array(dtype=wp.vec3), C_out: wp.array(dtype=wp.mat33),
        missed: wp.array(dtype=wp.int32)):
    p = wp.tid()
    g = gid[p]
    xp = (x[p] - origin) * inv_dx
    bi = int(xp[0] - 0.5)
    bj = int(xp[1] - 0.5)
    bk = int(xp[2] - 0.5)
    fx = xp - wp.vec3(float(bi), float(bj), float(bk))
    W = qweights(fx)
    nv = wp.vec3(0.0)
    nC = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = nidx(bmap, g, bi + i, bj + j, bk + k)
                if idx < 0:
                    wp.atomic_add(missed, 1, 1)
                else:
                    dpos = wp.vec3(float(i), float(j), float(k)) - fx
                    w = W[i, 0] * W[j, 1] * W[k, 2]
                    gvv = gv[idx]
                    nv = nv + w * gvv
                    nC = nC + (4.0 * inv_dx * w) * wp.outer(gvv, dpos)
    v_out[p] = nv
    C_out[p] = nC
    x_out[p] = x[p] + dt * nv


@wp.kernel
def advance_t(t_arr: wp.array(dtype=float), dt: float):
    t_arr[0] = t_arr[0] + dt


@dataclass
class State:
    x: wp.array
    v: wp.array
    C: wp.array
    b: wp.array

    @staticmethod
    def alloc(n, device, requires_grad=False):
        return State(wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=requires_grad),
                     wp.zeros(n, dtype=wp.mat33, device=device, requires_grad=requires_grad))

    def copy(self, requires_grad=False):
        out = State.alloc(self.x.shape[0], self.x.device, requires_grad)
        for a, b_ in zip((out.x, out.v, out.C, out.b), (self.x, self.v, self.C, self.b)):
            wp.copy(a, b_)
        return out


class MPM:
    """Batched sparse-block MLS-MPM. Coordinates: board plane z = 0; origin is the
    world position of grid node (0, 0, 0) and sits floor_k cells below the board.
    The untaped substeps of a frame run as one captured CUDA graph on fixed buffers
    (particle count fixed per instance; grid capacity grows on demand -> recapture)."""

    def __init__(self, G, n_particles, lo_xy, hi_xy, dx=0.003, nz_blocks=2, floor_k=2, dt=1e-4, r_grasp=None,
                 grav=9.81, damp=1.0, device="cuda", use_graph=True):
        self.G, self.dx, self.inv_dx, self.dt, self.floor_k = G, dx, 1.0 / dx, dt, floor_k
        self.device, self.grav, self.damp, self.P = device, grav, damp, n_particles
        self.r_grasp = 1.5 * dx if r_grasp is None else r_grasp
        lo = np.asarray(lo_xy, float) - 16 * dx
        self.origin = wp.vec3(float(lo[0]), float(lo[1]), -floor_k * dx)
        self.NBX = int(math.ceil((hi_xy[0] - lo[0]) / dx / 4)) + 5
        self.NBY = int(math.ceil((hi_xy[1] - lo[1]) / dx / 4)) + 5
        self.NBZ = nz_blocks
        self.missed = wp.zeros(2, dtype=wp.int32, device=device)
        self.use_graph = use_graph
        self.bmap_t = torch.full((G, self.NBX, self.NBY, self.NBZ), -1, dtype=torch.int32, device=device)
        self.bmap = wp.from_torch(self.bmap_t)
        self.cap = 0
        self.stA, self.stB = State.alloc(n_particles, device), State.alloc(n_particles, device)
        self.gp0 = wp.zeros(G * 2, dtype=wp.vec3, device=device)
        self.gvel = wp.zeros(G * 2, dtype=wp.vec3, device=device)
        self.gom = wp.zeros(G * 2, dtype=float, device=device)
        self.t_arr = wp.zeros(1, dtype=float, device=device)
        self.graphs = {}
        self.mats = None

    def set_gid(self, gid_wp):
        self.gid = gid_wp

    def set_mats(self, mats: dict):
        """material arrays used by the captured graph: copied into fixed buffers"""
        if self.mats is None:
            self.mats = {k: wp.zeros_like(v) for k, v in mats.items() if k != "gid"}
            self.mats["gid"] = self.gid
        for k, v in mats.items():
            if k != "gid":
                wp.copy(self.mats[k], v)
        self.mats_grad = mats

    def _ensure_cap(self, nb):
        if nb > self.cap:
            self.cap = int(nb * 1.3) + 16
            n = self.cap * 64
            self.grid = (wp.zeros(n, dtype=float, device=self.device), wp.zeros(n, dtype=wp.vec3, device=self.device),
                         wp.zeros(n, dtype=float, device=self.device), wp.zeros(n, dtype=wp.vec3, device=self.device))
            self.bcoord_t = torch.zeros(self.cap, 4, dtype=torch.int32, device=self.device)
            self.bcoord = wp.from_torch(self.bcoord_t, dtype=wp.vec4i)
            self.graphs = {}

    # ---------------------------------------------------------------- sparse blocks
    def build_blocks(self, x: torch.Tensor, gid: torch.Tensor, dil: int):
        """mark every block column (bx, by) within `dil` blocks of a particle's block column
        (all z blocks) in the fixed bmap/bcoord buffers. Returns the number of blocks."""
        o = torch.tensor([self.origin[0], self.origin[1]], device=x.device)
        bxy = ((x[:, :2] - o) * self.inv_dx / 4).floor().long()
        bx = bxy[:, 0].clamp(0, self.NBX - 1); by = bxy[:, 1].clamp(0, self.NBY - 1)
        mark = torch.zeros(self.G, self.NBX, self.NBY, device=x.device)
        mark[gid.long(), bx, by] = 1.0
        if dil > 0:
            mark = torch.nn.functional.max_pool2d(mark[:, None], 2 * dil + 1, 1, dil)[:, 0]
        act = mark.nonzero()
        A = act.shape[0]
        nb = A * self.NBZ
        self._ensure_cap(nb)
        self.bmap_t.fill_(-1)
        self.bcoord_t.zero_()
        for z in range(self.NBZ):
            ids = torch.arange(A, device=x.device) * self.NBZ + z
            self.bmap_t[act[:, 0], act[:, 1], act[:, 2], z] = ids.int()
            self.bcoord_t[ids, 0] = act[:, 0].int(); self.bcoord_t[ids, 1] = act[:, 1].int()
            self.bcoord_t[ids, 2] = act[:, 2].int(); self.bcoord_t[ids, 3] = z
        return nb

    # ---------------------------------------------------------------- one substep
    def substep(self, st: State, out: State, mats: dict, grid, n_nodes, t_arr):
        gm, gmv, gmu, gv = grid
        wp.launch(p2g, dim=self.P, inputs=[st.x, st.v, st.C, st.b, mats["gid"], mats["E"], mats["nu"], mats["rho"],
                                          mats["rhoi"], mats["mu"], mats["sy"], mats["vol"], self.grav, self.origin, self.inv_dx, self.dt,
                                          self.bmap, self.floor_k, gm, gmv, gmu, out.b, self.missed], device=self.device)
        wp.launch(grid_op, dim=n_nodes, inputs=[gm, gmv, gmu, self.bcoord, self.origin, self.dx, self.dt, self.grav,
                                               self.floor_k, self.gp0, self.gvel, self.gom, 2, t_arr, self.r_grasp, self.damp,
                                               gv], device=self.device)
        wp.launch(g2p, dim=self.P, inputs=[st.x, mats["gid"], gv, self.origin, self.inv_dx, self.dt, self.bmap,
                                          out.x, out.v, out.C, self.missed], device=self.device)

    def _untaped(self, n):
        """n substeps stA -> (stA if n even else stB) on the fixed buffers"""
        cur, nxt = self.stA, self.stB
        for _ in range(n):
            for arr in self.grid:
                arr.zero_()
            self.substep(cur, nxt, self.mats, self.grid, self.cap * 64, self.t_arr)
            wp.launch(advance_t, dim=1, inputs=[self.t_arr, self.dt], device=self.device)
            cur, nxt = nxt, cur

    def _set_drive(self, gp0, gvel, gom):
        wp.copy(self.gp0, wp.from_torch(gp0.contiguous(), dtype=wp.vec3))
        wp.copy(self.gvel, wp.from_torch(gvel.contiguous(), dtype=wp.vec3))
        wp.copy(self.gom, wp.from_torch(gom.float().contiguous()))

    def _run_graph(self, n):
        if not self.use_graph:
            self._untaped(n); return
        key = (n, float(self.dt))                             # dt is baked into the captured launches
        if key not in self.graphs:
            t_keep = wp.clone(self.t_arr)
            wp.capture_begin(device=self.device)
            try:
                self._untaped(n)
            finally:
                self.graphs[key] = wp.capture_end(device=self.device)
            wp.copy(self.t_arr, t_keep)                       # the capture did not execute
        wp.capture_launch(self.graphs[key])

    def frame(self, st: State, gp0: torch.Tensor, gvel: torch.Tensor, gom: torch.Tensor, frame_s, dil, W=0,
              ckpt_every=0):
        """advance one frame. st is copied into the fixed buffer stA (if it is not stA);
        grasp points move rigidly: centre gp0 + gvel * t ((G*2, 3) torch), yaw rate gom (G*2,) rad/s.
        W > 0: the last W substeps are recorded on a wp.Tape with fresh arrays (the paper's
        truncated gradient, Sec. 4.1) -> returns (state, tape).
        ckpt_every = K > 0 (even, divides n_sub): run the frame in K-substep segments and keep
        the segment-start states -> returns (state, record) for frame_backward (full-frame
        gradient by recomputation). Otherwise returns (state, None).
        The returned state is a fixed buffer (copy it if you keep it across frames)."""
        n_sub = int(round(frame_s / self.dt))
        W = min(W, n_sub)
        if st is not self.stA:
            for a, b_ in zip((self.stA.x, self.stA.v, self.stA.C, self.stA.b), (st.x, st.v, st.C, st.b)):
                wp.copy(a, b_)
        nb = self.build_blocks(wp.to_torch(self.stA.x), wp.to_torch(self.gid), dil)
        self._set_drive(gp0, gvel, gom)
        self.t_arr.zero_()
        if ckpt_every:
            K = ckpt_every
            assert n_sub % K == 0 and K % 2 == 0, (n_sub, K)
            ck = []
            for sgm in range(n_sub // K):
                ck.append(tuple(wp.to_torch(a).clone() for a in (self.stA.x, self.stA.v, self.stA.C, self.stA.b)))
                self._run_graph(K)
            rec = {"ckpts": ck, "bmap": self.bmap_t.clone(), "bcoord": self.bcoord_t[:nb].clone(), "nb": nb,
                   "drive": (gp0.clone(), gvel.clone(), gom.clone()), "K": K, "n_sub": n_sub}
            return self.stA, rec
        n_un = n_sub - W
        if n_un > 0:
            self._run_graph(n_un)
        cur = self.stA if n_un % 2 == 0 else self.stB
        tape = None
        if W > 0:
            tape = wp.Tape()
            with tape:
                for k in range(n_un, n_sub):
                    out = State.alloc(self.P, self.device, requires_grad=True)
                    g2 = self.alloc_grid(nb, True)
                    ta = wp.full(1, float(k * self.dt), dtype=float, device=self.device)
                    self.substep(cur, out, self.mats_grad, g2, nb * 64, ta)
                    cur = out
        return cur, tape

    def alloc_grid(self, nb, requires_grad=False):
        n = nb * 64
        return (wp.zeros(n, dtype=float, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=float, device=self.device, requires_grad=requires_grad),
                wp.zeros(n, dtype=wp.vec3, device=self.device, requires_grad=requires_grad))

    def frame_backward(self, rec, lam):
        """adjoint of one recorded frame. lam = (gx, gv, gC, gb) torch adjoints of the frame's
        final state (None entries = 0); the material gradients accumulate into
        self.mats_grad[k].grad. Returns the adjoint of the frame's start state."""
        nb, K = rec["nb"], rec["K"]
        self.bmap_t.copy_(rec["bmap"]); self.bcoord_t.zero_(); self.bcoord_t[:nb] = rec["bcoord"]
        self._set_drive(*rec["drive"])
        dts = (wp.vec3, wp.vec3, wp.mat33, wp.mat33)
        lam = list(lam)
        for sgm in reversed(range(len(rec["ckpts"]))):
            inp = State.alloc(self.P, self.device, requires_grad=True)
            for a, c, dt_ in zip((inp.x, inp.v, inp.C, inp.b), rec["ckpts"][sgm], dts):
                wp.copy(a, wp.from_torch(c.contiguous(), dtype=dt_))
            tape = wp.Tape()
            cur = inp
            with tape:
                for k in range(K):
                    out = State.alloc(self.P, self.device, requires_grad=True)
                    g2 = self.alloc_grid(nb, True)
                    ta = wp.full(1, float((sgm * K + k) * self.dt), dtype=float, device=self.device)
                    self.substep(cur, out, self.mats_grad, g2, nb * 64, ta)
                    cur = out
            grads = {}
            for a, l_, dt_ in zip((cur.x, cur.v, cur.C, cur.b), lam, dts):
                if l_ is not None:
                    grads[a] = wp.from_torch(l_.contiguous(), dtype=dt_)
            tape.backward(grads=grads)
            lam = [wp.to_torch(a.grad).clone() for a in (inp.x, inp.v, inp.C, inp.b)]
            del tape                       # no tape.reset(): it would zero the material gradients too
        return lam
