"""MPM digital twin of the tracked cables of one clip (PhysWorld Sec. 3.1 / App. B.2).

Parameterisation (all per particle in the end; the global stage ties them):
    E   = exp(lE_g + dlE_p)        Young's modulus [Pa]
    rho = exp(lrho_g + dlrho_p)    density [kg/m^3]
    mu  = exp(lmu_g + dlmu_p)      board friction coefficient
    nu  = nu_g                     Poisson ratio (global only, as in the paper's local stage)
    sy  = exp(lsy_g)               von Mises yield stress [Pa] (global only; +inf = elastic)
Global stage: lE_g, lrho_g, lmu_g, nu_g, lsy_g; local stage: dlE_p, dlrho_p, dlmu_p.
Loss per frame (App. B.2): Chamfer (both directions) between the simulated
centreline cloud and the observed one + 0.1 * smooth-L1 on the tracked nodes, in
metres, xy only (the video gives the board-plane track, no depth).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from .cable import SceneParticles, build_cable, densify_track
from .mpm import MPM, State

BOUNDS = {"lE": (math.log(1e3), math.log(3e6)), "lrho": (math.log(200.0), math.log(8000.0)),
          "lmu": (math.log(0.01), math.log(5.0)), "nu": (0.0, 0.45), "lsy": (math.log(1.0), math.log(1e7))}
RHO_PHYS = 0.044 / (math.pi * 0.003 ** 2)          # 1556 kg/m^3: measured 0.044 kg/m over a 6 mm cylinder


def smooth_l1(d):
    a = d.abs()
    return torch.where(a < 1.0, 0.5 * a * a, a - 0.5)


def chamfer2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()


MS_CFL = 0.5


def rho_for_dt(E, nu, dx, dt, cfl):
    """smallest (inertial) density for which the P-wave CFL number at time step dt is <= cfl"""
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return M * (dt / (cfl * dx)) ** 2


def max_stable_dt(E, rho, nu, dx, cfl=0.4):
    M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return cfl * dx / math.sqrt(M / rho)


def drive_from_nodes(E4: torch.Tensor):
    """E4 (4, T, 4, 2): per cable of ONE scene the nodes [n0, n1, n14, n15] per frame ->
    (ends (4, T, 2, 2) = n0, n15; yaw (4, T, 2) unwrapped). The hub is one rigid module:
    its yaw = direction of the mean (n1 - n0) over the 4 cables (as the MuJoCo/Newton
    mover drive, wire_model/yang/speed_warp.mover_poses); each plug's yaw = (n15 - n14)."""
    out = (E4[:, :, 1] - E4[:, :, 0]).mean(0)
    hy = torch.atan2(out[:, 1], out[:, 0])
    v = E4[:, :, 3] - E4[:, :, 2]
    py = torch.atan2(v[..., 1], v[..., 0])
    def unw(a):
        d = torch.remainder(torch.diff(a, dim=-1) + math.pi, 2 * math.pi) - math.pi
        return torch.cat([a[..., :1], a[..., :1] + torch.cumsum(d, -1)], -1)
    hy = unw(hy); py = unw(py)
    yaw = torch.stack([hy.expand_as(py), py], -1)
    return E4[:, :, [0, 3]], yaw


class Twin:
    def __init__(self, P: np.ndarray, cables=(0, 1, 2, 3), dx=0.003, dt=1e-4, frame_s=0.25, device="cuda",
                 damp=1.0, r_grasp=None, mass_scale_dt=None):
        """P (C, T, 16, 2) tracked nodes (metres) of ONE clip"""
        self.P = np.asarray(P, np.float32)[list(cables)]
        self.G, self.T = self.P.shape[0], self.P.shape[1]
        self.dx, self.dt, self.frame_s, self.device = dx, dt, frame_s, device
        self.ms_dt = mass_scale_dt      # quasi-static mass scaling to this time step (None = off)
        self.cab = [build_cable(self.P[g, 0], dx) for g in range(self.G)]
        self.sp = SceneParticles(self.cab, device)
        lo = self.P.reshape(-1, 2).min(0) - 0.1; hi = self.P.reshape(-1, 2).max(0) + 0.1
        self.NP = self.sp.pos0.shape[0]
        torch.cuda.set_device(torch.device(device))
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream(device)))   # torch and warp on one stream
        self.mpm = MPM(self.G, self.NP, lo, hi, dx=dx, dt=dt, device=device, damp=damp, r_grasp=r_grasp)
        self.gid_wp = wp.from_torch(self.sp.gid)
        self.mpm.set_gid(self.gid_wp)
        self.zc = self.cab[0]["zc"]
        self.obs_dense = [[torch.tensor(densify_track(self.P[g, t]), dtype=torch.float32, device=device) for t in range(self.T)]
                          for g in range(self.G)]
        self.Ptorch = torch.tensor(self.P, device=device)
        self.drive = drive_from_nodes(torch.tensor(np.asarray(P, np.float32), device=device)[..., [0, 1, 14, 15], :])
        self.drive = (self.drive[0][list(cables)], self.drive[1][list(cables)])
        self.NP = self.sp.pos0.shape[0]

    # ------------------------------------------------------------------ state / materials
    def init_state(self) -> State:
        st = State.alloc(self.NP, self.device)
        wp.copy(st.x, wp.from_torch(self.sp.pos0.contiguous(), dtype=wp.vec3))
        b = torch.eye(3, device=self.device).repeat(self.NP, 1, 1).contiguous()
        wp.copy(st.b, wp.from_torch(b, dtype=wp.mat33))
        return st

    def per_particle(self, glob: dict, loc: dict | None):
        """torch tensors (P,) of E, rho, mu, nu, sy"""
        one = torch.ones(self.NP, device=self.device)
        lE = glob["lE"] * one; lrho = glob["lrho"] * one; lmu = glob["lmu"] * one
        if loc is not None:
            lE = lE + loc["dlE"]; lrho = lrho + loc["dlrho"]; lmu = lmu + loc["dlmu"]
        sy = torch.exp(glob["lsy"]) * one if glob.get("elastic", None) is None else torch.full_like(one, 1e30)
        E, rho, nu = torch.exp(lE), torch.exp(lrho), glob["nu"] * one
        out = {"E": E, "rho": rho, "mu": torch.exp(lmu), "nu": nu, "sy": sy, "rhoi": rho}
        if self.ms_dt is not None:
            out["rhoi"] = torch.maximum(rho, rho_for_dt(E, nu, self.dx, self.ms_dt, MS_CFL))
        return out

    def to_warp(self, pp: dict, requires_grad=False):
        m = {k: wp.from_torch(v.detach().float().contiguous().clone(), requires_grad=requires_grad) for k, v in pp.items()}
        m["vol"] = wp.from_torch(self.sp.vol.contiguous())
        m["gid"] = self.gid_wp
        return m

    # ------------------------------------------------------------------ rollout
    def ends(self, t, ends=None):
        """(G, 2, 3) grasp points at frame t from the measured (or given (G, T, 2, 2)) end nodes"""
        E = self.drive[0] if ends is None else ends[0]
        e = E[:, t]
        return torch.cat([e, torch.full_like(e[..., :1], self.zc)], -1)

    def frame_loss(self, x: torch.Tensor, t: int, target=None):
        """paper loss (App. B.2) at frame t from particle positions x (P, 3)"""
        sc = self.sp.slice_centroids(x)
        nodes = self.sp.nodes(x)[..., :2]
        tgt = self.Ptorch[:, t] if target is None else target
        cd = 0.0
        for g in range(self.G):
            cd = cd + chamfer2(sc[self.sp.slice_gid == g, :2], self.obs_dense[g][t])
        cd = cd / self.G
        tr = smooth_l1(nodes - tgt).mean()
        return cd + 0.1 * tr, nodes

    def rollout(self, mats, T=None, ends=None, grad=None, loss_cb=None, K=100, W=10, frame_s=None,
                return_x=False, acc=None, st=None):
        """simulate frames 0..T-1 driving both ends of every cable along the measured
        (or given: (ends (G, T, 2, 2), yaw (G, T, 2))) end paths. Returns node positions
        (T, G, 16, 3) and info. Gradients (loss_cb(t, x (P, 3)) -> dL_t/dx or None):
          grad="trunc": through the last W substeps of each frame only (paper Sec. 4.1)
          grad="frame": through all substeps of each frame, frame-start state detached
          grad="full" : through the whole rollout (adjoint carried across frames)
        material gradients are added to acc[k] (torch (P,)) for k in acc."""
        acc = {} if acc is None else acc
        T = self.T if T is None else T
        fs = self.frame_s if frame_s is None else frame_s
        st = self.init_state() if st is None else st
        self.mpm.set_mats(mats)
        out = [self.sp.nodes(wp.to_torch(st.x)).detach().clone()]
        xs = [wp.to_torch(st.x).clone()] if return_x else None
        self.mpm.missed.zero_()
        bad = False
        recs = []
        for t in range(T - 1):
            p0 = self.ends(t, ends); p1 = self.ends(t + 1, ends)
            vel = (p1 - p0) / fs
            Y = self.drive[1] if ends is None else ends[1]
            om = (Y[:, t + 1] - Y[:, t]) / fs
            dmax = (p1 - p0).norm(dim=-1).max().item()
            dil = int(math.ceil((dmax + 3 * self.dx) / (4 * self.dx))) + 1
            st, rec = self.mpm.frame(st, p0.reshape(-1, 3), vel.reshape(-1, 3), om.reshape(-1), fs, dil,
                                     W=W if grad == "trunc" else 0,
                                     ckpt_every=K if grad in ("frame", "full") else 0)
            x = wp.to_torch(st.x)
            if not torch.isfinite(x).all() or x[:, 2].abs().max() > 0.5:
                bad = True
                break
            if grad is not None and loss_cb is not None:
                gx = loss_cb(t + 1, x)
                if grad == "trunc" and gx is not None:
                    rec.backward(grads={st.x: wp.from_torch(gx.contiguous(), dtype=wp.vec3)})
                    self._collect(mats, acc); rec.reset()
                elif grad == "frame" and gx is not None:
                    self.mpm.frame_backward(rec, (gx, None, None, None))
                    self._collect(mats, acc)
                elif grad == "full":
                    recs.append((rec, gx))
            out.append(self.sp.nodes(x).detach().clone())
            if return_x:
                xs.append(x.clone())
        if grad == "full" and not bad:
            lam = [None, None, None, None]
            for rec, gx in reversed(recs):
                if gx is not None:
                    lam[0] = gx if lam[0] is None else lam[0] + gx
                lam = self.mpm.frame_backward(rec, lam)
            self._collect(mats, acc)
        missed = self.mpm.missed.numpy().tolist()
        return torch.stack(out), {"missed": missed, "bad": bad, "xs": xs, "state": st}

    def _collect(self, mats, acc):
        for k in acc:
            acc[k] += wp.to_torch(mats[k].grad)
            mats[k].grad.zero_()
