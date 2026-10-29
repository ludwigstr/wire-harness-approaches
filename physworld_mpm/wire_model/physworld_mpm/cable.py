"""Volumetric MPM particles for the tracked cables (PhysWorld Sec. 3.1 "fill interior voids").

The 16 tracked nodes of a cable at frame 0 define its centreline (centripetal
Catmull-Rom spline through the nodes). The cable is a solid cylinder of
DIAMETER 6 mm (radius 3 mm, the radius every engine in this project uses;
0.044 kg/m measured mass -> 1556 kg/m^3), lying on the board (z = 0). Slices every
dp = dx/2 along the arc, each slice a disk lattice (spacing dp) in the plane
normal to the tangent -> 12 particles per slice at dx = 3 mm. Particle volume =
pi r^2 ds / n_slice (exact volume). Node j <-> the 3 slices nearest to its arc
length (end nodes: the 2 end slices); its position = mean of their particles."""
from __future__ import annotations

import numpy as np
import torch

RADIUS = 0.003


def catmull_rom(P: np.ndarray, n_per_seg: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """centripetal Catmull-Rom through P (N, 2) -> dense points (M, 2), knot index of each node (N,)"""
    P = np.asarray(P, float)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, knots = [], []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-9
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-9
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-9
        t = np.linspace(t1, t2, n_per_seg, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        knots.append(i * n_per_seg)
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    D = np.vstack(out + [P[-1:]])
    knots.append(len(D) - 1)
    return D, np.array(knots)


def resample(D: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dense curve -> points every ds of arc length, their arc length, cumulative arc of D"""
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = int(np.floor(s[-1] / ds)) + 1
    q = np.linspace(0, s[-1], n)
    x = np.interp(q, s, D[:, 0]); y = np.interp(q, s, D[:, 1])
    return np.stack([x, y], 1), q, s


def densify_track(nodes: np.ndarray, ds: float = 0.003) -> np.ndarray:
    """observed 'point cloud' of a tracked cable: the spline through its 16 nodes every ds"""
    D, _ = catmull_rom(nodes, 50)
    return resample(D, ds)[0]


def build_cable(nodes0: np.ndarray, dx: float = 0.003, radius: float = RADIUS):
    """nodes0 (16, 2) frame-0 track -> dict with particle positions (P, 3), slice id (P,),
    arc length (P,), volume (P,), node->slice map, z of the centreline"""
    dp = dx / 2
    D, knots = catmull_rom(nodes0, 200)
    C, q, s_dense = resample(D, dp)
    s_nodes = s_dense[knots]
    tang = np.gradient(C, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    k = np.arange(-int(np.ceil(radius / dp)), int(np.ceil(radius / dp))) + 0.5
    off = [(a * dp, b * dp) for a in k for b in k if (a * dp) ** 2 + (b * dp) ** 2 <= radius ** 2 + 1e-12]
    off = np.array(off)                                                     # (n_sec, 2): (lateral, vertical)
    zc = radius
    X = (C[:, None, :] + off[None, :, 0:1] * nrm[:, None, :])               # (S, n_sec, 2)
    Z = zc + np.broadcast_to(off[None, :, 1], X.shape[:2])
    pos = np.concatenate([X, Z[..., None]], -1).reshape(-1, 3)
    S = len(C)
    slice_id = np.repeat(np.arange(S), len(off))
    ds_eff = q[-1] / (S - 1)
    vol = np.full(len(pos), np.pi * radius ** 2 * ds_eff / len(off))
    node_slices = []
    for j, sj in enumerate(s_nodes):
        i = int(np.argmin(np.abs(q - sj)))
        if j == 0:
            sl = [0, 1]
        elif j == len(s_nodes) - 1:
            sl = [S - 2, S - 1]
        else:
            sl = [i - 1, i, i + 1]
        node_slices.append(sl)
    return {"pos": pos.astype(np.float32), "slice": slice_id, "arc": q[slice_id], "vol": vol.astype(np.float32),
            "n_slices": S, "n_sec": len(off), "node_slices": node_slices, "s_nodes": s_nodes, "zc": zc,
            "length": float(q[-1])}


class SceneParticles:
    """several cables (one grid each) concatenated into flat particle arrays"""

    def __init__(self, cables: list[dict], device):
        self.cables = cables
        self.G = len(cables)
        pos, gid, sl, arc, vol = [], [], [], [], []
        self.offsets = [0]; self.slice_off = [0]
        for g, c in enumerate(cables):
            pos.append(c["pos"]); gid.append(np.full(len(c["pos"]), g)); sl.append(c["slice"] + self.slice_off[-1])
            arc.append(c["arc"] / c["length"]); vol.append(c["vol"])
            self.offsets.append(self.offsets[-1] + len(c["pos"])); self.slice_off.append(self.slice_off[-1] + c["n_slices"])
        # slice-major particle order (spatially coherent; a random order made P2G 2-5x slower)
        perm = np.arange(self.offsets[-1])
        self.perm = perm
        self.pos0 = torch.tensor(np.concatenate(pos)[perm], device=device)
        self.gid = torch.tensor(np.concatenate(gid)[perm], dtype=torch.int32, device=device)
        self.slice = torch.tensor(np.concatenate(sl)[perm], dtype=torch.long, device=device)
        self.arc = torch.tensor(np.concatenate(arc)[perm], dtype=torch.float32, device=device)
        self.vol = torch.tensor(np.concatenate(vol)[perm], device=device)
        self.n_slices = self.slice_off[-1]
        cnt = torch.zeros(self.n_slices, device=device).index_add_(0, self.slice, torch.ones_like(self.vol))
        self.slice_cnt = cnt
        # node -> slices (G, 16, up to 3), padded with -1
        ns = torch.full((self.G, 16, 3), -1, dtype=torch.long)
        for g, c in enumerate(cables):
            for j, sls in enumerate(c["node_slices"]):
                for q, s_ in enumerate(sls):
                    ns[g, j, q] = s_ + self.slice_off[g]
        self.node_slices = ns.to(device)
        self.slice_gid = torch.tensor(np.concatenate([np.full(c["n_slices"], g) for g, c in enumerate(cables)]),
                                      device=device)

    def slice_centroids(self, x: torch.Tensor) -> torch.Tensor:
        """x (P, 3) -> (n_slices, 3)"""
        s = torch.zeros(self.n_slices, 3, device=x.device, dtype=x.dtype).index_add_(0, self.slice, x)
        return s / self.slice_cnt[:, None]

    def nodes(self, x: torch.Tensor) -> torch.Tensor:
        """x (P, 3) -> (G, 16, 3) node positions"""
        sc = self.slice_centroids(x)
        idx = self.node_slices.clamp_min(0)
        m = (self.node_slices >= 0).float()[..., None]
        return (sc[idx] * m).sum(2) / m.sum(2)
