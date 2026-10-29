"""The paper's state: s_t = [p0, q^1, ..., q^{L-1}] -- the first node's
position and one unit quaternion per link (Sec. IV-A). With a fixed link
length every node position follows by forward kinematics, so a decoded
state can never stretch or shrink the cable.

Convention used here (the paper leaves the first link's frame implicit):
  q^1       rotation taking the reference axis e_x to link 1 (world frame)
  q^i, i>1  rotation from link i-1 to link i, expressed in link i-1's frame
  R_1 = R(q^1),  R_i = R_{i-1} R(q^i),  p_i = p_{i-1} + ell * R_i e_x
Rotations are the shortest arc between consecutive link directions (a curve
sampled by positions carries no twist). Planar data (x, y) is lifted to
z = 0, where every quaternion is a rotation about z -- the general 3-D code
path is kept so Newton's 3-D output can use the same model later.

Quaternions are (w, x, y, z), normalised, and sign-canonicalised to w >= 0 so
the double cover never produces a +q / -q discontinuity in the targets.
"""
from __future__ import annotations

import torch
from torch import Tensor


def q_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def q_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def q_conj(q: Tensor) -> Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def q_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return q_mul(q_mul(q, qv), q_conj(q))[..., 1:]


def q_from_two_vectors(a: Tensor, b: Tensor, eps: float = 1e-9) -> Tensor:
    """Shortest-arc rotation taking unit vector a to unit vector b."""
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    w = 1.0 + (a * b).sum(-1, keepdim=True)
    xyz = torch.cross(a, b, dim=-1)
    # antiparallel: w ~ 0 and no cross product; rotate 180 deg about any
    # axis perpendicular to a (pick one built from the smaller components)
    anti = w.squeeze(-1) < 1e-6
    if bool(anti.any()):
        alt = torch.stack([-a[..., 1], a[..., 0], torch.zeros_like(a[..., 0])], -1)
        alt2 = torch.stack([torch.zeros_like(a[..., 0]), -a[..., 2], a[..., 1]], -1)
        use_alt = alt.norm(dim=-1, keepdim=True) > 1e-6
        axis = torch.where(use_alt, alt, alt2)
        xyz = torch.where(anti.unsqueeze(-1), axis, xyz)
        w = torch.where(anti.unsqueeze(-1), torch.zeros_like(w), w)
    return q_normalize(torch.cat([w, xyz], dim=-1))


def state_dim(n_nodes: int) -> int:
    return 3 + 4 * (n_nodes - 1)


def positions_to_chain(P: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """P (..., L, 2|3) node positions -> (state (..., 3 + 4(L-1)), link length (...),
    per-link lengths (..., L-1)). The link length is the mean segment length;
    the per-link lengths are returned so the caller can measure what the
    fixed-length assumption costs on its own data."""
    if P.shape[-1] == 2:
        P = torch.cat([P, torch.zeros_like(P[..., :1])], dim=-1)
    d = P[..., 1:, :] - P[..., :-1, :]                       # (..., L-1, 3)
    seg = d.norm(dim=-1)
    ell = seg.mean(dim=-1)
    u = d / seg.clamp_min(1e-9).unsqueeze(-1)
    ex = torch.zeros_like(u[..., 0, :]); ex[..., 0] = 1.0
    quats = []
    R = q_from_two_vectors(ex, u[..., 0, :])                  # q^1: e_x -> link 1
    quats.append(R)
    for i in range(1, u.shape[-2]):
        local = q_rotate(q_conj(R), u[..., i, :])             # link i in frame i-1
        qi = q_from_two_vectors(ex, local)
        quats.append(qi)
        R = q_normalize(q_mul(R, qi))
    q = torch.stack(quats, dim=-2)                            # (..., L-1, 4)
    s = torch.cat([P[..., 0, :], q.flatten(-2)], dim=-1)
    return s, ell, seg


def chain_to_positions(s: Tensor, ell: Tensor, n_nodes: int) -> Tensor:
    """Forward kinematics: state (..., 3 + 4(L-1)) and link length -> (..., L, 3).
    ell is one length per chain (...) as in the paper, or one per link
    (..., L-1): the tracker resamples the routes only approximately
    equidistantly, and the per-link lengths of a cable's first frame cut the
    representation floor from ~3 mm to well under 1 mm on the real clips."""
    p0 = s[..., :3]
    q = q_normalize(s[..., 3:].reshape(*s.shape[:-1], n_nodes - 1, 4))
    ex = torch.zeros_like(p0); ex[..., 0] = 1.0
    per_link = ell.ndim == s.ndim and ell.shape[-1] == n_nodes - 1
    L = (lambda i: ell[..., i]) if per_link else (lambda i: ell)
    pts = [p0]
    R = q[..., 0, :]
    p = p0 + L(0).unsqueeze(-1) * q_rotate(R, ex)
    pts.append(p)
    for i in range(1, n_nodes - 1):
        R = q_normalize(q_mul(R, q[..., i, :]))
        p = p + L(i).unsqueeze(-1) * q_rotate(R, ex)
        pts.append(p)
    return torch.stack(pts, dim=-2)


def normalize_state(s: Tensor, n_nodes: int) -> Tensor:
    """Project a decoded state back onto the manifold (unit quaternions, w >= 0)."""
    q = q_normalize(s[..., 3:].reshape(*s.shape[:-1], n_nodes - 1, 4))
    return torch.cat([s[..., :3], q.flatten(-2)], dim=-1)
