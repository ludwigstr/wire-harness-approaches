"""Polyline geometry helpers shared by the simulator and the perception stack.

The central invariant: a DLO configuration is a polyline whose N-1 edges all
have length l0 = length / (N-1). Violating it silently corrupts results --
a configuration built with the wrong edge lengths starts the chain under
spurious axial load, and any edge that is kinematically driven (the grasped
edge) keeps its wrong length for the whole rollout, because no spring force
can act on a node whose position is imposed.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def edge_lengths(p: Tensor) -> Tensor:
    """(B, N, 2) -> (B, N-1) length of each edge."""
    return (p[:, 1:] - p[:, :-1]).norm(dim=-1)


def arc_length(p: Tensor) -> Tensor:
    """(B, N, 2) -> (B,) total polyline length."""
    return edge_lengths(p).sum(-1)


def resample_polyline(p: Tensor, n_out: int) -> Tensor:
    """Resample a polyline to `n_out` points equally spaced along arc length.

    p: (B, M, 2) -> (B, n_out, 2). Endpoints are preserved exactly. This is how
    a perceived centerline of arbitrary point count becomes a valid N-node DLO
    configuration.
    """
    batch = p.shape[0]
    seg = (p[:, 1:] - p[:, :-1]).norm(dim=-1)                  # (B, M-1)
    cum = torch.cat([torch.zeros(batch, 1, device=p.device, dtype=p.dtype),
                     seg.cumsum(-1)], dim=-1)                  # (B, M)
    total = cum[:, -1:].clamp_min(1e-12)
    want = torch.linspace(0, 1, n_out, device=p.device, dtype=p.dtype)
    want = want.unsqueeze(0) * total                           # (B, n_out)

    out = torch.empty(batch, n_out, 2, device=p.device, dtype=p.dtype)
    for b in range(batch):
        idx = torch.searchsorted(cum[b], want[b].contiguous()).clamp(1, p.shape[1] - 1)
        lo, hi = idx - 1, idx
        span = (cum[b][hi] - cum[b][lo]).clamp_min(1e-12)
        t = ((want[b] - cum[b][lo]) / span).unsqueeze(-1)
        out[b] = p[b][lo] + t * (p[b][hi] - p[b][lo])
    return out


def _arc_from_turn(n: int, l0: float, phi: Tensor) -> Tensor:
    """Constant-turning-angle chain: (B,) turn angles -> (B, n, 2) polyline.

    Every edge has length exactly l0 by construction, for any phi.
    """
    k = torch.arange(n - 1, device=phi.device, dtype=phi.dtype)
    ang = (k.unsqueeze(0) - (n - 2) / 2.0) * phi.unsqueeze(-1)     # (B, n-1)
    e = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * l0     # (B, n-1, 2)
    zero = torch.zeros(phi.shape[0], 1, 2, device=phi.device, dtype=phi.dtype)
    return torch.cat([zero, e.cumsum(1)], 1)                       # (B, n, 2)


def polyline_from_angles(theta: Tensor, l0: float,
                         heading0: Tensor | float = 0.0) -> Tensor:
    """Turning angles -> polyline. Inverse of `DLOSimulator.turning_angles`.

    theta: (B, N-2) turning angle at each interior node.
    Returns (B, N, 2) starting at the origin, all edges exactly l0.

    This realises the shape a wire *wants* to take: feed it a rest-angle
    profile and you get that wire's natural, force-free shape.
    """
    batch, n_int = theta.shape
    dev, dt = theta.device, theta.dtype
    h0 = torch.as_tensor(heading0, device=dev, dtype=dt).expand(batch)
    # heading of edge j = h0 + sum of the turns at nodes 1..j
    headings = torch.cat([h0.unsqueeze(-1),
                          h0.unsqueeze(-1) + theta.cumsum(-1)], -1)  # (B, N-1)
    e = torch.stack([torch.cos(headings), torch.sin(headings)], -1) * l0
    zero = torch.zeros(batch, 1, 2, device=dev, dtype=dt)
    return torch.cat([zero, e.cumsum(1)], 1)


def random_rest_angle(
    batch: int,
    n_nodes: int,
    amp: tuple[float, float] = (0.0, 0.35),
    n_modes: int = 5,
    p_straight: float = 0.15,
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Smooth random natural-curvature profiles, (B, N-2).

    Built as a random Fourier series along the wire's arc length, with a random
    spectral decay per sample. This gives a family running from near-constant
    curvature (a wire that was coiled at one radius) to multi-bend profiles (a
    wire routed through several turns) -- rather than white noise, which would
    describe a wire kinked randomly at every node, which is not how a bundle
    acquires its set.

    `amp` bounds |theta_rest| per node. The upper default of 0.35 rad is not
    arbitrary: a node turns l0/R for a natural radius R, so with l0 = L/(N-1)
    ~ 3 cm, 0.35 rad corresponds to R ~ 8.5 cm -- about as tight as a bundle
    plausibly gets. It also keeps a *constant* profile's total turn,
    (N-2)*theta, near one full loop rather than coiling the wire onto itself.

    `p_straight` keeps a fraction of samples at exactly zero, so the paper's
    original straight-rest model stays inside the training distribution.
    """
    n_int = n_nodes - 2
    s = torch.linspace(0, 1, n_int, device=device, dtype=dtype)
    k = torch.arange(1, n_modes + 1, device=device, dtype=dtype)

    beta = 0.5 + 1.5 * torch.rand(batch, 1, generator=generator,
                                  device=device, dtype=dtype)
    coef = torch.randn(batch, n_modes, generator=generator, device=device,
                       dtype=dtype) * k.unsqueeze(0).pow(-beta)
    phase = 2 * torch.pi * torch.rand(batch, n_modes, generator=generator,
                                      device=device, dtype=dtype)

    basis = torch.sin(torch.pi * k.view(1, -1, 1) * s.view(1, 1, -1)
                      + phase.unsqueeze(-1))                    # (B, K, n_int)
    theta = (coef.unsqueeze(-1) * basis).sum(1)                 # (B, n_int)

    peak = theta.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    lo, hi = amp
    a = lo + (hi - lo) * torch.rand(batch, 1, generator=generator,
                                    device=device, dtype=dtype)
    theta = theta / peak * a

    straight = torch.rand(batch, 1, generator=generator, device=device,
                          dtype=dtype) < p_straight
    return torch.where(straight, torch.zeros_like(theta), theta)


def discrete_arc_batch(
    n: int,
    l0: float,
    chords: Tensor,
    iters: int = 60,
) -> Tensor:
    """Batched: polylines of n nodes, every edge exactly l0, given end-to-end.

    Bisects the turn angle for all B samples simultaneously. The per-sample
    python loop this replaces was the dominant cost of dataset generation --
    it issued O(B * iters) tiny GPU kernels and swamped the actual simulation.

    chords: (B,) desired end-to-end distances, each < (n-1)*l0.
    Returns (B, n, 2) starting at the origin, symmetric about the x-axis.
    """
    span = (n - 1) * l0
    if bool((chords > span).any()):
        bad = chords[chords > span][0].item()
        raise ValueError(f"chord {bad} exceeds the fully-straight span {span}")

    dev, dt = chords.device, chords.dtype
    lo = torch.zeros_like(chords)
    hi = torch.full_like(chords, 2.0 * torch.pi / max(n - 1, 1) * 0.999)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = _arc_from_turn(n, l0, mid)
        c = (p[:, -1] - p[:, 0]).norm(dim=-1)          # (B,)
        # phi=0 -> straight (chord = span); larger phi curls it up
        too_long = c > chords
        lo = torch.where(too_long, mid, lo)
        hi = torch.where(too_long, hi, mid)
    p = _arc_from_turn(n, l0, 0.5 * (lo + hi))
    return p - p[:, :1]


def discrete_arc(
    n: int,
    l0: float,
    chord: float,
    device="cpu",
    dtype=torch.float32,
    iters: int = 60,
) -> Tensor:
    """Single-sample convenience wrapper around `discrete_arc_batch`."""
    c = torch.tensor([chord], device=device, dtype=torch.float64)
    return discrete_arc_batch(n, l0, c, iters=iters).to(dtype)


def random_slack_wire(
    batch: int,
    n: int,
    length: float,
    slack: tuple[float, float] = (0.55, 0.95),
    device="cpu",
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Batch of buckled wires with exact edge lengths and random slack.

    `slack` is the end-to-end distance as a fraction of the rest length; 1.0
    would be fully straight. Each sample gets a random slack, a random overall
    rotation, and a random buckling sign.
    """
    l0 = length / (n - 1)
    lo, hi = slack
    fr = torch.rand(batch, generator=generator, device=device, dtype=torch.float64)
    fracs = lo + fr * (hi - lo)

    p = discrete_arc_batch(n, l0, chords=fracs * length)         # (B, n, 2)

    sign = torch.where(
        torch.rand(batch, 1, generator=generator, device=device, dtype=torch.float64) < 0.5,
        -1.0, 1.0)
    p[..., 1] *= sign

    ang = torch.rand(batch, generator=generator, device=device,
                     dtype=torch.float64) * 2 * torch.pi
    c, s = torch.cos(ang), torch.sin(ang)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    p = torch.einsum("bij,bkj->bki", rot, p)
    return p.to(dtype)
