"""Dataset generation from the analytical model (Caporali et al., Sec. II-B1).

The paper's recipe: initialise the DLO almost straight, sample a set of k
actions, apply them sequentially, and record (V_in, action, params) -> V_out
for each. Sequential chains matter -- they keep the training configurations on
the manifold the dynamics actually reach, which random shapes would not.

Adapted to the wire-harness videos: node 0 is pinned (the wire stays plugged
into the stationary connector) and the grasped edge is the terminal one (the
connector a hand carries across the board). Actions are sampled so the wire
stays slack, since a taut wire is governed by k_s -- which is fixed, not
estimated -- and carries no information about k_b.

A note on the settling horizon, which is a real modelling decision and not a
convergence knob: the paper damps viscously (-k_d p_dot), and viscous damping
does not change *where* a system comes to rest, only how fast it gets there.
Run to full relaxation and every k_d yields the same shape, i.e. k_d becomes
unidentifiable (measured: 34 mm shape spread at a short horizon vs 3.6 mm when
relaxed -- see scripts/probe_identifiability.py). Real dry friction *freezes* a
wire's shape; viscous damping plus a finite horizon is the paper's proxy for
that. So the horizon is held fixed here, matched to the video cadence, and k_d
absorbs the surface interaction. Horizon and k_d are partially degenerate by
construction: k_d is meaningful *relative to this horizon*, not absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import math
import torch

from .actions import (Action, grasped_node_targets, rollout,
                      terminal_edge_action)
from .dlo_model import DLOParams, DLOSimulator
from .geometry import discrete_arc_batch, random_rest_angle

Tensor = torch.Tensor


@dataclass
class SamplingRanges:
    """Physically plausible ranges the dataset is drawn from.

    Broad by design: the NN is conditioned on the parameters so that one model
    covers the real-world variability, per the paper. k_b is drawn
    log-uniformly because it spans two decades and the paper's own estimates
    (~0.06 to ~0.19) sit near the bottom of it.
    """

    length: tuple[float, float] = (0.25, 0.60)      # [m]
    mass: tuple[float, float] = (0.01, 0.08)        # [kg]
    # The ceiling stays at the old 1.0. Raising it to 2.0 doubled the bending
    # force against a FIXED k_s = 1e5, and a 300-chain probe then threw away
    # 18.9% of samples for stretching past 1% (max 31.7%) -- the inextensible-
    # wire assumption breaking exactly in the new regime. Only the floor moves
    # down, which softens rather than tensions. [0.004, 1.0] still spans 250x,
    # enough for the 26x..423x the physics fits show within one cable.
    k_b: tuple[float, float] = (0.004, 1.00)        # log-uniform, PER JOINT
    # Bending stiffness is not uniform along a real harness bundle, and every
    # physics engine here gains by fitting it per joint: the rod's row 11 -> 12
    # is the clean case (18.6 -> 17.0 mm, two gradient fits differing only by
    # the 14 scales). Within ONE fitted cable the spread reaches 159x
    # (Genesis), 423x (Newton row 8) and 26x (rod), so the dataset has to show
    # profiles of that order or the network never sees the regime its physics
    # counterparts settle in. The profile is SMOOTH along the wire -- the
    # fitted ones are, and independent per-joint draws would spend samples on
    # alternating stiff/soft joints that no fit has ever produced.
    k_b_spread: tuple[float, float] = (1.0, 400.0)  # max/min within one cable
    k_b_p_flat: float = 0.20                        # share with no profile at all
    k_d: tuple[float, float] = (2.0, 40.0)
    # Dry friction force per node [N]. Bounded by a measurement, not a guess:
    # scripts/probe_coulomb.py sweeps a bent wire and finds it stops moving
    # entirely at ~1 N, so anything above that is a frozen wire and carries no
    # information. Sampling to 0.6 keeps the whole transition from free to
    # nearly-stuck inside the training distribution, including k_c = 0 so the
    # paper's purely viscous case stays in-distribution.
    k_c: tuple[float, float] = (0.0, 0.60)
    k_s: float = 1.0e5                              # fixed; see DLOParams docstring
    rest_amp: tuple[float, float] = (0.0, 0.35)     # |theta_rest| per node [rad]
    rest_p_straight: float = 0.15                   # keep the paper's case in-distribution
    slack_init: tuple[float, float] = (0.55, 0.92)  # end-to-end / rest length
    slack_bounds: tuple[float, float] = (0.30, 0.94)  # enforced after actions
    max_disp: float = 0.30                          # per action, fraction of L
    max_rot: float = 1.20                           # per action [rad]
    # The hub connector is driven too, not held fixed. Measured on the clips it
    # travels a median 89 mm per 2 s against the mover's 58 mm, so it is the
    # LARGER of the two drives -- but it is carried by the rig rather than
    # thrown by a hand, so its motion is smoother and it barely rotates. Scaled
    # relative to the mover action rather than given absolute bounds, so the two
    # ends stay in a plausible ratio whatever the wire length.
    # Sampled as a fraction of the mover displacement, uniform on [0, this].
    # Set from the footage, not from intuition: the hub travels a median 89 mm
    # per 2 s against the mover's 58 mm, a ratio of ~1.5. An earlier value of
    # 0.55 put the mean ratio at 0.28 and left the real regime entirely outside
    # the training distribution -- the network would have been extrapolating on
    # every real frame.
    hub_disp_frac: float = 2.20                     # of the mover displacement
    hub_rot_frac: float = 0.15                      # of the mover rotation


@dataclass
class SimConfig:
    """Fixed simulation settings. n_move/n_settle are load-bearing -- see the
    module docstring on k_d identifiability."""

    n_nodes: int = 16
    dt: float = 4.0e-5      # set by k_s=1e5; see DLOParams docstring
    n_move: int = 15000     # 0.6 s: a hand carrying a connector
    n_settle: int = 10000   # 0.4 s: partial relaxation, as on a real surface


def _k_b_profile(base, batch, n_nodes, r: "SamplingRanges", device, generator):
    """A smooth bending-stiffness profile along the wire, (batch, n_nodes-2).

    Two low-order sinusoids with random phase give a shape that varies over the
    length without ever alternating joint to joint; its log-amplitude is drawn
    so the max/min ratio within one cable covers r.k_b_spread. A share
    r.k_b_p_flat keeps the uniform case -- the paper's own, and every earlier
    checkpoint's -- inside the distribution.
    """
    n_int = n_nodes - 2
    t = torch.linspace(0.0, 1.0, n_int, device=device).unsqueeze(0)

    def rnd(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    ph = rnd(batch, 2) * 2 * math.pi
    w = 1.0 + 2.0 * rnd(batch, 2)                      # 1..3 cycles over the wire
    shape = (torch.sin(w[:, :1] * 2 * math.pi * t + ph[:, :1])
             + 0.5 * torch.sin(w[:, 1:] * 2 * math.pi * t + ph[:, 1:]))
    shape = shape / shape.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

    lo, hi = math.log(r.k_b_spread[0]), math.log(r.k_b_spread[1])
    amp = torch.exp(lo + (hi - lo) * rnd(batch, 1)) .log() / 2.0   # +-amp in log
    flat = (rnd(batch, 1) < r.k_b_p_flat).to(amp.dtype)
    amp = amp * (1.0 - flat)
    prof = base.unsqueeze(1) * torch.exp(shape * amp)
    # Clamp back into the k_b box. Without it a base near either end plus a
    # large amplitude lands two decades outside, and the normaliser -- which
    # maps [k_b_lo, k_b_hi] to [0, 1] in log space -- would see values far
    # outside its own range. Clamping compresses the most extreme profiles
    # rather than letting them define a box nothing else occupies.
    return prof.clamp(r.k_b[0], r.k_b[1])


def sample_params(batch, r: SamplingRanges, device, n_nodes: int,
                  generator=None) -> DLOParams:
    def uni(lo, hi):
        return lo + (hi - lo) * torch.rand(batch, generator=generator,
                                           device=device)

    log_lo, log_hi = torch.log(torch.tensor(r.k_b, device=device))
    k_b_base = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(
        batch, generator=generator, device=device))
    k_b = _k_b_profile(k_b_base, batch, n_nodes, r, device, generator)

    return DLOParams(
        m=uni(*r.mass),
        k_b=k_b,
        k_d=uni(*r.k_d),
        k_c=uni(*r.k_c),
        k_s=torch.full((batch,), r.k_s, device=device),
        length=uni(*r.length),
        rest_angle=random_rest_angle(batch, n_nodes, amp=r.rest_amp,
                                     p_straight=r.rest_p_straight,
                                     device=device, generator=generator),
    )


def sample_connector_action(
    p: Tensor,
    length: Tensor,
    r: SamplingRanges,
    generator=None,
    max_tries: int = 12,
) -> Action:
    """Sample a terminal-edge (connector) action that keeps the wire slack.

    Rejection sampling, vectorised: every try resamples only the samples still
    violating the slack bounds. Samples that never satisfy it fall back to a
    zero action rather than silently training the NN on taut configurations.

    The slack test constrains node alpha, NOT the far end node alpha+1, and
    compares it against the *sub-chain* rest length alpha*l0 rather than the
    full L. The grasped edge is placed rigidly, so it contributes no compliance:
    only nodes 0..alpha can deform to absorb the span. Testing the end node
    against the full length admits actions that rotate the grasped edge back on
    itself, putting node alpha further from the anchor than node alpha+1 and
    leaving the sub-chain to cover more distance than it has material for. That
    silently tensions the whole wire (measured: every edge uniformly stretched
    ~7%, the rigid grasped edge alone at exactly l0) and the error compounds
    down the chain, since each V_out becomes the next V_in.

    p:      (B, N, 2) current configuration
    length: (B,) rest length per sample
    """
    batch, n_nodes, _ = p.shape
    dev = p.device
    lo, hi = r.slack_bounds
    alpha = n_nodes - 2
    sub_len = alpha * length / (n_nodes - 1)      # rest length of nodes 0..alpha

    dx = torch.zeros(batch, device=dev)
    dy = torch.zeros(batch, device=dev)
    dth = torch.zeros(batch, device=dev)
    todo = torch.ones(batch, dtype=torch.bool, device=dev)

    anchor = p[:, 0]                         # pinned connector
    for _ in range(max_tries):
        k = int(todo.sum())
        if k == 0:
            break
        mag = r.max_disp * length[todo] * torch.rand(k, generator=generator, device=dev)
        ang = 2 * torch.pi * torch.rand(k, generator=generator, device=dev)
        cand_dx, cand_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        cand_dth = r.max_rot * (2 * torch.rand(k, generator=generator, device=dev) - 1)

        # resolve where the grasped edge actually lands: rotating about the
        # midpoint moves both its nodes, so do not approximate this
        probe = Action(
            alpha=torch.full((k,), alpha, dtype=torch.long, device=dev),
            dx=cand_dx, dy=cand_dy, dtheta=cand_dth,
        )
        _, tgt = grasped_node_targets(p[todo], probe)
        span = (tgt[:, 0] - anchor[todo]).norm(dim=-1) / sub_len[todo]
        ok = (span > lo) & (span < hi)

        idx = torch.nonzero(todo, as_tuple=True)[0]
        good = idx[ok]
        dx[good], dy[good], dth[good] = cand_dx[ok], cand_dy[ok], cand_dth[ok]
        todo[good] = False

    return terminal_edge_action(n_nodes, dx, dy, dth, at_end=True)


def generate(
    n_chains: int,
    chain_len: int = 5,
    cfg: SimConfig | None = None,
    r: SamplingRanges | None = None,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Tensor]:
    """Generate n_chains * chain_len transitions.

    Returns tensors on CPU:
      V_in       (S, N, 2)   configuration before the action
      V_out      (S, N, 2)   configuration after it settles
      action     (S, 4)      [alpha, dx, dy, dtheta]
      params     (S, 4)      [m, k_b, k_d, length]
      rest_angle (S, N-2)    natural turning angle per interior node
    """
    cfg = cfg or SimConfig()
    r = r or SamplingRanges()
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)

    sim = DLOSimulator(n_nodes=cfg.n_nodes, dt=cfg.dt, device=dev)
    prm = sample_params(n_chains, r, dev, cfg.n_nodes, g)
    lengths = prm.length.clone()

    # each chain starts from its own slack, randomly-oriented wire
    p = _init_batch(n_chains, cfg.n_nodes, lengths, r, dev, g)

    # Node 0 is no longer pinned: it is the hub plug, and it moves. It is
    # driven by its own action instead (see hub_action below).
    static = torch.zeros(n_chains, cfg.n_nodes, dtype=torch.bool, device=dev)
    prm_b = prm.to_batched(n_chains, dev, n_nodes=cfg.n_nodes)

    # Burn-in. _init_batch lays down a constant-curvature arc, but a plugged-in
    # wire rests at a compromise between its natural shape and its endpoints.
    # A zero action pins both ends where they are and lets the interior relax,
    # so the first recorded V_in is a state the real setup could actually be in.
    zero = torch.zeros(n_chains, device=dev)
    p = rollout(sim, p, prm_b, terminal_edge_action(cfg.n_nodes, zero, zero, zero),
                static_mask=static, n_move=1, n_settle=cfg.n_settle,
                action2=terminal_edge_action(cfg.n_nodes, zero, zero, zero,
                                             at_end=False))

    V_in, V_out, acts, pars, rests = [], [], [], [], []
    for step in range(chain_len):
        act = sample_connector_action(p, lengths, r, generator=g)
        # Hub action: a scaled, independently-directed version of the mover's.
        # Independent direction matters -- if it merely copied the mover the
        # pair would be a rigid translation, which the old hub-fixed frame
        # already covered and which teaches nothing new.
        ang = 2 * torch.pi * torch.rand(n_chains, generator=g, device=dev)
        sc = r.hub_disp_frac * torch.rand(n_chains, generator=g, device=dev)
        mag = sc * torch.stack([act.dx, act.dy], -1).norm(dim=-1)
        hub_dx, hub_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        # `sample_connector_action` checked slack against a FIXED node 0, which
        # is no longer true once the hub is driven: both ends can now pull
        # apart at once and tension the wire, which drove 15.6% of samples past
        # the 1% stretch filter. Re-check the span with BOTH targets and shrink
        # the hub step until the wire is slack again -- shrinking the hub rather
        # than the mover keeps the mover's own accepted action intact.
        span_dir = (p[:, -1] + torch.stack([act.dx, act.dy], -1)) \
            - (p[:, 0] + torch.stack([hub_dx, hub_dy], -1))
        span = span_dir.norm(dim=-1)
        limit = r.slack_bounds[1] * lengths
        over = span > limit
        if bool(over.any()):
            shrink = torch.where(over, (limit / span.clamp_min(1e-9)), torch.ones_like(span))
            hub_dx = hub_dx * shrink
            hub_dy = hub_dy * shrink
        hub_act = terminal_edge_action(
            cfg.n_nodes, hub_dx, hub_dy, r.hub_rot_frac * act.dtheta,
            at_end=False)
        pf = rollout(sim, p, prm_b, act, static_mask=static,
                     n_move=cfg.n_move, n_settle=cfg.n_settle, action2=hub_act)

        finite = torch.isfinite(pf).all(dim=(1, 2))
        if not bool(finite.all()):
            n_bad = int((~finite).sum())
            raise RuntimeError(
                f"{n_bad}/{n_chains} rollouts diverged at chain step {step}; "
                f"check dt against DLOSimulator.stability_dt")

        V_in.append(p.cpu())
        V_out.append(pf.cpu())
        # 8-dim: the mover action followed by the hub action
        acts.append(torch.cat([act.as_vector(), hub_act.as_vector()],
                              dim=-1).cpu())
        # order matters: the net consumes params[:, :-1] as
        # [m, k_b x (N-2), k_d, k_c], and length stays LAST because it is not
        # an input (V_in's arc length already carries it). k_b occupies N-2
        # columns now instead of one -- torch.cat, not stack, since the blocks
        # have different widths.
        def col(x):
            return x if x.ndim == 2 else x.unsqueeze(-1)
        pars.append(torch.cat([col(prm.m), col(prm.k_b), col(prm.k_d),
                               col(prm.k_c), col(prm.length)], -1).cpu())
        rests.append(prm.rest_angle.cpu())
        p = pf
        if verbose:
            print(f"  chain step {step+1}/{chain_len} done", flush=True)

    return {
        "V_in": torch.cat(V_in),
        "V_out": torch.cat(V_out),
        "action": torch.cat(acts),
        "params": torch.cat(pars),
        "rest_angle": torch.cat(rests),
        "config": {**asdict(cfg), **asdict(r)},
    }


def generate_seq(
    n_chains: int,
    chain_len: int = 5,
    n_hist: int = 4,
    n_sub: int = 4,
    cfg: SimConfig | None = None,
    r: SamplingRanges | None = None,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Tensor]:
    """The SEQUENCE dataset: the same chains as `generate`, sampled at the
    video's frame cadence, with the preceding frames attached.

    `generate` records one transition per 1 s action and the network that
    learns from it is Markov in the configuration: it never sees how fast the
    wire was moving, although the simulator's state is (position, velocity)
    and 0.4 s of settling leaves a residual velocity in every V_in. Here every
    1 s action is recorded `n_sub` times (0.25 s apart at the defaults, the
    clips' 4 fps) and each consecutive pair becomes a transition that carries
    the `n_hist` configurations before it. Velocity and acceleration are then
    observable from the history, and a rollout on video steps one frame at a
    time with the frames it has already predicted as its own history.

    The sub-step action is the share of the 1 s action that falls into the
    sub-interval: the grasped edges are ramped linearly over n_move steps and
    held afterwards, so a sub-interval inside the move phase carries a
    proportional fraction and one inside the settle phase carries zero --
    those zero-action transitions are the free dynamics, which is exactly
    what the Markov dataset could never show.

    Returns tensors on CPU:
      hist       (S, n_hist, N, 2)  the n_hist configurations before V_in,
                                    oldest first; a chain's first entries are
                                    the burnt-in rest state repeated, i.e. a
                                    wire that has been standing still
      V_in       (S, N, 2)
      V_out      (S, N, 2)          one frame later
      action     (S, 8)             [alpha, dx, dy, dtheta] x (mover, hub)
                                    for THIS sub-interval
      params     (S, 3 + (N-2) + 1) [m, k_b x (N-2), k_d, k_c, length]
      rest_angle (S, N-2)
      t_sub      (S,)               sub-interval index within its 1 s action
    """
    cfg = cfg or SimConfig()
    r = r or SamplingRanges()
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)
    total = cfg.n_move + cfg.n_settle
    assert total % n_sub == 0, (total, n_sub)
    sub = total // n_sub
    record_at = [sub * (i + 1) for i in range(n_sub)]         # 6250, ..., 25000
    # share of the ramp completed at each record time; the ramp ends at n_move
    frac = [min(1.0, t / cfg.n_move) for t in record_at]
    share = [frac[0]] + [frac[i] - frac[i - 1] for i in range(1, n_sub)]

    sim = DLOSimulator(n_nodes=cfg.n_nodes, dt=cfg.dt, device=dev)
    prm = sample_params(n_chains, r, dev, cfg.n_nodes, g)
    lengths = prm.length.clone()
    p = _init_batch(n_chains, cfg.n_nodes, lengths, r, dev, g)
    static = torch.zeros(n_chains, cfg.n_nodes, dtype=torch.bool, device=dev)
    prm_b = prm.to_batched(n_chains, dev, n_nodes=cfg.n_nodes)
    zero = torch.zeros(n_chains, device=dev)
    p = rollout(sim, p, prm_b, terminal_edge_action(cfg.n_nodes, zero, zero, zero),
                static_mask=static, n_move=1, n_settle=cfg.n_settle,
                action2=terminal_edge_action(cfg.n_nodes, zero, zero, zero,
                                             at_end=False))

    def col(x):
        return x if x.ndim == 2 else x.unsqueeze(-1)
    par_row = torch.cat([col(prm.m), col(prm.k_b), col(prm.k_d),
                         col(prm.k_c), col(prm.length)], -1).cpu()
    rest_row = prm.rest_angle.cpu()

    # rolling window of the last n_hist frames, oldest first; starts at rest
    window = [p.cpu().clone() for _ in range(n_hist)]
    H, V_in, V_out, acts, pars, rests, tsub = [], [], [], [], [], [], []
    for step in range(chain_len):
        act = sample_connector_action(p, lengths, r, generator=g)
        ang = 2 * torch.pi * torch.rand(n_chains, generator=g, device=dev)
        sc = r.hub_disp_frac * torch.rand(n_chains, generator=g, device=dev)
        mag = sc * torch.stack([act.dx, act.dy], -1).norm(dim=-1)
        hub_dx, hub_dy = mag * torch.cos(ang), mag * torch.sin(ang)
        span_dir = (p[:, -1] + torch.stack([act.dx, act.dy], -1)) \
            - (p[:, 0] + torch.stack([hub_dx, hub_dy], -1))
        span = span_dir.norm(dim=-1)
        limit = r.slack_bounds[1] * lengths
        over = span > limit
        if bool(over.any()):
            shrink = torch.where(over, (limit / span.clamp_min(1e-9)), torch.ones_like(span))
            hub_dx = hub_dx * shrink
            hub_dy = hub_dy * shrink
        hub_act = terminal_edge_action(
            cfg.n_nodes, hub_dx, hub_dy, r.hub_rot_frac * act.dtheta,
            at_end=False)
        pf, recs = rollout(sim, p, prm_b, act, static_mask=static,
                           n_move=cfg.n_move, n_settle=cfg.n_settle,
                           action2=hub_act, record_at=record_at)
        finite = torch.isfinite(pf).all(dim=(1, 2))
        if not bool(finite.all()):
            raise RuntimeError(
                f"{int((~finite).sum())}/{n_chains} rollouts diverged at chain "
                f"step {step}; check dt against DLOSimulator.stability_dt")

        full = torch.cat([act.as_vector(), hub_act.as_vector()], dim=-1).cpu()
        cur = p.cpu().clone()
        for i in range(n_sub):
            nxt = recs[:, i].cpu()
            a = full.clone()
            a[:, [1, 2, 3, 5, 6, 7]] *= share[i]          # alphas stay
            H.append(torch.stack(window, dim=1))
            V_in.append(cur)
            V_out.append(nxt)
            acts.append(a)
            pars.append(par_row)
            rests.append(rest_row)
            tsub.append(torch.full((n_chains,), i, dtype=torch.long))
            window = window[1:] + [cur]
            cur = nxt
        p = pf
        if verbose:
            print(f"  chain step {step+1}/{chain_len} done", flush=True)

    return {
        "hist": torch.cat(H),
        "V_in": torch.cat(V_in),
        "V_out": torch.cat(V_out),
        "action": torch.cat(acts),
        "params": torch.cat(pars),
        "rest_angle": torch.cat(rests),
        "t_sub": torch.cat(tsub),
        "config": {**asdict(cfg), **asdict(r), "n_hist": n_hist, "n_sub": n_sub,
                   "frame_s": cfg.dt * sub},
    }


def _init_batch(batch, n_nodes, lengths, r, dev, g):
    """Slack wires with per-sample rest lengths, all in one batched call."""

    # discrete_arc_batch takes one l0; group by doing it in normalised units
    # (unit rest length) and scaling per sample afterwards -- edge lengths stay
    # exactly proportional, so the invariant survives the scaling.
    lo, hi = r.slack_init
    fr = lo + (hi - lo) * torch.rand(batch, generator=g, device=dev,
                                     dtype=torch.float64)
    unit = discrete_arc_batch(n_nodes, 1.0 / (n_nodes - 1), chords=fr)
    p = unit * lengths.double().reshape(-1, 1, 1)

    sign = torch.where(torch.rand(batch, 1, generator=g, device=dev,
                                  dtype=torch.float64) < 0.5, -1.0, 1.0)
    p[..., 1] *= sign
    ang = 2 * torch.pi * torch.rand(batch, generator=g, device=dev,
                                    dtype=torch.float64)
    c, s = torch.cos(ang), torch.sin(ang)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return torch.einsum("bij,bkj->bki", rot, p).float()
