"""Discrete Elastic Rods (DER) simulator, PLANAR restriction.

Bergou, Wardetzky, Robinson, Audoly, Grinspun, "Discrete Elastic Rods"
(SIGGRAPH 2008) -- the rod theory DLO-Lab (arXiv:2606.04206) builds its own
solver on. This is NOT DLO-Lab's code: no Taichi, no Genesis, no material
frames or twisting. It is an independent, from-scratch implementation of the
PLANAR restriction of their bending formulation, built to be directly
comparable with this project's existing mass-spring-damper model
(dlo_model.py) on the same real 2D-tracked footage.

Why planar, not full 3D: every cable here is filmed from directly above a
flat board -- there is no out-of-plane rotation to observe, fit, or predict,
so the material-frame/twist machinery full DER needs in 3D would only add
degrees of freedom this project has no data to ground. What DOES carry over
and matters: DER writes bending energy per unit ARC LENGTH via the discrete
curvature binormal at each vertex, normalised by that vertex's own Voronoi
length, rather than as a bare per-node angular spring (dlo_model.py's
`bending_energy`, which sums 0.5*k_b*angle^2 over nodes with no length
term) -- so k_b there is only a stiffness *for the current n_nodes*: halving
the node count changes how bent the rod ends up under the same k_b, since
half as many spring terms resist the same physical bend. DER's Voronoi
normalisation removes that: k_b becomes a true material property [N m^2],
independent of how finely the rod happens to be discretised. That is the one
piece of DER worth having here, so it is the one piece this file adds
beyond the existing model -- plus the same bending-plasticity mechanism
already grafted onto dlo_model.py (wire_model.dlo_model.DLOParams), acting on
the curvature binormal instead of the raw angle.

Interface matches DLOSimulator exactly (same step(p, v, prm, constraints)
contract, same NodeConstraints) so it is a drop-in alternative anywhere
DLOSimulator is used -- wire_model.actions.rollout, wire_model.dataset.
generate, and this project's own long-horizon evaluation scripts all take
the simulator as a parameter and do not otherwise care which one it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dlo_model import NodeConstraints  # identical contract, reused as-is

Tensor = torch.Tensor


@dataclass
class DERParams:
    """Physical parameters, DER convention.

    m           total mass of the rod [kg]
    k_b         bending MODULUS [N m^2] -- energy density along the rod is
                0.5 * k_b * |kb - kb_rest|^2 / l_voronoi (Bergou et al. Sec.
                4.2), so unlike DLOParams.k_b this does not need retuning if
                n_nodes changes.
    k_d         viscous damping [N s/m]
    k_c         dry friction force per node [N]; 0 disables it
    k_s         axial stiffness [N/m]; kept high, not estimated
    length      rest length of the rod [m]
    kb_rest     rest curvature binormal per interior node (B, N-2) -- the DER
                analogue of DLOParams.rest_angle. |kb| = 2*tan(phi/2) for
                turning angle phi (Bergou et al. Eq. 1, restricted to the
                plane), so kb_rest = 0 means "wants to be straight",
                matching rest_angle = 0. Use `rest_angle_to_kb` to convert an
                existing angle-based estimate.
    k_yield     [dimensionless, kb units] curvature-binormal deviation from
                kb_rest below which no creep happens (elastic, recoverable
                bend). None (default) disables plasticity exactly.
    k_creep     how fast excess curvature is absorbed into a new permanent
                kb_rest once past k_yield, per second.
    """

    m: Tensor | float
    k_b: Tensor | float
    k_d: Tensor | float
    k_c: Tensor | float = 0.0
    k_s: Tensor | float = 1.0e5
    length: Tensor | float = 0.5
    kb_rest: Tensor | float = 0.0
    k_yield: Tensor | float | None = None
    k_creep: Tensor | float = 0.0

    def to_batched(self, batch: int, device, dtype=torch.float32,
                   n_nodes: int | None = None) -> "DERParams":
        def prep(x):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 0:
                t = t.expand(batch)
            return t.reshape(batch, 1, 1)

        def prep_vec(x):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 2:
                return t
            if n_nodes is None:
                raise ValueError("n_nodes needed to broadcast a scalar/1-D "
                                 "kb_rest to (B, N-2)")
            n_int = n_nodes - 2
            if t.ndim == 0:
                return t.expand(batch, n_int)
            if t.shape[0] != n_int:
                raise ValueError(f"kb_rest has {t.shape[0]} entries, "
                                 f"expected {n_int} interior nodes")
            return t.unsqueeze(0).expand(batch, -1)

        return DERParams(
            m=prep(self.m), k_b=prep(self.k_b), k_d=prep(self.k_d),
            k_c=prep(self.k_c), k_s=prep(self.k_s), length=prep(self.length),
            kb_rest=prep_vec(self.kb_rest),
            k_yield=None if self.k_yield is None else prep(self.k_yield),
            k_creep=prep(self.k_creep),
        )


def rest_angle_to_kb(rest_angle) -> Tensor:
    """Convert our existing turning-angle convention (DLOParams.rest_angle,
    and every real-data script's own curvature estimate) to DER's curvature-
    binormal one, so an existing estimate can seed a DERParams directly
    instead of being re-derived: kb = 2*tan(angle/2)."""
    return 2.0 * torch.tan(torch.as_tensor(rest_angle) * 0.5)


class DERSimulator:
    """Semi-implicit Euler integrator, planar Discrete Elastic Rod.

    Shares its integration scheme, Coulomb-friction impulse, and plasticity
    update verbatim with DLOSimulator (dlo_model.py) -- see that class's
    docstrings for why each is written the way it is (implicit damping for
    stability at high k_d; impulse-with-clamp friction for stiction without
    a separate stick/slip branch). The bending energy is the one thing that
    differs; see the module docstring.
    """

    def __init__(self, n_nodes: int, dt: float = 4.0e-5,
                device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                symplectic: bool = True, damping_mode: str = "implicit",
                eps: float = 1.0e-9, kb_clamp: float = 10.0) -> None:
        if n_nodes < 3:
            raise ValueError("n_nodes must be >= 3 for bending to be defined")
        if damping_mode not in ("implicit", "explicit"):
            raise ValueError("damping_mode must be 'implicit' or 'explicit'")
        self.n_nodes = n_nodes
        self.dt = dt
        self.device = torch.device(device)
        self.dtype = dtype
        self.symplectic = symplectic
        self.damping_mode = damping_mode
        self.eps = eps
        # kb = 2*tan(phi/2) diverges as the turning angle phi -> pi (a rod
        # folding fully back on itself) -- a real cable in this rig's footage
        # never does that, but a bad transient state during integration
        # could pass through something close to it and blow the force up.
        # Clamping kb caps that at a phi of about 2*atan(kb_clamp/2) =~ 157
        # degrees at the default 10.0, well past any bend these clips show,
        # while keeping the force finite if the integrator ever overshoots.
        self.kb_clamp = kb_clamp

    # ---- forces -------------------------------------------------------

    def axial_forces(self, p: Tensor, k_s: Tensor, l0: Tensor) -> Tensor:
        """f^s: Hookean springs along each edge, rest length l0. Identical
        to DLOSimulator.axial_forces -- stretching is not a DER-specific
        concern, this project's high-k_s near-inextensibility choice
        (see DLOParams' docstring) applies equally here."""
        d = p[:, 1:] - p[:, :-1]
        length = d.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        direction = d / length
        f_edge = k_s * (length - l0) * direction
        f = torch.zeros_like(p)
        f[:, :-1] += f_edge
        f[:, 1:] -= f_edge
        return f

    def curvature_binormal(self, p: Tensor) -> Tensor:
        """kb_i at each interior vertex (Bergou et al. Eq. 1, z-component
        only -- the in-plane components vanish for a planar curve):

            kb_i = 2 (e_{i-1} x e_i) / (|e_{i-1}| |e_i| + e_{i-1}.e_i)

        Equivalent to 2*tan(phi_i/2) for the turning angle phi_i used
        elsewhere in this project (see rest_angle_to_kb), but written this
        way because it is what DER's bending energy is conventionally
        defined in terms of, and it avoids turning_angles' atan2 branch cut
        mattering for the energy itself (only for interpreting kb after).
        """
        e = p[:, 1:] - p[:, :-1]                        # (B, N-1, 2)
        e_prev, e_next = e[:, :-1], e[:, 1:]             # (B, N-2, 2)
        cross = e_prev[..., 0] * e_next[..., 1] - e_prev[..., 1] * e_next[..., 0]
        dot = (e_prev * e_next).sum(-1)
        len_prod = e_prev.norm(dim=-1) * e_next.norm(dim=-1)
        kb = 2.0 * cross / (len_prod + dot).clamp_min(self.eps)
        return kb.clamp(-self.kb_clamp, self.kb_clamp)

    def voronoi_length(self, p: Tensor) -> Tensor:
        """l_i = 0.5*(|e_{i-1}| + |e_i|) per interior vertex -- the length of
        rod each vertex's bending energy is normalised by (Bergou et al.
        Sec. 4.2). This is what makes k_b resolution-independent: double
        n_nodes and each l_i halves, so each vertex's energy contribution
        scales down to match -- the SUM stays the same physical bending
        energy the rod actually stores, which a plain per-node sum (as in
        dlo_model.py) does not guarantee.
        """
        e_len = (p[:, 1:] - p[:, :-1]).norm(dim=-1)      # (B, N-1)
        return 0.5 * (e_len[:, :-1] + e_len[:, 1:])       # (B, N-2)

    def bending_energy(self, p: Tensor, k_b: Tensor,
                       kb_rest: Tensor | float = 0.0) -> Tensor:
        """E^b = sum_i 0.5 k_b (kb_i - kb_rest_i)^2 / l_voronoi_i."""
        kb = self.curvature_binormal(p)
        l_vor = self.voronoi_length(p).clamp_min(self.eps)
        d = kb - kb_rest
        return (0.5 * k_b.squeeze(-1) * d.pow(2) / l_vor).sum()

    def bending_forces(self, p: Tensor, k_b: Tensor,
                       kb_rest: Tensor | float = 0.0) -> Tensor:
        """f^b = -dE^b/dp, via autograd -- identical pattern to
        DLOSimulator.bending_forces, for the same reason (deriving this by
        hand is error-prone; autograd is exact by construction)."""
        with torch.enable_grad():
            p_ = p.detach().requires_grad_(True)
            energy = self.bending_energy(p_, k_b, kb_rest)
            (grad,) = torch.autograd.grad(energy, p_)
        return -grad

    def elastic_forces(self, p: Tensor, prm: DERParams) -> Tensor:
        l0 = prm.length / (self.n_nodes - 1)
        return (self.axial_forces(p, prm.k_s, l0)
                + self.bending_forces(p, prm.k_b, prm.kb_rest))

    def total_forces(self, p: Tensor, v: Tensor, prm: DERParams) -> Tensor:
        return self.elastic_forces(p, prm) - prm.k_d * v

    # ---- integration ----------------------------------------------------

    def _apply_coulomb(self, v_new: Tensor, m_node: Tensor, prm: DERParams) -> Tensor:
        """Dry (Coulomb) friction against the board -- identical mechanism
        and reasoning to DLOSimulator._apply_coulomb (see its docstring):
        an impulse with a clamp, not a force, so it reproduces stiction
        without chattering at v ~ 0."""
        k_c = getattr(prm, "k_c", 0.0)
        if not torch.is_tensor(k_c):
            if k_c == 0.0:
                return v_new
            k_c = torch.as_tensor(k_c, dtype=v_new.dtype, device=v_new.device)
        elif not bool((k_c != 0).any()):
            return v_new
        dv = self.dt * k_c / m_node
        speed = v_new.norm(dim=-1, keepdim=True)
        scale = (1.0 - dv / speed.clamp_min(self.eps)).clamp_min(0.0)
        return v_new * scale

    def _apply_plasticity(self, p_new: Tensor, prm: DERParams) -> None:
        """Bending plasticity in the curvature-binormal representation --
        same yield-threshold + creep-rate mechanism as DLOSimulator's (see
        DLOParams' docstring for the physical motivation and DLO-Lab
        reference), acting on kb_rest instead of rest_angle. A no-op when
        prm.k_yield is None."""
        if prm.k_yield is None:
            return
        k_yield = torch.as_tensor(prm.k_yield)
        k_yield = k_yield.squeeze(-1) if k_yield.ndim == 3 else k_yield
        k_creep = torch.as_tensor(prm.k_creep)
        k_creep = k_creep.squeeze(-1) if k_creep.ndim == 3 else k_creep
        kb = self.curvature_binormal(p_new)
        d = kb - prm.kb_rest
        excess = torch.sign(d) * (d.abs() - k_yield).clamp_min(0.0)
        prm.kb_rest = prm.kb_rest + self.dt * k_creep * excess

    def step(
        self,
        p: Tensor,
        v: Tensor,
        prm: DERParams,
        constraints: NodeConstraints | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step -- identical structure to DLOSimulator.step."""
        m_node = prm.m / self.n_nodes
        f_el = self.elastic_forces(p, prm)

        if self.damping_mode == "implicit":
            v_new = (m_node * v + self.dt * f_el) / (m_node + self.dt * prm.k_d)
        else:
            v_new = v + self.dt * (f_el - prm.k_d * v) / m_node

        v_new = self._apply_coulomb(v_new, m_node, prm)

        if self.symplectic:
            p_new = p + self.dt * v_new
        else:
            p_new = p + self.dt * v

        if constraints is not None:
            mask = constraints.mask.unsqueeze(-1)
            v_driven = (constraints.target - p) / self.dt
            p_new = torch.where(mask, constraints.target, p_new)
            v_new = torch.where(mask, v_driven, v_new)

        self._apply_plasticity(p_new, prm)
        return p_new, v_new

    def stability_dt(self, prm: DERParams) -> Tensor:
        """Same two fast modes bound dt here as in DLOSimulator (axial
        spring frequency, and the damping bound under explicit
        integration) -- the bending term does not add a third, since its
        stiffness acts through the same node masses the axial term already
        bounds. See DLOSimulator.stability_dt for the derivation."""
        m_node = prm.m / self.n_nodes
        dt_axial = 2.0 / (prm.k_s / m_node).sqrt()
        if self.damping_mode == "implicit":
            return dt_axial.reshape(-1)
        dt_damp = 2.0 * m_node / prm.k_d.clamp_min(self.eps)
        return torch.minimum(dt_axial, dt_damp).reshape(-1)
