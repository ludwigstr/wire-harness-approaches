"""Analytical DLO model: mass-spring-damper serial chain.

Reimplements the model of Caporali et al., "Deformable Linear Objects
Manipulation with Online Model Parameters Estimation", Sec. II-A:

    m_i p_i_ddot = -k_d p_i_dot + f_i^s + f_i^b                       (Eq. 1)

Nodes carry mass and are linked by axial springs (k_s, held stiff so axial
deformation is negligible). Torsional springs at interior nodes provide the
bending stiffness k_b. Damping is proportional to node velocity and absorbs
the DLO-surface interaction -- this is why the paper estimates a different
k_d per surface (cloth vs cardboard).

Everything is planar (2D), batched over B independent DLOs, and written in
torch so dataset generation runs on the GPU.

Deviation from the paper, required by the wire-harness videos: any subset of
nodes can be *pinned* to prescribed positions. The paper's rope has two free
ends and only the grasped edge is driven; a harness wire is plugged into
connectors at both ends, so its terminal nodes are kinematically constrained.
See `NodeConstraints`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class DLOParams:
    """Physical parameters of a DLO.

    Each field is either a python float (shared across the batch) or a
    tensor of shape (B,) giving a per-sample value.

    m           total mass of the DLO [kg]
    k_b         bending stiffness of the torsional springs [N m / rad]
    k_d         viscous damping coefficient [N s / m]
    k_c         dry (Coulomb) friction force per node [N]; 0 disables it
    k_s         axial stiffness [N / m]; kept high, not estimated
    length      rest length of the DLO [m]
    rest_angle  natural turning angle per interior node [rad], (B, N-2)

    `rest_angle` extends the paper, which implicitly fixes it to zero: its
    bending energy is 0.5*k_b*theta^2, so the zero-energy shape is a straight
    line and any curvature is stored energy that pushes back. Real wire
    bundles, coiled and routed for months, take a permanent set -- they hold
    curvature with no restoring force -- and that set is NOT uniform along the
    wire. Hence a per-node profile rather than a scalar: energy is
    0.5*k_b*(theta_i - theta_rest_i)^2, so the wire's free shape is an
    arbitrary curve rather than a line.

    Leaving this at zero does not make the error go away, it relocates it: a
    wire that holds a bend the model can only explain as "wants to be straight
    but friction pins it" forces the estimator to inflate k_d, so the estimated
    friction silently absorbs a material property.

    The k_s default of 1e5 is calibrated, not guessed. The paper only says "a
    high value (negligible axial deformation)". Two mechanisms set the bar:

    - Statically, at 1e3 a compressed wire relieves load by shortening (~4%
      arc-length error) instead of buckling, which corrupts k_b: the shape then
      reflects the wrong mechanism. scripts/probe_ks.py shows 1e4 already fixes
      this for slow, gentle actions (<0.05% error).
    - Dynamically, viscous drag during a fast drag genuinely tensions the wire
      (drag force per node ~ k_d * v), and the resulting strain scales as 1/k_s.
      Under the dataset's actual action distribution 1e4 leaves 6.8% of samples
      strained past 1% (max 5.3%), which a real, near-inextensible wire would
      never show. 1e5 drops that to 0.00% (max 0.9%) for 2.4x the compute.

    The static probe alone would have wrongly cleared 1e4 -- gentle actions do
    not excite the mechanism that matters. Requires dt <= ~4e-5.
    """

    m: Tensor | float
    k_b: Tensor | float
    k_d: Tensor | float
    k_c: Tensor | float = 0.0
    k_s: Tensor | float = 1.0e5
    length: Tensor | float = 0.5
    rest_angle: Tensor | float = 0.0

    # Bending PLASTICITY: rest_angle creeps toward the current curvature once
    # held past a yield threshold, rather than staying fixed for the whole
    # rollout. Mirrors DLO-Lab's (arXiv:2606.04206) yield-threshold + creep-
    # rate mechanism for their Discrete Elastic Rods solver, grafted onto our
    # own two-ended mass-spring-damper model instead of adopting their whole
    # DER/Taichi stack: `rest_angle` is already exactly the per-node state
    # such a rule would act on (this project's own measurement found it the
    # DOMINANT shape parameter -- see plastic-set memory note). k_yield=None
    # (the default) disables the mechanism exactly, so every existing caller
    # is unaffected.
    #   k_yield  [rad]   turning-angle deviation from rest_angle below which
    #                    no creep happens (an elastic, fully-recoverable bend)
    #   k_creep  [rad/s per rad of excess]  how fast the excess is absorbed
    #                    into a new permanent rest_angle once past yield
    k_yield: Tensor | float | None = None
    k_creep: Tensor | float = 0.0

    def to_batched(self, batch: int, device, dtype=torch.float32,
                   n_nodes: int | None = None) -> "DLOParams":
        """Broadcast scalars to (B, 1, 1), and rest_angle to (B, N-2).

        rest_angle keeps its own shape because it is per-node, not per-DLO;
        `n_nodes` is required unless it is already batched.
        """

        def prep(x, per_node=None):
            """Scalars become (B, 1, 1). A PROFILE stays a profile.

            k_b and k_d are the two stiffnesses every physics engine in this
            project fits per joint; the force terms already broadcast
            element-wise (bending_energy multiplies k_b.squeeze(-1) against a
            (B, N-2) angle difference, damping multiplies k_d against a
            (B, N, 2) velocity), so a per-node profile needs no kernel change
            at all -- only this reshape had been flattening it to one value.
            """
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if per_node is not None:
                # A 1-D tensor is ambiguous: length `batch` means one scalar
                # per sample, length `per_node` means one profile shared by the
                # batch. Decide by the expected node count, the way prep_angle
                # does, instead of by "is it longer than 1" -- that read the
                # batch dimension as a profile and silently built (B, B, 1).
                if t.ndim == 2:
                    return t.reshape(batch, t.shape[-1], 1)
                if t.ndim == 1 and t.shape[0] == per_node and batch != per_node:
                    return t.unsqueeze(0).expand(batch, -1).reshape(batch, per_node, 1)
            if t.ndim == 0:
                t = t.expand(batch)
            return t.reshape(batch, 1, 1)

        def prep_angle(x):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 2:
                return t                                  # (B, N-2)
            if n_nodes is None:
                raise ValueError("n_nodes needed to broadcast a scalar/1-D "
                                 "rest_angle to (B, N-2)")
            n_int = n_nodes - 2
            if t.ndim == 0:
                return t.expand(batch, n_int)
            if t.shape[0] != n_int:
                raise ValueError(f"rest_angle has {t.shape[0]} entries, "
                                 f"expected {n_int} interior nodes")
            return t.unsqueeze(0).expand(batch, -1)       # shared profile

        return DLOParams(
            m=prep(self.m),
            k_b=prep(self.k_b, per_node=None if n_nodes is None else n_nodes - 2),
            k_d=prep(self.k_d, per_node=n_nodes),
            k_c=prep(self.k_c),
            k_s=prep(self.k_s),
            length=prep(self.length),
            rest_angle=prep_angle(self.rest_angle),
            k_yield=None if self.k_yield is None else prep(self.k_yield),
            k_creep=prep(self.k_creep),
        )


@dataclass
class NodeConstraints:
    """Kinematically driven nodes.

    mask    (B, N) bool -- True where the node position is prescribed
    target  (B, N, 2)   -- position to drive those nodes to

    Driven nodes ignore Eq. (1): their position is imposed and their velocity
    is set from the actual displacement, so the rest of the chain feels a
    physically consistent motion.
    """

    mask: Tensor
    target: Tensor


class DLOSimulator:
    """Euler integrator for the mass-spring-damper DLO of Eq. (1).

    Position update is semi-implicit (symplectic) Euler -- velocity first,
    then position -- the standard choice for mass-spring systems.

    `damping_mode` controls how -k_d v is integrated:

    "implicit" (default)
        v_new = (m v + dt f_elastic) / (m + dt k_d), i.e. the damping term is
        evaluated at the new velocity. Unconditionally stable and identical to
        the explicit form as dt -> 0.
    "explicit"
        The literal forward-Euler form of the paper.

    The default deviates from the paper's stated "forward Euler" deliberately.
    At the paper's own parameters (m = 0.02 kg over ~16 nodes, k_d = 14..30)
    the node damping timescale m_node/k_d is ~1e-4 s, so explicit integration
    needs dt < 2 m_node / k_d -- a bound that *tightens as k_d grows*. Since
    dataset generation sweeps k_d over a broad range, an explicit integrator
    would silently diverge on exactly the high-damping samples. The systems in
    this regime are strongly overdamped, where inertia is negligible and the
    implicit form converges to the same quasi-static trajectory.
    """

    def __init__(
        self,
        n_nodes: int,
        dt: float = 4.0e-5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        symplectic: bool = True,
        damping_mode: str = "implicit",
        eps: float = 1.0e-9,
    ) -> None:
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

    # ---- forces -----------------------------------------------------------

    def axial_forces(self, p: Tensor, k_s: Tensor, l0: Tensor) -> Tensor:
        """f^s: Hookean springs along each edge, rest length l0."""
        d = p[:, 1:] - p[:, :-1]                       # (B, N-1, 2)
        length = d.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        direction = d / length
        f_edge = k_s * (length - l0) * direction       # (B, N-1, 2)

        f = torch.zeros_like(p)
        f[:, :-1] += f_edge                            # pull node i towards i+1
        f[:, 1:] -= f_edge
        return f

    def turning_angles(self, p: Tensor) -> Tensor:
        """Signed angle between consecutive edges at each interior node."""
        u = p[:, 1:] - p[:, :-1]                       # (B, N-1, 2)
        u_prev, u_next = u[:, :-1], u[:, 1:]           # (B, N-2, 2)
        cross = u_prev[..., 0] * u_next[..., 1] - u_prev[..., 1] * u_next[..., 0]
        dot = (u_prev * u_next).sum(-1)
        return torch.atan2(cross, dot)                 # (B, N-2)

    def bending_energy(self, p: Tensor, k_b: Tensor,
                       rest_angle: Tensor | float = 0.0) -> Tensor:
        """E^b = sum_i 0.5 k_b (theta_i - theta_rest_i)^2 over interior nodes.

        rest_angle is (B, N-2): the wire's natural bend at each interior node.
        The difference is wrapped to (-pi, pi] so that a rest angle near +pi and
        a turning angle near -pi count as adjacent, not as a 2*pi error.
        """
        d = self.turning_angles(p) - rest_angle
        d = torch.atan2(torch.sin(d), torch.cos(d))
        return (0.5 * k_b.squeeze(-1) * d.pow(2)).sum()

    def bending_forces(self, p: Tensor, k_b: Tensor,
                       rest_angle: Tensor | float = 0.0) -> Tensor:
        """f^b = -dE^b/dp, via autograd on the energy.

        Deriving the torsional-spring force by hand is error-prone; autograd on
        the energy is exact by construction. `test_dlo_model.py` cross-checks it
        against finite differences. The graph is local to this call, so long
        rollouts do not accumulate one.
        """
        with torch.enable_grad():
            p_ = p.detach().requires_grad_(True)
            energy = self.bending_energy(p_, k_b, rest_angle)
            (grad,) = torch.autograd.grad(energy, p_)
        return -grad

    def elastic_forces(self, p: Tensor, prm: DLOParams) -> Tensor:
        """f^s + f^b -- the velocity-independent part of Eq. (1)."""
        l0 = prm.length / (self.n_nodes - 1)
        return (self.axial_forces(p, prm.k_s, l0)
                + self.bending_forces(p, prm.k_b, prm.rest_angle))

    def total_forces(self, p: Tensor, v: Tensor, prm: DLOParams) -> Tensor:
        """Right-hand side of Eq. (1): -k_d v + f^s + f^b."""
        return self.elastic_forces(p, prm) - prm.k_d * v

    # ---- integration ------------------------------------------------------

    def _apply_coulomb(self, v_new: Tensor, m_node: Tensor, prm: DLOParams) -> Tensor:
        """Dry (Coulomb) friction against the board, as a velocity-level impulse.

        The viscous term -k_d*v already in Eq. (1) is proportional to speed, so
        it vanishes as the wire slows and can never hold anything in place. The
        wire, though, LIES on the board: its friction is dry, so it has a
        roughly speed-independent magnitude and -- the part that matters here --
        a breakaway threshold. Below it the contact sticks and that stretch of
        wire simply does not move while the mover pulls at the far end, which is
        what these clips show constantly and what the viscous model cannot
        represent at all.

        Written as a force, -k_c*sign(v) is a trap: at v ~ 0 it flips sign every
        step and the node chatters (or explodes, since the magnitude does not
        shrink as dt does). So it is applied as an IMPULSE with a clamp instead.
        Friction may remove at most the momentum the node actually has:

            dv = min(dt*k_c/m_node, |v|)  against the direction of motion

        A node that would coast on keeps moving, minus a fixed speed decrement;
        a node whose motion friction can fully absorb is set exactly to rest and
        stays there until the elastic forces push it past the threshold again.
        Stiction is therefore reproduced without a separate stick/slip branch,
        and the scheme cannot overshoot into a reversal no matter how large k_c
        or dt are.

        k_c is a force per node [N]. Gravity is not modelled (the board is
        flat), so the normal load is implicit and k_c absorbs mu*N as one
        estimated quantity rather than two unidentifiable ones.
        """
        k_c = getattr(prm, "k_c", 0.0)
        if not torch.is_tensor(k_c):
            if k_c == 0.0:
                return v_new
            k_c = torch.as_tensor(k_c, dtype=v_new.dtype, device=v_new.device)
        elif not bool((k_c != 0).any()):
            return v_new
        # k_c and m_node arrive as (B,1,1) from to_batched, so dv broadcasts
        # against speed (B,N,1) without any reshaping.
        dv = self.dt * k_c / m_node
        speed = v_new.norm(dim=-1, keepdim=True)
        scale = (1.0 - dv / speed.clamp_min(self.eps)).clamp_min(0.0)
        return v_new * scale

    def step(
        self,
        p: Tensor,
        v: Tensor,
        prm: DLOParams,
        constraints: NodeConstraints | None = None,
        f_ext: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One Euler step of Eq. (1).

        `f_ext` (B, N, 2) adds a per-node external force to the elastic ones.
        Eq. (1) has no such term -- it exists so a caller can DRIVE the rod
        toward a target shape without pinning nodes outright, which is what
        PIEGraph's guidance step (Orozco et al., CoRL 2025, Eq. 2/4) needs:
        the network proposes a setpoint, external forces pull the rod at it,
        and the rod's own springs decide how much of that proposal is
        physically reachable. Pinning (`constraints`) cannot express that --
        it imposes positions exactly, overriding the physics rather than
        negotiating with it. Default None leaves the dynamics untouched.
        """
        m_node = prm.m / self.n_nodes
        f_el = self.elastic_forces(p, prm)
        if f_ext is not None:
            f_el = f_el + f_ext

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
            mask = constraints.mask.unsqueeze(-1)       # (B, N, 1)
            # Driven nodes: impose position, and set velocity to the realised
            # displacement so neighbours see a consistent motion.
            v_driven = (constraints.target - p) / self.dt
            p_new = torch.where(mask, constraints.target, p_new)
            v_new = torch.where(mask, v_driven, v_new)

        self._apply_plasticity(p_new, prm)
        return p_new, v_new

    def _apply_plasticity(self, p_new: Tensor, prm: DLOParams) -> None:
        """Bending plasticity: creep rest_angle toward the post-step curvature
        past a yield threshold. Mutates `prm.rest_angle` (rebinds the
        attribute to a new tensor; the caller's DLOParams sees the update
        since it is the same object) -- a no-op when prm.k_yield is None, so
        every caller that never sets it is exactly unaffected. See DLOParams'
        docstring for the mechanism and why it is here rather than in a
        separate solver.
        """
        if prm.k_yield is None:
            return
        # k_yield/k_creep arrive as (B,1,1) from to_batched (the same shape as
        # every other scalar DLOParams field); rest_angle/theta are (B, N-2).
        # Squeezing the trailing dim to (B,1) is required for these to
        # broadcast against (B, N-2) correctly -- left as (B,1,1) they would
        # broadcast against the WRONG axis (numpy/torch pad missing dims on
        # the left, not align them with an existing batch dim).
        k_yield = torch.as_tensor(prm.k_yield)
        k_yield = k_yield.squeeze(-1) if k_yield.ndim == 3 else k_yield
        k_creep = torch.as_tensor(prm.k_creep)
        k_creep = k_creep.squeeze(-1) if k_creep.ndim == 3 else k_creep
        theta = self.turning_angles(p_new)                       # (B, N-2)
        d = theta - prm.rest_angle
        d = torch.atan2(torch.sin(d), torch.cos(d))               # wrap to (-pi, pi]
        excess = torch.sign(d) * (d.abs() - k_yield).clamp_min(0.0)
        prm.rest_angle = prm.rest_angle + self.dt * k_creep * excess

    def stability_dt(self, prm: DLOParams) -> Tensor:
        """Largest stable dt, per batch item, as the min over both fast modes.

        Axial:   omega = sqrt(k_s / m_node)  ->  dt < 2 / omega
        Damping: dt < 2 m_node / k_d, and this is the binding one in the
                 paper's regime -- it is why `damping_mode="implicit"` is the
                 default. With implicit damping only the axial bound applies.
        """
        m_node = prm.m / self.n_nodes
        dt_axial = 2.0 / (prm.k_s / m_node).sqrt()
        if self.damping_mode == "implicit":
            return dt_axial.reshape(-1)
        dt_damp = 2.0 * m_node / prm.k_d.clamp_min(self.eps)
        return torch.minimum(dt_axial, dt_damp).reshape(-1)
