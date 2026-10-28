"""Action parametrisation and rollout (Caporali et al., Sec. II-A).

The manipulation action is a pick-and-place on one *edge* of the DLO:

    a = [alpha, dx, dy, dtheta]

alpha indexes the grasped edge {p_alpha, p_alpha+1}; dx, dy translate it and
dtheta rotates it about its midpoint. The grasped edge is driven kinematically,
the rest of the chain answers through Eq. (1).

In the wire-harness videos the "gripper" is the human hand moving a connector,
and the grasped edge is the wire's terminal edge -- so alpha is fixed to the
terminal index and (dx, dy, dtheta) come from tracking the connector pose.
`static_mask` holds the opposite connector fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dlo_model import DLOParams, DLOSimulator, NodeConstraints

Tensor = torch.Tensor


@dataclass
class Action:
    """Pick-and-place action, batched over B samples.

    alpha   (B,) long   -- index of the grasped edge, in [0, N-2]
    dx, dy  (B,)        -- translation applied to the grasped edge [m]
    dtheta  (B,)        -- rotation about the edge midpoint [rad]
    """

    alpha: Tensor
    dx: Tensor
    dy: Tensor
    dtheta: Tensor

    def as_vector(self) -> Tensor:
        """(B, 4) tensor [alpha, dx, dy, dtheta], alpha cast to float."""
        return torch.stack(
            [self.alpha.to(self.dx.dtype), self.dx, self.dy, self.dtheta], dim=-1
        )

    @staticmethod
    def from_vector(vec: Tensor) -> "Action":
        return Action(
            alpha=vec[:, 0].round().long(),
            dx=vec[:, 1],
            dy=vec[:, 2],
            dtheta=vec[:, 3],
        )

    def to(self, device) -> "Action":
        return Action(
            self.alpha.to(device), self.dx.to(device),
            self.dy.to(device), self.dtheta.to(device),
        )


def grasped_node_targets(p: Tensor, action: Action) -> tuple[Tensor, Tensor]:
    """Where the two grasped nodes must end up.

    Returns (idx, target) with idx (B, 2) the node indices {alpha, alpha+1}
    and target (B, 2, 2) their commanded positions.
    """
    batch = p.shape[0]
    ar = torch.arange(batch, device=p.device)
    idx = torch.stack([action.alpha, action.alpha + 1], dim=-1)      # (B, 2)

    pa = p[ar, action.alpha]                                          # (B, 2)
    pb = p[ar, action.alpha + 1]                                      # (B, 2)
    centre = 0.5 * (pa + pb)                                          # (B, 2)

    c, s = torch.cos(action.dtheta), torch.sin(action.dtheta)
    rot = torch.stack(
        [torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2
    )                                                                 # (B, 2, 2)

    edge = torch.stack([pa, pb], dim=1) - centre.unsqueeze(1)         # (B, 2, 2)
    rotated = torch.einsum("bij,bkj->bki", rot, edge)
    shift = torch.stack([action.dx, action.dy], dim=-1)               # (B, 2)
    target = rotated + centre.unsqueeze(1) + shift.unsqueeze(1)
    return idx, target


def rollout(
    sim: DLOSimulator,
    p0: Tensor,
    prm: DLOParams,
    action: Action,
    static_mask: Tensor | None = None,
    n_move: int = 4000,
    n_settle: int = 8000,
    return_traj: bool = False,
    action2: "Action | None" = None,
    record_at: "list[int] | None" = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Simulate one action to rest and return the final configuration.

    The grasped edge is ramped linearly to its target over `n_move` steps
    (a hand/gripper moves gradually; snapping it there injects a shock the
    real system never sees), then held for `n_settle` steps so the chain can
    come to rest under damping.

    `action2` drives a SECOND edge at the same time. The paper's rope has one
    grasped end and one free end, but a harness wire is plugged in at both, and
    in these clips the hub connector travels a median 89 mm per 2 s -- more than
    the mover it is being predicted against. Treating it as static, or folding
    it away by working in a hub-fixed frame, therefore discards the larger of
    the two drives.

    Folding it away used to be exact: with only springs and viscous damping the
    dynamics are translation-invariant, so a moving hub frame changes nothing.
    Dry friction breaks that -- it acts against the BOARD, which is a fixed
    frame -- so once k_c is in the model the hub must be driven explicitly.

    p0          (B, N, 2)  initial configuration
    static_mask (B, N) bool -- nodes held at their p0 position. May overlap with
                a grasped edge only if you intend the grasp to win.

    Returns final (B, N, 2), or (final, traj) with traj (B, T, N, 2) if
    `return_traj`, sampled every `max(1, (n_move+n_settle)//200)` steps.

    `record_at`: exact step counts at which to record the configuration
    (the state AFTER that many steps), returned as (final, recs) with recs
    (B, len(record_at), N, 2). The sequence dataset uses it to sample a
    1 s action at the video's frame cadence, so a transition there is one
    frame and the frames before it are the history the network sees.
    """
    batch, n_nodes, _ = p0.shape
    device = p0.device
    ar = torch.arange(batch, device=device)

    drives = [grasped_node_targets(p0, action)]
    if action2 is not None:
        drives.append(grasped_node_targets(p0, action2))

    grasp_mask = torch.zeros(batch, n_nodes, dtype=torch.bool, device=device)
    starts = []
    for idx, _ in drives:
        grasp_mask[ar.unsqueeze(-1), idx] = True
        starts.append(torch.stack([p0[ar, idx[:, 0]], p0[ar, idx[:, 1]]], dim=1))

    if static_mask is None:
        static_mask = torch.zeros_like(grasp_mask)
    # the grasp wins wherever the two overlap
    hold_mask = static_mask & ~grasp_mask
    full_mask = hold_mask | grasp_mask

    p, v = p0.clone(), torch.zeros_like(p0)
    total = n_move + n_settle
    stride = max(1, total // 200)
    traj = []
    record_set = set(record_at) if record_at is not None else set()
    recs = []

    for t in range(total):
        frac = min(1.0, (t + 1) / max(n_move, 1))
        full_target = p0.clone()                                # holds static nodes
        for (idx, target), start in zip(drives, starts):
            full_target[ar.unsqueeze(-1), idx] = start + frac * (target - start)

        p, v = sim.step(p, v, prm, NodeConstraints(full_mask, full_target))
        if return_traj and t % stride == 0:
            traj.append(p.clone())
        if record_at is not None and (t + 1) in record_set:
            recs.append(p.clone())

    if record_at is not None:
        return p, torch.stack(recs, dim=1)
    if return_traj:
        return p, torch.stack(traj, dim=1)
    return p


def terminal_edge_action(
    n_nodes: int,
    dx: Tensor,
    dy: Tensor,
    dtheta: Tensor,
    at_end: bool = True,
) -> Action:
    """Action that grasps a terminal edge -- the harness-video case.

    at_end=True grasps edge N-2 (nodes N-2, N-1); False grasps edge 0.
    """
    alpha_val = n_nodes - 2 if at_end else 0
    alpha = torch.full_like(dx, alpha_val, dtype=torch.long)
    return Action(alpha=alpha, dx=dx, dy=dy, dtheta=dtheta)
