"""PhysWorld GNN world model (paper Sec. 3.3, Eq. 6-8) on the 16-node cable graph.

X_{t+1} = X_t + f_theta(X_{t-h:t}, a_t, Phi)
  vertices : the 16 tracked nodes of one cable; the two end nodes are the controller
             (grasp) vertices, the 14 others object vertices
  vertex encoder input (no positions, translation equivariant): type one-hot,
             action a_t of controller vertices (displacement of the grasp to the next
             frame + yaw change of its module; zero on object vertices), and the
             per-vertex physical properties Phi_i = (log E_i, log rho_i), normalised
  edges    : bidirectional, every pair within d_e = 0.12 m plus chain neighbours;
             edge encoder input = relative positions x_i - x_j at t-h..t ("history
             distance") + edge type one-hot (object-object / object-controller)
  propagator: p = 7 message-passing steps with unshared MLPs (paper p = 7)
  decoder  : per-vertex displacement; controllers are set by the action.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from wire_model.adaptigraph.model import mlp

SCALE = 0.1          # m per feature unit for positions
ACT = 0.01           # m per feature unit for actions
YAW = 0.1            # rad per feature unit


class PhysWorldGNN(nn.Module):
    def __init__(self, n_nodes=16, h=3, width=128, steps=7, radius=0.12, n_phi=2):
        super().__init__()
        self.N, self.h, self.radius, self.steps = n_nodes, h, radius, steps
        d_v = 2 + 3 + n_phi
        d_e = 2 * (h + 1) + 2
        self.enc_v, self.enc_e = mlp(d_v, width, width, 1), mlp(d_e, width, width, 1)
        self.prop_e = nn.ModuleList([mlp(3 * width, width, width, 1) for _ in range(steps)])
        self.prop_v = nn.ModuleList([mlp(2 * width, width, width, 1) for _ in range(steps)])
        self.dec = mlp(width, width, 2, 1)
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)
        ctrl = torch.zeros(n_nodes, dtype=torch.bool); ctrl[0] = True; ctrl[-1] = True
        self.register_buffer("ctrl", ctrl)

    def forward(self, hist: Tensor, X: Tensor, phi: Tensor, dp: Tensor, dyaw: Tensor) -> Tensor:
        """hist (B, h, N, 2) previous frames (oldest first); X (B, N, 2) current; phi (B, N, 2)
        normalised; dp (B, N, 2) displacement of the controller vertices to the next frame (zero
        elsewhere); dyaw (B, N) yaw change of the controller's module (zero elsewhere)
        -> X_{t+1} (B, N, 2) with the controllers set by the action"""
        B, N, _ = X.shape
        allx = torch.cat([hist, X.unsqueeze(1)], 1)                                   # (B, h+1, N, 2)
        rel = (allx.unsqueeze(3) - allx.unsqueeze(2)) / SCALE                          # (B, h+1, N, N, 2) x_i - x_j
        rel = rel.permute(0, 2, 3, 1, 4).reshape(B, N, N, -1)
        d = torch.cdist(X, X)
        adj = d < self.radius
        ar = torch.arange(N - 1, device=X.device)
        chain = torch.zeros(N, N, dtype=torch.bool, device=X.device); chain[ar, ar + 1] = True; chain[ar + 1, ar] = True
        adj = (adj | chain) & ~torch.eye(N, dtype=torch.bool, device=X.device)
        typ = torch.stack([~self.ctrl, self.ctrl], -1).to(X.dtype).unsqueeze(0).expand(B, N, 2)
        oc = (self.ctrl.unsqueeze(0) | self.ctrl.unsqueeze(1)).to(X.dtype)            # edge touches a controller
        etype = torch.stack([1 - oc, oc], -1).unsqueeze(0).expand(B, N, N, 2)
        act = torch.cat([dp / ACT, (dyaw / YAW).unsqueeze(-1)], -1) * self.ctrl.view(1, N, 1)
        hv = self.enc_v(torch.cat([typ, act, phi], -1))
        m = adj.unsqueeze(-1).to(X.dtype)
        he = self.enc_e(torch.cat([rel, etype], -1)) * m
        for l in range(self.steps):
            he = he + self.prop_e[l](torch.cat([he, hv.unsqueeze(2).expand(-1, -1, N, -1),
                                                hv.unsqueeze(1).expand(-1, N, -1, -1)], -1)) * m
            hv = hv + self.prop_v[l](torch.cat([hv, he.sum(2)], -1))
        out = X + self.dec(hv) * SCALE * 0.1
        return torch.where(self.ctrl.view(1, N, 1), X + dp, out)
