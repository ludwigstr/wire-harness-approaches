"""AdaptiGraph's material-conditioned GBND (paper Sec. III-B), on the cable graph.

Vertex i at time t: history positions over h = 3 steps (as displacements
from the current position, which is what makes the model translation-
equivariant), the particle type (object / end-effector), the material type
(one here: rope) and the physical property variable phi. The two driven end
nodes are the "end-effector" particles; their displacement to the next frame
(the action) is a vertex feature of theirs, zero on object particles.
Edges: every pair within the connection radius d (chain neighbours always),
feature = relative position receiver - sender + edge type. Encoders -> L
rounds of edge/vertex propagation (message passing) -> vertex decoder ->
displacement of every object particle (end nodes are set by the action).

phi for a rope = normalised stiffness in (0, 1): here the object's mean
log-stiffness mapped from the simulator's sampled range [0.004, 1.0]. At
test time phi is estimated by minimising the prediction error over a few
observed interactions (paper Sec. III-C: BO / CMA-ES on phi within
[phi_min - 0.2 (phi_max - phi_min), phi_max + 0.2 (...)] = [-0.2, 1.2]).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

SCALE = 0.1                       # metres per feature unit


def mlp(d_in, d_h, d_out, n=2):
    layers, d = [], d_in
    for _ in range(n):
        layers += [nn.Linear(d, d_h), nn.ReLU()]; d = d_h
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class AdaptiGraph(nn.Module):
    def __init__(self, n_nodes=16, n_hist=3, width=128, layers=4, radius=0.12, n_material=1):
        super().__init__()
        self.n_nodes, self.n_hist, self.radius, self.L = n_nodes, n_hist, radius, layers
        d_v = 2 * n_hist + 2 + n_material + 1 + 2          # hist disp, type one-hot, material, phi, action disp
        d_e = 2 + 2                                          # relative position, edge type one-hot
        self.enc_v, self.enc_e = mlp(d_v, width, width, 1), mlp(d_e, width, width, 1)
        self.prop_e = nn.ModuleList([mlp(3 * width, width, width, 1) for _ in range(layers)])
        self.prop_v = nn.ModuleList([mlp(2 * width, width, width, 1) for _ in range(layers)])
        self.dec = mlp(width, width, 2, 1)
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)   # start at "no change"
        self.register_buffer("eef", torch.zeros(n_nodes, dtype=torch.bool)); self.eef[0] = True; self.eef[-1] = True

    def edges(self, X: Tensor):
        """pairs (receiver, sender) within the radius, chain neighbours always; (B, E, 2) index tensors + mask"""
        B, N, _ = X.shape
        d = torch.cdist(X, X)                                                     # (B, N, N)
        adj = d < self.radius
        chain = torch.zeros(N, N, dtype=torch.bool, device=X.device)
        ar = torch.arange(N - 1, device=X.device); chain[ar, ar + 1] = True; chain[ar + 1, ar] = True
        adj = adj | chain
        adj &= ~torch.eye(N, dtype=torch.bool, device=X.device)
        return adj                                                                # (B, N, N) receiver i, sender j

    def forward(self, hist: Tensor, X: Tensor, phi: Tensor, dp_eef: Tensor, material: Tensor | None = None) -> Tensor:
        """hist (B, n_hist, N, 2) preceding frames; X (B, N, 2) current; phi (B,) in (0,1);
        dp_eef (B, N, 2) displacement of the end-effector nodes to the next frame (zero elsewhere)
        -> predicted next positions (B, N, 2); the end nodes are set exactly by the action."""
        B, N, _ = X.shape
        hd = ((hist - X.unsqueeze(1)) / SCALE).permute(0, 2, 1, 3).reshape(B, N, -1)
        typ = F.one_hot(self.eef.long(), 2).to(X.dtype).unsqueeze(0).expand(B, N, 2)
        mat = torch.ones(B, N, 1, device=X.device, dtype=X.dtype) if material is None else material
        ph = phi.view(B, N, 1) if phi.dim() == 2 else phi.view(B, 1, 1).expand(B, N, 1)   # per-node (PhysWorld) or global (AdaptiGraph)
        v = torch.cat([hd, typ, mat, ph, dp_eef / SCALE], -1)
        adj = self.edges(X)
        rel = (X.unsqueeze(2) - X.unsqueeze(1)) / SCALE                              # (B, N, N, 2) receiver - sender
        same = (self.eef.unsqueeze(0) == self.eef.unsqueeze(1)).to(X.dtype)         # (N, N)
        etype = torch.stack([same, 1 - same], -1).unsqueeze(0).expand(B, N, N, 2)
        he = self.enc_e(torch.cat([rel, etype], -1)) * adj.unsqueeze(-1)            # (B, N, N, W)
        hv = self.enc_v(v)                                                          # (B, N, W)
        for l in range(self.L):
            he = he + self.prop_e[l](torch.cat([he, hv.unsqueeze(2).expand(-1, -1, N, -1), hv.unsqueeze(1).expand(-1, N, -1, -1)], -1)) * adj.unsqueeze(-1)
            hv = hv + self.prop_v[l](torch.cat([hv, he.sum(2)], -1))
        dx = self.dec(hv) * SCALE
        out = X + dx
        return torch.where(self.eef.view(1, N, 1), X + dp_eef, out)
