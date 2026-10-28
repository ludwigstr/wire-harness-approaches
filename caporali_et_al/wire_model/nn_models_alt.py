"""Alternative surrogate architectures, benchmarked against the paper's MLP.

The paper's network (Fig. 3, `DLONetwork`) flattens the 16x2 configuration into
one 32-vector and feeds it to a dense block. That works, but it discards the
one thing we know for certain about the input: a DLO is a CHAIN. Node i is
coupled to i-1 and i+1 and to nothing else directly, the bending energy is a
local three-node term, and the same physics applies at every position along the
wire. A dense layer has to rediscover all of that from data, separately for
each node index, and it has no way to share what it learns at node 5 with
node 6.

Each architecture here keeps the sequence axis instead:

  DLOBiLSTM      recurrent both ways along the chain. A disturbance at the
                 grasped end propagates down the wire, which is literally a
                 sequence problem; bidirectional because the wire is pinned at
                 BOTH ends, so information flows from both.
  DLOConv1d      dilated 1-D convolutions along the chain. Weight-shared over
                 node index (the physics is position-invariant) and explicitly
                 local, with dilation to reach the whole wire in a few layers.
  DLOTransformer self-attention over nodes. Drops the locality assumption
                 entirely -- included as the control: if it wins, long-range
                 coupling matters more than the chain prior.

All three share the conditioning scheme and the interface of `DLONetwork`
(forward(V_in, a_n, p_n) -> dV), so `DLOPredictor`, the estimator and the
training loop take them unchanged.

Conditioning: the action and the scalar parameters are global, the rest-angle
profile is per-node. The globals are embedded once and injected as FiLM
(per-channel scale and shift) at every node, which is the sequence-model
counterpart of the multiplicative gating in the paper's Fig. 3; the rest angles
are simply concatenated to the node features, where they belong -- theta_rest,i
is a property of node i.

All predict dV and are zero-initialised at the head, exactly as the MLP is, so
every model starts from the identity and the comparison is not confounded by
initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class _Film(nn.Module):
    """Embed the global conditioning into per-channel (scale, shift)."""

    def __init__(self, n_global: int, width: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_global, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * width))
        self.width = width

    def forward(self, g: Tensor) -> tuple[Tensor, Tensor]:
        gs = self.net(g)
        scale, shift = gs[:, :self.width], gs[:, self.width:]
        # 1 + scale so an untrained FiLM is the identity rather than a
        # near-zero gate that would kill the signal before training starts
        return (1.0 + scale).unsqueeze(1), shift.unsqueeze(1)


class _NodeFeatures(nn.Module):
    """(V_in, rest) -> per-node feature sequence (B, N, C).

    The rest-angle vector has N-2 entries (interior nodes only); it is padded
    at both ends so it lines up with the node axis rather than being silently
    misaligned by one.
    """

    def __init__(self, n_nodes: int, n_rest: int, width: int):
        super().__init__()
        self.n_nodes, self.n_rest = n_nodes, n_rest
        self.proj = nn.Linear(3 if n_rest else 2, width)

    def forward(self, V_in: Tensor, rest: Tensor | None) -> Tensor:
        if self.n_rest:
            pad = torch.zeros(V_in.shape[0], 1, device=V_in.device,
                              dtype=V_in.dtype)
            r = torch.cat([pad, rest, pad], dim=1).unsqueeze(-1)
            x = torch.cat([V_in, r], dim=-1)
        else:
            x = V_in
        return self.proj(x)


class _AltBase(nn.Module):
    """Shared plumbing: split p_n into scalars and rest, build features, decode."""

    def __init__(self, n_nodes: int, width: int, n_action: int, n_params: int,
                 n_rest: int):
        super().__init__()
        self.n_nodes, self.width = n_nodes, width
        self.n_params, self.n_rest = n_params, n_rest
        self.n_action = n_action
        self.feat = _NodeFeatures(n_nodes, n_rest, width)
        self.film = _Film(n_action + n_params, width)
        self.head = nn.Linear(width, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _prepare(self, V_in: Tensor, a_n: Tensor, p_n: Tensor):
        scal, rest = p_n[:, :self.n_params], p_n[:, self.n_params:]
        h = self.feat(V_in, rest if self.n_rest else None)
        scale, shift = self.film(torch.cat([a_n, scal], dim=-1))
        return h * scale + shift

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        h = self._prepare(V_in, a_n, p_n)
        return self.head(self.body(h))


class DLOBiLSTM(_AltBase):
    """Bidirectional LSTM along the node chain."""

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 2):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.lstm = nn.LSTM(width, width // 2, num_layers=layers,
                            batch_first=True, bidirectional=True)

    def body(self, h: Tensor) -> Tensor:
        out, _ = self.lstm(h)
        return out


class DLOConv1d(_AltBase):
    """Dilated 1-D residual convolutions along the node chain."""

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14,
                 dilations=(1, 2, 4, 8)):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=d, dilation=d))
            for d in dilations])

    def body(self, h: Tensor) -> Tensor:
        x = h.transpose(1, 2)                     # (B, C, N) for conv1d
        for blk in self.blocks:
            x = torch.relu(x + blk(x))            # residual: keeps depth safe
        return x.transpose(1, 2)


class DLOTransformer(_AltBase):
    """Self-attention over nodes, with a learned positional embedding."""

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 3,
                 heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def body(self, h: Tensor) -> Tensor:
        return self.enc(h + self.pos)


class DLOTransformerHist(DLOTransformer):
    """DLOTransformer + a learned dynamics embedding from real recent history.

    DyWA (Lyu et al. 2025, "Dynamics-adaptive World Action Model") encodes a
    short history of real observation-action pairs into a "dynamics
    embedding" via a small trainable adaptation module, FiLM-conditioning the
    main network on it. Their ablation (Table II) shows this beats supplying
    physical parameters directly, and beats the module alone without also
    predicting outcomes -- the embedding is LEARNED end-to-end across many
    examples, not fit per-instance.

    This is the direct answer to a measured failure on this project: fitting
    (k_b, k_d, k_c) per cable via gradient descent on 6-8 real calibration
    frames overfit badly (fitted worse than the fixed NOMINAL on 38/76 cables,
    see long_horizon_fitp.py). A learned encoder, trained across ALL clips,
    should generalise from a short noisy history instead of overfitting to it.

    Each history step is summarised rotation/translation-invariantly (turning
    angles need no canonicalisation, see nn_model.py's rest_angle docstring)
    as [canonical action (8), turning_angles(V_in) (n_rest), turning_angles
    (V_out) (n_rest)] -- what happened, and how the cable actually responded.
    A per-step MLP embeds each entry, a mask-aware mean pools over the
    (padded) history length, and a final linear layer produces the z_dim
    embedding. Its FiLM head is zero-initialised, so a freshly grafted model
    (old weights loaded, this branch random-but-zeroed) starts IDENTICAL to
    the no-history model -- fine-tuning has to earn any deviation from it.
    """

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                n_params: int = 4, n_rest: int = 14, layers: int = 3,
                heads: int = 4, hist_dim: int | None = None,
                hist_hidden: int = 64, z_dim: int = 16):
        super().__init__(n_nodes, width, n_action, n_params, n_rest, layers, heads)
        self.z_dim = z_dim
        self.hist_dim = hist_dim if hist_dim is not None else n_action + 2 * n_rest
        self.hist_mlp = nn.Sequential(
            nn.Linear(self.hist_dim, hist_hidden), nn.ReLU(),
            nn.Linear(hist_hidden, hist_hidden), nn.ReLU())
        self.hist_out = nn.Linear(hist_hidden, z_dim)
        self.film_hist = _Film(z_dim, width)
        # Zero-initialised so an untrained (or history-less) forward pass is
        # exactly identical to the plain DLOTransformer -- see class docstring.
        nn.init.zeros_(self.film_hist.net[-1].weight)
        nn.init.zeros_(self.film_hist.net[-1].bias)

    def embed_history(self, hist: Tensor, hist_mask: Tensor) -> Tensor:
        """hist (B, L, hist_dim), hist_mask (B, L) 1=real/0=pad -> (B, z_dim)."""
        he = self.hist_mlp(hist)                                  # (B, L, H)
        m = hist_mask.unsqueeze(-1)                                # (B, L, 1)
        pooled = (he * m).sum(1) / m.sum(1).clamp_min(1.0)         # (B, H)
        return self.hist_out(pooled)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor,
               hist: Tensor | None = None, hist_mask: Tensor | None = None) -> Tensor:
        h = self._prepare(V_in, a_n, p_n)
        if hist is not None:
            z = self.embed_history(hist, hist_mask)
            scale, shift = self.film_hist(z)
            h = h * scale + shift
        return self.head(self.body(h))


class _GNNLayer(nn.Module):
    """One round of chain-graph message passing: each node sends a message to
    its two neighbours (i-1, i+1), messages are summed, and a node update MLP
    combines the aggregate with the node's own state. Residual, so depth is
    safe and an untrained layer starts close to identity."""

    def __init__(self, width: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width))
        self.upd = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, h: Tensor) -> Tensor:
        left = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)     # edge i -> i+1
        right = torch.cat([h[:, 1:], h[:, :-1]], dim=-1)    # edge i+1 -> i
        m_fwd, m_bwd = self.msg(left), self.msg(right)
        agg = torch.zeros_like(h)
        agg[:, :-1] += m_bwd     # node i receives from i+1
        agg[:, 1:] += m_fwd      # node i receives from i-1
        return h + self.upd(torch.cat([h, agg], dim=-1))


class DLOGNN(_AltBase):
    """Explicit message-passing GNN over the chain graph (node i <-> i-1, i+1
    only), the literal graph-neural-network counterpart to DLOConv1d's
    weight-shared local convolutions and DLOTransformer's all-pairs attention.

    Where DLOConv1d gets locality "for free" from a shared convolution kernel,
    this builds it from the graph-network primitives (message function, sum
    aggregation, node update, both learned) -- the same recipe the survey's
    "GNN-based IN-biLSTM" baseline uses in spirit (see DLOJacobian's
    docstring), without the recurrent stage. `layers` rounds of message
    passing give an (layers)-hop receptive field; 4 layers reaches end to end
    on the 16-node chain, same as DLOConv1d's dilation schedule.
    """

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.layers = nn.ModuleList([_GNNLayer(width) for _ in range(layers)])

    def body(self, h: Tensor) -> Tensor:
        for layer in self.layers:
            h = layer(h)
        return h


def _turning_angles(p: Tensor) -> Tensor:
    """(B, N, 2) -> (B, N-2) signed turning angle at each interior node.
    Same four lines as nn_model._turning_angles_batch, kept local so this
    module stays free of that import."""
    u = p[:, 1:] - p[:, :-1]
    cross = u[:, :-1, 0] * u[:, 1:, 1] - u[:, :-1, 1] * u[:, 1:, 0]
    dot = (u[:, :-1] * u[:, 1:]).sum(-1)
    return torch.atan2(cross, dot)


class DLOTransformerSeq(_AltBase):
    """The transformer with a PHYSICAL per-node input: relations and history.

    Two objections to DLOTransformer (user, 2026-10-20) drive this design.
    First, its node feature is a position in the canonical frame, although
    the physics of a chain is relational -- bending is a three-node term,
    stretch a two-node term, the same at every joint. Second, and worse, it
    is Markov in the configuration: the simulator's state is (position,
    velocity), 0.4 s of settling leaves a residual velocity in every V_in,
    and on the clips a 1 s step sees the cables mid-swing -- but the network
    never sees how fast anything was moving.

    Per node i the feature is therefore
      position (2)            canonical frame, kept so the drive at the
                              terminal edges stays locatable
      edge to i-1, to i+1 (4) zero at the ends
      turning angle (1)       the current bend, zero at the ends
      rest angle (1)          the bend it would return to, zero at the ends
      k_b,i (1)               ITS OWN bending stiffness -- a node property,
                              not a global FiLM vector the network has to
                              un-mix by column index
      history (2 x n_hist)    the n_hist preceding frames as displacements
                              from the current position, oldest first:
                              velocity and acceleration, made observable
    The globals stay global: action (8) and [m, k_d, k_c] via FiLM.
    Predicts dV like every other model here, head zero-initialised.
    """
    takes_node_hist = True

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 17, n_rest: int = 14, n_hist: int = 4,
                 layers: int = 3, heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.n_hist = n_hist
        self.n_kb = n_params - 3            # [m, k_b x n_kb, k_d, k_c]
        assert self.n_kb in (1, n_rest), (n_params, n_rest)
        n_feat = 2 + 4 + 1 + 1 + 1 + 2 * n_hist
        self.feat = nn.Linear(n_feat, width)          # replaces _NodeFeatures
        self.film = _Film(n_action + 3, width)        # replaces the n_params FiLM
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    @staticmethod
    def _pad_ends(x: Tensor) -> Tensor:
        """(B, N-2) interior quantity -> (B, N, 1) on the node axis."""
        z = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([z, x, z], dim=1).unsqueeze(-1)

    def node_features(self, V_in: Tensor, p_n: Tensor,
                      hist: Tensor | None) -> Tensor:
        B, N, _ = V_in.shape
        m_n = p_n[:, 0:1]
        kb_n = p_n[:, 1:1 + self.n_kb]
        kd_kc = p_n[:, 1 + self.n_kb:3 + self.n_kb]
        rest = p_n[:, 3 + self.n_kb:3 + self.n_kb + self.n_rest]
        if self.n_kb == 1:
            kb_n = kb_n.expand(B, self.n_rest)
        e = V_in[:, 1:] - V_in[:, :-1]                            # (B, N-1, 2)
        z2 = torch.zeros(B, 1, 2, device=V_in.device, dtype=V_in.dtype)
        e_prev = torch.cat([z2, e], dim=1)                        # edge into node i
        e_next = torch.cat([e, z2], dim=1)                        # edge out of node i
        theta = self._pad_ends(_turning_angles(V_in))
        if hist is None:                                          # standing still
            hist = torch.zeros(B, self.n_hist, N, 2, device=V_in.device,
                               dtype=V_in.dtype)
        h = hist.permute(0, 2, 1, 3).reshape(B, N, 2 * self.n_hist)
        x = torch.cat([V_in, e_prev, e_next, theta, self._pad_ends(rest),
                       self._pad_ends(kb_n), h], dim=-1)
        return x, torch.cat([m_n, kd_kc], dim=-1)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor,
               hist: Tensor | None = None) -> Tensor:
        x, glob = self.node_features(V_in, p_n, hist)
        h = self.feat(x)
        scale, shift = self.film(torch.cat([a_n, glob], dim=-1))
        h = h * scale + shift
        return self.head(self.enc(h + self.pos))


ARCHITECTURES = {
    "bilstm": DLOBiLSTM,
    "conv1d": DLOConv1d,
    "transformer": DLOTransformer,
    "transformer_hist": DLOTransformerHist,
    "gnn": DLOGNN,
}


class DLOJacobian(_AltBase):
    """Deformation Jacobian: dV = J(state) @ a, exactly linear in the action.

    From Gu et al. (Neurocomputing 696, 2026), Sec. 3.2.1. The Jacobian
    formulation is reported there to beat MLP, biLSTM and GNN-based IN-biLSTM
    on the same offline data (1.5 cm vs ~2.0 cm at 60k samples) -- not through
    capacity, but through a prior none of those architectures carry: at a fixed
    configuration, the map from end-effector motion to cable deformation is
    LINEAR. Every other model here is free to be arbitrarily non-linear in the
    action and has to learn that structure from data.

    So the network never sees the action. It reads the configuration and the
    material parameters and emits a per-node 2x6 block; the action is then
    applied by matrix multiplication. The 6 continuous action components are
    (dx, dy, dtheta) for each of the two driven ends. The alpha columns are
    edge INDICES, not quantities, and are dropped rather than multiplied --
    they are constant in this setup anyway (terminal edges at both ends).

    The survey lists "only models small deformation" as the standing limitation
    of this family. That is a real risk here, since a 2 s forecast moves a plug
    up to 148 mm, and it is exactly what the benchmark has to settle.
    """

    N_CONT = 3          # (dx, dy, dtheta) per driven end

    def __init__(self, n_nodes: int = 16, width: int = 128, n_action: int = 8,
                 n_params: int = 4, n_rest: int = 14, layers: int = 3,
                 heads: int = 4):
        super().__init__(n_nodes, width, n_action, n_params, n_rest)
        self.n_ends = max(1, n_action // 4)
        self.n_cols = self.n_ends * self.N_CONT
        self.pos = nn.Parameter(torch.zeros(1, n_nodes, width))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            batch_first=True, norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        # emits the Jacobian block per node instead of a displacement
        self.head = nn.Linear(width, 2 * self.n_cols)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _state_film(self, p_n: Tensor) -> tuple[Tensor, Tensor]:
        """Condition on the parameters only -- the action is applied later."""
        scal = p_n[:, :self.n_params]
        pad = torch.zeros(scal.shape[0], self.n_action, device=scal.device,
                          dtype=scal.dtype)
        return self.film(torch.cat([pad, scal], dim=-1))

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        rest = p_n[:, self.n_params:]
        h = self.feat(V_in, rest if self.n_rest else None)
        scale, shift = self._state_film(p_n)
        h = self.enc(h * scale + shift + self.pos)

        B, N = V_in.shape[0], self.n_nodes
        J = self.head(h).view(B, N, 2, self.n_cols)
        # drop the alpha column of each 4-wide action block
        cont = torch.cat([a_n[:, c + 1:c + 4]
                          for c in range(0, a_n.shape[1], 4)], dim=-1)
        return torch.einsum("bnij,bj->bni", J, cont)


ARCHITECTURES["jacobian"] = DLOJacobian
ARCHITECTURES["transformer_seq"] = DLOTransformerSeq
