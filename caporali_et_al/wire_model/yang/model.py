"""IN-biLSTM of Yang et al. (RAS 2022, Sec. 3.2, Fig. 4), planar version.

Graph: the 16 tracked nodes of one cable as a chain, a directed edge in both
directions between neighbours. Relative ("rel") edge encoding, Eq. (11)
without the twist terms:
    e_ij = (p_i - p_j, pdot_i - pdot_j, a_i, a_j, [static node attributes i, j])
f_e (4-layer MLP, 150 units) per edge, summed per receiver (Eq. 4) -> k_i;
a 2-layer bi-directional LSTM (150 units) runs ALONG THE CHAIN over k_0..k_N
(Eqs. 6-9, zero initial states at both ends); the last-layer hidden states of
both directions are concatenated (not summed, as in the paper) and handed to
f_v = state encoder (4 layers) + effect predictor (4 layers, linear output),
which predicts the next state of every node (Eq. 5).

State per node: position + velocity in 2-D (the tracking is planar, so the
twist angle psi of the paper's PT encoding is unobservable and dropped). The
net outputs the displacement over one frame, dp = p_{t+1} - p_t, which fixes
both the next position and the next (finite-difference) velocity dp / dt.

Deviation (documented): every node also carries its REST turning angle (the
clip's frame-0 shape). The black box being imitated -- MuJoCo's cable -- is
stress-free in that shape, so without it the next state is not a function of
the inputs. The paper's AGX cable is straight at rest and needs no such input.

Actions: the measured motion of the two cable ends (end-node displacement and
end-edge rotation, 3 numbers), zero on interior nodes, plus an end flag.
"""
import torch
import torch.nn as nn


def mlp(n_in, n_hidden, n_out, n_layers=4, out_act=True):
    layers, d = [], n_in
    for i in range(n_layers):
        last = i == n_layers - 1
        layers.append(nn.Linear(d, n_out if last else n_hidden))
        if not last or out_act:
            layers.append(nn.ReLU())
        d = n_hidden
    return nn.Sequential(*layers)


def turning(P):
    """signed turning angles at the interior nodes, (..., N, 2) -> (..., N-2)"""
    u = P[..., 1:, :] - P[..., :-1, :]
    a, b = u[..., :-1, :], u[..., 1:, :]
    return torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))


def end_actions(P0, P1):
    """end-node displacement + end-edge rotation from frame P0 to P1, (B, N, 2) -> (B, N, 3); zero inside"""
    B, N, _ = P0.shape
    act = torch.zeros(B, N, 3, device=P0.device, dtype=P0.dtype)

    def ang(P, i, j):
        e = P[:, j] - P[:, i]
        return torch.atan2(e[:, 1], e[:, 0])
    for node, (i, j) in ((0, (0, 1)), (N - 1, (N - 2, N - 1))):
        act[:, node, :2] = P1[:, node] - P0[:, node]
        dth = ang(P1, i, j) - ang(P0, i, j)
        act[:, node, 2] = torch.atan2(torch.sin(dth), torch.cos(dth))
    return act


class INBiLSTM(nn.Module):
    def __init__(self, n_nodes=16, hidden=150, lstm_layers=2, use_rest=True):
        super().__init__()
        self.N, self.use_rest = n_nodes, use_rest
        n_static = 2 if use_rest else 1                  # [rest angle], end flag
        self.n_node = 2 + 3 + n_static                   # velocity, action, static
        n_edge = 2 + 2 + 2 * 3 + 2 * n_static            # dp, dv, a_i, a_j, static i, j
        self.f_e = mlp(n_edge, hidden, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.enc = mlp(self.n_node, hidden, hidden)
        self.pred = mlp(hidden + 2 * hidden, hidden, 2, out_act=False)
        # normalisation constants (set from the training data, saved with the weights)
        for name, dim in (("s_pos", 1), ("s_vel", 1), ("s_act", 3), ("s_out", 1)):
            self.register_buffer(name, torch.ones(dim))

    def forward(self, p, v, act, rest):
        """p, v: (B, N, 2) position [m], velocity [m/s]; act: (B, N, 3); rest: (B, N) rest
        turning angle (0 at the ends). Returns dp (B, N, 2) in metres."""
        B, N, _ = p.shape
        flag = torch.zeros(B, N, 1, device=p.device, dtype=p.dtype)
        flag[:, 0] = flag[:, -1] = 1.0
        static = torch.cat([rest[..., None], flag], -1) if self.use_rest else flag
        pn, vn, an = p / self.s_pos, v / self.s_vel, act / self.s_act
        # directed edges i <- j for j = i - 1 and j = i + 1
        r = torch.arange(N, device=p.device)
        recv = torch.cat([r[1:], r[:-1]]); send = torch.cat([r[:-1], r[1:]])
        e = torch.cat([pn[:, recv] - pn[:, send], vn[:, recv] - vn[:, send], an[:, recv], an[:, send],
                       static[:, recv], static[:, send]], -1)
        eff = self.f_e(e)                                                   # (B, E, H)
        k = torch.zeros(B, N, eff.shape[-1], device=p.device, dtype=p.dtype).index_add_(1, recv, eff)
        h, _ = self.lstm(k)                                                 # (B, N, 2H): [h^l, h^r]
        x = torch.cat([vn, an, static], -1)
        out = self.pred(torch.cat([self.enc(x), h], -1))
        return out * self.s_out
