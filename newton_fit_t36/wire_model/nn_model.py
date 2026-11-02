"""NN surrogate of the DLO dynamics (Caporali et al., Sec. II-B, Fig. 3).

Four blocks of Linear+ReLU -- DLO block, action block, parameters block,
prediction block. The DLO block embeds the flattened initial configuration;
the action and parameter blocks embed a = [alpha, dx, dy, dtheta] and
p = [m, k_b, k_d]; their embeddings are combined and decoded into the change
in node coordinates:

    V_pred = F(V_in, a, p) = dV(V_in, a, p) + V_in

Predicting the *delta* rather than absolute coordinates is the paper's design
and matters: it makes the identity (do nothing) the zero output, so the network
spends its capacity on the deformation instead of re-encoding the input.

Why a surrogate at all: it is >1 order of magnitude faster than the analytical
model at inference, and -- the real reason -- it is cheaply differentiable with
respect to *both* the action and the parameters, which is what makes the
gradient-based estimation of Sec. II-C possible (see estimation.py).

The paper's Fig. 3 shows the action and parameter embeddings joining the DLO
embedding by concatenation and element-wise multiplication. The figure marks
element-wise products on the prediction-block skip paths; `combine="film"`
reproduces that conditioning style (a multiplicative gate per feature, in the
spirit of FiLM) and `combine="concat"` is the plain alternative. Both are
provided because the figure is not unambiguous on this point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class Normalizer:
    """Feature scaling to roughly [0, 1] / [-1, 1], per the paper's Sec. II-B2.

    Positions are handled separately by `canonicalize`, which removes the
    translation/rotation symmetry. This only rescales actions and parameters,
    whose raw units (k_d ~ 40 vs m ~ 0.02) differ by three orders of magnitude
    and would otherwise let k_d dominate the first layer purely by scale.

    `param_log` marks parameters normalised in log space. k_b needs it: it is
    sampled log-uniformly over two decades, so linear normalisation would pile
    most of the data into the bottom few percent of the range and leave the
    gradient-based estimator of Sec. II-C badly conditioned exactly where the
    real ropes sit (the paper's own k_b estimates are ~0.06-0.19 out of a
    [0.01, 1.0] range).
    """

    action_lo: Tensor
    action_hi: Tensor
    param_lo: Tensor
    param_hi: Tensor
    pos_scale: float = 1.0
    param_log: Tensor | None = None
    rest_scale: float = 0.35   # |theta_rest| bound; see SamplingRanges.rest_amp

    def _fwd(self, p: Tensor) -> Tensor:
        return torch.where(self.param_log, p.clamp_min(1e-12).log(), p)

    def _lo_hi(self) -> tuple[Tensor, Tensor]:
        return self._fwd(self.param_lo), self._fwd(self.param_hi)

    def norm_action(self, a: Tensor) -> Tensor:
        return 2 * (a - self.action_lo) / (self.action_hi - self.action_lo) - 1

    def norm_params(self, p: Tensor) -> Tensor:
        if self.param_log is None:
            return (p - self.param_lo) / (self.param_hi - self.param_lo)
        lo, hi = self._lo_hi()
        return (self._fwd(p) - lo) / (hi - lo)

    def denorm_params(self, p_n: Tensor) -> Tensor:
        if self.param_log is None:
            return p_n * (self.param_hi - self.param_lo) + self.param_lo
        lo, hi = self._lo_hi()
        raw = p_n * (hi - lo) + lo
        return torch.where(self.param_log, raw.exp(), raw)

    def norm_rest(self, r: Tensor) -> Tensor:
        return r / self.rest_scale

    def to(self, device) -> "Normalizer":
        return Normalizer(
            self.action_lo.to(device), self.action_hi.to(device),
            self.param_lo.to(device), self.param_hi.to(device), self.pos_scale,
            None if self.param_log is None else self.param_log.to(device),
            self.rest_scale)


def canonicalize(V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Remove the translation/rotation symmetry of a DLO configuration.

    The paper (Sec. II-B2) normalises by "finding a transformation that makes
    V_in aligned to the x-axis and mean-centered". Concretely: centre on the
    centroid, then rotate so the first-to-last node chord lies along +x.

    This is data augmentation done exactly rather than by sampling -- the
    dynamics are equivariant to rigid motions, so a network that must rediscover
    that from random rotations wastes capacity on a symmetry we can quotient out
    for free.

    Returns (V_canon, centre, angle) so the prediction can be mapped back.
    """
    centre = V.mean(dim=1, keepdim=True)                    # (B, 1, 2)
    Vc = V - centre
    chord = Vc[:, -1] - Vc[:, 0]                            # (B, 2)
    angle = torch.atan2(chord[:, 1], chord[:, 0])           # (B,)
    return rotate(Vc, -angle), centre, angle


def rotate(V: Tensor, angle: Tensor) -> Tensor:
    """Rotate (B, N, 2) by per-sample angle (B,)."""
    c, s = torch.cos(angle), torch.sin(angle)
    rot = torch.stack([torch.stack([c, -s], -1),
                       torch.stack([s, c], -1)], dim=-2)     # (B, 2, 2)
    return torch.einsum("bij,bkj->bki", rot, V)


def rotate_vec(v: Tensor, angle: Tensor) -> Tensor:
    """Rotate (B, 2) by per-sample angle (B,)."""
    return rotate(v.unsqueeze(1), angle).squeeze(1)


def _mlp(sizes: list[int], out_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_act or i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DLONetwork(nn.Module):
    """F(V_in, a, p) -> V_pred, per Fig. 3.

    n_nodes  nodes in the DLO
    width    hidden width (256 in the paper)
    combine  "film" (element-wise gating, per Fig. 3) or "concat"
    """

    def __init__(
        self,
        n_nodes: int = 16,
        width: int = 256,
        n_action: int = 8,
        n_params: int = 4,
        n_rest: int = 0,
        combine: str = "film",
    ) -> None:
        super().__init__()
        if combine not in ("film", "concat"):
            raise ValueError("combine must be 'film' or 'concat'")
        self.n_nodes, self.width, self.combine = n_nodes, width, combine
        self.n_params = n_params      # read back by the estimator
        self.n_action = n_action

        self.n_rest = n_rest
        self.dlo_block = _mlp([n_nodes * 2, width, width, width])
        self.action_block = _mlp([n_action, width, width])
        # The rest-angle profile enters through the parameters block, not as a
        # separate branch: it IS a physical parameter of the wire, on the same
        # footing as k_b (which scales the very force whose target it sets).
        self.param_block = _mlp([n_params + n_rest, width, width])

        fused = width if combine == "film" else 3 * width
        self.prediction_block = _mlp([fused, width, width], out_act=True)
        self.head = nn.Linear(width, n_nodes * 2)

        # start at the identity: dV = 0. A DLO barely moves under a small
        # action, so an untrained net that already predicts "no change" starts
        # near the data mean instead of scattering nodes across the board.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, V_in: Tensor, a_n: Tensor, p_n: Tensor) -> Tensor:
        """V_in (B,N,2) canonical; a_n (B,4), p_n (B, n_params+n_rest) normalised.
        Returns dV (B,N,2) in the same canonical frame."""
        h_dlo = self.dlo_block(V_in.flatten(1))
        h_act = self.action_block(a_n)
        h_par = self.param_block(p_n)

        if self.combine == "film":
            h = h_dlo * h_act * h_par
        else:
            h = torch.cat([h_dlo, h_act, h_par], dim=-1)

        return self.head(self.prediction_block(h)).view(-1, self.n_nodes, 2)


def _turning_angles_batch(p: Tensor) -> Tensor:
    """Signed turning angle at each interior node, batched (B, N, 2) -> (B, N-2).

    Mirrors DLOSimulator.turning_angles (dlo_model.py) without needing a
    simulator instance -- duplicated rather than imported to keep nn_model.py
    free of a dependency on the physics module for what is a four-line
    geometric identity.
    """
    u = p[:, 1:] - p[:, :-1]
    u_prev, u_next = u[:, :-1], u[:, 1:]
    cross = u_prev[..., 0] * u_next[..., 1] - u_prev[..., 1] * u_next[..., 0]
    dot = (u_prev * u_next).sum(-1)
    return torch.atan2(cross, dot)


class DLOPredictor(nn.Module):
    """DLONetwork wrapped in the canonicalisation and normalisation it expects.

    Takes and returns configurations in world coordinates, so callers never
    have to remember the frame convention. This is the F(.) used by both the
    training loop and the gradient-based estimators.
    """

    def __init__(self, net: DLONetwork, norm: Normalizer) -> None:
        super().__init__()
        self.net = net
        self.norm = norm

    def forward(self, V_in: Tensor, action: Tensor, params: Tensor,
                rest_angle: Tensor | None = None,
                hist_V_in: Tensor | None = None, hist_V_out: Tensor | None = None,
                hist_action: Tensor | None = None,
                hist_mask: Tensor | None = None,
                hist_V: Tensor | None = None) -> Tensor:
        """V_in (B,N,2) world; action (B,4) raw; params (B,3) raw.

        rest_angle (B, N-2) is the wire's natural turning angle per interior
        node. It needs no canonicalisation: turning angles are already invariant
        to rigid motion, which is exactly why the model is parametrised by them
        rather than by a natural shape in coordinates.

        hist_* (B, L, ...), world coordinates, only used by DLOTransformerHist:
        up to L real PRECEDING transitions of the SAME cable (hist_mask marks
        padding). Each is canonicalised and normalised independently -- its own
        V_in defines its own frame, exactly as the current step's does -- since
        a history entry from earlier in the clip has no reason to share this
        step's frame. See DLOTransformerHist's docstring for why.
        """
        Vc, centre, angle = canonicalize(V_in)
        scale = self.norm.pos_scale
        Vc = Vc / scale

        # The action's linear part lives in world coordinates -- rotate it into
        # the canonical frame too, or the network sees a displacement that
        # disagrees with the configuration it is attached to. dtheta and alpha
        # are frame-invariant. Built with cat rather than in-place writes so
        # gradients flow cleanly back to `action` for the estimator in
        # estimation.py.
        # One 4-block per driven end: [alpha, dx, dy, dtheta]. Every block's
        # displacement needs the same rotation -- handling only the first would
        # feed the hub's motion in world coordinates while the configuration is
        # canonical, i.e. a drive pointing the wrong way.
        blocks = []
        for c in range(0, action.shape[1], 4):
            disp = rotate_vec(action[:, c + 1:c + 3], -angle) / scale
            blocks += [action[:, c:c + 1], disp, action[:, c + 3:c + 4]]
        a = torch.cat(blocks, dim=-1)
        a_n = self.norm.norm_action(a)
        p_n = self.norm.norm_params(params)
        if rest_angle is not None:
            p_n = torch.cat([p_n, self.norm.norm_rest(rest_angle)], dim=-1)

        net_kwargs = {}
        if getattr(self.net, "takes_node_hist", False):
            # Node history for DLOTransformerSeq: the n_hist preceding frames
            # (B, n_hist, N, 2) in WORLD coordinates, mapped into the CURRENT
            # frame -- the same centre, angle and scale as V_in -- and handed
            # over as displacements from the current position. Unlike the
            # DyWA history below, each frame is not canonicalised on its own:
            # a velocity only means something relative to where the node is
            # now. None = the wire has been standing still.
            if hist_V is not None:
                B, L, N, _ = hist_V.shape
                Vh = rotate((hist_V - centre.unsqueeze(1)).reshape(B * L, N, 2),
                            -angle.repeat_interleave(L)).reshape(B, L, N, 2)
                net_kwargs["hist"] = Vh / scale - Vc.unsqueeze(1)
        if hist_V_in is not None:
            B, L, N = hist_V_in.shape[0], hist_V_in.shape[1], hist_V_in.shape[2]
            Vh_in = hist_V_in.reshape(B * L, N, 2)
            Vh_out = hist_V_out.reshape(B * L, N, 2)
            Ah = hist_action.reshape(B * L, hist_action.shape[-1])
            Vhc, _, angh = canonicalize(Vh_in)
            hblocks = []
            for c in range(0, Ah.shape[1], 4):
                disp = rotate_vec(Ah[:, c + 1:c + 3], -angh) / scale
                hblocks += [Ah[:, c:c + 1], disp, Ah[:, c + 3:c + 4]]
            Ahn = self.norm.norm_action(torch.cat(hblocks, dim=-1))
            ta_in = self.norm.norm_rest(_turning_angles_batch(Vh_in))
            ta_out = self.norm.norm_rest(_turning_angles_batch(Vh_out))
            hist_feat = torch.cat([Ahn, ta_in, ta_out], dim=-1).reshape(B, L, -1)
            net_kwargs = {"hist": hist_feat, "hist_mask": hist_mask}

        dV = self.net(Vc, a_n, p_n, **net_kwargs)
        V_pred_c = (Vc + dV) * scale
        return rotate(V_pred_c, angle) + centre
