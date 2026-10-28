"""Gaussian Process per-node dynamics model.

The Transformer surrogate treats the whole 16-node chain as one 58-dim
input -> 32-dim output regression. A GP over that space is intractable: too
few real examples (781-1272 transitions) to cover a 58-dim input, and a
32-dim correlated output needs a full multi-output GP to do properly.

Instead this fits ONE (well, one pair, dx/dy) sparse variational GP over a
PER-NODE regression: every interior node of every transition becomes its own
training example, featurised locally (chain position, local curvature, the
driven-end actions) rather than the whole configuration. That turns ~800
transitions into ~11000 (node, transition) examples -- a far more favourable
regime for a GP -- at the cost of the node not seeing the REST of the chain's
current shape directly (only its own local curvature and position along it).
Physically defensible for a chain where bending is a local phenomenon (see
dlo_model.py's bending energy: a three-node local term), but a real
simplification against the Transformer, which attends over the whole chain.

Every feature is computed in the SAME canonical frame the Transformer uses
(wire_model.nn_model.canonicalize): centred on the centroid, rotated so the
hub->mover chord lies along +x. Rotation/translation invariance is exact,
not learned, exactly as for the Transformer -- a GP with a stationary kernel
has no other way to generalise across the wildly different board positions in
this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import gpytorch

from .nn_model import canonicalize, rotate_vec

Tensor = torch.Tensor


def _turning_angles(p: np.ndarray) -> np.ndarray:
    u = np.diff(p, axis=0)
    u_prev, u_next = u[:-1], u[1:]
    cross = u_prev[:, 0] * u_next[:, 1] - u_prev[:, 1] * u_next[:, 0]
    dot = (u_prev * u_next).sum(-1)
    return np.arctan2(cross, dot)


def node_features(V_in: np.ndarray, action: np.ndarray, pos_scale: float,
                  action_lo: np.ndarray, action_hi: np.ndarray):
    """(V_in (N,2), action (8,)) -> per-interior-node features (N-2, 11) and
    the canonicalisation (centre, angle) needed to map predictions back.

    Feature layout: [alpha, turning_angle_i / pi, hub-block(4 normalised),
    mover-block(4 normalised)]. `alpha` is the node's position along the
    chain in [0,1] -- the chain-position analogue of the paper's per-edge
    alpha flag, letting the GP learn that nodes near a driven end move more.
    """
    Vc = V_in - V_in.mean(0, keepdims=True)
    chord = Vc[-1] - Vc[0]
    angle = np.arctan2(chord[1], chord[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    Vc = Vc @ R.T / pos_scale

    n = V_in.shape[0]
    ta = _turning_angles(Vc) / np.pi                       # (n-2,)

    blocks = []
    for k in range(0, action.shape[0], 4):
        disp = R @ (action[k + 1:k + 3])
        blk = np.array([action[k], disp[0] / pos_scale, disp[1] / pos_scale,
                        action[k + 3]], np.float32)
        blocks.append((blk - action_lo[k:k + 4]) /
                      (action_hi[k:k + 4] - action_lo[k:k + 4]) * 2 - 1)
    a_n = np.concatenate(blocks)                            # (8,)

    alpha = np.linspace(0, 1, n)[1:-1]                       # (n-2,)
    feat = np.stack([alpha, ta], axis=1)                    # (n-2, 2)
    feat = np.concatenate([feat, np.tile(a_n, (n - 2, 1))], axis=1)  # (n-2,10)
    return feat.astype(np.float32), (V_in.mean(0), angle)


def unmap_displacement(dVc: np.ndarray, angle: float, pos_scale: float) -> np.ndarray:
    """Canonical-frame node displacement -> world-frame displacement."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (dVc * pos_scale) @ R.T


class NodeGP(gpytorch.models.ApproximateGP):
    """Sparse variational GP, batched over the 2 output dims (dx, dy)."""

    def __init__(self, inducing_points: Tensor):
        # inducing_points: (2, M, D) -- one set per output task, batched
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([2]))
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([2]),
                                          ard_num_dims=inducing_points.size(-1)),
            batch_shape=torch.Size([2]))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


@dataclass
class GPDynamics:
    """Trained GP + everything needed to featurise/roll out with it."""
    gp: NodeGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    pos_scale: float
    action_lo: np.ndarray
    action_hi: np.ndarray
    device: str

    def predict_step(self, V_in: np.ndarray, action: np.ndarray):
        """-> (dV mean (N,2), dV std (N,2)) in WORLD units, driven ends 0."""
        feat, (centre, angle) = node_features(V_in, action, self.pos_scale,
                                              self.action_lo, self.action_hi)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).expand(2, -1, -1)                # (2, n-2, D) batch
        self.gp.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(x))
            mean = pred.mean.cpu().numpy()                  # (2, n-2)
            std = pred.stddev.cpu().numpy()                 # (2, n-2)
        dVc = np.stack([mean[0], mean[1]], axis=1)           # (n-2, 2)
        sVc = np.stack([std[0], std[1]], axis=1)
        dV_world = unmap_displacement(dVc, angle, self.pos_scale)
        # std is a scale, not a signed displacement -- rotate its magnitude
        # the same way, abs() since a rotation can flip a component's sign.
        sV_world = np.abs(unmap_displacement(sVc, angle, self.pos_scale))
        n = V_in.shape[0]
        full_mean = np.zeros((n, 2), np.float32)
        full_std = np.zeros((n, 2), np.float32)
        full_mean[1:-1] = dV_world
        full_std[1:-1] = sV_world
        return full_mean, full_std
