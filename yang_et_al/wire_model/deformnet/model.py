"""DeformNet, adapted to tracked cable nodes.

Paper (Sec. III): a PointNet encoder turns the RGB-D point cloud into an
embedding e (split into deformation and appearance parts), a conditional NeRF
decodes images from it, and a Dreamer-v2 RSSM learns the dynamics IN
EMBEDDING SPACE:
    recurrent model      h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    representation       z_t ~ q(z_t | h_t, e_t)          categorical, 32 x 32
    transition           z^_t ~ p(z^_t | h_t)
    embedding predictor  e^_t ~ p(e^_t | h_t, z_t)
    loss  ||e_t - e^_t||^2 + KL(q || p) with KL balancing   (+ reward term, unused here)
Training: encoder/decoder first, then frozen while the RSSM is trained.

Here: the observation is the set of 16 tracked node positions (planar, z = 0),
the PointNet reads them as an unordered set (as in the paper), the decoder
emits the 16 ORDERED positions (our NeRF replacement), the embedding keeps
only the deformation part (no appearance in a point set). Actions are the
measured displacements of the two driven ends, like every other row.

Exposes the RopeDreamer trainer/evaluator API (forward / warmup / imagine /
positions / set_state_stats) so scripts/ropedreamer_train.py --model deformnet
runs it unchanged with --repr positions.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def mlp(d_in, d_hidden, d_out, n_hidden=2):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.ELU()]
        d = d_hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class PointNetEncoder(nn.Module):
    """Per-point MLP, max-pool over the set, MLP head -> embedding (paper: PointNet)."""

    def __init__(self, d_point=3, d_embed=64, width=128):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(d_point, 64), nn.ELU(), nn.Linear(64, width), nn.ELU(), nn.Linear(width, width))
        self.head = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, d_embed))

    def forward(self, P: Tensor) -> Tensor:                 # (..., N, 3) -> (..., d_embed)
        return self.head(self.point(P).max(dim=-2).values)


class ActionEncoder(nn.Module):
    def __init__(self, n_links, d_action, d_disp=3):
        super().__init__()
        self.emb = nn.Embedding(n_links + 1, d_action, padding_idx=n_links); self.n_links = n_links
        self.disp = mlp(d_disp, 128, d_action, n_hidden=1)

    def forward(self, idx, dp):
        pad = idx < 0; idx = torch.where(pad, torch.full_like(idx, self.n_links), idx)
        return (self.emb(idx) + self.disp(dp) * (~pad).unsqueeze(-1)).sum(1)


class DeformNet(nn.Module):
    repr = "positions"

    def __init__(self, n_nodes=16, d_embed=64, d_rnn=256, n_cat=32, n_cls=32, d_action=128, d_hidden=256,
                 beta=1.0, kl_balance=0.8, free_nats=0.0, pos_weight=0.0, rollout_weight=0.0, warm=5, **_):
        super().__init__()
        self.n_nodes, self.beta, self.kl_balance, self.free_nats = n_nodes, beta, kl_balance, free_nats
        self.pos_weight, self.rollout_weight, self.warm = pos_weight, rollout_weight, warm
        self.d_state = 3 * n_nodes
        self.d_embed, self.n_cat, self.n_cls, self.d_rnn = d_embed, n_cat, n_cls, d_rnn
        self.d_z = n_cat * n_cls
        self.register_buffer("s_mean", torch.zeros(self.d_state)); self.register_buffer("s_std", torch.ones(self.d_state))
        self.enc = PointNetEncoder(3, d_embed)
        self.dec = mlp(d_embed, d_hidden, self.d_state)
        self.act = ActionEncoder(n_nodes - 1, d_action)
        self.inp = nn.Sequential(nn.Linear(self.d_z + d_action, d_hidden), nn.ELU())
        self.gru = nn.GRUCell(d_hidden, d_rnn)
        self.prior = mlp(d_rnn, d_hidden, self.d_z, n_hidden=1)
        self.post = mlp(d_rnn + d_embed, d_hidden, self.d_z, n_hidden=1)
        self.emb_pred = mlp(d_rnn + self.d_z, d_hidden, d_embed, n_hidden=1)
        self.ae_frozen = False

    # ------------------------------------------------------------ representation
    def set_state_stats(self, mean, std, std_floor=0.05):
        self.s_mean.copy_(mean); self.s_std.copy_(std.clamp_min(std_floor))

    def _points(self, s: Tensor) -> Tensor:
        return ((s - self.s_mean) / self.s_std).reshape(*s.shape[:-1], self.n_nodes, 3)

    def encode(self, s: Tensor) -> Tensor:
        return self.enc(self._points(s))

    def decode_state(self, e: Tensor) -> Tensor:                # embedding -> raw flattened positions
        return self.dec(e) * self.s_std + self.s_mean

    def positions(self, s: Tensor, ell=None) -> Tensor:
        return s.reshape(*s.shape[:-1], self.n_nodes, 3)

    def pretrain_ae(self, S: Tensor, device, epochs=200, lr=1e-3, log=print):
        """Stage 1 of the paper: encoder + decoder alone (reconstruction), then frozen."""
        S = S.to(device); opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        for ep in range(epochs):
            perm = torch.randperm(S.shape[0], device=device)
            for i in range(0, S.shape[0], 512):
                s = S[perm[i:i + 512]]
                loss = F.mse_loss(self.decode_state(self.encode(s)), s)
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    err = (self.positions(self.decode_state(self.encode(S)))[..., :2] - self.positions(S)[..., :2]).norm(dim=-1).mean() * 1000
                log(f"  AE ep {ep}: reconstruction {err:.2f} mm")
        for p in list(self.enc.parameters()) + list(self.dec.parameters()):
            p.requires_grad_(False)
        self.ae_frozen = True

    # ------------------------------------------------------------ latent
    def _logits(self, x):
        return x.view(*x.shape[:-1], self.n_cat, self.n_cls)

    def _sample(self, logits, sample=True):
        """straight-through categorical sample (Dreamer-v2); mode when sample=False"""
        probs = F.softmax(logits, -1)
        if sample:
            idx = torch.distributions.Categorical(probs=probs).sample()
        else:
            idx = probs.argmax(-1)
        one_hot = F.one_hot(idx, self.n_cls).to(probs.dtype)
        z = one_hot + probs - probs.detach()
        return z.flatten(-2)

    @staticmethod
    def _kl(lq, lp):
        """KL(q || p) for the batch of categoricals, summed over variables -> (B,)"""
        q = F.softmax(lq, -1); return (q * (F.log_softmax(lq, -1) - F.log_softmax(lp, -1))).sum((-1, -2))

    def step(self, h, z, a):
        return self.gru(self.inp(torch.cat([z, a], -1)), h)

    def initial(self, B, device):
        return torch.zeros(B, self.d_rnn, device=device), torch.zeros(B, self.d_z, device=device)

    # ------------------------------------------------------------ training pass
    def forward(self, s, idx, dp, sample=True, ell=None):
        B, T, _ = s.shape
        e = self.encode(s)
        a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1)
        h, z = self.initial(B, s.device)
        recon_e, pred_e, kls = [], [], []
        for t in range(T):
            if t > 0:
                h = self.step(h, z, a[:, t - 1])
            lp = self._logits(self.prior(h)); lq = self._logits(self.post(torch.cat([h, e[:, t]], -1)))
            z_prior = self._sample(lp, sample); z = self._sample(lq, sample)
            recon_e.append(self.emb_pred(torch.cat([h, z], -1)))
            if t > 0:
                pred_e.append(self.emb_pred(torch.cat([h, z_prior], -1)))
            # KL balancing (Dreamer-v2): alpha on the prior side, 1-alpha on the posterior side
            kls.append(self.kl_balance * self._kl(lq.detach(), lp) + (1 - self.kl_balance) * self._kl(lq, lp.detach()))
        recon_e = torch.stack(recon_e, 1); pred_e = torch.stack(pred_e, 1); kl = torch.stack(kls, 1)
        l_recon = F.mse_loss(recon_e, e.detach()); l_pred = F.mse_loss(pred_e, e[:, 1:].detach())
        l_kl = torch.clamp(kl, min=self.free_nats).mean()
        loss = l_recon + l_pred + self.beta * l_kl
        s_recon = self.decode_state(recon_e); s_pred = self.decode_state(pred_e)
        l_pos = torch.zeros((), device=s.device)
        if self.pos_weight > 0:
            l_pos = F.mse_loss(self.positions(s_recon), self.positions(s)) + F.mse_loss(self.positions(s_pred), self.positions(s[:, 1:]))
            loss = loss + self.pos_weight * l_pos
        l_roll = torch.zeros((), device=s.device)
        if self.rollout_weight > 0 and T > self.warm:
            l_roll = self.open_loop_loss(s, idx, dp, e, a); loss = loss + self.rollout_weight * l_roll
        return {"loss": loss, "recon": l_recon, "pred": l_pred, "kl": kl.mean(), "pos": l_pos, "roll": l_roll,
                "s_recon": s_recon, "s_pred": s_pred}

    def open_loop_loss(self, s, idx, dp, e, a):
        B, T, _ = s.shape; warm = min(self.warm, T - 1)
        h, z = self.initial(B, s.device)
        for t in range(warm):
            if t > 0:
                h = self.step(h, z, a[:, t - 1])
            z = self._sample(self._logits(self.post(torch.cat([h, e[:, t]], -1))), sample=False)
        preds = []
        for t in range(warm, T):
            h = self.step(h, z, a[:, t - 1]); z = self._sample(self._logits(self.prior(h)), sample=False)
            preds.append(self.decode_state(self.emb_pred(torch.cat([h, z], -1))))
        P_pred = self.positions(torch.stack(preds, 1)); P_true = self.positions(s[:, warm:])
        return F.mse_loss(P_pred[..., 1:-1, :], P_true[..., 1:-1, :])

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def warmup(self, s, idx, dp, sample=False):
        B, n, _ = s.shape; e = self.encode(s); a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, n, -1)
        h, z = self.initial(B, s.device)
        for t in range(n):
            if t > 0:
                h = self.step(h, z, a[:, t - 1])
            z = self._sample(self._logits(self.post(torch.cat([h, e[:, t]], -1))), sample)
        return h, z

    @torch.no_grad()
    def imagine(self, h, z, idx, dp, sample=False):
        B, T = idx.shape[:2]; a = self.act(idx.flatten(0, 1), dp.flatten(0, 1)).view(B, T, -1); out = []
        for t in range(T):
            h = self.step(h, z, a[:, t]); z = self._sample(self._logits(self.prior(h)), sample)
            out.append(self.decode_state(self.emb_pred(torch.cat([h, z], -1))))
        return torch.stack(out, 1)
