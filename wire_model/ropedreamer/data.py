"""Sequences for RopeDreamer from the tracked clips.

Every (clip, cable) is one trajectory: positions (T, 16, 2) in metres from a
track cache (X-Planar_vids/mujoco_videos/tracks/*.npz or data/tracks14/*.npz)
-> chain states (T, 63), per-link lengths from the first frame (15,), and the
action after each frame: the two driven end links (hub = link 0, plug =
link 14) with their measured displacement to the next frame, i.e. the
paper's (i_g, dp) for two simultaneous grasps. Training samples are random
windows of `seq_len` frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .kinematics import positions_to_chain, chain_to_positions

N_NODES = 16
GRASPS = (0, N_NODES - 2)          # links driven: first (hub side) and last (plug side)


def load_tracks(dirs: list[str | Path]) -> list[dict]:
    """-> list of trajectories: {clip, cable, P (T,16,2) m, fps}."""
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            for k in range(4):
                out.append({"clip": f.stem, "cable": k, "P": z[f"c{k}"].astype(np.float32),
                            "fps": float(z["fps"]), "set": Path(d).name})
    return out


REPR = "chain"          # "chain" (the paper) or "positions" (control: 16 x 3 flattened, no kinematic chain)


def trajectory_tensors(P: np.ndarray) -> dict:
    """One trajectory -> states, per-link lengths, actions (as tensors)."""
    Pt = torch.tensor(P, dtype=torch.float32)                      # (T, 16, 2)
    s, ell_mean, seg = positions_to_chain(Pt)                       # (T, 63), (T,), (T, 15)
    if REPR == "positions":
        s = torch.cat([Pt, torch.zeros_like(Pt[..., :1])], -1).flatten(1)   # (T, 48)
    # ONE link length per trajectory, the mean over all frames and links (the
    # paper's constant link length). Measured on the X-Planar tracks: this
    # floor is 3.2 mm mean node error; per-link lengths taken from frame 0
    # are WORSE (5.2 mm) because the tracker's resampling moves the node
    # spacing along the cable from frame to frame.
    ell = seg.mean()
    P3 = torch.cat([Pt, torch.zeros_like(Pt[..., :1])], -1)         # (T, 16, 3)
    nodes = [0, N_NODES - 1]
    dp = P3[1:, nodes] - P3[:-1, nodes]                             # (T-1, 2, 3)
    dp = torch.cat([dp, torch.zeros_like(dp[:1])], 0)               # last frame: no action
    idx = torch.tensor(GRASPS, dtype=torch.long).expand(Pt.shape[0], 2).clone()
    return {"s": s, "ell": ell, "idx": idx, "dp": dp, "P": Pt}


class WindowDataset(torch.utils.data.Dataset):
    """Random windows of seq_len consecutive frames over all trajectories;
    `stride` windows per trajectory epoch (start positions re-drawn each epoch)."""

    def __init__(self, trajs: list[dict], seq_len: int, windows_per_traj: int | None = None, seed: int = 0):
        # trajectories shorter than one window cannot be collated (the
        # trimmed 14-clip tracks have 27-45 frames); they are skipped here and
        # still evaluated in full by ropedreamer_eval.py
        self.T = [trajectory_tensors(t["P"]) for t in trajs if t["P"].shape[0] >= seq_len]
        self.seq_len = seq_len
        self.n_win = [max(1, (t["s"].shape[0] - seq_len) // (seq_len // 2) + 1) if windows_per_traj is None
                      else windows_per_traj for t in self.T]
        self.index = [(i, w) for i, n in enumerate(self.n_win) for w in range(n)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, _ = self.index[j]
        t = self.T[i]
        T = t["s"].shape[0]
        start = int(self.rng.integers(0, max(1, T - self.seq_len + 1)))
        sl = slice(start, start + self.seq_len)
        return t["s"][sl], t["idx"][sl], t["dp"][sl], t["ell"]


def state_offset(trajs: list[dict]) -> Tensor:
    """Mean base position over the data (subtracted from p0 so the state is
    centred); quaternion entries are left as they are."""
    p0 = np.concatenate([t["P"][:, 0] for t in trajs]).mean(0)
    if REPR == "positions":
        off = torch.zeros(3 * N_NODES); off[0::3], off[1::3] = float(p0[0]), float(p0[1])
        return off
    off = torch.zeros(3 + 4 * (N_NODES - 1)); off[0], off[1] = float(p0[0]), float(p0[1])
    return off


def state_stats(trajs: list[dict]) -> tuple[Tensor, Tensor]:
    """Per-dimension mean and std of the chain state over all frames."""
    S = torch.cat([trajectory_tensors(t["P"])["s"] for t in trajs])
    return S.mean(0), S.std(0)
