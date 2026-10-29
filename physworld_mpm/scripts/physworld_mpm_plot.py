"""Plots for the PhysWorld MPM rebuild (2026-10-28): fitted twin rollout (nodes) next to the
real track at 4 frames, and optionally the particles of one frame (top + side view).
usage: physworld_mpm_plot.py <twin .pt> <out .png> [--particles]"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]

tf = torch.load(sys.argv[1], weights_only=False)
out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
z = np.load(ROOT / f"data/tracks14/{tf['clip']}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])
N = tf["nodes_final"].numpy(); T = N.shape[0]
fr = np.linspace(0, T - 1, 5).astype(int)[1:]
m = np.array(tf["twin_final"])
fig, ax = plt.subplots(1, 4, figsize=(20, 6))
for a, t in zip(ax, fr):
    for c in range(4):
        a.plot(P[c, t, :, 0], P[c, t, :, 1], "k.-", lw=1, label="real" if c == 0 else None)
        a.plot(N[t, c, :, 0], N[t, c, :, 1], "r.-", lw=1, label="MPM twin" if c == 0 else None)
    a.set_aspect("equal"); a.set_title(f"frame {t}: {m[t]:.1f} mm")
ax[0].legend()
fig.suptitle(f"{tf['clip']} {tf['model']}: twin {m[5:].mean():.2f} mm/node (frames 5..{T-1}); "
             f"E {np.exp(tf['glob']['lE'].item()):.3g} rho {np.exp(tf['glob']['lrho'].item()):.4g} "
             f"mu {np.exp(tf['glob']['lmu'].item()):.3g} nu {tf['nu']:.3f} sy {tf['sy']:.3g}")
plt.tight_layout(); plt.savefig(out, dpi=70)
print(out)
