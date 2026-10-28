"""Overlay of ALL generated twins14 rollouts per clip at equal timestamps
(colored = varied rollouts, black = exact MuJoCo twin _000, red dashed = real track).
    python scripts/overlay_twins14_rollouts.py <out_dir> [--synth data/wm/ropedreamer_twins14_synth]"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COL = ["#2a6fdb", "#e0732c", "#2e9e5b", "#b0389c"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--synth", default="data/wm/ropedreamer_twins14_synth")
ap.add_argument("--tracks", default="data/tracks14")
ap.add_argument("--max-draw", type=int, default=300, help="varied rollouts drawn per panel (all are counted)")
a = ap.parse_args()
out = ROOT / a.out
out.mkdir(parents=True, exist_ok=True)

files = defaultdict(list)
for f in sorted(glob.glob(str(ROOT / a.synth / "*.npz"))):
    stem = Path(f).stem
    files[stem.rsplit("_", 1)[0]].append(f)

clips = sorted(files)
fig, axes = plt.subplots(len(clips), 5, figsize=(17, 3.3 * len(clips)))
total = 0
for row, clip in enumerate(clips):
    fs = files[clip]
    z = np.load(ROOT / a.tracks / f"{clip}.npz")
    real = np.stack([z[f"c{k}"] for k in range(4)], 1)
    twin = next((np.load(f)["V"] for f in fs if f.endswith("_000.npz")), None)
    varied = [f for f in fs if not f.endswith("_000.npz")]
    total += len(varied)
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(varied), min(len(varied), a.max_draw), replace=False)) if varied else []
    Vs = [np.load(varied[i])["V"] for i in pick]
    T = len(real)
    ts = np.linspace(0, T - 1, 5).round().astype(int)
    allP = np.concatenate([real.reshape(-1, 2)] + [V.reshape(-1, 2) for V in Vs] +
                          ([twin.reshape(-1, 2)] if twin is not None else []))
    lo, hi = allP.min(0) - 0.03, allP.max(0) + 0.03
    al = float(np.clip(6.0 / max(len(Vs), 1), 0.02, 0.5))
    for c, t in enumerate(ts):
        ax = axes[row, c]
        live = [V[t] for V in Vs if t < len(V)]
        for P in live:
            for k in range(4):
                ax.plot(P[k, :, 0], P[k, :, 1], "-", color=COL[k], alpha=al, lw=1.0)
        if twin is not None and t < len(twin):
            for k in range(4):
                ax.plot(twin[t, k, :, 0], twin[t, k, :, 1], "-", color="k", lw=1.0)
        for k in range(4):
            ax.plot(real[t, k, :, 0], real[t, k, :, 1], "--", color="#d62728", lw=0.9)
        sd = 1000 * np.linalg.norm(np.stack(live).std(0), axis=-1).mean() if len(live) > 1 else 0.0
        ax.set_title(f"t = {t / 4:.2f} s   spread {sd:.1f} mm", fontsize=8)
        ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    axes[row, 0].set_ylabel(f"{clip}\n{len(varied)} varied", fontsize=9)
    print(clip, len(varied), "varied rollouts", flush=True)
fig.suptitle(f"twins14 training data, all {len(clips)} clips ({total} varied rollouts; up to {a.max_draw} drawn per clip, colored), "
             f"exact MuJoCo twin (black), real track (red dashed)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(out / "all_clips_overlay_rollouts.png", dpi=90)
