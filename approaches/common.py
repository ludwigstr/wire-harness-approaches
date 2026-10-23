"""Shared pieces of all ten approach scripts: the clip set, the track caches,
the table metric and the result format.

Data
----
data/tracks14/<clip>.npz      the 14 evaluation clips (already trimmed)
data/tracks_xplanar/<clip>.npz the 103 X-Planar clips (full length)
Each file holds c0..c3: (T, 16, 2) node positions in metres of the four cables
(c0 = the cable at the hub), plus t0, t1 (video frames) and fps (4).

Metric
------
mm/node = mean over the four cables of the mean distance between predicted and
tracked node, per scored frame, averaged over the frames of a clip; the table
reports the mean (and median) over clips. Which frames are scored depends on
the approach family and is stated in each script:
  * physics engines and PhysTwin: every frame from frame 0;
  * Caporali: every 4th frame (1 s steps) plus the last, from frame 0;
  * RSSM / graph world models: every frame after a 5-frame warm-up.

Result files use the table's JSON format {clip: {start, ende, mittel, n_frames, video}}
("mittel" = the clip mean; the German keys are kept on purpose).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACKS14 = ROOT / "data" / "tracks14"
TRACKS_XPLANAR = ROOT / "data" / "tracks_xplanar"

# clip -> trim (seconds of video kept, None = full clip). The track caches are
# already cut to these; kept here for reference and for re-tracking from video.
CLIP_TRIMS = {
    "20260602_102713": None, "20260602_114019": 13, "20260602_121006": 11, "20260602_134613": None,
    "20260602_141210": 13, "20260602_162650": 12, "20260602_165113": 13, "20260602_170108": 9,
    "20260602_172435": 19, "20260602_175847": 14, "20260602_183620": 20, "20260602_193329": 15,
    "20260602_193953": 12, "20260605_112939": None,
}


def load_clip(path: Path) -> dict:
    z = np.load(path)
    return {"clip": path.stem, "P": np.stack([z[f"c{k}"] for k in range(4)]).astype(np.float32),   # (4, T, 16, 2)
            "fps": float(z["fps"]), "t0": int(z["t0"]), "t1": int(z["t1"])}


def load_clips(tracks: str | Path = TRACKS14, clips: str | None = None) -> list[dict]:
    """All clips of a track directory (optionally a comma-separated subset)."""
    d = Path(tracks)
    d = d if d.is_absolute() else ROOT / d
    out = [load_clip(p) for p in sorted(d.glob("*.npz"))]
    if clips:
        keep = set(clips.split(","))
        out = [c for c in out if c["clip"] in keep]
    return out


def frame_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """pred, truth (4, F, 16, 2) in metres -> (F,) mean over cables of the mean node distance, in mm."""
    return np.linalg.norm(pred - truth, axis=-1).mean(-1).mean(0) * 1000.0


def record(errs: np.ndarray, **extra) -> dict:
    e = np.asarray(errs, dtype=float)
    return {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()), "n_frames": int(len(e)), "video": "", **extra}


def summarise(results: dict, label: str) -> str:
    v = np.array([r["mittel"] for r in results.values()])
    return f"{label}: {len(v)} clips, mean {v.mean():.2f} mm/node, median {np.median(v):.2f}"


def write_results(results: dict, out: str | Path, label: str) -> None:
    out = Path(out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(summarise(results, label))
    print(f"-> {out}")


def turning_angles(P: np.ndarray) -> np.ndarray:
    """(..., N, 2) -> (..., N-2) signed turning angle at every interior node."""
    u = np.diff(P, axis=-2)
    return np.arctan2(u[..., :-1, 0] * u[..., 1:, 1] - u[..., :-1, 1] * u[..., 1:, 0], (u[..., :-1, :] * u[..., 1:, :]).sum(-1))
