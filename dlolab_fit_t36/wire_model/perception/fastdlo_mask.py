"""Wire segmentation via FASTDLO instead of the black-hat morphology.

The black-hat mask keys on "darker than its surroundings", so it also fires on
shadows, the board's printed markings and the module bezels. FASTDLO (Caporali
et al., lar-unibo) is a learned DLO segmenter with a siamese network that
resolves crossings, and on these frames it is visibly cleaner.

What is NOT taken from FASTDLO is the instance decomposition. It returns 8-12
instances per frame -- the wires of the bundle, split further at every occlusion
-- with no notion of which plug pair each belongs to. Rebuilding the four cables
from those fragments would mean re-deriving the connectivity that the existing
topology tracer already computes reliably. So FASTDLO supplies the mask, and
`trace_cable_goal` still walks each mover to the hub on it. The rig constraint
(1 hub + 4 movers, 4 cables) continues to come from `harness.py`, never from
whatever the segmenter happened to return.

The network runs on a 640x848 downscale of the 1600x2128 frame; the mask is
upsampled back. At ~2.5 px/px that costs about one wire radius of edge
precision, well under the tracer's tolerance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .rectify import board_mask

# The clone lives outside the project tree; allow an override so this does not
# hard-depend on one machine's scratchpad.
FASTDLO_DIR = Path(os.environ.get(
    "FASTDLO_DIR",
    "./fastdlo"))

NET_W, NET_H = 640, 848

_PIPE = None


def available() -> bool:
    return (FASTDLO_DIR / "weights" / "CP_seg.pth").exists()


def _pipeline():
    """Load the network once; it is ~550 MB of weights."""
    global _PIPE
    if _PIPE is None:
        if not available():
            raise RuntimeError(f"FASTDLO-Gewichte nicht gefunden in {FASTDLO_DIR}")
        sys.path.insert(0, str(FASTDLO_DIR))
        cwd = os.getcwd()
        os.chdir(FASTDLO_DIR)          # the repo resolves its configs relatively
        try:
            from fastdlo.core import Pipeline
            _PIPE = Pipeline(
                checkpoint_siam=str(FASTDLO_DIR / "weights" / "CP_siam.pth"),
                checkpoint_seg=str(FASTDLO_DIR / "weights" / "CP_seg.pth"),
                img_w=NET_W, img_h=NET_H)
        finally:
            os.chdir(cwd)
    return _PIPE


def fastdlo_wires(frame, tf, conn_radius=135, module_centres=(), close=9,
                  mask_th=77):
    """Boolean wire mask at full frame resolution.

    `module_centres` are punched out with `conn_radius`, exactly as the
    black-hat path does: the plug bodies are wire-coloured and would otherwise
    fuse all four cables into one blob at the hub.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    out, _ = _pipeline().run(source_img=small, mask_th=mask_th)

    m_small = np.any(np.asarray(out) > 0, axis=2) if out.ndim == 3 else out > 0
    m = cv2.resize(m_small.astype(np.uint8), (w, h),
                   interpolation=cv2.INTER_NEAREST).astype(bool)

    m &= board_mask(frame, tf, inset_mm=30)
    if len(module_centres):
        yy, xx = np.mgrid[0:h, 0:w]
        for c in np.asarray(module_centres, float).reshape(-1, 2):
            m &= ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) >= conn_radius ** 2

    if close > 1:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((close, close), np.uint8)).astype(bool)
    return m


def instances(frame):
    """Per-instance point sequences in FULL-frame pixel coordinates, (x, y).

    Not used for cable assignment (see the module docstring) -- kept for
    inspection and for overlaying what the network actually separated.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    p = _pipeline()
    p.run(source_img=small, mask_th=77)
    sx, sy = w / NET_W, h / NET_H
    res = []
    for v in getattr(p, "last_splines", {}).values():
        pts = np.asarray(v["points"], float)      # (row, col)
        res.append(np.stack([pts[:, 1] * sx, pts[:, 0] * sy], -1))
    return res
