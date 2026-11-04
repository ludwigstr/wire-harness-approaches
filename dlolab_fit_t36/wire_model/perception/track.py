"""Extract a single wire strand's shape across the frames of a video.

Ties the perception stack together: rectify (px->mm, once per video, since the
camera is static) -> segment (wires vs modules) -> hub detection -> per-frame
strand centerlines in metric board coordinates. The output is a sequence of
(n_nodes, 2) configurations in millimetres, ready to hand to the estimator as
the observed V of one deformable wire.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .centerline import strand_centerline, strand_quality
from .rectify import BoardTransform, find_board
from .segment import segment, wire_components


def detect_hub(components, module_centres) -> np.ndarray:
    """The fan-out module: the one nearest the most component endpoints.

    Every strand has one end at the hub, so the hub collects the most votes from
    component extremities, while each peripheral module collects at most one.
    """
    votes = np.zeros(len(module_centres))
    for _, m in components:
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], -1).astype(float)
        for end in (pts[pts[:, 0].argmin()], pts[pts[:, 0].argmax()],
                    pts[pts[:, 1].argmin()], pts[pts[:, 1].argmax()]):
            votes[int(np.linalg.norm(module_centres - end, axis=1).argmin())] += 1
    return module_centres[int(votes.argmax())]


def strand_target(mask, hub_xy, module_centres) -> np.ndarray:
    """The peripheral module a component runs to (nearest its far-from-hub end)."""
    peripheral = np.array([m for m in module_centres
                           if np.linalg.norm(m - hub_xy) > 1.0])
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], -1).astype(float)
    far = pts[int(np.linalg.norm(pts - hub_xy, axis=1).argmax())]
    return peripheral[int(np.linalg.norm(peripheral - far, axis=1).argmin())]


@dataclass
class FrameStrands:
    frame_idx: int
    hub_px: np.ndarray
    strands_mm: list          # (n_nodes, 2) per strand, board millimetres
    targets_mm: list          # target module centre per strand, board mm
    quality: list             # strand_quality dict per strand


def extract_frame(
    frame: np.ndarray,
    tf: BoardTransform,
    n_nodes: int,
    frame_idx: int = 0,
    min_area: int = 800,
) -> FrameStrands:
    seg = segment(frame, tf)
    comps = wire_components(seg, min_area=min_area)
    if len(seg.module_centres) < 2 or not comps:
        return FrameStrands(frame_idx, np.zeros(2), [], [], [])
    hub = detect_hub(comps, seg.module_centres)

    strands, targets, quals = [], [], []
    for _, m in comps:
        mod = strand_target(m, hub, seg.module_centres)
        cl_px = strand_centerline(m, n_nodes, hub, mod)
        strands.append(tf.to_mm(cl_px) / 1000.0)      # -> metres
        targets.append(tf.to_mm(mod) / 1000.0)
        quals.append(strand_quality(cl_px, tf))
    return FrameStrands(frame_idx, hub, strands, targets, quals)


def match_strand(prev_mm: np.ndarray, frame: FrameStrands) -> int | None:
    """Index of the strand in `frame` best matching a previous shape.

    Matched by target-module proximity first (the plug a strand runs to is the
    stablest label across frames), then by mean node distance as a tiebreak.
    """
    if not frame.strands_mm:
        return None
    prev_end = prev_mm[-1]
    best, best_d = None, np.inf
    for i, (s, tgt) in enumerate(zip(frame.strands_mm, frame.targets_mm)):
        d = np.linalg.norm(tgt - prev_end)
        if d < best_d:
            best, best_d = i, d
    return best


def track_strand(
    video_path: str,
    target_module_mm: np.ndarray,
    n_nodes: int = 16,
    stride: int = 1,
    min_slack: float = 0.4,
    max_frames: int | None = None,
) -> dict:
    """Follow the strand running to `target_module_mm` through a whole video.

    Returns dict with:
      shapes    (T, n_nodes, 2) metric configurations, node 0 at the hub
      frames    the frame index of each shape
      arc       (T,) arc length [m] per frame
      target    the tracked module centre [m]
      dropped   count of frames rejected by the slack gate
    The camera is static, so the board transform is found once on frame 0.
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video_path}")
    tf = find_board(frame0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    shapes, idxs, arcs = [], [], []
    dropped = 0
    idx = 0
    while True:
        if max_frames and len(idxs) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or idx >= n_total:
            break
        fs = extract_frame(frame, tf, n_nodes, idx)
        if fs.strands_mm:
            j = int(np.linalg.norm(
                np.array(fs.targets_mm) - target_module_mm, axis=1).argmin())
            q = fs.quality[j]
            if q["slack"] >= min_slack and q["monotonic"]:
                shapes.append(fs.strands_mm[j])
                idxs.append(idx)
                arcs.append(q["arc"] / 1000.0)
            else:
                dropped += 1
        idx += stride

    cap.release()
    return {"shapes": np.array(shapes), "frames": np.array(idxs),
            "arc": np.array(arcs), "target": target_module_mm,
            "dropped": dropped, "transform": tf}
