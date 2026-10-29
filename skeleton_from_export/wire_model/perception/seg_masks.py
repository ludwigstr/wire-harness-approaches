"""Precomputed per-clip segmentation masks from WireSegmentation/.

Each file there (`<clip>_replay_raw_mask_1*.mp4`) is a side-by-side render:
left panel is the raw frame, right panel is a learned class map -- green blobs
for the five module plugs, yellow for cable, black otherwise. On the frames
checked it is a clean, single connected strand per cable with none of the
fragmentation FASTDLO produced and none of the shadow/glare false positives
the black-hat mask picked up, so it replaces both as the wire/module source
wherever a clip has one.

Two things stop this being a drop-in video swap:

1. Both panels are 2000x2660, the source frame 1600x2128 -- exactly a 0.8
   scale in both axes (verified against tracked module pixel positions, ~2-16
   px residual, consistent with the frames not being the identical instant --
   see point 2). No crop or offset to correct for.
2. The mask video runs at 3.0 fps / ~81 frames against the source's 4.0 fps /
   85 frames, and the overlay text it burns in (`source=frame_000040.png` at
   mask-video frame 36) confirms frames were DROPPED, not resampled on a fixed
   ratio -- the export pipeline fell behind inference at points and skipped
   frames rather than slow down. So mask-frame index i does not correspond to
   source-frame index i, or to any fixed affine reindexing of i. There is no
   OCR available in this environment to read the burned-in frame counter, so
   the correspondence is instead recovered by content: the raw panel is
   (modulo the 0.8 scale) a copy of the source frame, so matching it against
   every source frame by downsampled MSE picks out the right one directly --
   validated to land within 1 frame of the OCR-legible ground truth above.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

MASK_DIR = Path(os.environ.get(
    "WIRE_SEG_DIR", "./Skeletonization"))

PANEL_TO_SOURCE_SCALE = 0.8     # 2000x2660 class-map panel -> 1600x2128 source
_TEXT_BAND_FRAC = 0.075         # burned-in counter occupies the top ~7.5%

_video_path: dict[str, Path | None] = {}
_small_stack: dict[str, list] = {}          # clip -> [downsampled raw panels]
_result_cache: dict[tuple, tuple | None] = {}   # (clip, orig_idx) -> masks


def _find_video(clip: str) -> Path | None:
    # The Skeletonization/ folder holds three products per clip -- the class
    # map (<clip>_replay_raw_mask_1 (1).mp4), the skeleton, and a canvas render
    # -- and some names double the clip prefix; the `*_replay_raw_mask_1*` glob
    # picks the class map for either naming and never the other two products.
    if clip not in _video_path:
        hits = sorted(MASK_DIR.glob(f"{clip}*_replay_raw_mask_1*.mp4"))
        _video_path[clip] = hits[0] if hits else None
    return _video_path[clip]


def available(clip: str) -> bool:
    return _find_video(clip) is not None


def _small(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = g[int(_TEXT_BAND_FRAC * g.shape[0]):]
    return cv2.resize(g, (80, 106), interpolation=cv2.INTER_AREA).astype(np.float32)


def _raw_panel_stack(clip: str) -> list:
    """Downsampled raw panel of every mask-video frame, decoded once per clip."""
    if clip in _small_stack:
        return _small_stack[clip]
    cap = cv2.VideoCapture(str(_find_video(clip)))
    stack = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        stack.append(_small(fr[:, :fr.shape[1] // 2]))
    cap.release()
    _small_stack[clip] = stack
    return stack


def _best_mask_frame(clip: str, source_frame: np.ndarray) -> int | None:
    q = _small(source_frame)
    stack = _raw_panel_stack(clip)
    if not stack:
        return None
    d = [float(np.mean((s - q) ** 2)) for s in stack]
    return int(np.argmin(d))


def class_masks(clip: str, orig_idx: int, source_frame: np.ndarray):
    """(mover_mask, wire_mask) booleans at SOURCE frame resolution, or None."""
    key = (clip, orig_idx)
    if key in _result_cache:
        return _result_cache[key]
    if not available(clip):
        _result_cache[key] = None
        return None
    mi = _best_mask_frame(clip, source_frame)
    if mi is None:
        _result_cache[key] = None
        return None
    cap = cv2.VideoCapture(str(_find_video(clip)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mi)
    ok, frM = cap.read()
    cap.release()
    if not ok:
        _result_cache[key] = None
        return None
    right = frM[:, frM.shape[1] // 2:]
    g, r = right[..., 1].astype(int), right[..., 2].astype(int)
    mover = (g > 180) & (r < 60)
    wire = (g > 180) & (r > 180)
    H, W = source_frame.shape[:2]
    mover_r = cv2.resize(mover.astype(np.uint8), (W, H),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    wire_r = cv2.resize(wire.astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    out = (mover_r, wire_r)
    _result_cache[key] = out
    return out


_skel_path: dict[str, Path | None] = {}


def _find_skeleton(clip: str) -> Path | None:
    if clip not in _skel_path:
        hits = sorted(MASK_DIR.glob(f"{clip}*_replay_full_skeleton*.mp4"))
        _skel_path[clip] = hits[0] if hits else None
    return _skel_path[clip]


def skeleton_frame(clip, orig_idx, source_frame):
    """Boolean skeleton at SOURCE resolution, or None -- for display only.

    The skeleton video is the same replay as the class map (same frame count),
    so the frame the class map matched to this source frame is reused directly
    rather than content-matched again -- the skeleton panel has no photo half to
    match against. White pixels are the thinned strand; upscaled to the source
    frame so it overlays the same coordinates as everything else.
    """
    sk = _find_skeleton(clip)
    if sk is None:
        return None
    mi = _best_mask_frame(clip, source_frame)
    if mi is None:
        return None
    cap = cv2.VideoCapture(str(sk))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mi)
    ok, frS = cap.read()
    cap.release()
    if not ok:
        return None
    white = cv2.cvtColor(frS, cv2.COLOR_BGR2GRAY) > 128
    H, W = source_frame.shape[:2]
    return cv2.resize(white.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def module_centres(mover_mask: np.ndarray, want: int = 5) -> np.ndarray:
    """Centroids of the `want` largest mover blobs.

    Real plugs form a tight cluster of near-equal areas (~52k px measured);
    anything much smaller is network noise, not a sixth module, so the area
    gap -- not just rank -- decides how many are kept when fewer than `want`
    are actually visible.
    """
    n, _, stats, cent = cv2.connectedComponentsWithStats(
        mover_mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((0, 2))
    areas = stats[1:, 4]
    order = np.argsort(-areas)[:want]
    if len(order) == 0:
        return np.zeros((0, 2))
    top_area = areas[order[0]]
    keep = [i for i in order if areas[i] > 0.4 * top_area]
    return cent[1:][keep]
