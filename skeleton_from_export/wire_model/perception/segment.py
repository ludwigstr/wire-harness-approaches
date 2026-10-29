"""Segment wire bundles and connector modules from a backlit board frame.

The scene is favourable in one way and hostile in another. Favourable: the
board is a backlight, so everything on it is dark against a bright field.
Hostile: the wires are thin and, once blurred by the optics, a wire pixel is
merely *grey* (150-200) against a ~251 background -- not dark. A global
threshold that catches the wires therefore also catches half the board, and one
that spares the board loses most of the wire.

So a global threshold is used only for the connector modules, which really are
solidly dark, and the wires are found with a morphological black-hat, which
responds to local contrast and does not care about the absolute level.

Measured on 20260617_183034: p1 = 30, p5 = 116, p50 = 251 inside the board.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rectify import BoardTransform, board_mask


@dataclass
class Segmentation:
    wires: np.ndarray        # bool, thin dark structures on the board
    connectors: np.ndarray   # bool, the modules (dilated to swallow their bezel)
    board: np.ndarray        # bool, board interior
    connector_boxes: list    # (x, y, w, h, area, cx, cy) per module, area-sorted
    module_centres: np.ndarray  # (K, 2) module centres in pixels


def detect_modules(
    g: np.ndarray,
    board: np.ndarray,
    dark_thresh: int = 130,
    open_kernel: int = 140,
    min_core_area: int = 2000,
) -> np.ndarray:
    """Locate connector modules as filled dark squares. Returns (K, 2) centres.

    Two facts make plain connected-components unreliable and drive this design:

    - A module is a dark rounded-square *frame* with a bright interior (a
      fiducial card), so it is not a solid dark blob. Its contour is filled
      before anything else, or a large square never fits inside it.
    - Modules merge with the wire stub plugged into them and with each other in
      the fan near the hub, and rig hardware and corner glare add dark blobs of
      their own. A morphological OPEN with a ~module-sized square keeps only
      regions that actually contain a full module-sized square: thin wires and
      narrow rig bars cannot survive it, so the count is right for the right
      reason rather than by area tuning.

    dark_thresh=130 (not the 100 used for wire masking) is deliberately loose:
    one real module has a bright marker surface and only a faint dark border, so
    it needs the higher threshold to close its ring -- at 100 it is missed, at
    150 glare starts to qualify.
    """
    solid = ((g < dark_thresh) & board).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    filled = np.zeros_like(solid)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    op = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                          np.ones((open_kernel, open_kernel), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(op, 8)

    # A connector module is a square of very consistent size. Measured across
    # clips, real modules come out 194-199 x 194-203 px, aspect 0.96-1.01,
    # area 37.4k-39.9k -- tight enough to filter on directly. Two other things
    # otherwise slip through and were doing real damage:
    #   * the rig hardware at the top of the board (151x100 / 218x129, aspect
    #     1.5-1.7, area 15k-26k). It was being counted as a module, which made
    #     it "mover 0" -- so every figure's labels started at M1 and the real
    #     first mover went unnumbered.
    #   * a hand or forearm reaching in (358x409, 582x243, 565x445; area
    #     100k-152k), which is 3-4x a module.
    centres = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w_ / max(h_, 1)
        if a < max(min_core_area, 30000) or a > 50000:
            continue
        if not (0.80 <= aspect <= 1.25):
            continue
        centres.append(cent[i])
    return np.array(centres, dtype=float).reshape(-1, 2)


def _solid_mask(frame, tf, dark_thresh=130):
    """Filled dark regions inside the board: modules, a hand, the rig mount."""
    from .rectify import board_mask
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bm = board_mask(frame, tf, inset_mm=30)
    solid = ((g < dark_thresh) & bm).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    filled = np.zeros_like(solid)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled, g, bm


def static_mask(cap, tf, read_fn, n_frames, samples=9, dark_thresh=130):
    """Dark regions present in MOST frames of a clip -- i.e. not a hand.

    Needed because the rig mount hangs into the top of the board and reads as
    one large filled dark region, indistinguishable by size from an arm: on
    20260602_114019 it measures 109k px against the 70k "that must be a hand"
    threshold. Judging intervention by absolute blob size therefore condemned
    entire clips of perfectly clean frames -- 230 of them in one batch run, all
    of them false positives.

    What actually separates the two is time, not size: the mount is in the same
    place in every frame, an arm is there for a few seconds. So the persistent
    part is measured once per clip and discounted, leaving the transient part
    to be judged.
    """
    acc, used = None, 0
    for idx in np.linspace(0, max(0, n_frames - 1), samples).astype(int):
        fr = read_fn(cap, int(idx))
        if fr is None:
            continue
        f, _, _ = _solid_mask(fr, tf, dark_thresh)
        acc = f.astype(np.int32) if acc is None else acc + f
        used += 1
    if not used:
        return None
    # present in more than half the sampled frames -> static
    return (acc > used // 2).astype(np.uint8)


def scene_quality(frame, tf, over_thresh=0.40, blob_area=70000,
                  dark_thresh=130, static=None, modules=None,
                  module_radius=170.0):
    """Flag frames unusable as evidence: blown out, or a hand reaching in.

    Both were found by inspecting rejected situations:
      * overexposure -- a washed-out board where the cables are barely visible;
        85% of board pixels saturated versus 1-9% in usable frames.
      * human intervention -- an arm entering the frame appears as a filled
        dark region 3-4x a module's area.

    The size test only means anything once everything legitimately dark has
    been accounted for, and there are two such things:

      `static`  the clip's persistent dark content (`static_mask`) -- above all
                the rig mount, 109-157k px of it, which alone condemned 230
                clean frames in one batch run.
      `modules` the five tracked plug positions. They move, so they cannot be
                in the static mask, and two that drift close together merge
                into an ~86k blob -- still a false alarm, just a different one.

    With both discounted, a blob that remains really is foreign to the rig.
    Returns (ok, info).
    """
    filled, g, bm = _solid_mask(frame, tf, dark_thresh)
    over = float((g[bm] >= 250).mean()) if bm.any() else 1.0
    if static is not None:
        filled = (filled.astype(bool) & ~static.astype(bool)).astype(np.uint8)
    if modules is not None:
        # uint8 rather than bool: cv2.circle rejects bool images
        drop = np.zeros(filled.shape, np.uint8)
        for c in np.asarray(modules, float).reshape(-1, 2):
            if not np.isfinite(c).all():
                continue
            cv2.circle(drop, (int(round(c[0])), int(round(c[1]))),
                       int(module_radius), 1, thickness=-1)
        filled = (filled.astype(bool) & ~drop.astype(bool)).astype(np.uint8)
    op = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((140, 140), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(op, 8)
    biggest = max([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)],
                  default=0)

    info = {"overexposed": over, "biggest_blob": biggest}
    ok = over <= over_thresh and biggest <= blob_area
    return ok, info


def segment(
    frame: np.ndarray,
    tf: BoardTransform,
    inset_mm: float = 30.0,
    dark_thresh: int = 100,
    module_min_area: int = 8000,
    module_dilate: int = 45,
    blackhat_kernel: int = 31,
    wire_percentile: float = 99.0,
    wire_min_response: int = 30,
) -> Segmentation:
    """Split a frame into wires, connector modules and board.

    `module_dilate` deliberately over-grows the connector regions. It costs the
    last centimetre of each wire, right where it enters its plug -- but that
    stretch is the one most corrupted by the module's own dark bezel and by the
    plug shadow, and the wire's endpoint is better recovered from the module
    pose than from pixels there.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board = board_mask(frame, tf, inset_mm=inset_mm)

    module_centres = detect_modules(g, board)

    # Connector *regions* to mask out of the wire search: everything solidly
    # dark that a detected module sits inside. Grown generously, since the last
    # centimetre of wire at each plug is corrupted by the bezel and plug shadow
    # and is better recovered from the module pose than from those pixels.
    solid = ((g < dark_thresh) & board).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE,
                             np.ones((31, 31), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(solid, 8)

    conn = np.zeros_like(solid)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < module_min_area:
            continue
        # keep this dark blob only if a detected module lives in it
        holds_module = any(lab[int(cy), int(cx)] == i
                           for cx, cy in module_centres
                           if 0 <= int(cy) < lab.shape[0]
                           and 0 <= int(cx) < lab.shape[1])
        if holds_module:
            conn[lab == i] = 1
            boxes.append((int(x), int(y), int(w), int(h), int(a),
                          float(cent[i][0]), float(cent[i][1])))
    boxes.sort(key=lambda b: -b[4])
    conn_d = cv2.dilate(conn, np.ones((module_dilate, module_dilate), np.uint8))

    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          np.ones((blackhat_kernel, blackhat_kernel), np.uint8))
    valid = board & (conn_d == 0)
    if valid.sum() < 100:
        raise RuntimeError("almost no board left after masking connectors")
    thr = max(wire_min_response, float(np.percentile(bh[valid], wire_percentile)))
    wires = (bh > thr) & valid
    wires = cv2.morphologyEx(wires.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8)).astype(bool)

    return Segmentation(wires=wires, connectors=conn_d.astype(bool),
                        board=board, connector_boxes=boxes,
                        module_centres=module_centres)


def wire_components(seg: Segmentation, min_area: int = 1200):
    """Connected wire components, largest first, as (area, mask) pairs.

    Each is one bundle running between two connector regions -- the dilated
    connector mask is what cuts the harness into separate strands.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        seg.wires.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            out.append((a, lab == i))
    out.sort(key=lambda t: -t[0])
    return out
