"""The harness has a KNOWN structure: 1 hub + 4 movers, 4 cables. Always.

This is a hard constraint from the rig, not something to be discovered per
frame. Treating the counts as an outcome of detection was wrong and produced
scenes with 2 of 4 cables:

  * the module filter (core area 30-50k px) dropped one or two real modules in
    many frames, leaving 3-4 instead of 5;
  * a slack gate (>= 0.55) then discarded most traced cables, even though the
    paths were found -- measured slack on real cables runs 0.37-0.46, so the
    gate was rejecting correct traces, not bad ones.

So: find the five modules ONCE on a frame that shows them clearly, then carry
them through the clip by template tracking; and always return four cables, one
per mover, falling back to a coarser extraction rather than returning nothing.
Every cable carries a `method` and `quality` so a weak one stays visible as
weak instead of silently vanishing.
"""

from __future__ import annotations

import cv2
import numpy as np

from .centerline import _resample_pixels, strand_centerline, strand_quality
from .rectify import board_mask
from .segment import detect_modules, segment
from .tracer import build_graph, trace_cable_goal
from . import seg_masks as _seg

N_MOVERS = 4          # rig constant: four peripheral connectors
N_MODULES = N_MOVERS + 1


def wire_mask(frame, tf, conn_radius=135, close=11, clip=None, orig_idx=None):
    """Boolean wire mask, from the precomputed class map when the clip has one.

    Falls back to the black-hat mask only for clips WireSegmentation/ does not
    cover. The class map's strand is clean but occasionally gapped where two
    cables cross or one goes briefly out of the network's confidence -- a
    closing (widened to 15 px, see WireSegmentation notes) bridges resize
    nicks but not those larger real gaps, which is why `four_cables` still
    keeps its corridor/straight fallback below.
    """
    if clip is not None and orig_idx is not None and _seg.available(clip):
        r = _seg.class_masks(clip, orig_idx, frame)
        if r is not None:
            _, wire = r
            # The class map is frame-wide and occasionally calls rig hardware
            # above the board "wire" -- the same reason board_mask exists for
            # the black-hat path below. Harmless most of the time (too far
            # from any plug to be seeded), but it adds graph clutter for
            # nothing, so it is cut the same way.
            wire = wire & board_mask(frame, tf, inset_mm=30)
            k = max(close, 15)
            wire = cv2.morphologyEx(wire.astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((k, k), np.uint8))
            return wire.astype(bool), None
    seg = segment(frame, tf, module_dilate=12)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bm = board_mask(frame, tf, inset_mm=30)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    conn = np.zeros(g.shape, bool)
    for c in seg.module_centres:
        conn |= ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) < conn_radius ** 2
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    valid = bm & (~conn)
    thr = max(30.0, float(np.percentile(bh[valid], 99.0)))
    w = ((bh > thr) & valid).astype(np.uint8)
    w = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return w.astype(bool), seg


def modules_relaxed(frame, tf, want=N_MODULES, clip=None, orig_idx=None):
    """Find exactly `want` modules, relaxing the size filter until they appear.

    The strict filter is right for rejecting the rig and a hand, but too tight
    when a module is partly shadowed. Rather than accept whatever count comes
    out, widen the area window step by step and stop at the first setting that
    yields the known number.

    When the clip has a precomputed class map, its green-blob centroids are
    used instead -- exact segmentation beats a size/aspect heuristic, and it
    is what `wire_mask` above uses too, so hub and cable geometry come from
    the same source.
    """
    if clip is not None and orig_idx is not None and _seg.available(clip):
        r = _seg.class_masks(clip, orig_idx, frame)
        if r is not None:
            mover, _ = r
            mc = _seg.module_centres(mover, want=want)
            return mc, {"relaxed": False, "incomplete": len(mc) != want,
                        "source": "seg_mask"}
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bm = board_mask(frame, tf, inset_mm=30)
    best = None
    for lo, hi, thr in ((30000, 50000, 130), (24000, 58000, 130),
                        (20000, 65000, 140), (16000, 75000, 150)):
        mc = detect_modules(g, bm, dark_thresh=thr, min_core_area=lo)
        mc = np.array([c for c in mc], dtype=float).reshape(-1, 2)
        if len(mc) == want:
            return mc, {"relaxed": lo != 30000}
        if best is None or abs(len(mc) - want) < abs(len(best) - want):
            best = mc
    return best, {"relaxed": True, "incomplete": True}


def _hub_connectivity(wires, modules, dist):
    """Per module: summed mask coverage of traces to the four other modules.

    The hub is the module all four cables connect to, so it is found by
    testing exactly that: treat each module in turn as the hub, grow the cable
    trace to the other four, and sum how much of each route lies on the mask.
    The real hub links to all four with well-supported routes; a module that
    cables only pass near links to none. Local ring or endpoint counts cannot
    make this distinction, because they see a passing cable the same as an
    attached one -- this uses the actual topology instead.
    """
    from .tracking import (_grow_from_hub, hub_candidates, _coverage)
    from .tracer import build_graph
    from .centerline import _resample_pixels
    modules = np.asarray(modules, float).reshape(-1, 2)
    gr = build_graph(wires)
    out = np.zeros(len(modules))
    if not gr[0]:
        return out
    for hi in range(len(modules)):
        gd, gp, gx = _grow_from_hub(gr, modules[hi])
        tot = 0.0
        for k in range(len(modules)):
            if k == hi:
                continue
            c = hub_candidates(gr, modules[hi], modules[k], gd, gp, gx, k=3)
            if not c:
                continue
            poly = c[0]["poly"]
            if np.linalg.norm(poly[0] - modules[hi]) > \
               np.linalg.norm(poly[-1] - modules[hi]):
                poly = poly[::-1]
            cl = _resample_pixels(
                np.vstack([modules[hi][None], poly, modules[k][None]]), 16)
            tot += _coverage(cl, dist)
        out[hi] = tot
    return out


def _strand_counts(wires, modules, r_in=140.0, r_out=200.0):
    """Number of separate wire strands crossing an annulus around each module."""
    modules = np.asarray(modules, float).reshape(-1, 2)
    out = np.zeros(len(modules))
    if wires is None or not wires.any():
        return out
    h, w = wires.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for i, c in enumerate(modules):
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        ring = wires & (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        n, _ = cv2.connectedComponents(ring.astype(np.uint8), 8)
        out[i] = n - 1
    return out


def identify_hub(wires, modules, r_in=140.0, r_out=200.0):
    """The hub is the module with the most cables ATTACHED, counted as strands.

    Counting wire pixels within a radius, as this used to, measures traffic
    rather than attachment: with the class map every connector sits in a ~105 px
    ring the segmenter labels as plug, so a module surrounded by empty ring
    scores low while one that unrelated cables merely pass near scores high.
    Measured, that put the hub on the centre module of 20260605_112939 instead
    of the left one, and mask coverage fell from 80% to 45%.

    Counting how many SEPARATE strands cross an annulus just outside the plug
    asks the right question. Four cables leave the hub, one leaves a mover, and
    a cable passing by crosses the annulus twice -- entering and leaving -- so
    it cannot masquerade as four attachments.
    """
    modules = np.asarray(modules, float).reshape(-1, 2)
    if wires is None or not wires.any():
        return 0
    h, w = wires.shape
    yy, xx = np.mgrid[0:h, 0:w]
    counts = []
    for c in modules:
        d2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2
        ring = wires & (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        n, _ = cv2.connectedComponents(ring.astype(np.uint8), 8)
        counts.append(n - 1)
    return int(np.argmax(counts))


def _path_overlap(p1, p2, tol=6.0, stride=3):
    """Fraction of the shorter polyline lying within `tol` px of the other.

    Two cables leaving the same hub legitimately share a few px right at the
    connector, but nothing beyond that -- they are physically different
    strands. A high fraction here means the two independent traces rode the
    same segment chain for a real stretch, which only happens when a crossing
    merged two cables in the mask and one trace's start seed landed on the
    other cable's segment instead of its own.
    """
    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
        return 0.0
    a = np.asarray(p1, float)[::stride]
    b = np.asarray(p2, float)[::stride]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return max(float((d.min(axis=1) < tol).mean()),
               float((d.min(axis=0) < tol).mean()))


def corridor_cable(wires, hub_px, mod_px, half_width=100):
    """Fallback: pixels in a straight corridor between the two known plugs."""
    ys, xs = np.nonzero(wires)
    if len(xs) < 40:
        return None
    P = np.stack([xs, ys], -1).astype(float)
    axis = np.asarray(mod_px, float) - np.asarray(hub_px, float)
    L = float(np.linalg.norm(axis))
    if L < 1:
        return None
    u = axis / L
    rel = P - hub_px
    t = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    sel = (t >= -20) & (t <= L + 20) & (perp < half_width)
    if int(sel.sum()) < 40:
        return None
    m = np.zeros(wires.shape, bool)
    pts = P[sel].astype(int)
    m[pts[:, 1], pts[:, 0]] = True
    return strand_centerline(m, 16, hub_px, mod_px)


def four_cables(frame, tf, n_nodes=16, conn_radius=135, modules=None,
                hub_index=None, use_fastdlo=False, clip=None, orig_idx=None):
    """ALWAYS return four cables, hub-first, spanning plug centre to centre.

    Order of attempts per mover: topology tracer -> straight-corridor
    extraction -> straight line between the plugs. The last is a placeholder,
    not a measurement, and is marked as such so it can be excluded from scoring
    while still keeping the cable present in the figure.
    """
    # Modules first, cables second: the plug centres define where every cable
    # must start and end, and they punch the connector bodies out of the mask.
    if modules is None:
        modules, _ = modules_relaxed(frame, tf, clip=clip, orig_idx=orig_idx)
    modules = np.asarray(modules, float).reshape(-1, 2)
    if len(modules) < 2:
        return None
    if use_fastdlo:
        from .fastdlo_mask import fastdlo_wires
        w = fastdlo_wires(frame, tf, conn_radius, module_centres=modules)
    else:
        w, _ = wire_mask(frame, tf, conn_radius, clip=clip, orig_idx=orig_idx)
    # If the caller already knows which module is the hub (because it tracked
    # them in a fixed order), honour it. Re-identifying the hub here silently
    # permutes the mover order, so cable k stopped corresponding to mover k --
    # which showed up as movers reporting 0 mm while their cable clearly moved.
    hi = identify_hub(w, modules) if hub_index is None else int(hub_index)
    hub = modules[hi]
    movers = [modules[i] for i in range(len(modules)) if i != hi]

    gr = build_graph(w)
    polys = []
    for m in movers:
        poly = trace_cable_goal(w, np.asarray(m, float), np.asarray(hub, float),
                                graph=gr)
        polys.append(poly if (poly is not None and len(poly) >= 20) else None)

    # A crossing the mask merged can make two different movers' searches
    # settle on the SAME segment chain -- measured, an unrelated pair came back
    # 97-100% identical instead of the near-0% two real cables share. Neither
    # can be trusted then, so both drop to the corridor fallback rather than
    # showing a confident, possibly-wrong route (or two cables on top of each
    # other, which is worse: it looks plausible at a glance).
    bad = set()
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            if _path_overlap(polys[a], polys[b]) > 0.4:
                bad.add(a); bad.add(b)
    for i in bad:
        polys[i] = None

    out = []
    for m, poly in zip(movers, polys):
        method, cl = "tracer", None
        if poly is not None:
            if np.linalg.norm(poly[0] - hub) > np.linalg.norm(poly[-1] - hub):
                poly = poly[::-1]
            cl = _resample_pixels(np.vstack([hub[None, :], poly, m[None, :]]),
                                 n_nodes)
        if cl is None:
            cc = corridor_cable(w, hub, m)
            if cc is not None:
                cl, method = cc, "korridor"
        if cl is None:
            cl = _resample_pixels(np.vstack([hub[None, :], m[None, :]]), n_nodes)
            method = "gerade (Platzhalter)"
        q = strand_quality(cl, tf)
        out.append({"shape_mm": tf.to_mm(cl) / 1000.0, "px": cl,
                    "mover_px": np.asarray(m, float),
                    "hub_px": np.asarray(hub, float),
                    "method": method, "slack": q["slack"]})
    return {"hub_px": hub, "movers_px": np.array(movers), "cables": out,
            "wires": w, "modules": modules}


def find_anchor_frame(cap, tf, n_frames, read_fn, tries=14, clip=None,
                      static=None):
    """A frame that shows all five modules. Returns (idx, modules) or None.

    Needed because detection fails on individual frames (shadow, a module half
    under a cable), while the rig always has five. Anchoring on a frame where
    all five are visible and tracking outward is more reliable than trying to
    force the count frame by frame.

    `static` is the clip's persistent dark content. Without it the quality gate
    fires on the rig mount and rejects every candidate, so no anchor is found
    and the ENTIRE clip is skipped -- measured, that lost 33 of 56 clips whose
    class maps showed all five modules in every sampled frame. It is computed
    here when not supplied so no caller can forget it again.
    """
    from .segment import scene_quality, static_mask
    if static is None:
        static = static_mask(cap, tf, read_fn, n_frames)
    best = None
    for idx in np.linspace(0.15 * n_frames, 0.9 * n_frames, tries).astype(int):
        fr = read_fn(cap, int(idx))
        if fr is None:
            continue
        ok, _ = scene_quality(fr, tf, static=static)
        if not ok:
            continue
        mc, info = modules_relaxed(fr, tf, clip=clip, orig_idx=int(idx))
        if mc is not None and len(mc) == N_MODULES:
            return int(idx), mc
        if mc is not None and (best is None or len(mc) > len(best[1])):
            best = (int(idx), mc)
    return best


def track_harness(cap, tf, n_frames, read_fn, half=95, search=90, clip=None,
                  static=None):
    """Carry all five modules through the clip.

    Delegates to `track_harness_permask` whenever the clip has a class map:
    detecting the plugs in every frame beats correlating one anchor forward,
    and the template scheme was measured drifting 142 px on 20260602_102713 --
    the hub marker sat beside its module, so every cable was anchored wrongly
    at both ends. The template path below remains for clips without a map.
    """
    if clip is not None:
        from . import seg_masks as _sm
        if _sm.available(clip):
            r = track_harness_permask(cap, tf, n_frames, read_fn, clip,
                                      static=static)
            if r[0] is not None:
                return r

    """Carry all five modules through the clip from an anchor frame.

    Template matching, walked outward in both directions from the anchor, so
    every frame gets five module positions even where per-frame detection would
    have found only three. Returns (tracks (T,5,2), hub_index, anchor_idx).
    """
    found = find_anchor_frame(cap, tf, n_frames, read_fn, clip=clip,
                              static=static)
    if found is None:
        return None, None, None
    anchor, mods = found

    # Vote the hub over several frames. Deciding it from the anchor frame alone
    # is not stable: on 20260605_112939 anchor 12 gives the (correct) left
    # module and anchor 60 the centre one, because a cable draped across an
    # idle module can briefly out-count the real hub. Picking the wrong hub is
    # silent -- all four traces simply aim at the wrong goal -- so it must not
    # hinge on one frame.
    votes = []
    for idx in {max(0, min(n_frames - 1, anchor + d)) for d in (-8, -4, 0, 4, 8)}:
        fr_ = read_fn(cap, int(idx))
        if fr_ is None:
            continue
        w_, _ = wire_mask(fr_, tf, clip=clip, orig_idx=int(idx))
        votes.append(identify_hub(w_, mods))
    if not votes:
        return None, None, None
    hub_i = int(np.bincount(votes, minlength=len(mods)).argmax())

    fr = read_fn(cap, anchor)

    g0 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    tmpl = []
    for p in mods:
        x, y = int(round(p[0])), int(round(p[1]))
        tmpl.append(g0[max(0, y - half):y + half, max(0, x - half):x + half])

    tracks = np.full((n_frames, len(mods), 2), np.nan)
    tracks[anchor] = mods

    def walk(rng):
        pos = np.asarray(mods, float).copy()
        for idx in rng:
            fr_ = read_fn(cap, idx)
            if fr_ is None:
                continue
            g = cv2.cvtColor(fr_, cv2.COLOR_BGR2GRAY)
            for k in range(len(pos)):
                t = tmpl[k]
                cx, cy = int(round(pos[k][0])), int(round(pos[k][1]))
                wx0, wy0 = max(0, cx - half - search), max(0, cy - half - search)
                wx1 = min(g.shape[1], cx + half + search)
                wy1 = min(g.shape[0], cy + half + search)
                win = g[wy0:wy1, wx0:wx1]
                if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                    continue
                res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                pos[k] = [wx0 + loc[0] + t.shape[1] / 2.0,
                          wy0 + loc[1] + t.shape[0] / 2.0]
            tracks[idx] = pos

    walk(range(anchor + 1, n_frames))
    walk(range(anchor - 1, -1, -1))
    return tracks, hub_i, anchor


def _clockwise_order(hub, movers):
    """Mover indices ordered clockwise around the hub, starting at 12 o'clock.

    The old ordering was whatever `connectedComponentsWithStats` happened to
    return, i.e. blob discovery order -- so M0 meant a different plug in every
    clip and the labels carried no information. Clockwise from north is
    reproducible: it depends only on geometry, not on how the segmenter
    enumerated the blobs.

    Image coordinates have y pointing DOWN, so a clockwise sweep on screen is
    the direction of increasing atan2(dx, -dy).
    """
    d = np.asarray(movers, float) - np.asarray(hub, float)
    ang = np.arctan2(d[:, 0], -d[:, 1])          # 0 = up, grows clockwise
    ang = np.mod(ang, 2 * np.pi)
    return list(np.argsort(ang))


def track_harness_permask(cap, tf, n_frames, read_fn, clip, static=None):
    """Per-frame module positions from the class map, with stable identities.

    Replaces the anchor-plus-template-matching scheme for clips that have a
    class map. That scheme fixed the modules once and carried them by image
    correlation, which drifts: measured on 20260602_102713 the hub marker ended
    up beside its module rather than on it, and every cable anchored to it was
    wrong at both ends.

    The class map locates all five plugs in every frame directly, so tracking
    is only needed to keep the IDENTITIES consistent, not the positions. Frame
    one fixes the numbering (hub first, then movers clockwise); every later
    frame assigns its detections to the previous frame's by nearest neighbour,
    which is unambiguous because plugs move ~15 px between frames while sitting
    hundreds of pixels apart.

    Returns (tracks (T,5,2), 0, anchor_idx) -- hub is always column 0, so the
    hub index it reports is 0 by construction.
    """
    from .segment import scene_quality, static_mask
    if static is None:
        static = static_mask(cap, tf, read_fn, n_frames)

    det = {}
    for idx in range(n_frames):
        fr = read_fn(cap, idx)
        if fr is None:
            continue
        mc, _ = modules_relaxed(fr, tf, clip=clip, orig_idx=idx)
        if mc is not None and len(mc) == N_MODULES:
            det[idx] = np.asarray(mc, float)
    if len(det) < 3:
        return None, None, None

    # Order of operations matters here. Identities must be stabilised FIRST and
    # the hub voted on afterwards: averaging hub POSITIONS across frames, as an
    # earlier version did, lands between two modules whenever the vote is split
    # and then picks whichever happens to be nearest -- measured, that chose the
    # centre module instead of the left one on 20260605_112939 and dropped mask
    # coverage from 80% to 45%. Voting over stable slots cannot do that: the
    # answer is a slot index, so it is either one module or another.
    idxs = sorted(det)
    anchor = idxs[len(idxs) // 2]
    from scipy.optimize import linear_sum_assignment

    tracks = np.full((n_frames, N_MODULES, 2), np.nan)
    tracks[anchor] = det[anchor]                    # arbitrary order for now

    def walk(seq):
        prev = det[anchor].copy()
        for idx in seq:
            if idx not in det:
                continue
            cur = det[idx]
            # Optimal (Hungarian) assignment frame-to-frame, minimising total
            # displacement, so two modules passing close cannot swap the way a
            # greedy nearest-neighbour match did.
            cost = np.linalg.norm(prev[:, None, :] - cur[None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(cost)
            row = np.zeros((N_MODULES, 2))
            for slot, j in zip(rows, cols):
                row[slot] = cur[j]
            tracks[idx] = row
            prev = row
    walk(range(anchor + 1, n_frames))
    walk(range(anchor - 1, -1, -1))

    # Hub by START POSITION. Every recording begins from the same reset
    # configuration, and in it the hub mover ALWAYS parks bottom-left
    # (operator-confirmed rig convention, 2026-08-19) -- so the hub is
    # simply the module nearest the frame's bottom-left corner at the start
    # of the clip. This replaces the connectivity vote, which mis-crowned a
    # single-cable mover as hub on 20260602_172926: a mover attached to the
    # hub reaches every other module along real cables (through the hub),
    # so its routes score near-perfect mask coverage too and noise decides
    # the vote. (The ring/strand count in identify_hub is ALSO unreliable
    # -- wrong on 20260602_121006 and 20260605_112939 -- hence the position
    # prior rather than yet another image heuristic; identities are already
    # stabilised across frames above, so the start-frame decision carries
    # through the whole clip.)
    #
    # "Start" means the first stabilised frame AT OR AFTER frame 8, not
    # frame 0: the first few frames of these recordings are unreliable
    # (measured on 20260602_110528, frame 0's module detections bear no
    # relation to frame 4/8's -- the same reason every pipeline in this
    # project tracks from t0=8).
    first = next((i for i in idxs if i >= 8 and np.isfinite(tracks[i]).all()),
                 next((i for i in idxs if np.isfinite(tracks[i]).all()), None))
    if first is None:
        return None, None, None
    fr0 = read_fn(cap, first)
    h_img = fr0.shape[0] if fr0 is not None else float(np.nanmax(tracks) * 1.05)
    corner = np.array([0.0, float(h_img)])
    hi = int(np.argmin(np.linalg.norm(tracks[first] - corner[None, :], axis=1)))

    movers = [j for j in range(N_MODULES) if j != hi]
    ref = tracks[anchor]
    order = [hi] + [movers[j] for j in _clockwise_order(ref[hi], ref[movers])]
    return tracks[:, order, :], 0, anchor
