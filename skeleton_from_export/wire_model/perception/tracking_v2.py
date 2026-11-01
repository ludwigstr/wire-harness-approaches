"""Four-cable tracking, second version (user, 2026-11-01: "Tracking 1000% verbessern").

What was wrong with `tracking.py`, measured on the 107 export clips of 2026-10-21/23/26/28:

1. ONE ROUTE PER TERMINAL. `_grow_from_hub` is a single Dijkstra from the hub, so every segment
   has exactly one back-pointer -- the cheapest way there. Where two cables run through a loop
   (two bows between the hub and the fan-out point) both bows lead to the same downstream segments
   and Dijkstra keeps only the cheaper bow for everything beyond. The other bow never exists as a
   candidate: measured on 20261021_180139 frame 130, cables 0 and 1 each had ONE candidate, the
   wrong one (545/985 mm instead of 620/897). No scoring can pick a route that was never offered.
   Fix: `hub_candidates_diverse` re-grows from the hub with the segments of already-accepted routes
   penalised (the diversification `candidates()` always used in the other direction), so the second
   bow comes out as a genuinely different route to the same terminal.

2. THE LENGTH PRIOR FOLLOWED THE ERROR. The arc-length target was the median over an 8-frame
   window, so a wrong route that persists for 40 frames became its own target. A cable's length is
   a constant of the rig: the target is now one value per cable per clip, the median over the
   better-covered half of the whole clip, with a tight tolerance.

3. NO TIME. With `solo_seed=True` every frame was solved on its own; only a post-hoc guard
   compared neighbours, with a 70 mm threshold a 10 % strand swap does not reach. Here the per-frame
   solver returns its top-M joint assignments and a Viterbi pass over the whole clip picks the
   sequence that minimises per-frame cost plus the shape change between consecutive frames (after
   the mover's own displacement). Single-frame flickers cost one transition each way and lose.

4. THE MASK WAS CUT AT THE TOP. `wire_mask` keeps only the board interior inset by 30 mm; the board's
   top edge sits ~100 px below the image top, so a cable running along the top edge under the gantry
   lost both its class-map pixels and any chance of being traced. The class map itself also has a
   hole there (the network does not segment the cable in front of the gantry), although the cable is
   plainly visible as a dark line. `augmented_wire_mask` keeps an 8 mm inset and adds the dark-cable
   pixels of a black-hat filter where they connect to the class map's strands.

The per-frame cost terms (coverage, overlap, turning) are reused from `tracking.py` unchanged.
"""
from __future__ import annotations

import heapq
from itertools import product

import cv2
import numpy as np

from .centerline import _resample_pixels
from .rectify import board_mask
from . import tracking as T

W_T = 1.0            # transition cost per PRIOR_SCALE px of shape change between frames
W_LENGTH = 8.0       # per unit of relative deviation from the cable's (global) length
LEN_TOL = 0.06       # free band; the clean clips' projected length varies 2-10 % over a clip
MISSING = 5.0        # per unassigned cable per frame
TOP_M = 12           # joint assignments kept per frame for the Viterbi pass
REPAIR_TOL = 0.08    # beyond this relative length error a route is replaced (see track_clip)
CARRY_COST = T.CARRY_COST


# ----------------------------------------------------------------------------- mask

def augmented_wire_mask(raw, class_cable, tf, bh_thr=30, inset_mm=8.0, text_frac=0.07,
                        plug_px=(), plate_radius=140):
    """Class-map cable pixels plus connected dark-cable pixels, inside the board.

    `raw` and `class_cable` at the same (tracker) resolution. The black-hat picks the cable up as
    a dark line on the light board; it also picks up bezel shadows and the plugs, so only its
    components that touch the class map's own strands are accepted, and the plug discs are cut.
    """
    h, w = class_cable.shape
    cab = class_cable.copy()
    cab[: int(text_frac * h)] = False
    bm = board_mask(raw, tf, inset_mm=inset_mm)
    g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    dark = (bh > bh_thr) & bm
    dark[: int(text_frac * h)] = False
    # The mover/hub plates (150 mm squares) carry black rims and printed markings that the
    # black-hat picks up as strongly as a cable; measured on 20261021_180139 they connected to the
    # strands and routes looped round the plates. No dark pixel inside a plate counts -- the class
    # map's own cable pixels there are kept.
    plates = np.zeros((h, w), np.uint8)
    for p in plug_px:
        cv2.circle(plates, (int(p[0]), int(p[1])), int(plate_radius), 1, -1)
    dark &= plates == 0
    both = cv2.morphologyEx((cab | dark).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab = cv2.connectedComponents(both, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[cab])] = True
    keep[0] = False
    wire = keep[lab] & (cab | dark)
    wire = wire & bm
    wire = cv2.morphologyEx(wire.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    return wire


def cut_plugs(wire, plug_px, radius=55):
    """Remove the plug bodies (class-map movers dilated) so they do not act as strands."""
    out = wire.copy()
    for p in plug_px:
        cv2.circle(out.view(np.uint8), (int(p[0]), int(p[1])), int(radius), 0, -1)
    return out.astype(bool)


# ----------------------------------------------------------------------------- candidates

def grow_from_hub(graph, hub_px, seg_penalty=None, start_radius=200.0, straightness=0.2,
                  max_gap=180.0, turn_weight=1.0, gap_weight=0.5, near_gap=30.0):
    """`tracking._grow_from_hub` plus a per-segment entry penalty (the diversification lever)."""
    segs, jcent, inc = graph
    hub_px = np.asarray(hub_px, float)
    pen = seg_penalty or {}
    if not segs:
        return {}, {}, {}

    def exit_of(si, e):
        xe = 0 if e == -1 else -1
        return segs[si].endpoint(xe), segs[si].direction_at(xe)

    def successors(si, e):
        xpt, xdir = exit_of(si, e)
        out, seen = [], set()
        if len(jcent):
            dj = np.linalg.norm(jcent - xpt, axis=1)
            j = int(dj.argmin())
            if dj[j] <= 15:
                for (s2, e2) in inc.get(j, []):
                    if s2 == si:
                        continue
                    align = float(-np.dot(xdir, segs[s2].direction_at(e2)))
                    if align >= straightness:
                        out.append((s2, e2, turn_weight * (1.0 - align)))
                        seen.add(s2)
        for s2, s2o in enumerate(segs):
            if s2 == si or s2 in seen:
                continue
            for e2 in (0, -1):
                p2 = s2o.endpoint(e2)
                gap = float(np.linalg.norm(p2 - xpt))
                if gap > max_gap or gap < 1e-6:
                    continue
                fwd = float(np.dot((p2 - xpt) / gap, xdir))
                if fwd < (-0.2 if gap < near_gap else 0.5):
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align) + gap_weight * gap / max_gap))
        return out

    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - hub_px))
            if d > start_radius:
                continue
            outward = s.endpoint(0 if e == -1 else -1) - s.endpoint(e)
            no = np.linalg.norm(outward)
            if no < 1e-9:
                continue
            radial = s.endpoint(e) - hub_px
            nr = np.linalg.norm(radial)
            align = float(np.dot(outward / no, radial / nr)) if nr > 1e-9 else 1.0
            if align < 0.0:
                continue
            c = d / start_radius + 0.5 * (1.0 - align) + pen.get(si, 0.0)
            st = (si, e)
            if c < dist.get(st, np.inf):
                dist[st] = c
                prev[st] = None
                heapq.heappush(pq, (c, st))
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, np.inf):
            continue
        for (s2, e2, w) in successors(*st):
            nc = c + w + pen.get(s2, 0.0)
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))
    xpts = {st: exit_of(*st)[0] for st in dist}
    return dist, prev, xpts


def hub_candidates_diverse(graph, hub_px, mover_px, k=8, rounds=4, per_round=3, penalty=1.5,
                           radius=230.0, same_tol=10.0):
    """Up to `k` genuinely different hub->mover routes.

    Round 0 is the plain growth. Each further round re-grows with the segments of every accepted
    route penalised, so a route through the OTHER bow of a loop, or along a parallel strand, is
    found even though it shares the terminal segment with an accepted one. Routes whose resampled
    shape is within `same_tol` px of an accepted one are dropped as duplicates.
    """
    hub_px = np.asarray(hub_px, float); mover_px = np.asarray(mover_px, float)
    out, pen = [], {}
    for r in range(rounds):
        dist, prev, xpts = grow_from_hub(graph, hub_px, seg_penalty=pen)
        cands = T.hub_candidates(graph, hub_px, mover_px, dist, prev, xpts, k=per_round, radius=radius)
        new = 0
        for c in cands:
            poly = c["poly"]
            if np.linalg.norm(poly[0] - hub_px) > np.linalg.norm(poly[-1] - hub_px):
                poly = poly[::-1]
            shape = _resample_pixels(np.vstack([hub_px[None], poly, mover_px[None]]), 24)
            if any(float(np.linalg.norm(shape - o["shape24"], axis=1).mean()) < same_tol for o in out):
                continue
            # the penalty is what made this route show up; score it by its own unpenalised cost
            raw_cost = c["cost"] - sum(pen.get(si, 0.0) for si in c["segs"])
            out.append({"poly": poly, "cost": raw_cost, "segs": c["segs"], "shape24": shape})
            for si in c["segs"]:
                pen[si] = pen.get(si, 0.0) + penalty
            new += 1
            if len(out) >= k:
                break
        if len(out) >= k or new == 0:
            break
    if not out:
        for c in T.candidates(graph, mover_px, hub_px, k=per_round):
            poly = c["poly"]
            if np.linalg.norm(poly[0] - hub_px) > np.linalg.norm(poly[-1] - hub_px):
                poly = poly[::-1]
            out.append({"poly": poly, "cost": c["cost"], "segs": c["segs"],
                        "shape24": _resample_pixels(np.vstack([hub_px[None], poly, mover_px[None]]), 24)})
    return out


# ----------------------------------------------------------------------------- per-frame solve

def solve_frame_topm(graph, hub_px, movers_px, dist, lengths=None, priors=None, k=8, n_nodes=16,
                     top_m=TOP_M):
    """All valid joint assignments of one frame, best `top_m` first.

    Returns (combos, fin): combos = [(score, [cand index or None per mover])], fin[i] = list of
    finished centrelines (hub -> route -> mover, n_nodes) for mover i.
    """
    n = len(movers_px)
    hub_px = np.asarray(hub_px, float)
    fin, base = [], []
    for i in range(n):
        m = np.asarray(movers_px[i], float)
        frow, brow = [], []
        for c in hub_candidates_diverse(graph, hub_px, m, k=k):
            cl = _resample_pixels(np.vstack([hub_px[None], c["poly"], m[None]]), n_nodes)
            s = c["cost"] + T.W_COVER * (1.0 - T._coverage(cl, dist))
            if lengths is not None and lengths[i]:
                rel = abs(T._arc_length(cl) - lengths[i]) / lengths[i]
                s += W_LENGTH * max(0.0, rel - LEN_TOL)
            if priors is not None and priors[i] is not None:
                d = T._shape_distance(cl, priors[i])
                if d is not None:
                    s += T.W_PRIOR * d / T.PRIOR_SCALE
            frow.append(cl); brow.append(s)
        if priors is not None and priors[i] is not None:
            cl = np.asarray(priors[i], float).copy(); cl[0], cl[-1] = hub_px, m
            frow.append(cl); brow.append(CARRY_COST + T.W_COVER * (1.0 - T._coverage(cl, dist)))
        fin.append(frow); base.append(brow)
    ov = {}
    for a in range(n):
        ra = T.FAN_FRAC * float(np.linalg.norm(np.asarray(movers_px[a]) - hub_px))
        for b in range(a + 1, n):
            rb = T.FAN_FRAC * float(np.linalg.norm(np.asarray(movers_px[b]) - hub_px))
            r = min(ra, rb)
            for ia, ca in enumerate(fin[a]):
                for ib, cb in enumerate(fin[b]):
                    ov[(a, ia, b, ib)] = T._path_overlap(ca, cb, tol=10.0, stride=1, hub=hub_px, min_r=r)
    choices = [list(range(len(fin[i]))) + [None] for i in range(n)]
    combos = []
    for combo in product(*choices):
        score, ok = 0.0, True
        for i, ci in enumerate(combo):
            score += MISSING if ci is None else base[i][ci]
        for a in range(n):
            for b in range(a + 1, n):
                if combo[a] is None or combo[b] is None:
                    continue
                f = ov[(a, combo[a], b, combo[b])]
                if f > T.MAX_FAR_OVERLAP:
                    ok = False; break
                score += T.W_OVERLAP * f
            if not ok:
                break
        if ok:
            combos.append((score, combo))
    combos.sort(key=lambda x: x[0])
    return combos[:top_m], fin


# ----------------------------------------------------------------------------- the clip

def track_clip(frames, tf, k=8, n_nodes=16, verbose=False):
    """frames: list of dicts {wire (bool mask), graph, dist, hub (px), movers (4,2 px)} in time order.
    Returns routes[t][j] (n_nodes,2 px) or None, and the global lengths used."""
    n = 4
    # pass 1: solo solve, no length prior -> global length per cable
    obs = [[] for _ in range(n)]
    for f in frames:
        combos, fin = solve_frame_topm(f["graph"], f["hub"], f["movers"], f["dist"], k=k, n_nodes=n_nodes, top_m=1)
        if combos:
            for j, ci in enumerate(combos[0][1]):
                if ci is not None:
                    cl = fin[j][ci]
                    obs[j].append((T._coverage(cl, f["dist"]), T._arc_length(cl)))
    lengths = [None] * n
    for j in range(n):
        if len(obs[j]) >= 3:
            cov_cut = np.median([c for c, _ in obs[j]])
            good = [L for c, L in obs[j] if c >= cov_cut]
            lengths[j] = float(np.median(good)) if good else None
    if verbose:
        print("global lengths px:", [None if L is None else round(L) for L in lengths], flush=True)
    # pass 2: top-M assignments per frame under the length prior
    per_frame = []
    for f in frames:
        combos, fin = solve_frame_topm(f["graph"], f["hub"], f["movers"], f["dist"], lengths=lengths,
                                       k=k, n_nodes=n_nodes)
        states = []
        for score, combo in combos:
            states.append((score, [None if ci is None else fin[j][ci] for j, ci in enumerate(combo)]))
        if not states:
            states = [(MISSING * n, [None] * n)]
        per_frame.append(states)
    # Viterbi over the states
    Tn = len(frames)
    cost = [np.array([s for s, _ in per_frame[0]])]
    back = [None]
    for t in range(1, Tn):
        prev_states, cur_states = per_frame[t - 1], per_frame[t]
        shift = frames[t]["movers"] - frames[t - 1]["movers"]
        c = np.full(len(cur_states), np.inf); b = np.zeros(len(cur_states), int)
        for ic, (sc, routes) in enumerate(cur_states):
            best, bi = np.inf, 0
            for ip, (sp, proutes) in enumerate(prev_states):
                tr = 0.0
                for j in range(n):
                    if routes[j] is None or proutes[j] is None:
                        tr += 1.0
                    else:
                        d = T._shape_distance(routes[j], proutes[j] + shift[j])
                        tr += W_T * (d or 0.0) / T.PRIOR_SCALE
                v = cost[t - 1][ip] + tr
                if v < best:
                    best, bi = v, ip
            c[ic] = best + sc; b[ic] = bi
        cost.append(c); back.append(b)
    seq = [int(np.argmin(cost[-1]))]
    for t in range(Tn - 1, 0, -1):
        seq.append(int(back[t][seq[-1]]))
    seq = seq[::-1]
    routes = [list(per_frame[t][seq[t]][1]) for t in range(Tn)]
    # LENGTH REPAIR. A cable is inextensible, so a route more than REPAIR_TOL off its clip length is
    # on the wrong strand or cut short where the cable is hidden (under the gantry carriage, measured
    # on 20261028_123828: physically occluded, no segmentation can see it). Such frames, and frames
    # with no route, take the shape of the nearest frame whose route has the right length, mapped
    # onto the current hub->mover chord. `carried` records every replaced (frame, cable).
    carried = np.zeros((Tn, n), bool)
    for j in range(n):
        Lj = lengths[j]

        def ok(t):
            r = routes[t][j]
            return r is not None and (Lj is None or abs(T._arc_length(r) - Lj) / Lj <= REPAIR_TOL)
        good = [t for t in range(Tn) if ok(t)]
        if not good:
            good = [t for t in range(Tn) if routes[t][j] is not None]
        if not good:
            continue
        good_set = set(good)
        for t in range(Tn):
            if t in good_set:
                continue
            ref = min(good, key=lambda u: abs(u - t))
            routes[t][j] = _chord_map(routes[ref][j], frames[ref]["hub"], frames[ref]["movers"][j],
                                      frames[t]["hub"], frames[t]["movers"][j])
            carried[t, j] = True
    return routes, lengths, carried


def _chord_map(r, h0, m0, h1, m1):
    r = np.asarray(r, float); h0 = np.asarray(h0, float); m0 = np.asarray(m0, float)
    h1 = np.asarray(h1, float); m1 = np.asarray(m1, float)
    d0 = m0 - h0; L0 = float(np.hypot(*d0))
    if L0 < 1e-6:
        return r + (m1 - m0)
    a0 = np.arctan2(d0[1], d0[0]); R0 = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
    q = (r - h0) @ R0.T / L0
    d1 = m1 - h1; L1 = float(np.hypot(*d1)); a1 = np.arctan2(d1[1], d1[0])
    R1 = np.array([[np.cos(a1), -np.sin(a1)], [np.sin(a1), np.cos(a1)]])
    return (q * L1) @ R1.T + h1
