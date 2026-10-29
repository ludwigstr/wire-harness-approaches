"""Topology-aware DLO tracer: follow each cable through crossings.

The problem every simpler method failed on: where two cables overlap they merge
into one connected region, so a per-component centerline wanders across both
(measured slack drops from ~0.9 to 0.44), and a corridor around the straight
hub->module line only works while no other cable happens to lie inside it.

This follows the FASTDLO / RT-DLO idea, which the Caporali paper itself cites:

  1. skeletonise the wire mask,
  2. cut it at junction pixels into simple SEGMENTS,
  3. at each junction, decide which segment-ends belong to the SAME cable by
     direction continuity -- a cable crossing another one keeps going straight,
     so the correct pairing is the one whose tangents are most anti-parallel,
  4. chain segments into one path per cable, anchored at the known plug ends.

The pairing rule is geometric rather than learned (FASTDLO uses a small network
for it). That is deliberate: here the cables are smooth and low-curvature, only
a handful meet at once, and both endpoints of every cable are known from module
detection, so a learned predictor would add opacity without adding information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(m * 255) > 0
    from skimage.morphology import skeletonize as sk_skel   # pragma: no cover
    return sk_skel(m > 0)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)


# ring of 8 neighbours in cyclic order, needed for the crossing number
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def crossing_number(skel: np.ndarray) -> np.ndarray:
    """Rutovitz crossing number: 1 = endpoint, 2 = path pixel, >=3 = branch.

    Counting raw 8-neighbours instead is a trap: on a diagonal 1-px line every
    staircase step has THREE 8-neighbours, so an ordinary line pixel is flagged
    as a junction. Measured on one frame that labelled 1664 of 4075 skeleton
    pixels (41%) as branches, shattering the segment graph into unusable
    fragments. The crossing number counts 0->1 transitions around the ring and
    is immune to that.
    """
    s = skel.astype(np.int16)
    ring = [np.roll(np.roll(s, dy, axis=0), dx, axis=1) for dy, dx in _RING]
    cn = np.zeros_like(s)
    for i in range(len(ring)):
        cn += np.abs(ring[i] - ring[(i + 1) % len(ring)])
    return (cn // 2) * skel


@dataclass
class Segment:
    """A simple (junction-free) chain of skeleton pixels."""
    pts: np.ndarray                    # (P, 2) as (x, y), ordered
    ends: tuple = field(default=())     # the two endpoint coordinates

    def direction_at(self, which: int, span: int = 18) -> np.ndarray:
        """Unit tangent pointing OUT of the segment at end `which` (0 or -1)."""
        p = self.pts
        n = min(span, len(p) - 1)
        if n < 1:
            return np.zeros(2)
        if which == 0:
            v = p[0] - p[n]
        else:
            v = p[-1] - p[-1 - n]
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-9 else np.zeros(2)

    def endpoint(self, which: int) -> np.ndarray:
        return self.pts[0] if which == 0 else self.pts[-1]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())


def _order_chain(pix: np.ndarray) -> np.ndarray:
    """Order the pixels of a simple path component into a chain."""
    if len(pix) <= 2:
        return pix
    key = {(int(x), int(y)): i for i, (x, y) in enumerate(pix)}
    deg = np.zeros(len(pix), int)
    nbrs = [[] for _ in range(len(pix))]
    for i, (x, y) in enumerate(pix):
        for dx, dy in _NEIGH:
            j = key.get((int(x) + dx, int(y) + dy))
            if j is not None:
                nbrs[i].append(j)
                deg[i] += 1
    start = int(np.argmin(deg)) if (deg == 1).any() else 0
    order, seen = [start], {start}
    cur = start
    while True:
        nxt = [j for j in nbrs[cur] if j not in seen]
        if not nxt:
            break
        cur = nxt[0]
        seen.add(cur)
        order.append(cur)
    return pix[order]


def build_segments(mask: np.ndarray, min_len: int = 8):
    """Skeletonise and split at junctions.

    Returns (segments, junction_labels, junction_centres).
    """
    skel = skeletonize(mask)
    cn = crossing_number(skel)
    junction = skel & (cn >= 3)
    # junction pixels come in small clumps; dilate slightly so a clump is one node
    jd = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nj, jlab = cv2.connectedComponents(jd.astype(np.uint8), 8)
    jcent = []
    for i in range(1, nj):
        ys, xs = np.nonzero(jlab == i)
        jcent.append(np.array([xs.mean(), ys.mean()]))

    body = skel & (~jd)
    nseg, slab = cv2.connectedComponents(body.astype(np.uint8), 8)
    segs = []
    for i in range(1, nseg):
        ys, xs = np.nonzero(slab == i)
        if len(xs) < min_len:
            continue
        pts = _order_chain(np.stack([xs, ys], -1).astype(float))
        segs.append(Segment(pts=pts, ends=(pts[0], pts[-1])))
    return segs, jlab, (np.array(jcent) if jcent else np.zeros((0, 2)))


def _incident(segs, jcent, radius: float = 9.0):
    """Which segment-ends touch which junction. -> dict junction -> [(seg, end)]"""
    inc = {j: [] for j in range(len(jcent))}
    for si, s in enumerate(segs):
        for e in (0, -1):
            p = s.endpoint(e)
            if len(jcent) == 0:
                continue
            d = np.linalg.norm(jcent - p, axis=1)
            j = int(d.argmin())
            if d[j] <= radius + 6:
                inc[j].append((si, e))
    return inc


def build_graph(mask, min_len: int = 12):
    """Segment graph of one frame, reusable for every cable in that frame.

    Building it inside the per-cable trace re-skeletonises the whole image once
    per cable; with 5 cables that is a 5x cost for identical work and it
    dominated batch runtime.
    """
    segs, _, jcent = build_segments(mask, min_len=min_len)
    return segs, jcent, _incident(segs, jcent)


def trace_cable_goal(mask, start_px, goal_px, min_seg_len: int = 12,
                     straightness: float = 0.2, max_gap: float = 60.0,
                     start_radius: float = 170.0, goal_radius: float = 200.0,
                     turn_weight: float = 1.0, gap_weight: float = 0.5,
                     graph=None):
    """Trace a cable by searching for the least-turning path to a KNOWN goal.

    The greedy "always continue straightest" walk has one failure it cannot
    escape: where the whole harness leaves the hub, every cable points the same
    way, so continuity alone cannot say which strand is which and the trace
    hops onto a neighbour. Measured, that put one cable's trace on top of
    another's, and it still looked healthy by shape metrics (slack 0.91,
    monotonic) because a wrong-but-smooth curve passes those.

    Carrying the destination fixes it: both ends of every cable are known, so
    this searches the segment graph (Dijkstra over states = (segment, entry
    end)) for the cheapest path that actually ARRIVES near `goal_px`, with cost
    charged for turning and for bridging gaps. A path onto the neighbouring
    cable is no longer free -- it ends up somewhere else, so it is simply not a
    solution.

    `start_radius` is the one parameter that must stay tight. Seeding every
    segment end within a generous radius lets the search simply START on the
    neighbouring cable when that runs straighter to the hub -- measured, at 260
    px one cable's trace sat on another's the whole way. Just past the 135 px
    connector mask (170 px) the only reachable seeds belong to the cable that
    actually leaves this plug: the start distance then pins to 125-136 px
    (mask edge), versus 140-182 px when the search is free to drift.

    Returns (P, 2) pixel polyline from start to goal, or None if no path
    reaches the goal.
    """
    import heapq

    if graph is None:
        segs, _, jcent = build_segments(mask, min_len=min_seg_len)
        inc = _incident(segs, jcent)
    else:
        segs, jcent, inc = graph
    if not segs:
        return None
    start_px = np.asarray(start_px, float)
    goal_px = np.asarray(goal_px, float)
    to_goal = goal_px - start_px
    ng = np.linalg.norm(to_goal)
    to_goal = to_goal / ng if ng > 1e-9 else np.zeros(2)

    def exit_of(si, e):
        """Exit end index, point and outward tangent after traversing seg si."""
        xe = 0 if e == -1 else -1
        return xe, segs[si].endpoint(xe), segs[si].direction_at(xe)

    def successors(si, e):
        """States reachable after traversing (si, e), with transition cost."""
        _, xpt, xdir = exit_of(si, e)
        out = []
        seen = set()
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
                if float(np.dot((p2 - xpt) / gap, xdir)) < 0.5:
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align)
                            + gap_weight * gap / max_gap))
        return out

    # seed with every segment end near the start, cheaper if it heads goalward
    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - start_px))
            if d > start_radius:
                continue
            heading = -s.direction_at(e)
            c = d / start_radius + 0.5 * (1.0 - float(np.dot(heading, to_goal)))
            st = (si, e)
            if c < dist.get(st, np.inf):
                dist[st] = c
                prev[st] = None
                heapq.heappush(pq, (c, st))
    if not pq:
        return None

    goal_state, best_goal = None, np.inf
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, np.inf):
            continue
        si, e = st
        _, xpt, _ = exit_of(si, e)
        if float(np.linalg.norm(xpt - goal_px)) <= goal_radius and c < best_goal:
            best_goal, goal_state = c, st
            continue                      # a cheaper arrival may still appear
        if c > best_goal:
            continue
        for (s2, e2, w) in successors(si, e):
            nc = c + w
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))

    if goal_state is None:
        return None
    chain, st = [], goal_state
    while st is not None:
        si, e = st
        pts = segs[si].pts
        chain.append(pts if e == 0 else pts[::-1])
        st = prev.get(st)
    poly = np.concatenate(chain[::-1], axis=0)
    if np.linalg.norm(poly[0] - start_px) > np.linalg.norm(poly[-1] - start_px):
        poly = poly[::-1]
    return poly


def trace_cable(mask, start_px, goal_px, min_seg_len: int = 8,
                straightness: float = 0.35, max_gap: float = 60.0,
                start_radius: float = 260.0, max_steps: int = 40):
    """Trace one cable from `start_px` to `goal_px` through crossings.

    Walks the segment graph: enter the segment whose end lies nearest the start,
    and at every junction continue into the segment-end whose tangent best
    OPPOSES the incoming one (i.e. carries straight on through the crossing).
    `straightness` is the minimum |cos| required to accept a continuation; below
    it the trace stops rather than turning a corner into a different cable.

    Returns (P, 2) pixel polyline, or None if no usable segment was found.
    """
    segs, jlab, jcent = build_segments(mask, min_len=min_seg_len)
    if not segs:
        return None
    inc = _incident(segs, jcent)
    start_px = np.asarray(start_px, float)
    goal_px = np.asarray(goal_px, float)

    # Choose the segment to start from by proximity AND substance AND heading.
    # Nearest-end alone is a trap: right at a plug the cable breaks into a
    # cluster of 15-70 px fragments (module edge, shadow), so the closest end
    # belongs to a stub and the trace dies inside it -- measured, one cable
    # stopped after 99 px while the real 390 px segment lay 196 px away.
    to_goal = goal_px - start_px
    ng = np.linalg.norm(to_goal)
    to_goal = to_goal / ng if ng > 1e-9 else np.zeros(2)
    best, bscore = None, -np.inf
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - start_px))
            if d > start_radius:
                continue
            heading = -s.direction_at(e)          # direction INTO the segment
            score = (float(np.dot(heading, to_goal))
                     + min(1.0, s.length / 250.0)
                     - d / start_radius)
            if score > bscore:
                bscore, best = score, (si, e)
    if best is None:                               # nothing in range: fall back
        bd = np.inf
        for si, s in enumerate(segs):
            for e in (0, -1):
                d = np.linalg.norm(s.endpoint(e) - start_px)
                if d < bd:
                    bd, best = d, (si, e)
    if best is None:
        return None

    si, e = best
    path = []
    used = set()
    for _ in range(max_steps):
        if si in used:
            break
        used.add(si)
        pts = segs[si].pts
        chain = pts if e == 0 else pts[::-1]      # walk away from the entry end
        path.append(chain)
        exit_pt = chain[-1]
        exit_dir = segs[si].direction_at(0 if e == -1 else -1)

        cand, cbest = None, -1.0

        # (a) continuation through a junction we have arrived at
        if len(jcent):
            dj = np.linalg.norm(jcent - exit_pt, axis=1)
            j = int(dj.argmin())
            if dj[j] <= 15:
                for (s2, e2) in inc.get(j, []):
                    if s2 == si or s2 in used:
                        continue
                    d2 = segs[s2].direction_at(e2)
                    # exit_dir points along travel; a straight continuation
                    # leaves the junction pointing back the opposite way
                    align = float(-np.dot(exit_dir, d2))
                    if align > cbest:
                        cbest, cand = align, (s2, e2)

        # (b) bridge a gap. A cable is not always skeletonised as one connected
        # piece: glare, a plug shadow or a neighbouring bundle can break it, and
        # stopping there truncated real cables (measured: one trace ended 667 px
        # short of the hub). Accept a jump to any unused segment end that lies
        # AHEAD in the travel direction, within `max_gap`, and continues straight.
        if cand is None or cbest < straightness:
            for s2, s2obj in enumerate(segs):
                if s2 == si or s2 in used:
                    continue
                for e2 in (0, -1):
                    p2 = s2obj.endpoint(e2)
                    gap = float(np.linalg.norm(p2 - exit_pt))
                    if gap > max_gap or gap < 1e-6:
                        continue
                    to_cand = (p2 - exit_pt) / gap
                    if float(np.dot(to_cand, exit_dir)) < 0.5:
                        continue                   # not ahead of us
                    d2 = s2obj.direction_at(e2)
                    align = float(-np.dot(exit_dir, d2))
                    # prefer straight and near: mild distance penalty
                    score = align - 0.25 * gap / max_gap
                    if score > cbest:
                        cbest, cand = score, (s2, e2)

        if cand is None or cbest < straightness:
            break
        si, e = cand

    if not path:
        return None
    poly = np.concatenate(path, axis=0)
    # orient hub->goal
    if np.linalg.norm(poly[0] - start_px) > np.linalg.norm(poly[-1] - start_px):
        poly = poly[::-1]
    return poly


def trace_all(mask, hub_px, module_pxs, **kw):
    """Trace one cable per peripheral module, each anchored hub -> module."""
    out = []
    for m in module_pxs:
        p = trace_cable(mask, m, hub_px, **kw)     # start at the plug, walk to hub
        out.append(None if p is None else p[::-1])  # return hub-first
    return out
