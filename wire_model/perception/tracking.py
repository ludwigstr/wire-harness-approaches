"""Track all four cables jointly, through time.

The previous tracer solved the wrong problem. It answered "what is the cheapest
route from mover k to the hub?" once per mover, once per frame -- four
independent questions on every frame, with nothing tying the answers together.
Two failure modes follow directly from that, and both were observed:

  * Where a crossing merges two cables in the mask, the cheapest route for
    mover A and the cheapest route for mover B can be THE SAME chain of
    segments. Measured, an unrelated pair came back 97-100% identical. Nothing
    in a per-cable search can notice this, because each search is individually
    optimal.
  * A frame where all four cables trace perfectly tells the next frame
    nothing. The movers shift ~15 px between frames and the cables barely
    change, yet every frame re-derived its answer from scratch and was free to
    pick a different branch at the same ambiguous crossing.

So this module changes the question to the one the rig actually poses:

    Which ASSIGNMENT of four disjoint routes -- one per mover, all ending at
    the hub -- best explains this frame, given what the cables looked like in
    the previous frame?

Two mechanisms implement it:

1. JOINT ASSIGNMENT (`solve_frame`). Each mover contributes several diverse
   candidate routes rather than one winner. The combination is then scored as
   a whole, with a penalty for candidates that share a path, and combinations
   where any pair overlaps badly are rejected outright. A route that steals
   another cable's segments is no longer individually optimal -- it makes the
   assignment worse -- so it loses to the correct one instead of having to be
   detected and discarded afterwards.

2. TEMPORAL PRIOR (`HarnessTracker`). Frames are processed in order from an
   anchor frame outward, and each frame's candidates are scored against the
   previous frame's cable, shifted by that mover's measured displacement. At
   an ambiguous crossing the cable that went left last frame still goes left,
   which is exactly the information a per-frame search does not have.

The anchor is the frame in the window with the most confident joint solution,
not the first one -- the same reasoning as `track_harness` uses for modules,
for the same reason: starting from a frame that happens to be ambiguous poisons
everything downstream of it.
"""

from __future__ import annotations

import heapq
from itertools import product

import cv2

import numpy as np

from .centerline import _resample_pixels, strand_quality
from .harness import N_MOVERS, corridor_cable, wire_mask
from .tracer import build_graph

# Scoring weights. path costs are in "turn units" (0 = perfectly straight
# continuation, ~1 per right-angle turn); the other terms are normalised to the
# same scale so no single term can dominate the assignment.
W_PRIOR = 1.0         # per PRIOR_SCALE px of drift from the previous frame
W_COVER = 6.0         # per unit of route NOT supported by mask evidence
W_OVERLAP = 3.0       # per unit of shared path in the fanned-out region
PRIOR_SCALE = 60.0    # px; cables move far less than this between frames
COVER_TOL = 14.0      # px; how far off the mask a node may sit and still count
FAN_FRAC = 0.45       # beyond this fraction of the hub->mover distance the
                      # cables have separated, so sharing is no longer legitimate
MAX_FAR_OVERLAP = 0.5  # sharing THAT much of the fanned-out region = same cable
# Price of explaining a cable by memory instead of evidence. It has to exceed
# a typical trace's total cost (~2-3.5 in these units), or memory wins even
# when a perfectly good fresh trace exists and the tracker degenerates into
# rigidly translating one anchor frame -- measured at 0.8, where 30 of 40
# cables came back carried. Set above that band, memory wins only when the
# fresh evidence is materially worse, which is the intent.
CARRY_COST = 2.5

# A cable is INEXTENSIBLE: its arc length is the same in every frame of a clip,
# whatever the movers do. That is the strongest constraint available here and
# nothing else in the cost function encodes it -- which is why truncated routes
# kept winning. Every observed failure of this kind was a route far SHORTER
# than the cable it claims to be: a stub ending in mid-air, or a straight
# shortcut across a loop the real cable makes. Both are impossible once length
# is pinned, so they stop being candidates rather than having to be out-scored.
W_LENGTH = 5.0        # per unit of relative deviation from the cable's length
LEN_TOL = 0.12        # deviations below this are free (tracing noise, slack)



def _path_overlap(p1, p2, tol=6.0, stride=3, hub=None, min_r=0.0):
    """Fraction of one route lying within `tol` px of the other.

    With `hub`/`min_r`, only points farther than `min_r` from the hub count.
    That distinction matters more than it looks: the four cables leave the hub
    as a BUNDLE and run as one thick strand for a stretch before fanning out,
    so near the hub two routes sharing segments is the physically correct
    answer, not an error. An earlier version penalised all sharing equally and
    pushed one of every bundled pair onto a straight-line fallback -- the exact
    artefact that made cables to M1/M2 look wrong. Out in the fan, where the
    strands are visibly separate, sharing does mean one route stole the other's.
    """
    if p1 is None or p2 is None or len(p1) < 2 or len(p2) < 2:
        return 0.0
    a = np.asarray(p1, float)[::stride]
    b = np.asarray(p2, float)[::stride]
    if hub is not None and min_r > 0:
        a = a[np.linalg.norm(a - hub, axis=1) > min_r]
        b = b[np.linalg.norm(b - hub, axis=1) > min_r]
        if len(a) < 2 or len(b) < 2:
            return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return max(float((d.min(axis=1) < tol).mean()),
               float((d.min(axis=0) < tol).mean()))


def _coverage(centerline, dist, tol=COVER_TOL, samples=200):
    """Fraction of the finished cable actually supported by mask evidence.

    This is the criterion that decides whether a route is real, and it replaces
    "do two routes overlap?" as the primary quality signal. A route that runs
    along a genuine strand scores near 1.0; one that leaves the mask and
    straight-lines across empty board to reach its mover -- which is what every
    corridor/placeholder fallback does, and what a trace that stops early does
    too -- scores low in exact proportion to how much of it was invented.

    `dist` is the distance transform of the INVERTED mask, so a lookup gives
    the distance from that pixel to the nearest wire pixel.
    """
    if centerline is None or len(centerline) < 2:
        return 0.0
    P = _resample_pixels(np.asarray(centerline, float), samples)
    h, w = dist.shape
    x = np.clip(P[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(P[:, 1].round().astype(int), 0, h - 1)
    return float((dist[y, x] <= tol).mean())


def _shape_distance(poly, prior, n=16):
    """Mean node distance (px) between a route and a prior shape."""
    if poly is None or prior is None or len(poly) < 2 or len(prior) < 2:
        return None
    a = _resample_pixels(np.asarray(poly, float), n)
    b = _resample_pixels(np.asarray(prior, float), n)
    return float(np.linalg.norm(a - b, axis=1).mean())


def _search(graph, start_px, goal_px, seg_penalty=None, straightness=0.2,
            max_gap=60.0, start_radius=170.0, goal_radius=200.0,
            turn_weight=1.0, gap_weight=0.5):
    """Cheapest route start->goal over the segment graph.

    Same cost model as the single-cable tracer (turning and gap bridging), plus
    `seg_penalty`: extra cost for entering a given segment. That is what makes
    DIVERSE candidates possible -- re-running with the previous winner's
    segments penalised yields the best genuinely different route rather than a
    trivial variation of the same one.

    Returns (polyline, cost, set_of_segment_indices) or (None, inf, set()).
    """
    segs, jcent, inc = graph
    if not segs:
        return None, np.inf, set()
    pen = seg_penalty or {}
    start_px = np.asarray(start_px, float)
    goal_px = np.asarray(goal_px, float)
    to_goal = goal_px - start_px
    ng = np.linalg.norm(to_goal)
    to_goal = to_goal / ng if ng > 1e-9 else np.zeros(2)

    def exit_of(si, e):
        xe = 0 if e == -1 else -1
        return xe, segs[si].endpoint(xe), segs[si].direction_at(xe)

    def successors(si, e):
        _, xpt, xdir = exit_of(si, e)
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
                if float(np.dot((p2 - xpt) / gap, xdir)) < 0.5:
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align)
                            + gap_weight * gap / max_gap))
        return out

    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - start_px))
            if d > start_radius:
                continue
            heading = -s.direction_at(e)
            c = (d / start_radius + 0.5 * (1.0 - float(np.dot(heading, to_goal)))
                 + pen.get(si, 0.0))
            st = (si, e)
            if c < dist.get(st, np.inf):
                dist[st] = c
                prev[st] = None
                heapq.heappush(pq, (c, st))
    if not pq:
        return None, np.inf, set()

    goal_state, best_goal = None, np.inf
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, np.inf):
            continue
        si, e = st
        _, xpt, _ = exit_of(si, e)
        if float(np.linalg.norm(xpt - goal_px)) <= goal_radius and c < best_goal:
            best_goal, goal_state = c, st
            continue
        if c > best_goal:
            continue
        for (s2, e2, w) in successors(si, e):
            nc = c + w + pen.get(s2, 0.0)
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))

    if goal_state is None:
        return None, np.inf, set()
    chain, used, st = [], set(), goal_state
    while st is not None:
        si, e = st
        used.add(si)
        pts = segs[si].pts
        chain.append(pts if e == 0 else pts[::-1])
        st = prev.get(st)
    poly = np.concatenate(chain[::-1], axis=0)
    if np.linalg.norm(poly[0] - start_px) > np.linalg.norm(poly[-1] - start_px):
        poly = poly[::-1]
    return poly, float(best_goal), used


def _grow_from_hub(graph, hub_px, start_radius=200.0, straightness=0.2,
                   max_gap=180.0, turn_weight=1.0, gap_weight=0.5,
                   near_gap=30.0):
    """Grow outward from the hub over the whole segment graph, once.

    This is the reverse of searching mover -> hub, and the direction matters
    because the evidence is not symmetric. The class map labels a ~105 px ring
    around EVERY connector as plug rather than wire, so the mover end of a
    cable is precisely where there is least to go on -- a search seeded there
    starts half-blind, and on a short cable (hub->M2 is 338 px, of which 217 px
    fall inside those two rings) it is starting on almost nothing.

    At the hub the four cables leave as one thick, unambiguous bundle. Growing
    outward from there follows the real topology: the bundle splits at genuine
    branch points, and each branch eventually arrives somewhere. Which mover it
    arrives at is then read off at the end rather than assumed at the start.

    One growth serves all four cables, so shared trunk segments are traversed
    once and cost nothing extra -- bundling is free, as it physically should be.

    `max_gap` is deliberately large. The class map is clean but breaks the
    cables into ~11 pieces per frame, mostly where a cable runs beneath a plug
    body and gets labelled as module rather than wire; measured, those breaks
    run 150-180 px. At the old 60 px the search simply could not cross them and
    the route dropped to a straight shortcut -- one cable scored 17% coverage
    for exactly this reason, and rose to 72% once the bridge was allowed. A
    generous bridge is only safe because W_LENGTH and the coverage term now
    police it: a jump onto the wrong strand changes the arc length and leaves
    the mask, so it costs more than it saves.

    Returns (cost, back-pointer, exit-point) maps keyed by state (seg, entry_end).
    """
    segs, jcent, inc = graph
    hub_px = np.asarray(hub_px, float)
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
                # The "is it in front of me?" test only means something once
                # the gap is longer than the noise in an endpoint's position.
                # Over 8 px the gap vector is essentially pixel noise, and
                # demanding it point within 60 deg of the tangent rejected a
                # perfectly good continuation at 0.45 against a 0.50 bar --
                # measured, that stalled the growth at 5 of 17 segments and
                # left the M0 cable unreachable. Below `near_gap` only a
                # genuine U-turn is excluded; beyond it the full test applies.
                fwd = float(np.dot((p2 - xpt) / gap, xdir))
                if fwd < (-0.2 if gap < near_gap else 0.5):
                    continue
                align = float(-np.dot(xdir, s2o.direction_at(e2)))
                if align < straightness:
                    continue
                out.append((s2, e2, turn_weight * (1.0 - align)
                            + gap_weight * gap / max_gap))
        return out

    pq, dist, prev = [], {}, {}
    for si, s in enumerate(segs):
        for e in (0, -1):
            d = float(np.linalg.norm(s.endpoint(e) - hub_px))
            if d > start_radius:
                continue
            # heading must lead AWAY from the hub: we are growing outward
            outward = s.endpoint(0 if e == -1 else -1) - s.endpoint(e)
            no = np.linalg.norm(outward)
            if no < 1e-9:
                continue
            radial = (s.endpoint(e) - hub_px)
            nr = np.linalg.norm(radial)
            align = float(np.dot(outward / no, radial / nr)) if nr > 1e-9 else 1.0
            if align < 0.0:
                continue
            c = d / start_radius + 0.5 * (1.0 - align)
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
            nc = c + w
            st2 = (s2, e2)
            if nc < dist.get(st2, np.inf):
                dist[st2] = nc
                prev[st2] = st
                heapq.heappush(pq, (nc, st2))

    xpts = {st: exit_of(*st)[0] for st in dist}
    return dist, prev, xpts


def hub_candidates(graph, hub_px, mover_px, dist, prev, xpts, k=6,
                   radius=230.0):
    """Routes from the hub growth that arrive near `mover_px`, cheapest first.

    Arrival is judged generously (`radius` covers the connector's drait-free
    ring), and candidates are deduplicated by the segment they end on, so the
    k returned are genuinely different branches rather than k truncations of
    one branch.
    """
    segs = graph[0]
    mover_px = np.asarray(mover_px, float)
    by_terminal = {}
    for st, c in dist.items():
        d = float(np.linalg.norm(xpts[st] - mover_px))
        if d > radius:
            continue
        score = c + d / radius
        cur = by_terminal.get(st[0])
        if cur is None or score < cur[0]:
            by_terminal[st[0]] = (score, st)

    out = []
    for score, st in sorted(by_terminal.values())[:k]:
        chain, used, cur = [], set(), st
        while cur is not None:
            si, e = cur
            used.add(si)
            pts = segs[si].pts
            chain.append(pts if e == 0 else pts[::-1])
            cur = prev.get(cur)
        poly = np.concatenate(chain[::-1], axis=0)
        out.append({"poly": poly, "cost": score, "segs": used})
    return out


def candidates(graph, start_px, goal_px, k=4, penalty=1.5, **kw):
    """Up to `k` diverse routes start->goal, cheapest first.

    Diversity comes from penalising the segments each accepted route used, so
    the next search must find a materially different way through rather than
    re-deriving the same chain.
    """
    out, pen = [], {}
    for _ in range(k):
        poly, cost, used = _search(graph, start_px, goal_px, seg_penalty=pen, **kw)
        if poly is None or len(poly) < 20:
            break
        out.append({"poly": poly, "cost": cost, "segs": used})
        for si in used:
            pen[si] = pen.get(si, 0.0) + penalty
    return out


def _arc_length(poly):
    return float(np.linalg.norm(np.diff(np.asarray(poly, float), axis=0),
                                axis=1).sum())


def solve_frame(graph, hub_px, movers_px, dist=None, priors=None, k=6,
                n_nodes=16, lengths=None):
    """Choose one route per mover as a JOINT assignment.

    `dist` is the distance transform of the inverted wire mask; it is what
    makes the primary quality term -- mask coverage -- computable. `priors` is
    an optional list (one per mover, may contain None) of the previous frame's
    route already shifted by that mover's displacement.

    Candidates are scored as the FINISHED cable (hub -> traced route -> mover),
    not as the bare traced fragment, because that is what the pipeline consumes
    and it is where an unsupported straight-line shortcut becomes visible.

    Returns (routes, score); routes[i] is a polyline or None.
    """
    n = len(movers_px)
    # ONE growth outward from the hub serves every mover: the cables share a
    # trunk, so searching each of them inward from its own plug re-solved the
    # same bundle four times over -- from the end where the connector ring
    # leaves least to see. See `_grow_from_hub`.
    gdist, gprev, gxpts = _grow_from_hub(graph, hub_px)

    # Finished centreline per candidate, plus its coverage: both are needed
    # repeatedly below, and both are independent of the chosen combination.
    #
    # The carried-over shape from the previous frame is entered here as just
    # another candidate rather than as a fallback used when tracing fails.
    # That matters: as a fallback it was never checked against the current
    # frame, so a cable that had genuinely moved kept being redrawn where it
    # used to be (measured: 40% mask coverage against ~78% for a real trace --
    # the artefact behind the wrong M2 cable). Scored as a candidate it must
    # beat the fresh evidence on the SAME cost function, coverage included, so
    # it survives exactly when it is still the better explanation.
    fin, base, kind = [], [], []
    for i in range(n):
        m = movers_px[i]
        frow, brow, krow = [], [], []
        cands = hub_candidates(graph, hub_px, m, gdist, gprev, gxpts, k=k)
        if not cands:
            # No branch of the hub growth reaches this mover -- the mask is
            # broken between them. Fall back to searching inward from the
            # mover, which can still pick up an isolated fragment near it.
            cands = candidates(graph, m, hub_px, k=k)
        for c in cands:
            poly = c["poly"]
            if np.linalg.norm(poly[0] - hub_px) > np.linalg.norm(poly[-1] - hub_px):
                poly = poly[::-1]
            cl = _resample_pixels(
                np.vstack([hub_px[None, :], poly, m[None, :]]), n_nodes)
            s = c["cost"]
            if dist is not None:
                s += W_COVER * (1.0 - _coverage(cl, dist))
            if priors is not None and priors[i] is not None:
                d = _shape_distance(cl, priors[i])
                if d is not None:
                    s += W_PRIOR * d / PRIOR_SCALE
            if lengths is not None and lengths[i]:
                rel = abs(_arc_length(cl) - lengths[i]) / lengths[i]
                s += W_LENGTH * max(0.0, rel - LEN_TOL)
            frow.append(cl); brow.append(s); krow.append("trace")
        if priors is not None and priors[i] is not None:
            cl = np.asarray(priors[i], float).copy()
            cl[0], cl[-1] = hub_px, m       # re-anchor to the current plugs
            s = CARRY_COST
            if dist is not None:
                s += W_COVER * (1.0 - _coverage(cl, dist))
            frow.append(cl); brow.append(s); krow.append("carry")
        fin.append(frow); base.append(brow); kind.append(krow)


    # Pairwise overlap in the FANNED-OUT region only -- see `_path_overlap`.
    # Cached because the brute-force product revisits the same pairs.
    ov = {}
    for a in range(n):
        ra = FAN_FRAC * float(np.linalg.norm(movers_px[a] - hub_px))
        for b in range(a + 1, n):
            rb = FAN_FRAC * float(np.linalg.norm(movers_px[b] - hub_px))
            r = min(ra, rb)
            for ia, ca in enumerate(fin[a]):
                for ib, cb in enumerate(fin[b]):
                    ov[(a, ia, b, ib)] = _path_overlap(
                        ca, cb, tol=10.0, stride=1, hub=hub_px, min_r=r)

    # With <=6 candidates per mover this is at most 7^4 combinations -- small
    # enough that exhaustive search beats any cleverer method, and it gives the
    # true optimum rather than a greedy approximation.
    choices = [list(range(len(fin[i]))) + [None] for i in range(n)]
    best, best_score = None, np.inf
    for combo in product(*choices):
        score, ok = 0.0, True
        for i, ci in enumerate(combo):
            if ci is None:
                score += 5.0        # leaving a cable unassigned is expensive,
                continue            # but not impossible -- gaps do happen
            score += base[i][ci]
        for a in range(n):
            for b in range(a + 1, n):
                if combo[a] is None or combo[b] is None:
                    continue
                f = ov[(a, combo[a], b, combo[b])]
                if f > MAX_FAR_OVERLAP:
                    ok = False
                    break
                score += W_OVERLAP * f
            if not ok:
                break
        if ok and score < best_score:
            best_score, best = score, combo

    if best is None:
        return [None] * n, [None] * n, np.inf
    # Finished centrelines, not raw fragments: the temporal prior, the
    # carry-forward through mask gaps and the pipeline output are then all the
    # same representation, so none of them needs re-deriving (and mis-aligning)
    # the hub and mover endpoints.
    return ([fin[i][ci] if ci is not None else None for i, ci in enumerate(best)],
            [kind[i][ci] if ci is not None else None for i, ci in enumerate(best)],
            best_score)


class HarnessTracker:
    """Sequential four-cable tracker over a contiguous range of frames.

    Usage: `run()` over the frames a situation needs. Frames are solved in
    order outward from the most confident one, so the temporal prior always
    flows from a frame that was solved well.
    """

    def __init__(self, tf, clip=None, n_nodes=16, conn_radius=135, k=6,
                 max_age=4, jump_mm=70.0, solo_seed=True, no_guard=False):
        self.tf = tf
        self.clip = clip
        self.n_nodes = n_nodes
        self.conn_radius = conn_radius
        self.k = k
        self.max_age = max_age
        # A step between adjacent frames counts as a wrong-strand jump when the
        # cable's nodes move more than max(jump_mm, 2.5 x mover displacement)
        # from the mover-shifted previous shape. Above the ~30 mm real
        # frame-to-frame drift, below the ~120 mm leaps measured on a snap.
        self.jump_mm = jump_mm
        # solo_seed: use the per-frame solo solve (no carried prior) as the base
        # result. The solo solve depends only on the frame, so the SAME frame
        # gives the SAME shape regardless of which situation's span it lands in
        # -- the cross-situation identity the forecasts need, since one image's
        # target frame is the next image's start. It also avoids the sweep's
        # carried prior dragging a cable onto a longer wrong strand the frame's
        # own evidence rejects. The continuity guard still repairs solo snaps.
        self.solo_seed = solo_seed
        self.no_guard = no_guard

    def _graph(self, frame, idx):
        w, _ = wire_mask(frame, self.tf, self.conn_radius,
                         clip=self.clip, orig_idx=idx)
        # Distance to the nearest wire pixel, for the coverage term. Computed
        # once per frame here rather than per candidate, where it would run
        # tens of times over the same 3.4 Mpixel image.
        dist = cv2.distanceTransform((~w).astype(np.uint8), cv2.DIST_L2, 3)
        return w, build_graph(w), dist

    def run(self, cap, read_fn, tracks, indices):
        """-> {frame_idx: [cable dict per mover]}, hub-first ordering.

        `tracks` must already be hub-first (column 0 = hub), as produced by
        `track_harness` after reordering.
        """
        indices = sorted(indices)
        prepared = {}
        for idx in indices:
            fr = read_fn(cap, idx)
            if fr is None or not np.isfinite(tracks[idx]).all():
                continue
            w, gr, dist = self._graph(fr, idx)
            prepared[idx] = {"wires": w, "graph": gr, "dist": dist,
                             "mods": tracks[idx]}
        if not prepared:
            return {}

        # Pick the anchor: solve every frame WITHOUT a prior and keep the one
        # whose joint solution is most confident. Starting at the first frame
        # instead would propagate that frame's ambiguity through the window.
        solo, obs = {}, [[] for _ in range(N_MOVERS)]
        for idx, p in prepared.items():
            routes, kinds, score = solve_frame(
                p["graph"], p["mods"][0], p["mods"][1:], dist=p["dist"],
                k=self.k, n_nodes=self.n_nodes)
            n_found = sum(1 for r in routes if r is not None)
            solo[idx] = (routes, score, n_found)
            for j, r in enumerate(routes):
                if r is not None and j < N_MOVERS:
                    obs[j].append((idx, _coverage(r, p["dist"]), _arc_length(r)))
        anchor = min(solo, key=lambda i: (-solo[i][2], solo[i][1]))

        # Pin each cable's arc length LOCALLY: the median of the better-covered
        # solo observations within a window of the frame being solved (only the
        # better-covered half votes, since a route that follows the strand is at
        # least as long as one that cuts the corner). A single global median
        # over the whole clip biases a cable that runs long in part of the clip
        # onto an over-extended route everywhere -- measured as a cable cutting
        # across empty space to reach an inflated length target while the strand
        # it should follow lies elsewhere. A local window tracks the true length
        # as the cable slackens and keeps each frame's target honest, without
        # giving up the single-pass consistency the window edges need.
        LW = 8

        def local_lengths(center):
            out = [None] * N_MOVERS
            for j, o in enumerate(obs):
                near = [(c, L) for (fi, c, L) in o if abs(fi - center) <= LW]
                if len(near) < 3:
                    near = [(c, L) for (_, c, L) in o]          # global fallback
                if len(near) < 3:
                    continue
                cov_cut = np.median([c for c, _ in near])
                good = [L for c, L in near if c >= cov_cut]
                if good:
                    out[j] = float(np.median(good))
            return out

        # Per cable we carry (route, age): age counts frames since the route
        # was last backed by fresh evidence rather than by memory.
        # Re-solve the anchor under the length constraint too: it seeded the
        # whole window, so leaving it as the unconstrained solution would let a
        # truncated route propagate outward as the prior.
        pa = prepared[anchor]
        a_routes, _, _ = solve_frame(
            pa["graph"], pa["mods"][0], pa["mods"][1:], dist=pa["dist"],
            k=self.k, n_nodes=self.n_nodes, lengths=local_lengths(anchor))
        result = {anchor: [(r, 0 if r is not None else None) for r in a_routes]}
        if getattr(self, "solo_seed", False):
            # Consistency mode: use the per-frame solo solve as the result. The
            # solo solve depends only on the frame (no anchor, no carried prior)
            # so the SAME frame yields the SAME shape no matter which situation's
            # span it is tracked in -- exactly the cross-situation identity the
            # forecasts need (a target frame of one 2 s image is the start of the
            # next). The sweep's carried prior, by contrast, can drag a cable
            # onto a longer wrong strand that the frame's own evidence rejects
            # (measured: M1 in 20260602_193953). The continuity guard below still
            # repairs the occasional solo snap.
            result = {idx: [(r, 0 if r is not None else None)
                            for r in solo[idx][0]] for idx in prepared}
        else:
          order = ([i for i in indices if i > anchor],
                   [i for i in indices if i < anchor][::-1])
          for seq in order:
            prev_idx = anchor
            for idx in seq:
                if idx not in prepared:
                    continue
                p = prepared[idx]
                prev = result.get(prev_idx)
                priors, prev_age = None, [None] * N_MOVERS
                if prev is not None:
                    # Shift the previous route by this mover's displacement so
                    # the prior describes where the cable should be NOW, not
                    # where it was. A shape already carried for max_age frames
                    # is withheld: the shift only moves it rigidly, so past
                    # that point it stops describing a real observation.
                    shift = p["mods"] - prepared[prev_idx]["mods"]
                    priors = []
                    for j, (r, age) in enumerate(prev):
                        stale = age is None or age >= self.max_age
                        priors.append(None if (r is None or stale)
                                      else r + shift[j + 1])
                        prev_age[j] = age
                routes, kinds, _ = solve_frame(
                    p["graph"], p["mods"][0], p["mods"][1:], dist=p["dist"],
                    priors=priors, k=self.k, n_nodes=self.n_nodes,
                    lengths=local_lengths(idx))

                # `kind` says whether the winning explanation was fresh
                # evidence or memory, so the age counter follows the actual
                # decision instead of guessing from whether a trace existed.
                merged = []
                for j, r in enumerate(routes):
                    if r is None:
                        merged.append((None, None))
                    elif kinds[j] == "carry":
                        merged.append((r, (prev_age[j] or 0) + 1))
                    else:
                        merged.append((r, 0))
                result[idx] = merged
                prev_idx = idx

        # Frame-to-frame continuity guard, per cable. A cable's test points may
        # only move about as far as its mover does between adjacent frames; a
        # fresh trace that leaps far beyond the rigid mover shift has snapped
        # onto the wrong strand (a crossing, or the shared bundle near the hub).
        # Such steps split the timeline into jump-free segments. The LONGEST
        # segment is trusted, and from its middle the good shape is carried
        # outward across every jump -- so a wrong run of any length, even one
        # that reaches the clip's end, is overwritten by the last good shape
        # instead of the reverse. Seeding from the longest segment (not the
        # tracker's confidence anchor, which can itself sit inside a bad run)
        # makes the direction of repair correct regardless of where the jump is.
        _run_guard = not getattr(self, "no_guard", False)

        def _dev(ref, cur):
            """(mm node deviation of cur from ref shifted by the mover, mover mm)."""
            shift = prepared[cur]["mods"][j + 1] - prepared[ref]["mods"][j + 1]
            d = (self.tf.to_mm(result[cur][j][0])
                 - self.tf.to_mm(result[ref][j][0] + shift))
            mv = float(np.linalg.norm(
                self.tf.to_mm(prepared[cur]["mods"][j + 1][None])[0]
                - self.tf.to_mm(prepared[ref]["mods"][j + 1][None])[0]))
            return float(np.mean(np.linalg.norm(d, axis=1))), mv

        for j in range(N_MOVERS if _run_guard else 0):
            allf = sorted(result)
            have = [i for i in allf if result[i][j][0] is not None]
            if len(have) < 3:
                continue

            # Largest run of consecutive present frames with no jump between
            # them -- the stretch we trust and seed the repair from.
            segs = [[have[0]]]
            for a, b in zip(have, have[1:]):
                ok = b == a + 1
                if ok:
                    dev, mv = _dev(a, b)
                    ok = dev <= max(self.jump_mm, 2.5 * mv)
                segs[-1].append(b) if ok else segs.append([b])
            anchor = max(segs, key=len)[len(max(segs, key=len)) // 2]

            def _carry(cur, ref):
                # Map the ref shape onto cur's hub->mover chord (translate,
                # rotate, scale) rather than translating it rigidly by the mover
                # displacement. A rigid mover-shift moves the hub end too, which
                # the plug never does, so materialise then re-pins the hub and
                # stretches the first segment -- ballooning the arc length when
                # ref is several frames away. Chord re-instantiation preserves
                # the shape and lands both ends where they belong.
                r = result[ref][j][0]
                h0, m0 = r[0], r[-1]
                d0 = m0 - h0
                L0 = float(np.hypot(d0[0], d0[1]))
                h1 = prepared[cur]["mods"][0]
                m1 = prepared[cur]["mods"][j + 1]
                if L0 < 1e-6:
                    result[cur][j] = (r + (m1 - m0), result[cur][j][1])
                    return
                a0 = np.arctan2(d0[1], d0[0])
                q = (r - h0) @ np.array([[np.cos(-a0), -np.sin(-a0)],
                                         [np.sin(-a0), np.cos(-a0)]]).T / L0
                d1 = m1 - h1
                L1 = float(np.hypot(d1[0], d1[1]))
                a1 = np.arctan2(d1[1], d1[0])
                out = (q * L1) @ np.array([[np.cos(a1), -np.sin(a1)],
                                           [np.sin(a1), np.cos(a1)]]).T + h1
                result[cur][j] = (out, result[cur][j][1])

            def _good(ref, cur):                # present AND no big jump from ref
                if result[cur][j][0] is None:
                    return False
                dev, mv = _dev(ref, cur)
                return dev <= max(self.jump_mm, 2.5 * mv)

            # Carry the trusted shape outward across every jump AND every frame
            # where the cable was lost (route None), so a gap fills with the
            # last good shape rather than a corridor/straight placeholder that
            # would itself read as a jump.
            ref = anchor
            for cur in [f for f in allf if f > anchor]:
                if not _good(ref, cur):
                    _carry(cur, ref)
                ref = cur
            ref = anchor
            for cur in [f for f in allf if f < anchor][::-1]:
                if not _good(ref, cur):
                    _carry(cur, ref)
                ref = cur

        # Materialise into the cable dicts the rest of the pipeline expects.
        out = {}
        for idx, routes in result.items():
            p = prepared[idx]
            hub, mods = p["mods"][0], p["mods"]
            cables = []
            for j, (poly, age) in enumerate(routes):
                m = mods[j + 1]
                method = "tracker" if age == 0 else f"fortgeschrieben({age})"
                # `poly` is already a finished hub->mover centreline; a carried
                # one is re-anchored to the CURRENT plug centres, since the
                # shift that produced it only moved it rigidly.
                cl = None
                if poly is not None:
                    cl = np.asarray(poly, float).copy()
                    cl[0], cl[-1] = hub, m
                if cl is None:
                    cc = corridor_cable(p["wires"], hub, m)
                    if cc is not None:
                        cl, method = cc, "korridor"
                if cl is None:
                    cl = _resample_pixels(np.vstack([hub[None, :], m[None, :]]),
                                          self.n_nodes)
                    method = "gerade (Platzhalter)"
                q = strand_quality(cl, self.tf)
                cables.append({"shape_mm": self.tf.to_mm(cl) / 1000.0, "px": cl,
                               "mover_px": np.asarray(m, float),
                               "hub_px": np.asarray(hub, float),
                               "method": method, "slack": q["slack"]})
            out[idx] = cables
        return out
