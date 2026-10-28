"""Extract an ordered centerline polyline from a wire-bundle mask.

A single bundle mask is skeletonised and turned into a pixel graph. The clean
case is a simple path between two endpoints; the messy case -- the one that
actually occurs -- is a skeleton with spurs (thinning artefacts) and junctions
(where a bundle's strands split or two bundles graze). Both are handled by
finding the longest geodesic path through the skeleton graph:

  - spurs are short dead-end branches, so they are never on the longest path;
  - at a junction the longest path takes the through-route, which for a wire
    running between two connectors is the physically correct one.

The endpoints are then snapped to the two nearest connector centres, because a
bundle's true ends are inside the plugs, which the segmentation masked out.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    _HAVE_SKIMAGE = True
except ImportError:                                   # pragma: no cover
    _HAVE_SKIMAGE = False


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """1px-wide skeleton of a boolean mask."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return (cv2.ximgproc.thinning(m * 255) > 0)
    if _HAVE_SKIMAGE:
        return skeletonize(m > 0)
    raise RuntimeError("need cv2.ximgproc.thinning or skimage for skeletonize")


_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(skel: np.ndarray):
    """Pixel adjacency graph of a skeleton.

    Returns (nodes, adj) where nodes is an (M, 2) int array of (y, x) skeleton
    pixels and adj maps a node index to its 8-neighbour indices with euclidean
    edge weights.
    """
    ys, xs = np.nonzero(skel)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    nodes = np.stack([ys, xs], -1)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(ys))}
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in _NEIGH:
            j = idx.get((y + dy, x + dx))
            if j is not None:
                adj[i].append((j, float(np.hypot(dy, dx))))
    return nodes, adj


def _bfs_farthest(adj, src, n):
    """Dijkstra from src; return (farthest node, dist array, parent array)."""
    import heapq
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    reach = np.where(np.isfinite(dist), dist, -1)
    return int(reach.argmax()), dist, parent


def _trace_path(parent, target):
    path = []
    v = int(target)
    while v != -1:
        path.append(v)
        v = int(parent[v])
    return path[::-1]


def longest_path_pixels(mask: np.ndarray) -> np.ndarray:
    """Longest geodesic path through a bundle mask, as (P, 2) (x, y) pixels.

    Double-BFS for the graph diameter: the farthest node from an arbitrary
    start, then the farthest from that node -- the standard tree-diameter trick,
    which here also skips every spur because a spur is never an endpoint of the
    diameter.
    """
    skel = _skeleton(mask)
    if skel.sum() < 2:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("empty wire mask")
        return np.stack([xs, ys], -1)[:1].astype(float)

    nodes, adj = _skeleton_graph(skel)
    n = len(nodes)
    a, _, _ = _bfs_farthest(adj, 0, n)
    b, dist, parent = _bfs_farthest(adj, a, n)
    pix = nodes[_trace_path(parent, b)]     # (P, 2) as (y, x)
    return pix[:, ::-1].astype(float)       # -> (x, y)


def anchored_path_pixels(mask: np.ndarray, src_xy, dst_xy) -> np.ndarray:
    """Shortest skeleton path between the points nearest src_xy and dst_xy.

    This is the right tracer for a strand whose two ends are KNOWN (a hub and
    one peripheral module): anchoring both ends removes the two failure modes of
    a blind longest-path search on a multi-wire bundle --

    - hairpins, where the diameter runs out along one strand of the bundle and
      back along the parallel one (a hairpin is always longer, so the SHORTEST
      path never takes it);
    - endpoint snapping to the wrong module, since the destination is specified
      rather than guessed from a nearest-neighbour after the fact.

    Falls back to the longest path if either anchor has no skeleton nearby.
    """
    skel = _skeleton(mask)
    if skel.sum() < 2:
        return longest_path_pixels(mask)
    nodes, adj = _skeleton_graph(skel)
    nodes_xy = nodes[:, ::-1].astype(float)             # (M, 2) as (x, y)

    def nearest(pt):
        return int(np.linalg.norm(nodes_xy - np.asarray(pt, float), axis=1).argmin())

    src, dst = nearest(src_xy), nearest(dst_xy)
    _, _, parent = _bfs_farthest(adj, src, len(nodes))  # Dijkstra tree from src
    if parent[dst] == -1 and dst != src:
        return longest_path_pixels(mask)
    return nodes_xy[_trace_path(parent, dst)]


def _resample_pixels(xy: np.ndarray, n_out: int) -> np.ndarray:
    """Resample an (P, 2) pixel polyline to n_out arc-length-equal points."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(cum[-1], 1e-9)
    want = np.linspace(0.0, total, n_out)
    x = np.interp(want, cum, xy[:, 0])
    y = np.interp(want, cum, xy[:, 1])
    return np.stack([x, y], -1)


def _smooth(xy: np.ndarray, win: int = 5) -> np.ndarray:
    """Moving-average smooth, endpoints preserved.

    Uses edge-replicated padding, NOT np.convolve's zero padding: zero padding
    averages the first/last points against (0, 0), i.e. the image origin, which
    yanks the wire ends toward the top-left corner and manufactures a spike that
    inflates the arc length several-fold. (That bug turned clean 300 mm strands
    into 1300 mm zigzags -- worth stating so it is not reintroduced.)
    """
    n = xy.shape[0]
    if win < 3 or n < win:
        return xy
    half = win // 2
    sm = xy.copy()
    for c in range(2):
        padded = np.pad(xy[:, c], half, mode="edge")
        kernel = np.ones(win) / win
        sm[:, c] = np.convolve(padded, kernel, mode="valid")
    sm[0], sm[-1] = xy[0], xy[-1]
    return sm


def strand_centerline(
    mask: np.ndarray,
    n_nodes: int,
    hub_xy: np.ndarray,
    module_xy: np.ndarray,
    smooth_win: int = 5,
    reach_ends: bool = True,
    n_strips: int = 40,
) -> np.ndarray:
    """Centerline of one hub->module strand, as (n_nodes, 2) (x, y) pixels.

    The wire pixels are projected onto the hub->module chord, binned into strips
    perpendicular to it, and each strip is reduced to its pixel centroid. Node 0
    is the hub end (the stationary connector the pinned-end DLO model expects).

    Why this rather than a skeleton graph path: a wire bundle is several wires
    running side by side, so its thinned skeleton is a ladder of rungs, not a
    line, and every graph path (shortest or longest) zigzags across the rungs --
    producing arc lengths of 1300 mm for a 300 mm strand. Averaging pixels in
    a perpendicular strip collapses the whole ribbon to its midline in one step,
    and the chord axis fixes a monotstrand ordering for free. The one assumption
    is that the wire progresses monotonically along the chord, which a slack
    wire strung between two plugs satisfies; `strand_quality` flags the rare
    case that folds back.

    With `reach_ends`, the terminal nodes are pulled onto the plug centres,
    recovering the last centimetre the dilated connector mask removed.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], -1).astype(float)
    axis = np.asarray(module_xy, float) - np.asarray(hub_xy, float)
    length = np.linalg.norm(axis)
    if length < 1.0 or len(pts) < 4:
        return _resample_pixels(np.stack([hub_xy, module_xy]), n_nodes)
    u = axis / length

    t = (pts - hub_xy) @ u
    edges = np.linspace(t.min(), t.max(), n_strips + 1)
    centres = []
    for i in range(n_strips):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        if int(sel.sum()) >= 2:
            centres.append(pts[sel].mean(0))
    raw = np.array([hub_xy] + centres + [module_xy])
    raw = _smooth(raw, smooth_win)
    if reach_ends:
        raw[0] = hub_xy
        raw[-1] = module_xy
    return _resample_pixels(raw, n_nodes)


def strand_quality(centerline_px: np.ndarray, tf=None) -> dict:
    """Cheap sanity metrics for a traced strand.

    slack = chord / arc. A folded or contaminated trace has a low slack
    (the centerline is much longer than the straight-line distance); a clean
    slack wire sits near 0.5-1.0. `monotonic` checks the centerline never
    reverses along its own chord, which a good strip trace should not.
    """
    cl = centerline_px if tf is None else tf.to_mm(centerline_px)
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    arc = float(seg.sum())
    chord = float(np.linalg.norm(cl[-1] - cl[0]))
    u = (cl[-1] - cl[0]) / max(chord, 1e-9)
    prog = np.diff((cl - cl[0]) @ u)
    return {"arc": arc, "chord": chord,
            "slack": chord / arc if arc > 0 else 0.0,
            "monotonic": bool((prog > -1e-6).all())}


def assign_strands_to_modules(components, hub_xy, module_centres):
    """Match each wire component to the peripheral module it runs to.

    Returns a list of (component_mask, module_xy), one per strand. The hub
    module is excluded as a destination; a component's target is the non-hub
    module nearest the component pixel farthest from the hub -- that far pixel is
    the plug end, which points unambiguously at one module even when the fan is
    crowded near the hub.
    """
    peripheral = [m for m in module_centres
                  if np.linalg.norm(m - hub_xy) > 1.0]
    out = []
    for area, mask in components:
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], -1).astype(float)
        far = pts[np.linalg.norm(pts - hub_xy, axis=1).argmax()]
        j = int(np.linalg.norm(np.array(peripheral) - far, axis=1).argmin())
        out.append((mask, peripheral[j]))
    return out
