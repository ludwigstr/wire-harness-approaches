"""Dedicated tracer for the thin cable 3 (pink in the check videos), user 2026-11-01.

Cable 3 is a loose bundle of 2-3 thin blue/grey wires (a few px wide) that the segmentation pipeline's
class map misses over long stretches. It is found directly in the camera image: a black-hat filter picks
up thin dark lines; the route is the cheapest path from the hub to the cable-3 mover through those pixels
(skimage MCP), with extra cost on the three thick cables (so it cannot follow them, crossing is allowed),
uniform cost inside the module plates (their printing is not wire), and a soft pull towards the previous
frame's route (no frame-to-frame jumps).
"""
import cv2
import numpy as np
from skimage.graph import MCP_Geometric

BH_KERNEL = 25
BH_THR = 18          # black-hat threshold for "wire pixel" (thin wires are low contrast)
C_WIRE = 1.0
C_EMPTY = 30.0       # cost per px over empty board
C_THICK = 8.0        # extra cost per px on one of the three thick cables
C_PLATE = 4.0        # cost per px inside a module plate
PLATE_R = 120        # px, plate half-size around hub/mover centre (0.8-scale image)
CORE_R = 45          # px, plug core: uniform cost (the wire ends here)
C_RING = 8.0         # cost per px on the plate outside the core where no wire is seen (wires may hook in)
PULL = 0.04          # cost per px of distance to the previous route, per px of path


def wire_evidence(raw):
    g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BH_KERNEL,) * 2))
    return cv2.GaussianBlur(bh, (3, 3), 0) > BH_THR


def cost_image(raw, hub, movers, thick_routes, prev_route=None, board_poly=None):
    H, W = raw.shape[:2]
    ev = wire_evidence(raw)
    cost = np.where(ev, C_WIRE, C_EMPTY).astype(np.float64)
    thick = np.zeros((H, W), np.uint8)
    for r in thick_routes:
        if r is not None:
            cv2.polylines(thick, [np.asarray(r, np.int32).reshape(-1, 1, 2)], False, 1, 22)
    cost[thick > 0] += C_THICK
    plates = np.zeros((H, W), np.uint8)
    for c in [hub] + list(movers):
        x, y = int(c[0]), int(c[1]); cv2.rectangle(plates, (x - PLATE_R, y - PLATE_R), (x + PLATE_R, y + PLATE_R), 1, -1)
    core = np.zeros((H, W), np.uint8)
    for c in [hub] + list(movers):
        x, y = int(c[0]), int(c[1]); cv2.rectangle(core, (x - CORE_R, y - CORE_R), (x + CORE_R, y + CORE_R), 1, -1)
    ring = (plates > 0) & (core == 0)
    cost[ring & ~ev] = C_RING
    cost[core > 0] = C_PLATE
    if prev_route is not None:
        pr = np.zeros((H, W), np.uint8)
        cv2.polylines(pr, [np.asarray(prev_route, np.int32).reshape(-1, 1, 2)], False, 1, 1)
        d = cv2.distanceTransform((pr == 0).astype(np.uint8), cv2.DIST_L2, 3)
        cost += PULL * d
    if board_poly is not None:
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [board_poly.astype(np.int32)], 1)
        cost[m == 0] = 1e4
    return cost, ev


def trace(cost, a, b):
    """Cheapest 8-connected path from point a to point b (x, y), returned as (N, 2) float x, y."""
    H, W = cost.shape
    a = (int(np.clip(a[1], 0, H - 1)), int(np.clip(a[0], 0, W - 1)))
    b = (int(np.clip(b[1], 0, H - 1)), int(np.clip(b[0], 0, W - 1)))
    m = MCP_Geometric(cost, fully_connected=True)
    m.find_costs([a], [b])
    p = np.array(m.traceback(b), float)[:, ::-1]
    return p


def smooth_route(p, n_nodes=16, win=15):
    if len(p) > win:
        k = np.ones(win) / win
        q = np.stack([np.convolve(np.pad(p[:, i], win // 2, mode="edge"), k, "valid") for i in range(2)], 1)
        q[0], q[-1] = p[0], p[-1]; p = q
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n_nodes)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
