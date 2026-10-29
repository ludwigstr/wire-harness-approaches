"""Fit each cable's parameters from a whole clip, then score held-out forecasts.

Everything before this estimated a cable's rest-angle profile from ONE frame
(rest_angle ~ the observed curvature) with a nominal k_b/k_d. That is a
stopgap: it assumes the observed shape is the wire's natural shape, which is
only true when the wire hangs slack. Here the parameters are FITTED from many
observed transitions of the clip, which is what the paper's Sec. II-C does.

Three things this gets right that the single-frame estimate cannot:

  * the hub moves, so every action is expressed in the hub's frame and the
    prediction is transformed back (the harness hub travels ~80 mm here);
  * cables are traced with the topology-aware tracer, so a strand is followed
    through crossings instead of jumping onto its neighbour;
  * cables span plug centre to plug centre, so the terminal node is exactly the
    point the mover moves.

Reports a train/held-out split and a learning curve over the number of fitted
transitions, so "how much data do we need" is answered by measurement.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast_frames import (hub_of, module_centres_px, traced_cable,
                             wire_mask_of)
from wire_model.estimation import (estimate_parameters, feasible_w,
                                   load_predictor, uw_to_kb_kd)
from wire_model.perception.rectify import find_board

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = ROOT / "data" / "dlo_nn_rest.pt"
N = 16


def read(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def track_all_modules(cap, tf, start_pts, n_frames, ref, half=95, search=90):
    """Template-track every module through the clip, outward from `ref`.

    Templates MUST be cut at the reference frame, not at frame 0: early in a
    clip the harness is not assembled yet (the board is nearly empty), so
    templates taken there match nothing and every later position is garbage --
    which silently produced zero traced cables on the first run.
    """
    fr_ref = read(cap, ref)
    g_ref = cv2.cvtColor(fr_ref, cv2.COLOR_BGR2GRAY)
    tmpl = []
    for p in start_pts:
        x, y = int(round(p[0])), int(round(p[1]))
        tmpl.append(g_ref[max(0, y - half):y + half, max(0, x - half):x + half])

    out = np.full((n_frames, len(start_pts), 2), np.nan)
    out[ref] = np.asarray(start_pts, float)

    def walk(rng):
        pos = np.asarray(start_pts, float).copy()
        for idx in rng:
            fr = read(cap, idx)
            if fr is None:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
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
            out[idx] = pos

    walk(range(ref + 1, n_frames))
    walk(range(ref - 1, -1, -1))
    return out


def _edge_action(V_in, V_out, at_end=True):
    """Displacement and rotation of one terminal edge between two shapes."""
    def edge(V):
        a, b = (V[-2], V[-1]) if at_end else (V[0], V[1])
        return 0.5 * (a + b), np.arctan2(b[1] - a[1], b[0] - a[0])
    m_in, a_in = edge(V_in)
    m_out, a_out = edge(V_out)
    dth = np.arctan2(np.sin(a_out - a_in), np.cos(a_out - a_in))
    alpha = (N - 2) if at_end else 0
    return np.array([alpha, m_out[0] - m_in[0], m_out[1] - m_in[1], dth],
                    dtype=np.float32)


def terminal_action(V_in, V_out, both_ends=False):
    """Action vector describing how the driven end(s) moved.

    `both_ends` returns 8 numbers: the mover edge followed by the hub edge.
    The hub used to be removed by working in a hub-fixed frame, which was exact
    while the physics was translation-invariant. Dry friction acts against the
    board and breaks that invariance, and the hub moves further than the mover
    anyway (median 89 mm vs 58 mm per 2 s), so it is now described explicitly.
    """
    a = _edge_action(V_in, V_out, at_end=True)
    if not both_ends:
        return a
    return np.concatenate([a, _edge_action(V_in, V_out, at_end=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref-frame", type=int, default=36)
    ap.add_argument("--mass", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--min-move", type=float, default=0.010)
    ap.add_argument("--conn-radius", type=int, default=135)
    ap.add_argument("--out", default="data/fit_clip_result.pt")
    args = ap.parse_args()

    vid = args.video
    cap = cv2.VideoCapture(vid)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tf = find_board(read(cap, 0))

    ref = read(cap, args.ref_frame)
    w_ref, seg_ref = wire_mask_of(ref, tf, args.conn_radius)
    hub_ref = hub_of(w_ref, seg_ref.module_centres)
    peripheral = [c for c in seg_ref.module_centres
                  if np.linalg.norm(c - hub_ref) > 60]
    order = [hub_ref] + peripheral
    print(f"{Path(vid).stem}: {n_frames} Frames, Hub + {len(peripheral)} Mover")

    tracks = track_all_modules(cap, tf, order, n_frames, args.ref_frame)
    print(f"Modul-Tracking: {tracks.shape[0]} Frames verfolgt")

    # trace every cable in every frame
    shapes = {k: {} for k in range(len(peripheral))}
    for idx in range(tracks.shape[0]):
        if not np.isfinite(tracks[idx]).all():
            continue
        fr = read(cap, idx)
        if fr is None:
            break
        w, _ = wire_mask_of(fr, tf, args.conn_radius)
        hub_px = tracks[idx, 0]
        for k in range(len(peripheral)):
            sh, q = traced_cable(w, hub_px, tracks[idx, k + 1], tf)
            if sh is None:
                continue
            if q["slack"] >= 0.55:
                shapes[k][idx] = sh
        if idx % 20 == 0:
            print(f"  Frame {idx}: " +
                  " ".join(f"K{k}{'+' if idx in shapes[k] else '-'}"
                           for k in range(len(peripheral))), flush=True)
    cap.release()
    for k in shapes:
        print(f"  Kabel {k}: {len(shapes[k])} Frames getract")

    model, _ = load_predictor(CKPT, DEV)

    # ---- transitions, expressed in the hub frame ----
    def transitions(k):
        keys = sorted(shapes[k])
        Vi, Vo, A = [], [], []
        for a, b in zip(keys[:-1], keys[1:]):
            if b - a > 2:
                continue
            s0, s1 = shapes[k][a], shapes[k][b]
            hub_d = (tf.to_mm(tracks[b, 0]) - tf.to_mm(tracks[a, 0])) / 1000.0
            if np.linalg.norm(s1[-1] - s0[-1] - hub_d) < args.min_move:
                continue
            # move into the hub's frame: subtract the hub's own displacement
            s1_rel = s1 - hub_d
            Vi.append(s0)
            Vo.append(s1_rel)
            A.append(terminal_action(s0, s1_rel))
        if not Vi:
            return None
        return (torch.tensor(np.array(Vi), dtype=torch.float32, device=DEV),
                torch.tensor(np.array(A), dtype=torch.float32, device=DEV),
                torch.tensor(np.array(Vo), dtype=torch.float32, device=DEV))

    results = {}
    print("\n--- Fit pro Kabel (60% Fit / 40% Held-out) ---")
    for k in range(len(peripheral)):
        tr = transitions(k)
        if tr is None or tr[0].shape[0] < 6:
            print(f"Kabel {k}: zu wenige Transitionen"); continue
        Vi, A, Vo = tr
        K = Vi.shape[0]
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(K, generator=g)
        nfit = max(4, int(0.6 * K))
        fit_i, test_i = perm[:nfit], perm[nfit:]

        est = estimate_parameters(model, Vi[fit_i], A[fit_i], Vo[fit_i],
                                  args.mass, n_iters=args.iters)
        u = torch.log(torch.tensor(est.ratio, device=DEV))
        k_b, k_d = uw_to_kb_kd(u, feasible_w(u))

        def score(idx_sel):
            if len(idx_sel) == 0:
                return float("nan")
            n = len(idx_sel)
            rest = est.rest_angle.unsqueeze(0).expand(n, -1)
            par = torch.stack([torch.full((n,), args.mass, device=DEV),
                               k_b.expand(n), k_d.expand(n)], -1)
            with torch.no_grad():
                pred = model(Vi[idx_sel], A[idx_sel], par, rest)
            return 1000 * (pred - Vo[idx_sel]).norm(dim=-1).mean().item()

        naive = 1000 * (Vi[test_i] - Vo[test_i]).norm(dim=-1).mean().item()
        print(f"Kabel {k}: {K} Transitionen | Fit {score(fit_i):5.1f} | "
              f"Held-out {score(test_i):5.1f} | naiv {naive:5.1f} mm/Knoten | "
              f"k_b/k_d {est.ratio:.4f}")
        results[k] = {"rest": est.rest_angle.cpu(), "ratio": est.ratio,
                      "k_b": float(k_b), "k_d": float(k_d),
                      "n_trans": K, "heldout": score(test_i), "naive": naive}

    # ---- learning curve: how many transitions are actually needed? ----
    print("\n--- Lernkurve (Held-out-Fehler je Anzahl Fit-Transitionen) ---")
    print(f"{'n_fit':>6} " + " ".join(f"Kabel{k}" for k in results))
    for nfit in (3, 5, 8, 12, 20):
        row = []
        for k in results:
            tr = transitions(k)
            Vi, A, Vo = tr
            K = Vi.shape[0]
            if K < nfit + 3:
                row.append("  n/a "); continue
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(K, generator=g)
            fi, ti = perm[:nfit], perm[int(0.6 * K):]
            est = estimate_parameters(model, Vi[fi], A[fi], Vo[fi], args.mass,
                                      n_iters=args.iters)
            u = torch.log(torch.tensor(est.ratio, device=DEV))
            kb, kd = uw_to_kb_kd(u, feasible_w(u))
            n = len(ti)
            with torch.no_grad():
                pred = model(Vi[ti], A[ti],
                             torch.stack([torch.full((n,), args.mass, device=DEV),
                                          kb.expand(n), kd.expand(n)], -1),
                             est.rest_angle.unsqueeze(0).expand(n, -1))
            row.append(f"{1000*(pred-Vo[ti]).norm(dim=-1).mean().item():6.1f}")
        print(f"{nfit:>6} " + " ".join(row))

    torch.save({"results": results, "video": vid,
                "ref_frame": args.ref_frame}, ROOT / args.out)
    print(f"\ngespeichert: {args.out}")


if __name__ == "__main__":
    main()
