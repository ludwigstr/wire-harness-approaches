"""Hybrid tracker (2026-11-01): TrackDLO-style EM follows the cables, the per-frame graph detector
(tracking_v2, npz from --det) re-initialises a chain when it leaves the evidence or the scene jumps.
Resets are stored per frame/cable in `reset` of the output npz.

TrackDLO-style 2D tracker (Xiang et al. RA-L 2023) for the 4 harness cables.

Core: CPD/GMM expectation-maximisation with the motion-coherence (MCT) prior on a geodesic Gaussian
kernel, LLE regularisation, Gaussian noise + uniform outlier weight, visibility weighting of nodes
(P_vis = exp(-k_vis * dist_to_evidence)), correspondence priors (ECPD alpha/omega term) for the pinned
ends, and a length-preserving (geodesic) reconstruction of occluded nodes after each EM. All four
cables are one GMM (block-diagonal priors, joint E-step), so crossings are shared softly.
Written from scratch in 2D after reading the (deleted) python/trackdlo.py and the current C++ core.
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa

SCALE = 0.8
COLS = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]

DEFAULT = dict(M=40, beta=120.0, lam=0.3, gamma=0.5, mu=0.1, k_vis=0.08, d_vis=14.0, omega=1e-3,
               max_iter=40, tol=0.2, tau_mm=120.0, stride_cm=4, stride_dark=2, dark_thr=28,
               gate_px=22.0, len_iters=0, sigma2_init=60.0, plug_dilate=8, use_dark=1, len_w=0.7, dark_in_cab=1, relay=1, resample_each=1, len_mode="hidden", smooth=0.5, fill_hidden=1, min_deficit=15.0, max_deficit=150.0,
               reset_off=0.25, det_off=0.08, det_len_tol=0.06, jump_mm=60.0)


# ----------------------------------------------------------------------------- data
def panels_of(clip):
    out = {}
    for d in glob.glob(str(ROOT / f"data/videos/{clip}/visualization/panels/*_frame_*")):
        out.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    return out


def load_frame(pan_dir):
    raw = cv2.imread(f"{pan_dir}/01_raw.png"); cm = cv2.imread(f"{pan_dir}/02_class_map.png")
    H, W = raw.shape[:2]; sz = (int(W * SCALE), int(H * SCALE))
    raw = cv2.resize(raw, sz, interpolation=cv2.INTER_AREA)
    cab = cv2.resize(np.all(cm == (0, 255, 255), axis=2).astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    plug = cv2.resize(np.all(cm == (0, 255, 0), axis=2).astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
    return raw, cab, plug


def dark_mask(raw, tf, thr, p_inset=60):
    g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = bh > thr
    ins = p_inset   # side/bottom inset as in the benchmark (60 mm); top inset only 10 mm so the cable under the gantry survives
    poly = tf.to_px(np.array(((ins, ins), (1200 - ins, ins), (1200 - ins, 1680 - 10), (ins, 1680 - 10)), float)).astype(np.int32)
    board = np.zeros(g.shape, np.uint8); cv2.fillPoly(board, [poly], 1)
    return m & (board > 0)


def text_mask(raw):
    """Yellow overlay text in the raw panel (top-left band), dilated."""
    yel = np.all(raw == (0, 255, 255), axis=2).astype(np.uint8)
    return cv2.dilate(yel, np.ones((9, 9), np.uint8)) > 0


def resample(poly, M):
    """Resample a polyline to M points equally spaced in arc length."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], M)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


# ----------------------------------------------------------------------------- model pieces
def geodesic_kernel(Y, beta):
    seg = np.linalg.norm(np.diff(Y, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    d = np.abs(s[:, None] - s[None, :])
    return np.exp(-d ** 2 / (2 * beta ** 2))


def lle_matrix(Y, k=2):
    M = len(Y); L = np.zeros((M, M))
    for i in range(M):
        idx = [j for j in range(i - k, i + k + 1) if j != i and 0 <= j < M]
        Xi = Y[idx]; comp = (Y[i] - Xi); G = comp @ comp.T
        G = G + 1e-3 * np.trace(G) * np.eye(len(idx)) + 1e-9 * np.eye(len(idx))
        w = np.linalg.solve(G, np.ones(len(idx))); w /= w.sum(); L[i, idx] = w
    I = np.eye(M); return (I - L).T @ (I - L)


def project_lengths_hidden(Y, rest, hidden, fixed_w, iters, skip=(0, None), smooth=0.5):
    """Inextensibility: visible-visible segments may only be shortened to rest; any segment touching a
    hidden node is set to its rest length (TrackDLO geodesic placement), hidden nodes are smoothed."""
    Y = Y.copy(); inv = 1.0 - fixed_w
    j0, j1 = skip; j1 = len(Y) - 1 if j1 is None else j1
    for _ in range(iters):
        # smooth hidden nodes towards their neighbours' midpoint (kills zigzags, keeps a bulge)
        if smooth > 0:
            mid = 0.5 * (Y[:-2] + Y[2:]); h = hidden[1:-1]
            Y[1:-1][h] = (1 - smooth) * Y[1:-1][h] + smooth * mid[h]
        for i in range(j0, j1):
            d = Y[i + 1] - Y[i]; n = np.linalg.norm(d) + 1e-9
            if not (hidden[i] or hidden[i + 1]) and n <= rest[i]:
                continue
            corr = (n - rest[i]) * d / n; w = inv[i] + inv[i + 1]
            if w < 1e-9:
                continue
            Y[i] += corr * inv[i] / w; Y[i + 1] -= corr * inv[i + 1] / w
    return Y


def project_lengths(Y, rest, fixed_w, iters, skip=(0, None)):
    """PBD-style distance projection; fixed_w in [0,1] = how immovable a node is (1 = pinned).
    Only segments i with skip[0] <= i < skip[1] are enforced (the plug interiors are left alone)."""
    Y = Y.copy(); inv = 1.0 - fixed_w
    j0, j1 = skip; j1 = len(Y) - 1 if j1 is None else j1
    for _ in range(iters):
        for i in range(j0, j1):
            d = Y[i + 1] - Y[i]; n = np.linalg.norm(d) + 1e-9; corr = (n - rest[i]) * d / n
            w = inv[i] + inv[i + 1]
            if w < 1e-9:
                continue
            Y[i] += corr * inv[i] / w; Y[i + 1] -= corr * inv[i + 1] / w
    return Y


def arc_points(a, b, n_inner, s_target, side):
    """n_inner points between a and b on a circular arc of length s_target (>= chord), bulging to `side`."""
    c = np.linalg.norm(b - a); n_pts = n_inner + 2
    if s_target <= c * 1.001 or c < 1e-6:
        return np.linspace(a, b, n_pts)[1:-1]
    lo, hi = 1e-4, 2.6                      # theta: half opening angle, sin(t)/t = c/s
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if np.sin(t) / t > c / s_target: lo = t
        else: hi = t
    t = 0.5 * (lo + hi); R = c / (2 * np.sin(t))
    e = (b - a) / c; nrm = np.array([-e[1], e[0]])
    if np.dot(nrm, side) < 0: nrm = -nrm
    mid = 0.5 * (a + b); centre = mid - nrm * R * np.cos(t)
    ang = np.linspace(-t, t, n_pts)[1:-1]
    return centre[None, :] + R * (np.cos(ang)[:, None] * nrm[None, :] + np.sin(ang)[:, None] * e[None, :])


def fill_hidden_runs(Y, Yprev, hidden, L_ref, min_deficit, max_deficit):
    """TrackDLO-style occluded-node placement: the chain is inextensible, so the length missing after the
    EM (L_ref - current) is put into the hidden interior runs as circular bulges on the side where the
    nodes were in the previous frame."""
    M = len(Y); runs = []; i = 1
    while i < M - 1:
        if hidden[i]:
            j = i
            while j < M - 1 and hidden[j]: j += 1
            if not hidden[i - 1] and not hidden[j]:
                runs.append((i - 1, j))
            i = j
        else:
            i += 1
    if not runs:
        return Y, 0.0
    L_now = np.linalg.norm(np.diff(Y, axis=0), axis=1).sum(); deficit = min(L_ref - L_now, max_deficit)
    if deficit < min_deficit:
        return Y, deficit
    chords = np.array([np.linalg.norm(Y[b] - Y[a]) for a, b in runs]); share = deficit * chords / chords.sum()
    Y = Y.copy()
    for (a, b), d in zip(runs, share):
        side = Yprev[a + 1:b].mean(0) - 0.5 * (Y[a] + Y[b])
        if np.linalg.norm(side) < 1e-3:
            side = np.array([0.0, -1.0])
        Y[a + 1:b] = arc_points(Y[a], Y[b], b - a - 1, chords[runs.index((a, b))] + d, side)
    return Y, deficit


class Chain:
    def __init__(self, Y0, p):
        self.Y = Y0.copy(); self.M = len(Y0); self.p = p
        self.L_ref = np.linalg.norm(np.diff(Y0, axis=0), axis=1).sum()
        self.rest = np.linalg.norm(np.diff(Y0, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(self.rest)]); self.s /= self.s[-1]
        self.G = geodesic_kernel(Y0, p["beta"])
        self.H = lle_matrix(Y0)


def em_joint(X, chains, pins, vis_d, p, sigma2):
    """One TrackDLO registration of all chains against point set X (N,2).
    pins: list per chain of (node_index, target_xy). vis_d: per chain node distance to evidence (px)."""
    D = 2; N = len(X); Ms = [c.M for c in chains]; M = sum(Ms)
    Y0 = np.concatenate([c.Y for c in chains]); Y = Y0.copy()
    off = np.cumsum([0] + Ms)
    Pvis = np.concatenate([np.exp(-p["k_vis"] * np.maximum(d - p["d_vis"], 0)) for d in vis_d])
    for it in range(p["max_iter"]):
        d2 = ((X[None, :, :] - Y[:, None, :]) ** 2).sum(2)
        P = np.exp(-d2 / (2 * sigma2)) * Pvis[:, None]
        c = (2 * np.pi * sigma2) ** (D / 2) * p["mu"] / (1 - p["mu"]) * M / N
        den = P.sum(0) + c; den[den == 0] = np.finfo(float).eps
        P /= den[None, :]
        P1 = P.sum(1); Pt1 = P.sum(0); Np = P1.sum(); PX = P @ X
        Ynew = np.empty_like(Y)
        for ci, ch in enumerate(chains):
            a, b = off[ci], off[ci + 1]
            A = np.diag(P1[a:b]) @ ch.G + p["lam"] * sigma2 * np.eye(ch.M) + sigma2 * p["gamma"] * ch.H @ ch.G
            B = PX[a:b] - (np.diag(P1[a:b]) + sigma2 * p["gamma"] * ch.H) @ Y0[a:b]
            if pins[ci]:
                Pt_1 = np.zeros(ch.M); PtX = np.zeros((ch.M, 2))
                for idx, tgt in pins[ci]:
                    Pt_1[idx] += 1; PtX[idx] += tgt
                A += sigma2 / p["omega"] * np.diag(Pt_1) @ ch.G
                B += sigma2 / p["omega"] * (PtX - np.diag(Pt_1) @ Y0[a:b])
            W = np.linalg.solve(A, B)
            Ynew[a:b] = Y0[a:b] + ch.G @ W
        trX = (Pt1[:, None] * X * X).sum(); trPXT = (PX * Ynew).sum(); trT = (P1[:, None] * Ynew * Ynew).sum()
        s2 = (trX - 2 * trPXT + trT) / (Np * D)
        sigma2 = max(float(s2), 1.0)
        dY = np.abs(Ynew - Y).max(); Y = Ynew
        if dY < p["tol"]:
            break
    return [Y[off[i]:off[i + 1]] for i in range(len(chains))], sigma2, it + 1


# ----------------------------------------------------------------------------- driver
def track_clip(clip, out_dir, p, sheet_frames=(), debug_dir=None, limit=None, det=None):
    z = np.load(ROOT / f"data/videos/skeletons_2026-10-30/{clip}.npz")
    zd = np.load(Path(det) / f"{clip}.npz")
    Dmm = np.stack([zd[f"c{k}"] for k in range(4)]) * 1000.0
    dcar = zd["carried"] if "carried" in zd.files else np.zeros((Dmm.shape[1], 4), bool)
    Ldet = np.linalg.norm(np.diff(Dmm, axis=2), axis=3).sum(2)          # (4,T) mm
    Lglob = np.array([np.median(Ldet[k][~dcar[:, k]]) if (~dcar[:, k]).any() else np.median(Ldet[k]) for k in range(4)])
    sf = [int(s) for s in z["source_frames"]]; Pmm = np.stack([z[f"c{k}"] for k in range(4)]) * 1000.0
    pan = panels_of(clip); T = len(sf) if limit is None else min(limit, len(sf))
    raw, cab, plug = load_frame(pan[sf[0]]); tf = find_board(raw); H, W = cab.shape
    chains = [Chain(resample(tf.to_px(Dmm[k, 0]), p["M"]), p) for k in range(4)]
    for k, ch in enumerate(chains):
        ch.L_ref = Lglob[k] / tf.mm_per_px
    reset = np.zeros((len(sf), 4), bool)
    out = np.zeros((4, T, 16, 2)); sigma2 = p["sigma2_init"]; t_start = time.time(); iters = []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p["plug_dilate"] + 1,) * 2)
    for i in range(T):
        n = sf[i]; raw, cab, plug = load_frame(pan[n])
        ends = [(tf.to_px(Pmm[k, i, 0]), tf.to_px(Pmm[k, i, 15])) for k in range(4)]
        if i == 0:
            for k, ch in enumerate(chains):
                out[k, 0] = resample(ch.Y, 16)
            prev_ends = ends; continue
        # evidence
        plug_d = cv2.dilate(plug.astype(np.uint8), ker) > 0
        txt = text_mask(raw)
        ev_cm = cab & ~txt & ~plug_d
        if p["use_dark"]:
            dk = dark_mask(raw, tf, p["dark_thr"]) & ~plug_d & ~txt
            # gate dark pixels: near previous nodes of any chain (predicted below) -- computed after prediction
        # prediction: move previous nodes with their end displacements (exp decay along the cable)
        for k, ch in enumerate(chains):
            dh = ends[k][0] - prev_ends[k][0]; dm = ends[k][1] - prev_ends[k][1]
            L_px = ch.rest.sum(); tau = p["tau_mm"] / tf.mm_per_px / L_px
            wh = np.exp(-ch.s / tau)[:, None]; wm = np.exp(-(1 - ch.s) / tau)[:, None]
            ch.Y = ch.Y + dh * wh + dm * wm
        # node image for gating
        node_img = np.zeros((H, W), np.uint8)
        for ch in chains:
            cv2.polylines(node_img, [ch.Y.astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
        d_nodes = cv2.distanceTransform((node_img == 0).astype(np.uint8), cv2.DIST_L2, 3)
        gate = d_nodes < p["gate_px"]
        ev = ev_cm & gate
        pts = [np.argwhere(ev[::p["stride_cm"], ::p["stride_cm"]])[:, ::-1] * p["stride_cm"]]
        if p["use_dark"]:
            ev2 = dk & gate & (~cab if not p["dark_in_cab"] else np.ones_like(cab))
            pts.append(np.argwhere(ev2[::p["stride_dark"], ::p["stride_dark"]])[:, ::-1] * p["stride_dark"])
            ev = ev | ev2
        X = np.concatenate(pts).astype(float)
        if len(X) < 50:
            X = np.concatenate([ch.Y for ch in chains])
        # visibility: distance of predicted node to nearest evidence pixel (class map or dark)
        de = cv2.distanceTransform((~(ev_cm | (dk if p["use_dark"] else ev_cm))).astype(np.uint8), cv2.DIST_L2, 3)
        vis_d = []
        for ch in chains:
            q = np.clip(ch.Y.round().astype(int), [0, 0], [W - 1, H - 1]); dd = de[q[:, 1], q[:, 0]]
            inplug = plug_d[q[:, 1], q[:, 0]]; dd = np.where(inplug, 1e3, dd); vis_d.append(dd)
        pins = [[(0, ends[k][0]), (chains[k].M - 1, ends[k][1])] for k in range(4)]
        Ys, sigma2, nit = em_joint(X, chains, pins, vis_d, p, sigma2); iters.append(nit)
        for k, ch in enumerate(chains):
            Y = Ys[k]; Y[0] = ends[k][0]; Y[-1] = ends[k][1]
            q = np.clip(Y.round().astype(int), [0, 0], [W - 1, H - 1]); dd = de[q[:, 1], q[:, 0]]
            inplug = plug_d[q[:, 1], q[:, 0]]
            # plug interior runs at the two ends (contiguous from the pinned node)
            j0 = 0
            while j0 + 1 < ch.M and inplug[j0 + 1]: j0 += 1
            j1 = ch.M - 1
            while j1 - 1 > j0 and inplug[j1 - 1]: j1 -= 1
            if p["len_iters"] > 0:
                fixed = np.where(dd <= p["d_vis"], p["len_w"], 0.0); fixed[0] = fixed[-1] = 1.0
                fixed[:j0 + 1] = 1.0; fixed[j1:] = 1.0
                if p["len_mode"] == "hidden":
                    hidden = dd > p["d_vis"]; hidden[:j0 + 1] = False; hidden[j1:] = False
                    Y = project_lengths_hidden(Y, ch.rest, hidden, fixed, p["len_iters"], skip=(j0, j1), smooth=p["smooth"])
                else:
                    Y = project_lengths(Y, ch.rest, fixed, p["len_iters"], skip=(j0, j1))
            if p["relay"]:
                Y[:j0 + 1] = np.linspace(Y[0], Y[j0], j0 + 1); Y[j1:] = np.linspace(Y[j1], Y[-1], ch.M - j1)
            if p["fill_hidden"]:
                hidden = dd > p["d_vis"]; hidden[:j0 + 1] = False; hidden[j1:] = False
                Y, deficit = fill_hidden_runs(Y, ch.Y, hidden, ch.L_ref, p["min_deficit"] / tf.mm_per_px, p["max_deficit"] / tf.mm_per_px)
            if p["resample_each"]:
                Y = resample(Y, ch.M)
            ch.Y = Y; out[k, i] = resample(Y, 16)
        # --- detector re-initialisation
        jump = max(max(np.linalg.norm(ends[k][0] - prev_ends[k][0]), np.linalg.norm(ends[k][1] - prev_ends[k][1])) for k in range(4)) * tf.mm_per_px
        def off_frac(Y):
            q = np.clip(Y.round().astype(int), [0, 0], [W - 1, H - 1])
            return float(((de[q[:, 1], q[:, 0]] > p["d_vis"]) & ~plug_d[q[:, 1], q[:, 0]]).mean())
        for k, ch in enumerate(chains):
            if dcar[i, k] or abs(Ldet[k, i] - Lglob[k]) > p["det_len_tol"] * Lglob[k]:
                continue                                   # detector route not trustworthy here
            Yd = resample(tf.to_px(Dmm[k, i]), ch.M); od = off_frac(Yd)
            if od > p["det_off"]:
                continue
            ot = off_frac(ch.Y); Lt = np.linalg.norm(np.diff(ch.Y, axis=0), axis=1).sum() * tf.mm_per_px
            if jump > p["jump_mm"] or ot > p["reset_off"] or abs(Lt - Lglob[k]) > 0.08 * Lglob[k]:
                ch.Y = Yd; out[k, i] = resample(Yd, 16); reset[i, k] = True
        prev_ends = ends
        if debug_dir is not None or n in sheet_frames:
            vis = raw.copy(); vis[ev] = (0, 255, 255)
            vis[X[:, 1].astype(int), X[:, 0].astype(int)] = (255, 255, 0)
            for k, ch in enumerate(chains):
                q = ch.Y.astype(np.int32); cv2.polylines(vis, [q.reshape(-1, 1, 2)], False, COLS[k], 2)
                for pt in q: cv2.circle(vis, tuple(pt), 4, COLS[k], -1)
            cv2.putText(vis, f"{clip} frame {n} s2={sigma2:.1f} it={nit} N={len(X)}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            if debug_dir is not None:
                cv2.imwrite(f"{debug_dir}/{n:04d}.jpg", cv2.resize(vis, (800, 1064)))
            if n in sheet_frames:
                sheet_frames_imgs[n] = vis
    rt = time.time() - t_start
    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {f"c{k}": np.stack([tf.to_mm(out[k, i]) for i in range(T)]) / 1000.0 for k in range(4)}
    if (out_dir / f"{clip}.npz").exists():
        raise SystemExit(f"{out_dir / clip}.npz exists -- pick a new --out")
    np.savez(out_dir / f"{clip}.npz", source_frames=np.array(sf[:T]), t0=z["t0"], t1=z["t1"], fps=z["fps"], reset=reset[:T], **res)
    print(f"resets per cable: {reset[:T].sum(0).tolist()} of {T} frames", flush=True)
    L = {k: (np.linalg.norm(np.diff(res[f"c{k}"], axis=1), axis=2).sum(1) * 1000) for k in range(4)}
    return rt, L, iters, sf[:T]


sheet_frames_imgs = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", type=int, nargs="*", default=[]); ap.add_argument("--debug-dir")
    ap.add_argument("--limit", type=int); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--det", required=True, help="folder with the tracking_v2 npz (per-frame detector)")
    a = ap.parse_args()
    p = dict(DEFAULT)
    for kv in a.set:
        k, v = kv.split("="); p[k] = type(DEFAULT[k])(float(v)) if isinstance(DEFAULT[k], (int, float)) else v
    out = Path(a.out)
    if a.debug_dir:
        Path(a.debug_dir).mkdir(parents=True, exist_ok=True)
    rt, L, iters, sf = track_clip(a.clip, out, p, set(a.sheet), a.debug_dir, a.limit, a.det)
    print(f"{a.clip}: {len(sf)} frames in {rt:.1f}s ({rt / max(len(sf) - 1, 1) * 1000:.0f} ms/frame), EM iters mean {np.mean(iters):.1f}")
    for k in range(4):
        print(f"  c{k} len mm: start {L[k][0]:.0f} median {np.median(L[k]):.0f} min {L[k].min():.0f} max {L[k].max():.0f}")
    if a.sheet:
        imgs = [cv2.resize(sheet_frames_imgs[n], (800, 1064)) for n in a.sheet if n in sheet_frames_imgs]
        cv2.imwrite(str(out / f"{a.clip}_sheet.png"), np.concatenate(imgs, 1))
    json.dump(p, open(out / f"{a.clip}_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()
