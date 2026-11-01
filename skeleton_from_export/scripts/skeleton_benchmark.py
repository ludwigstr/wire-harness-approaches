"""Benchmark for the export-folder skeletons (user, 2026-11-01: "Tracking 1000% verbessern").

No ground truth exists, so the score is built from what is physically certain and what the export
provides:
  bad_len   %frames where a cable's arc length is more than 8 % off that cable's clip median
            (a cable is inextensible: a wrong strand is nearly always a wrong length)
  jump95    95th percentile of the per-frame mean node displacement, mm (frame-to-frame jitter/jumps)
  jump_max  the largest such displacement, mm
  unexpl    %of class-map cable pixels (02_class_map.png, yellow, board region only) farther than
            TOL px from every route -- cable in the segmentation that no route explains
  off_mask  %of route samples farther than TOL px from any cable pixel of the class map OR of the
            dark-cable mask of the raw image -- route running over empty board
  pink_off  %of cable-3 route (the thin wire bundle, plug ends excluded) farther than 10 px from the
            thin-wire evidence (black-hat, wire_model.perception.thin_wire) -- pink cable off its wire
  flagged   for the reviewed clips: how many of the frames I marked as wrong (REVIEW_FRAMES) are
            still off by >8 % in length for the named cable

    python scripts/skeleton_benchmark.py --npz data/videos/skeletons_2026-10-30/*.npz --out bench.csv
    python scripts/skeleton_benchmark.py --npz exp/*.npz --baseline bench.csv      # delta per clip
"""
import argparse, csv, glob, os, re, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa: E402
from wire_model.perception import thin_wire as TW  # noqa: E402

SCALE = 0.8
TOL = 14.0
# cable index -> frame ranges I marked as wrong in the review (source frame numbers, inclusive)
REVIEW_FRAMES = {
    "20261021_160124": {0: [(88, 98)], 2: [(88, 95)]},
    "20261028_123828": {0: [(138, 160)]},
    "20261028_124048": {2: [(64, 69), (122, 130)]},
    "20261028_122521": {2: [(112, 123)]},
    "20261028_122147": {2: [(119, 123)]},
    "20261028_103236": {3: [(72, 83), (156, 173)]},
    "20261028_100222": {2: [(126, 133)]},
    "20261021_180139": {0: [(124, 165)], 1: [(121, 165)]},
    "20261021_175638": {2: [(90, 95), (136, 155)]},
    "20261021_175358": {0: [(129, 161)], 2: [(127, 147)]},
    "20261021_171620": {0: [(151, 158)]},
    "20261021_171354": {0: [(121, 154)], 2: [(74, 106)]},
}


def panels_of(folder):
    out = {}
    for d in glob.glob(str(folder / "visualization/panels/*_frame_*")):
        out.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    return out


def dark_mask(raw_small, tf):
    """Dark cable on the light board: black-hat of the grey image inside the board."""
    g = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = bh > 28
    poly = tf.to_px(np.array(((60, 60), (1140, 60), (1140, 1620), (60, 1620)), float)).astype(np.int32)
    board = np.zeros(g.shape, np.uint8); cv2.fillPoly(board, [poly], 1)
    return m & (board > 0)


def score_clip(npz, folder, stride=3):
    z = np.load(npz); P = np.stack([z[f"c{k}"] for k in range(4)]); sf = [int(s) for s in z["source_frames"]]
    T = P.shape[1]
    L = np.linalg.norm(np.diff(P, axis=2), axis=3).sum(2) * 1000
    d = np.linalg.norm(np.diff(P, axis=1), axis=3).mean(2) * 1000
    med = np.median(L, 1, keepdims=True); off = np.abs(L - med) > 0.08 * med
    r = {"clip": Path(npz).stem, "frames": T, "bad_len": 100 * off.any(0).mean(),
         "jump95": float(np.percentile(d, 95)), "jump_max": float(d.max())}
    pan = panels_of(folder); tf = None; unexpl = []; offm = []; pink = []
    for i in range(0, T, stride):
        n = sf[i]
        if n not in pan:
            continue
        raw = cv2.imread(f"{pan[n]}/01_raw.png"); cm = cv2.imread(f"{pan[n]}/02_class_map.png")
        raw = cv2.resize(raw, (int(raw.shape[1] * SCALE), int(raw.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        cab = cv2.resize(np.all(cm == (0, 255, 255), axis=2).astype(np.uint8), (raw.shape[1], raw.shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0
        cab[: int(0.07 * cab.shape[0])] = False
        if tf is None:
            tf = find_board(raw)
        routes = np.zeros(cab.shape, np.uint8)
        for k in range(4):
            px = tf.to_px(P[k, i] * 1000.0).astype(np.int32)
            cv2.polylines(routes, [px.reshape(-1, 1, 2)], False, 1, 1)
        dr = cv2.distanceTransform((routes == 0).astype(np.uint8), cv2.DIST_L2, 3)
        unexpl.append(100 * float((dr[cab] > TOL).mean()) if cab.any() else 0.0)
        ev = cab | dark_mask(raw, tf)
        de = cv2.distanceTransform((~ev).astype(np.uint8), cv2.DIST_L2, 3)
        samp = np.concatenate([tf.to_px(P[k, i] * 1000.0) for k in range(4)]).astype(int)
        samp = samp[(samp[:, 0] >= 0) & (samp[:, 1] >= 0) & (samp[:, 0] < de.shape[1]) & (samp[:, 1] < de.shape[0])]
        offm.append(100 * float((de[samp[:, 1], samp[:, 0]] > TOL).mean()))
        dw = cv2.distanceTransform((~TW.wire_evidence(raw)).astype(np.uint8), cv2.DIST_L2, 3)
        r3 = TW.smooth_route(tf.to_px(P[3, i] * 1000.0), 200, win=1)[25:-25].astype(int)
        r3 = r3[(r3[:, 0] >= 0) & (r3[:, 1] >= 0) & (r3[:, 0] < dw.shape[1]) & (r3[:, 1] < dw.shape[0])]
        pink.append(100 * float((dw[r3[:, 1], r3[:, 0]] > 10).mean()))
    r["unexpl"] = float(np.mean(unexpl)) if unexpl else np.nan
    r["off_mask"] = float(np.mean(offm)) if offm else np.nan
    r["pink_off"] = float(np.mean(pink)) if pink else np.nan
    rv = REVIEW_FRAMES.get(r["clip"]); flagged = tot = 0
    if rv:
        for k, rngs in rv.items():
            for a, b in rngs:
                idx = [i for i, n in enumerate(sf) if a <= n <= b]
                tot += len(idx); flagged += int(off[k, idx].sum())
    r["flagged"] = flagged; r["flagged_of"] = tot
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True); ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out"); ap.add_argument("--baseline"); ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    base = {}
    if a.baseline:
        base = {r["clip"]: r for r in csv.DictReader(open(a.baseline))}
    rows = []
    for f in a.npz:
        clip = Path(f).stem
        r = score_clip(f, ROOT / a.videos / clip, a.stride); rows.append(r)
        line = (f"{clip}  bad_len {r['bad_len']:5.1f}%  jump95 {r['jump95']:5.1f}  max {r['jump_max']:5.1f}  "
                f"unexpl {r['unexpl']:5.1f}%  off_mask {r['off_mask']:5.1f}%  pink_off {r['pink_off']:5.1f}%  flagged {r['flagged']}/{r['flagged_of']}")
        if clip in base:
            b = base[clip]
            line += (f"   | vs base: bad_len {float(b['bad_len']):.1f}  jump95 {float(b['jump95']):.1f}  "
                     f"unexpl {float(b['unexpl']):.1f}  off_mask {float(b['off_mask']):.1f}  flagged {b['flagged']}")
        print(line, flush=True)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if len(rows) > 1:
        print(f"\nMEAN over {len(rows)}: bad_len {np.mean([r['bad_len'] for r in rows]):.2f}%  "
              f"jump95 {np.mean([r['jump95'] for r in rows]):.1f}  unexpl {np.nanmean([r['unexpl'] for r in rows]):.2f}%  "
              f"off_mask {np.nanmean([r['off_mask'] for r in rows]):.2f}%  pink_off {np.nanmean([r['pink_off'] for r in rows]):.2f}%  flagged {sum(r['flagged'] for r in rows)}/{sum(r['flagged_of'] for r in rows)}")


if __name__ == "__main__":
    main()
