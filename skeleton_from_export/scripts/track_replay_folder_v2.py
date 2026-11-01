"""Re-track an export folder with wire_model.perception.tracking_v2 (user, 2026-11-01).

Reads the panels directly (01_raw.png at 0.8 scale, 02_class_map.png), takes the hub and mover
positions per frame from the existing v1 result (data/videos/skeletons_2026-10-30/<clip>.npz: node 0
and node 15 of each cable -- the module detection was never the problem), builds the augmented wire
mask and the segment graph, and runs the diverse-candidate + global-length + Viterbi tracker.

    python scripts/track_replay_folder_v2.py --clip 20261021_180139 --out data/videos/skeletons_2026-10-30/experiments/v2
"""
import argparse, glob, re, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa: E402
from wire_model.perception.tracer import build_graph  # noqa: E402
from wire_model.perception import tracking_v2 as V  # noqa: E402

SCALE = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="data/videos/skeletons_2026-10-30")
    ap.add_argument("--augment", action="store_true", help="add black-hat dark-cable pixels (tested worse on 20261021_180139)")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    clip = a.clip; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    if (out / f"{clip}.npz").exists():
        raise SystemExit(f"{out / clip}.npz exists -- pick a new --out")
    z = np.load(ROOT / a.base / f"{clip}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])
    sf = [int(s) for s in z["source_frames"]]
    panels = {}
    for d in glob.glob(str(ROOT / "data/videos" / clip / "visualization/panels/*_frame_*")):
        panels.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    t0 = time.time(); frames = []; tf = None
    for i, n in enumerate(sf):
        raw = cv2.imread(f"{panels[n]}/01_raw.png")
        raw = cv2.resize(raw, (int(raw.shape[1] * SCALE), int(raw.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        cm = cv2.imread(f"{panels[n]}/02_class_map.png")
        cm = cv2.resize(cm, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
        cab = np.all(cm == (0, 255, 255), axis=2); mov = np.all(cm == (0, 255, 0), axis=2)
        if tf is None:
            tf = find_board(raw)
        hub = tf.to_px(P[0, i, 0] * 1000.0); movers = np.stack([tf.to_px(P[k, i, -1] * 1000.0) for k in range(4)])
        if not a.augment:
            from wire_model.perception.rectify import board_mask
            cab[: int(0.07 * cab.shape[0])] = False
            wire = cab & board_mask(raw, tf, inset_mm=8.0)
            wire = cv2.morphologyEx(wire.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
        else:
            wire = V.augmented_wire_mask(raw, cab, tf, plug_px=[hub] + list(movers))
        wire = V.cut_plugs(wire, [hub] + list(movers))
        dist = cv2.distanceTransform((~wire).astype(np.uint8), cv2.DIST_L2, 3)
        frames.append({"wire": wire, "graph": build_graph(wire), "dist": dist, "hub": hub, "movers": movers})
    print(f"{clip}: {len(frames)} frames prepared in {time.time() - t0:.0f}s", flush=True)
    routes, lengths, carried = V.track_clip(frames, tf, k=a.k, verbose=True)
    print(f"carried (length repair): {int(carried.sum())} of {carried.size} cable-frames, per cable {carried.sum(0).tolist()}", flush=True)
    Q = np.zeros_like(P)
    for i in range(len(sf)):
        for k in range(4):
            r = routes[i][k].copy(); r[0], r[-1] = frames[i]["hub"], frames[i]["movers"][k]
            Q[k, i] = tf.to_mm(r) / 1000.0
    np.savez(out / f"{clip}.npz", t0=z["t0"], t1=z["t1"], fps=z["fps"], source_frames=np.array(sf), carried=carried,
             **{f"c{k}": Q[k] for k in range(4)})
    print(f"-> {out / clip}.npz  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
