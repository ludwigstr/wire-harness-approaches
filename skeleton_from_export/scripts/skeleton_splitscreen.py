"""Triple-screen check videos (user, 2026-11-01): left = our 16-node skeleton on the camera image (as in
skeleton_check_<clip>.mp4), middle = the segmentation pipeline's class map (02_class_map.png: yellow cable,
green movers), right = the pipeline's own raw skeleton panel (03_1_full_skeleton.png, white 1-px strand on
black, dilated for visibility), all from visualization/panels/<k>_frame_<n>/ of the export folder, so the
three can be compared frame by frame. The panels' yellow text band is masked out.

    python scripts/skeleton_splitscreen.py --npz data/videos/skeletons_2026-10-30/<clip>.npz \
        --folder data/videos/<clip> --out data/videos/skeletons_2026-10-30/splitscreen/skeleton_split_<clip>.mp4
"""
import argparse, glob, re, sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa: E402

SCALE = 0.8                     # 2000x2660 panel -> 1600x2128 source
COLORS = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]     # BGR per cable, as skeleton_check
W, H = 600, 798                 # each half


def draw(img, tf, P, label):
    for k in range(4):
        px = tf.to_px(P[k] * 1000.0).astype(int)
        cv2.polylines(img, [px.reshape(-1, 1, 2)], False, COLORS[k], 4, cv2.LINE_AA)
        for p in px:
            cv2.circle(img, tuple(int(v) for v in p), 7, COLORS[k], -1, cv2.LINE_AA)
    cv2.putText(img, label, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 6, cv2.LINE_AA)
    cv2.putText(img, label, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=float, default=4.0)
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists -- never overwrite rendered videos")
    out.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / a.npz); clip = Path(a.npz).stem
    P = np.stack([z[f"c{k}"] for k in range(4)]); sf = [int(s) for s in z["source_frames"]]
    panels = {}
    for d in glob.glob(str(ROOT / a.folder / "visualization/panels/*_frame_*")):
        panels.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)

    def raw(n):
        im = cv2.imread(f"{panels[n]}/01_raw.png")
        return cv2.resize(im, (int(im.shape[1] * SCALE), int(im.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)

    def label(img, text):
        cv2.putText(img, text, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 6, cv2.LINE_AA)
        cv2.putText(img, text, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    tf = find_board(raw(sf[0]))
    frames = []
    for i, n in enumerate(sf):
        left = draw(raw(n), tf, P[:, i], f"{clip}  frame {n}  ours")
        # middle: the pipeline's class map (yellow = cable, green = mover), text band masked
        cm = cv2.imread(f"{panels[n]}/02_class_map.png")
        cm[: int(0.06 * cm.shape[0])] = 0
        middle = label(cm, "segmentation")
        # right: the pipeline's raw skeleton alone (user, 2026-11-01: no overlay)
        sk = cv2.imread(f"{panels[n]}/03_1_full_skeleton.png", cv2.IMREAD_GRAYSCALE)
        sk[: int(0.06 * sk.shape[0])] = 0
        strand = cv2.dilate((sk > 128).astype(np.uint8), np.ones((5, 5), np.uint8))
        img = np.full((strand.shape[0], strand.shape[1], 3), 20, np.uint8)
        img[strand > 0] = (230, 230, 230)
        right = label(img, "raw skeleton")
        frames.append(cv2.cvtColor(np.concatenate([left, middle, right], 1), cv2.COLOR_BGR2RGB))
    iio.mimsave(out, frames, fps=a.fps, macro_block_size=1)
    print(f"-> {out}: {len(frames)} frames")


if __name__ == "__main__":
    main()
