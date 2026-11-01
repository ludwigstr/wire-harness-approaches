"""Render skeleton_check_<clip>.mp4 (+ middle still) from a tracking npz and the export panels, drawn exactly as
scripts/track_replay_folder.py does (2026-11-01, for the final_v1 tracking).

    python scripts/skeleton_check_from_npz.py --npz <dir>/<clip>.npz --folder data/videos/<clip> --out <outdir>
"""
import argparse, glob, re, sys
from pathlib import Path
import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa: E402

SCALE = 0.8
COLORS = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--folder", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    a = ap.parse_args()
    clip = Path(a.npz).stem; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    vid = out / f"skeleton_check_{clip}.mp4"
    if vid.exists():
        raise SystemExit(f"{vid} exists -- never overwrite rendered videos")
    z = np.load(ROOT / a.npz); sf = [int(s) for s in z["source_frames"]]
    pan = {}
    for d in glob.glob(str(ROOT / a.folder / "visualization/panels/*_frame_*")):
        pan.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    tf = None; imgs = []
    for i, n in enumerate(sf):
        img = cv2.imread(f"{pan[n]}/01_raw.png")
        img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        tf = tf or find_board(img)
        for k in range(4):
            px = tf.to_px(z[f"c{k}"][i] * 1000.0).astype(int)
            cv2.polylines(img, [px.reshape(-1, 1, 2)], False, COLORS[k], 4, cv2.LINE_AA)
            for p in px:
                cv2.circle(img, tuple(int(v) for v in p), 7, COLORS[k], -1, cv2.LINE_AA)
        txt = f"{clip}  source frame {n}"
        cv2.putText(img, txt, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(img, txt, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 2, cv2.LINE_AA)
        imgs.append(cv2.cvtColor(cv2.resize(img, (800, 1064), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    iio.mimsave(vid, imgs, fps=a.fps)
    iio.imwrite(out / f"skeleton_check_{clip}_frame{len(imgs) // 2:03d}.png", imgs[len(imgs) // 2])
    print(f"-> {vid}: {len(imgs)} frames")


if __name__ == "__main__":
    main()
