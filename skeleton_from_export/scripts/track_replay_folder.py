"""Cable skeletons (4 cables x 16 nodes, board metres) from a segmentation-pipeline export folder
such as data/videos/20261028_124829 (user, 2026-10-29), for testing algorithms later.

The folder has, per processed frame, the raw image and the class map (visualization/panels/
<k>_frame_<orig>/01_raw.png, 02_class_map.png; 2000x2660), but no source video. This script rebuilds
the two inputs the project's tracker expects -- a source video at 0.8 scale (1600x2128, 4 fps) and a
side-by-side "<clip>_replay_raw_mask_1.mp4" (raw | class map) in a private WIRE_SEG_DIR -- and then runs
exactly the tracks14 pipeline (find_board, track_harness via the class map, HarnessTracker).

Frames: the longest run of consecutive source frame numbers (duplicates dropped), so every step is one
source frame. Output: <out>/<clip>.npz in the data/tracks14 format (c0..c3 (T,16,2) m, t0, t1, fps)
plus a check video with the 16-node skeletons drawn on the raw frames.

    python scripts/track_replay_folder.py --folder data/videos/20261028_124829 --out data/videos/20261028_124829/skeleton
"""
import argparse, glob, os, re, sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALE = 0.8                      # 2000x2660 panel -> 1600x2128 source (seg_masks.PANEL_TO_SOURCE_SCALE)
COLORS = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]     # BGR per cable


def read(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    a = ap.parse_args()
    folder, out = ROOT / a.folder, ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists -- pick a new --out")
    out.mkdir(parents=True)
    clip = folder.name

    # longest run of consecutive source frames
    frames = {}
    for d in sorted(glob.glob(str(folder / "visualization/panels/*_frame_*"))):
        frames.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    nums = sorted(frames)
    runs, cur = [], [nums[0]]
    for n in nums[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            runs.append(cur); cur = [n]
    runs.append(cur)
    run = max(runs, key=len)
    print(f"{clip}: {len(nums)} distinct frames, using {run[0]}..{run[-1]} ({len(run)}); dropped {sorted(set(nums) - set(run))}",
          flush=True)

    # rebuild the tracker's inputs
    seg = out / "seg_input"; seg.mkdir()
    src = out / f"{clip}.mp4"
    ws = iio.get_writer(src, fps=a.fps, codec="libx264", quality=10, macro_block_size=1)
    wm = iio.get_writer(seg / f"{clip}_replay_raw_mask_1.mp4", fps=a.fps, codec="libx264", quality=10,
                        macro_block_size=1)
    for n in run:
        raw = cv2.imread(f"{frames[n]}/01_raw.png"); cm = cv2.imread(f"{frames[n]}/02_class_map.png")
        H, W = raw.shape[:2]
        ws.append_data(cv2.cvtColor(cv2.resize(raw, (int(W * SCALE), int(H * SCALE)), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB))
        wm.append_data(cv2.cvtColor(np.concatenate([raw, cm], 1), cv2.COLOR_BGR2RGB))
    ws.close(); wm.close()

    os.environ["WIRE_SEG_DIR"] = str(seg)           # read by seg_masks at import time
    from wire_model.perception.rectify import find_board
    from wire_model.perception.harness import track_harness
    from wire_model.perception.tracking import HarnessTracker

    cap = cv2.VideoCapture(str(src)); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tf = find_board(read(cap, 0))
    tracks, hub_i, _ = track_harness(cap, tf, T, read, clip=clip)
    if tracks is None:
        raise SystemExit("track_harness found no modules")
    order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
    tracks = tracks[:, order, :]
    tracked = HarnessTracker(tf, clip=clip).run(cap, read, tracks, range(T))
    ok = [f for f in range(T) if f in tracked]
    print(f"tracked {len(ok)} of {T} frames", flush=True)
    shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in ok]) for k in range(4)}
    np.savez(out / f"{clip}.npz", t0=ok[0], t1=ok[-1], fps=a.fps, source_frames=np.array(run)[ok],
             **{f"c{k}": shp[k] for k in range(4)})

    # check video: 16-node skeletons on the raw frames
    imgs = []
    for i, f in enumerate(ok):
        img = read(cap, f).copy()
        for k in range(4):
            px = tf.to_px(shp[k][i] * 1000.0).astype(int)
            cv2.polylines(img, [px.reshape(-1, 1, 2)], False, COLORS[k], 4, cv2.LINE_AA)
            for p in px:
                cv2.circle(img, tuple(int(v) for v in p), 7, COLORS[k], -1, cv2.LINE_AA)
        cv2.putText(img, f"{clip}  source frame {run[f]}", (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(img, f"{clip}  source frame {run[f]}", (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (20, 20, 20), 2, cv2.LINE_AA)
        imgs.append(cv2.cvtColor(cv2.resize(img, (800, 1064), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    iio.mimsave(out / f"skeleton_check_{clip}.mp4", imgs, fps=a.fps)
    iio.imwrite(out / f"skeleton_check_{clip}_frame{len(imgs) // 2:03d}.png", imgs[len(imgs) // 2])
    print(f"-> {out / (clip + '.npz')}: c0..c3 {shp[0].shape}")


if __name__ == "__main__":
    main()
