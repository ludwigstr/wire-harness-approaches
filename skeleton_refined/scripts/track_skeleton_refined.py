"""Skeleton-refined cable tracking: perception pipeline assignment + skeleton shape refinement.

Combines two sources:
- **Class map** (02_class_map.png, yellow channel) provides the wire mask for the
  perception pipeline's tracer -- fast, reliable 4-cable identity assignment.
- **X-planner skeleton** (03_1_full_skeleton.png) refines the cable SHAPE: each
  pipeline cable is attracted toward nearby skeleton pixels, then Gaussian-smoothed.

Result: 4 smooth, COMPLETE cables (hub to module) with no gaps, no branches,
no floating segments, and no sharp corners. NPZ output uses the pipeline's
16-node resampled shapes (in board metres), compatible with all evaluation
scripts in this repository (data/tracks14 format).

Usage (run from inside skeleton_refined/):

    python scripts/track_skeleton_refined.py --root /path/to/X-planner_skeletonization --out output
    python scripts/track_skeleton_refined.py --root /path/to/X-planner_skeletonization --out output --clip 20261021_160124
"""
import argparse
import glob
import os
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALE = 0.8  # 2000x2660 panel -> 1600x2128 source
COLORS_BGR = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]
COLORS_BRIGHT = [(80, 80, 255), (255, 80, 80), (80, 220, 80), (230, 80, 230)]
COLORS_GLOW = [(0, 0, 180), (180, 0, 0), (0, 110, 0), (140, 0, 140)]
VID_SIZE = (800, 1064)


def read_cap(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def build_frame_list(clip_dir):
    """Return sorted (seq, source_frame, panel_dir, raw_path, skel_path, cmap_path)."""
    panels = sorted(glob.glob(str(clip_dir / "visualization" / "panels" / "*_frame_*")))
    frames = {}
    for p in panels:
        m = re.search(r"(\d+)_frame_(\d+)$", os.path.basename(p))
        if not m:
            continue
        seq = int(m.group(1))
        src_frame = int(m.group(2))
        skel = os.path.join(p, "03_1_full_skeleton.png")
        cmap = os.path.join(p, "02_class_map.png")
        raw_path = str(clip_dir / "frames" / f"replay_{seq:06d}_PROCESSING.png")
        if os.path.exists(skel) and os.path.exists(cmap) and os.path.exists(raw_path):
            frames.setdefault(src_frame, (seq, src_frame, p, raw_path, skel, cmap))
    return sorted(frames.values(), key=lambda x: x[1])


def longest_run(nums):
    if not nums:
        return []
    runs, cur = [], [nums[0]]
    for n in nums[1:]:
        if n == cur[-1] + 1:
            cur.append(n)
        else:
            runs.append(cur)
            cur = [n]
    runs.append(cur)
    return max(runs, key=len)


def load_skeleton_mask(skel_path, source_shape):
    """Load skeleton PNG, mask text banner, resize to source resolution."""
    skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        return np.zeros(source_shape[:2], bool)
    text_h = int(0.06 * skel.shape[0])
    skel[:text_h, :] = 0
    skel_mask = skel > 80
    return cv2.resize(skel_mask.astype(np.uint8),
                      (source_shape[1], source_shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0


def densify_polyline(pts, n=300):
    """Resample a polyline to n evenly-spaced points via linear interpolation."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1))
    t_in = cum / total
    t_out = np.linspace(0, 1, n)
    return np.column_stack([
        np.interp(t_out, t_in, pts[:, 0]),
        np.interp(t_out, t_in, pts[:, 1])
    ])


def refine_cable(cable_16, skel_mask, n_dense=300, attract_r=35, sigma=5.0):
    """Refine a 16-node pipeline cable using skeleton attraction + smoothing.

    1. Densify the coarse 16-node polyline to n_dense points
    2. Attract each interior point toward the nearest skeleton pixel
    3. Gaussian-smooth to remove any sharp corners
    4. Re-anchor endpoints (hub and module positions stay exact)

    Result: a smooth, COMPLETE cable from hub to module that follows the
    skeleton's accurate shape where nearby, and stays continuous across gaps.
    """
    if cable_16 is None or len(cable_16) < 2:
        return cable_16

    hub = cable_16[0].copy()
    mover = cable_16[-1].copy()

    dense = densify_polyline(cable_16, n_dense)

    # Attract toward skeleton pixels
    sy, sx = np.nonzero(skel_mask)
    if len(sy) > 0:
        skel_pts = np.stack([sx, sy], axis=1).astype(float)
        tree = cKDTree(skel_pts)
        dists, idxs = tree.query(dense)
        margin = max(12, n_dense // 20)
        for i in range(margin, n_dense - margin):
            if dists[i] < attract_r:
                alpha = 0.85 * (1.0 - (dists[i] / attract_r) ** 2)
                dense[i] = (1 - alpha) * dense[i] + alpha * skel_pts[idxs[i]]

    # Gaussian smoothing (interior only)
    smoothed_x = gaussian_filter1d(dense[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter1d(dense[:, 1], sigma=sigma)
    dense[1:-1, 0] = smoothed_x[1:-1]
    dense[1:-1, 1] = smoothed_y[1:-1]

    # Re-anchor endpoints
    dense[0] = hub
    dense[-1] = mover

    # Smooth the first and last few points to blend anchors
    blend = min(20, n_dense // 10)
    for i in range(1, blend):
        w = i / blend
        dense[i] = (1 - w) * hub + w * dense[i]
    for i in range(n_dense - blend, n_dense - 1):
        w = (n_dense - 1 - i) / blend
        dense[i] = (1 - w) * mover + w * dense[i]

    return dense


def draw_cables(img, cable_px_list, skel_mask, tf, shp_frame):
    """Draw 4 refined cables on the image. No skeleton pixel coloring."""
    src_h, src_w = img.shape[:2]

    # Thin gray skeleton underlay for context
    skel_vis = np.zeros((src_h, src_w), np.uint8)
    skel_vis[skel_mask] = 60
    skel_dilated = cv2.dilate(skel_vis, np.ones((2, 2), np.uint8))
    gray_mask = skel_dilated > 0
    for c in range(3):
        img[:, :, c] = np.where(gray_mask,
                                np.clip(img[:, :, c].astype(int) + skel_dilated.astype(int), 0, 255).astype(np.uint8),
                                img[:, :, c])

    for k in range(4):
        cable_16 = cable_px_list[k]
        if cable_16 is None or len(cable_16) < 2:
            continue

        refined = refine_cable(cable_16, skel_mask)
        pts_int = refined.astype(np.int32).reshape(-1, 1, 2)

        # Glow (thick, darker)
        cv2.polylines(img, [pts_int], False, COLORS_GLOW[k], 6, cv2.LINE_AA)
        # Core (bright)
        cv2.polylines(img, [pts_int], False, COLORS_BRIGHT[k], 3, cv2.LINE_AA)

        # Hub and mover endpoint markers
        hub_pt = tuple(cable_16[0].astype(int))
        mov_pt = tuple(cable_16[-1].astype(int))
        cv2.circle(img, hub_pt, 6, COLORS_BGR[k], -1, cv2.LINE_AA)
        cv2.circle(img, mov_pt, 7, COLORS_BRIGHT[k], -1, cv2.LINE_AA)
        cv2.circle(img, mov_pt, 7, (255, 255, 255), 2, cv2.LINE_AA)


def process_clip(clip_dir, out_dir, fps=4.0):
    clip = clip_dir.name
    all_frames = build_frame_list(clip_dir)
    if not all_frames:
        return {"clip": clip, "status": "failed", "reason": "no frames found"}

    src_nums = [f[1] for f in all_frames]
    run = longest_run(src_nums)
    frame_map = {f[1]: f for f in all_frames}
    frames = [frame_map[n] for n in run if n in frame_map]

    print(f"  {len(all_frames)} total frames, using {run[0]}..{run[-1]} ({len(frames)})",
          flush=True)

    # Build source video + class-map side-by-side video (using original class map)
    seg_dir = out_dir / "_seg_tmp"
    seg_dir.mkdir(exist_ok=True)
    src_vid = out_dir / f"_tmp_{clip}.mp4"
    seg_vid = seg_dir / f"{clip}_replay_raw_mask_1.mp4"

    ws = iio.get_writer(str(src_vid), fps=fps, codec="libx264",
                        quality=5, macro_block_size=1)
    wm = iio.get_writer(str(seg_vid), fps=fps, codec="libx264",
                        quality=5, macro_block_size=1)

    for seq, src_frame, panel_dir, raw_path, skel_path, cmap_path in frames:
        raw = cv2.imread(raw_path)
        cmap = cv2.imread(cmap_path)
        H, W = raw.shape[:2]
        src_resized = cv2.resize(raw, (int(W * SCALE), int(H * SCALE)),
                                 interpolation=cv2.INTER_AREA)
        ws.append_data(cv2.cvtColor(src_resized, cv2.COLOR_BGR2RGB))
        side = np.concatenate([raw, cmap], axis=1)
        wm.append_data(cv2.cvtColor(side, cv2.COLOR_BGR2RGB))
    ws.close()
    wm.close()

    # Run perception pipeline with CLASS MAP as wire mask
    os.environ["WIRE_SEG_DIR"] = str(seg_dir)
    import wire_model.perception.seg_masks as _seg
    _seg.MASK_DIR = Path(str(seg_dir))
    _seg._video_path.clear()
    _seg._small_stack.clear()
    _seg._result_cache.clear()

    from wire_model.perception.rectify import find_board
    from wire_model.perception.harness import track_harness
    from wire_model.perception.tracking import HarnessTracker

    cap = cv2.VideoCapture(str(src_vid))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("  Finding board...", flush=True)
    tf = find_board(read_cap(cap, 0))

    print("  Tracking modules (class map)...", flush=True)
    tracks, hub_i, anchor = track_harness(cap, tf, T, read_cap, clip=clip)
    if tracks is None:
        cap.release()
        return {"clip": clip, "status": "failed", "reason": "no modules found"}

    order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
    tracks = tracks[:, order, :]

    print("  Tracing cables (class map, temporal prior)...", flush=True)
    tracker = HarnessTracker(tf, clip=clip, solo_seed=False, jump_mm=40.0)
    tracked = tracker.run(cap, read_cap, tracks, range(T))
    ok_frames = sorted(f for f in range(T) if f in tracked)
    if not ok_frames:
        cap.release()
        return {"clip": clip, "status": "failed", "reason": "no frames tracked"}

    print(f"  Tracked {len(ok_frames)}/{T} frames", flush=True)

    # Post-process: detect and fix cable swaps at crossings.
    print("  Fixing crossing swaps...", flush=True)
    swap_count = 0
    for fi in range(1, len(ok_frames)):
        f, fp = ok_frames[fi], ok_frames[fi - 1]
        for a in range(4):
            for b in range(a + 1, 4):
                pa = tracked[fp][a]["px"]
                pb = tracked[fp][b]["px"]
                ca = tracked[f][a]["px"]
                cb = tracked[f][b]["px"]
                if any(x is None for x in [pa, pb, ca, cb]):
                    continue
                mid_pa = pa[len(pa) // 2]
                mid_pb = pb[len(pb) // 2]
                mid_ca = ca[len(ca) // 2]
                mid_cb = cb[len(cb) // 2]
                d_aa = np.linalg.norm(mid_ca - mid_pa)
                d_ab = np.linalg.norm(mid_ca - mid_pb)
                d_ba = np.linalg.norm(mid_cb - mid_pa)
                d_bb = np.linalg.norm(mid_cb - mid_pb)
                if d_ab < d_aa and d_ba < d_bb:
                    tracked[f][a], tracked[f][b] = tracked[f][b], tracked[f][a]
                    swap_count += 1
    if swap_count:
        print(f"    Fixed {swap_count} cable swaps", flush=True)

    # Save NPZ
    shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in ok_frames]) for k in range(4)}
    np.savez(out_dir / f"{clip}.npz",
             t0=ok_frames[0], t1=ok_frames[-1], fps=fps,
             source_frames=np.array([run[f] for f in ok_frames]),
             **{f"c{k}": shp[k] for k in range(4)})

    # Generate combined video
    print("  Generating combined video...", flush=True)
    out_frames = []
    for i, f in enumerate(ok_frames):
        img = read_cap(cap, f).copy()
        src_h, src_w = img.shape[:2]

        # Get pipeline cable polylines in pixels
        cable_px_list = []
        for k in range(4):
            px = tf.to_px(shp[k][i] * 1000.0)
            cable_px_list.append(px)

        # Load skeleton mask for this frame
        skel_mask = np.zeros((src_h, src_w), bool)
        if f < len(frames):
            _, _, _, _, skel_path, _ = frames[f]
            skel_mask = load_skeleton_mask(skel_path, (src_h, src_w))

        # Draw refined cables
        draw_cables(img, cable_px_list, skel_mask, tf, shp)

        src_num = run[f] if f < len(run) else f
        cv2.putText(img, f"{clip}  frame {src_num}", (15, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, f"{clip}  frame {src_num}", (15, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)

        out_frames.append(cv2.resize(img, VID_SIZE, interpolation=cv2.INTER_AREA))
    cap.release()

    # Write video
    vid_path = out_dir / f"combined_{clip}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))
    for fr in out_frames:
        writer.write(fr)
    writer.release()

    mid = len(out_frames) // 2
    cv2.imwrite(str(out_dir / f"combined_{clip}_frame{mid:03d}.png"), out_frames[mid])

    # Clean up intermediate files
    for p in [src_vid, seg_vid]:
        try:
            os.remove(str(p))
        except OSError:
            pass

    lengths = {}
    for k in range(4):
        L = np.linalg.norm(np.diff(shp[k] * 1000, axis=1), axis=2).sum(axis=1)
        lengths[f"c{k}_mean_mm"] = round(float(L.mean()), 1)
        lengths[f"c{k}_std_mm"] = round(float(L.std()), 1)

    print(f"  -> {vid_path.name} ({len(out_frames)} frames, "
          f"{os.path.getsize(vid_path)/1e6:.1f} MB)")

    return {"clip": clip, "status": "ok", "frames": T, "tracked": len(ok_frames),
            **lengths}


def main():
    ap = argparse.ArgumentParser(description="Combined skeleton + perception pipeline")
    ap.add_argument("--root", required=True, help="X-planner skeletonization folder")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--clip", default=None, help="Process only this clip")
    ap.add_argument("--fps", type=float, default=4.0)
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.clip:
        clips = [root / a.clip]
    else:
        clips = sorted([d for d in root.iterdir() if d.is_dir()
                        and (d / "visualization" / "panels").exists()])

    print(f"Combined pipeline: {len(clips)} clip(s)")
    print(f"  Cable assignment: perception pipeline (class map)")
    print(f"  Wire shapes: skeleton-refined smooth cables")
    print(f"  Output: {out}\n")

    for i, cd in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {cd.name}")
        t0 = time.time()
        try:
            r = process_clip(cd, out, a.fps)
        except Exception as e:
            traceback.print_exc()
            r = {"clip": cd.name, "status": "error", "reason": str(e)[:200]}
        elapsed = time.time() - t0
        status = r.get("status", "?")
        if status == "ok":
            print(f"  ({elapsed:.0f}s)\n")
        else:
            print(f"  -> {status}: {r.get('reason', '?')} ({elapsed:.0f}s)\n")

    print("Done!")


if __name__ == "__main__":
    main()
