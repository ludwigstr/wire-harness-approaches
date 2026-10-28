"""Overlay video: Caporali per 1 s step (stride 4, data/bench_film_real.pt) and per video frame (stride 1,
data/bench_film_real_step1.pt) on the same real footage (user, 2026-10-28).

Both rollouts exactly as table row 5 / scripts/run_caporali_step1.sh: cached data/tracks14 tracks, online
estimation from 0.04/1.0/20/0.15, rest = frame-0 turning angles capped at 0.6, open loop, ends set to the
measured ends. Drawn on every video frame: measured shape green (solid with markers), per-frame prediction
blue (dashed), per-second prediction red (dashed) -- the per-second one only exists every 4th frame and is held
in between (its label says so).

    python scripts/caporali_overlay_video.py --clips 20260602_102713 20260602_134613 \
        --out data/figures/caporali_overlay_2026-10-28
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
os.environ.setdefault("WIRE_NOMINAL", "0.04,1.0,20,0.15")
os.environ["WIRE_REST_MODE"] = "frame0"; os.environ["WIRE_REST_CAP"] = "0.6"

import cv2  # noqa: E402
import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fit_clip import read  # noqa: E402
from wire_model.estimation import load_predictor  # noqa: E402
from wire_model.perception.rectify import find_board  # noqa: E402
from render_neural_video import DIR, draw_poly, roll_online_traced  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GREEN, BLUE, RED = (0, 170, 0), (230, 120, 0), (0, 0, 220)        # BGR


def model_of(ckpt):
    m, _ = load_predictor(ROOT / ckpt, DEV)
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def errs(trace, cables):
    return {fi: float(np.mean([np.linalg.norm(V[k] - cables[k][fi], axis=1).mean() for k in range(4)]) * 1000)
            for fi, V in trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--ckpt-sec", default="data/bench_film_real.pt")
    ap.add_argument("--ckpt-frame", default="data/bench_film_real_step1.pt")
    ap.add_argument("--tracks", default="data/tracks14")
    ap.add_argument("--fps-out", type=float, default=4.0)
    ap.add_argument("--frame-only", action="store_true", help="draw only measured + per-frame prediction")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists -- never overwrite rendered videos, pick a new --out")
    out.mkdir(parents=True)
    m_sec, m_frame = model_of(a.ckpt_sec), model_of(a.ckpt_frame)
    for clip in a.clips:
        d = np.load(ROOT / a.tracks / f"{clip}.npz")
        cables = [d[f"c{k}"].astype(np.float32) for k in range(4)]
        t0, n = int(d["t0"]), len(cables[0])
        tr_sec = roll_online_traced(m_sec, cables, cables[0].shape[1], 4, 0.05)
        tr_frame = roll_online_traced(m_frame, cables, cables[0].shape[1], 1, 0.05)
        e_sec, e_frame = errs(tr_sec, cables), errs(tr_frame, cables)
        print(f"{clip}: per second {np.mean(list(e_sec.values())):.1f}  per frame {np.mean(list(e_frame.values())):.1f} "
              f"mm/node (mean over each one's own frames)", flush=True)
        sec = dict(tr_sec); frame = dict(tr_frame)
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        tf = find_board(read(cap, 0))
        pxc = tf.to_px(np.array(((70, 70), (1130, 70), (1130, 1680), (70, 1680)), dtype=float))
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
        x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))
        px = lambda s: tf.to_px(s * 1000.0) - np.array([x0, y0])
        imgs, last = [], 0
        for fi in range(n):
            if fi in sec:
                last = fi
            img = read(cap, t0 + fi)[y0:y1, x0:x1].copy()
            for k in range(4):
                draw_poly(img, px(cables[k][fi]), GREEN, thick=3, marker=3)
                if not a.frame_only:
                    draw_poly(img, px(sec[last][k]), RED, thick=2, marker=0, dashed=True)
                draw_poly(img, px(frame[fi][k]), BLUE, thick=2, marker=0, dashed=True)
            held = "" if last == fi else f" (held from t={(t0 + last) / fps:.2f}s)"
            lines = [(f"{clip}  t={(t0 + fi) / fps:.2f}s  frame {fi}", (20, 20, 20)),
                     ("measured", GREEN),
                     (f"per frame (stride 1): {e_frame[fi]:.0f} mm/node", BLUE),
                     (f"per second (stride 4): {e_sec[last]:.0f} mm/node{held}", RED)][:3 if a.frame_only else 4]
            for i, (txt, col) in enumerate(lines):
                cv2.putText(img, txt, (16, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.putText(img, txt, (16, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()
        h = min(im.shape[0] for im in imgs); w = min(im.shape[1] for im in imgs)
        h, w = h - h % 2, w - w % 2
        p = out / f"overlay_{clip}.mp4"
        iio.mimsave(p, [im[:h, :w] for im in imgs], fps=a.fps_out)
        iio.imwrite(out / f"overlay_{clip}_last.png", imgs[-1][:h, :w])
        print(f"-> {p}", flush=True)


if __name__ == "__main__":
    main()
