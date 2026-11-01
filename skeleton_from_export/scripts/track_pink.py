"""Re-trace cable 3 (pink, the thin wire bundle) of a tracked clip with wire_model.perception.thin_wire
(user, 2026-11-01: "das pinke Kabel ... MUSS dein Fokus sein"). Cables 0-2, hub and movers are taken from
--base (npz of any tracker); only c3 is replaced. Writes <out>/<clip>.npz with c0..c3 and `pink_len_mm`.

    python scripts/track_pink.py --clip 20261021_173628 --base data/videos/skeletons_2026-10-30 --out <dir>
"""
import argparse, glob, re, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.perception.rectify import find_board  # noqa: E402
from wire_model.perception import thin_wire as TW  # noqa: E402

SCALE = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    if (out / f"{a.clip}.npz").exists():
        raise SystemExit(f"{out / a.clip}.npz exists -- pick a new --out")
    z = np.load(ROOT / a.base / f"{a.clip}.npz"); sf = [int(s) for s in z["source_frames"]]
    P = np.stack([z[f"c{k}"] for k in range(4)])
    pan = {}
    for d in glob.glob(str(ROOT / "data/videos" / a.clip / "visualization/panels/*_frame_*")):
        pan.setdefault(int(re.search(r"frame_(\d+)$", d).group(1)), d)
    tf = None; prev = None; Q = P.copy(); L = np.zeros(len(sf))
    for i, n in enumerate(sf):
        raw = cv2.imread(f"{pan[n]}/01_raw.png")
        raw = cv2.resize(raw, (int(raw.shape[1] * SCALE), int(raw.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
        if tf is None:
            tf = find_board(raw)
            bp = tf.to_px(np.array(((20, 20), (1180, 20), (1180, 1660), (20, 1660)), float))
        R = [tf.to_px(P[k, i] * 1000.0) for k in range(4)]
        hub = R[0][0]; mov = [R[k][-1] for k in range(4)]
        if prev is not None:   # move the previous route with its ends before using it as the pull
            prev = prev + (hub - prev[0]) * np.linspace(1, 0, len(prev))[:, None] + (mov[3] - prev[-1]) * np.linspace(0, 1, len(prev))[:, None]
        cost, _ = TW.cost_image(raw, hub, mov, R[:3], prev, bp)
        q = TW.smooth_route(TW.trace(cost, hub, mov[3]))
        Q[3, i] = tf.to_mm(q) / 1000.0; prev = q
        L[i] = np.linalg.norm(np.diff(Q[3, i], axis=0), axis=1).sum() * 1000
    np.savez(out / f"{a.clip}.npz", t0=z["t0"], t1=z["t1"], fps=z["fps"], source_frames=np.array(sf),
             pink_len_mm=L, **{k: z[k] for k in ("reset", "carried") if k in z.files}, **{f"c{k}": Q[k] for k in range(4)})
    print(f"{a.clip}: pink length median {np.median(L):.0f} mm, range {L.min():.0f}-{L.max():.0f}, "
          f">8% off median in {100 * (abs(L - np.median(L)) > 0.08 * np.median(L)).mean():.1f}% of frames", flush=True)


if __name__ == "__main__":
    main()
