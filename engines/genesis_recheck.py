"""Fast (no-video) re-validation of Genesis parameter choices after fixing
the get_links_pos() off-by-one bug in genesis_full_scene.py's error metric
(see genesis-metric-offbyone-bug memory) -- every number produced by this
project's Genesis work before that fix is suspect, this recomputes the key
comparisons cleanly via eval_clip (fast, no video) across all 16 clips.

Usage: python scripts/genesis_recheck.py --tag baseline --stiffness 0.02 --damping 0.02
       python scripts/genesis_recheck.py --tag tuned --stiffness 0.1 --damping 0.5
       python scripts/genesis_recheck.py --tag pinned --pin-cable-ends
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genesis_full_scene import eval_clip

ROOT = Path(__file__).resolve().parents[1]

from clip_set import CLIP_TRIMS  # the 14 evaluation clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stiffness", type=float, default=0.1)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--pin-cable-ends", action="store_true")
    ap.add_argument("--out-dir", default="data/genesis_recheck")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"

    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"[{args.tag}] Fortgesetzt: {len(results)} Clips schon vorhanden", flush=True)

    clips = list(CLIP_TRIMS.keys())
    t_start = time.time()
    done0 = len(results)
    for i, clip in enumerate(clips):
        if clip in results:
            continue
        t0 = time.time()
        try:
            errs = eval_clip(clip, t1_sec=CLIP_TRIMS[clip], stiffness=args.stiffness,
                             damping=args.damping, pin_cable_ends=args.pin_cable_ends)
        except Exception:
            print(f"[{args.tag}][{i+1}/{len(clips)}] {clip}: FEHLER\n{traceback.format_exc()}", flush=True)
            continue
        results[clip] = {"start": errs[0], "ende": errs[-1], "mittel": float(np.mean(errs)),
                         "n_frames": len(errs)}
        out_path.write_text(json.dumps(results, indent=1))
        n_done = len(results) - done0
        elapsed = time.time() - t_start
        eta = elapsed / max(1, n_done) * (len(clips) - len(results))
        print(f"[{args.tag}][{len(results)}/{len(clips)}] {clip}: Mittel {results[clip]['mittel']:.1f} mm/Knoten "
              f"({time.time()-t0:.0f}s, ETA {eta/60:.0f} min)", flush=True)

    mittel = np.array([v["mittel"] for v in results.values()])
    print(f"\n[{args.tag}] {len(mittel)} Clips, Median {np.median(mittel):.1f}  Mittel {mittel.mean():.1f} mm/Knoten")
    print(f"GENESIS_RECHECK_{args.tag.upper()}_FERTIG")


if __name__ == "__main__":
    main()
