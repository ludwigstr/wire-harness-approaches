"""Calibrate wire_model/ropedreamer/noise.py on the 14 clips: grid over the noise
parameters, augmented twins vs real tracks, smallest relative error over the
statistics in noise.stats. Writes the chosen parameters to --out.
    python3 scripts/ropedreamer_noise_calibrate.py --out data/wm/ropedreamer_twins14_noise_params.json"""
import argparse, glob, itertools, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wire_model.ropedreamer.noise import track_noise, stats  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
R, S = [], []
for f in sorted(glob.glob(str(ROOT / "data/tracks14/*.npz"))):
    c = Path(f).stem
    zr, zs = np.load(f), np.load(ROOT / f"data/wm/ropedreamer_twins14_twin_tracks/{c}.npz")
    for k in range(4):
        R.append(zr[f"c{k}"].astype(float)); S.append(zs[f"c{k}"].astype(float))
mean_stats = lambda L: {k: float(np.mean([stats(P)[k] for P in L])) for k in stats(L[0])}
real, twin = mean_stats(R), mean_stats(S)
# turn_mid is NOT a target: the real cables are smoother than their twins (11.7 vs 13.1 deg), a shape
# difference noise can only widen, so matching it would only push the noise towards zero
keys = ["turn_dt_mid_deg", "seg_dt_mid_mm", "seg_end_mm", "turn_end_deg", "ddp_mm"]
best = None
grid = itertools.product([0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0012], [0.0, 0.0002], [0.003, 0.0035],
                         [0.5, 0.7], [0.0, 0.0005])
for sp, stn, ee, st, se in grid:
    p = {"sigma_perp": sp, "sigma_tan": stn, "end_ext": ee, "end_ext_sd": 0.001, "straighten": st, "sigma_end": se}
    rng = np.random.default_rng(0)
    aug = mean_stats([track_noise(P, rng, p) for P in S])
    err = float(np.mean([abs(aug[k] - real[k]) / abs(real[k]) for k in keys]))
    if best is None or err < best[0]:
        best = (err, p, aug)
err, p, aug = best
print(f"{'statistic':18s} {'real':>8s} {'twin':>8s} {'twin+noise':>10s}")
for k in real:
    print(f"{k:18s} {real[k]:8.3f} {twin[k]:8.3f} {aug[k]:10.3f}")
print("params", p, "mean rel. error", round(err, 3))
(ROOT / a.out).write_text(json.dumps({"params": p, "real": real, "twin": twin, "twin_noise": aug, "rel_error": err}, indent=1))
