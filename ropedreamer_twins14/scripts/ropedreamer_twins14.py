"""Sim twins of the 14 evaluation clips (user, 2026-10-28: "Nimm die 14 Videos und
fuehre sie in MuJoCo zurueck. Dann variiere rund um diese 14 Simulationen.").

Each clip (data/tracks14/<clip>.npz, already trimmed per clip_set.CLIP_TRIMS) is
replayed in the MuJoCo general-fit env (data/mujoco_general_fit3.npz, EXACT
parameters, no randomization) with the real timing and real mover paths, twice:
  A  the fit's own evaluation: mujoco_general_fit_cmaes._eval_one's
     rollout_errs(make_model(shp, unpack(x)), shp, fps, per_cable=True)
  B  the dataset generator's replay: yang_make_dataset._mujoco_speed_run on the
     real-timing pose sequence (rollout 0 of --motion speed), the run the
     training data comes from.
Both are scored with the table metric (16 nodes, 4 cables, every frame incl.
frame 0 after the 300-step settle); A must equal the general-fit row
(data/mujoco_general_fit3_validate.json) and B must equal A, otherwise the twin
is not the fit's simulation. B's node positions are written in tracks format
(c0..c3, fps) to --out-tracks (for ropedreamer_eval.py), and every twin is
rendered with MuJoCo with the real tracked nodes drawn as green spheres
(contact sheet of 8 frames + mp4) into --render.

    MUJOCO_GL=egl python3 scripts/ropedreamer_twins14.py --out-tracks data/wm/ropedreamer_twins14_twin_tracks \
        --render data/figures/ropedreamer_twins14_2026-10-28 --out data/wm/ropedreamer_twins14_twins.json
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from clip_set import CLIP_TRIMS  # noqa: E402

TRACKS = ROOT / "data/tracks14"
FIT = ROOT / "data/mujoco_general_fit3.npz"


def _one(job):
    clip, out_tracks, render_dir = job
    import mujoco
    import mujoco_general_fit_cmaes as G
    import yang_make_dataset as Y
    from mujoco_full_loco import rollout_errs
    from wire_model.yang import speed_warp as SW
    t0 = time.time()
    z = np.load(TRACKS / f"{clip}.npz")
    F = np.stack([z[f"c{k}"] for k in range(4)], 1).astype(float)
    shp = {k: z[f"c{k}"] for k in range(4)}
    fps, T = float(z["fps"]), len(F)
    x = np.asarray(np.load(FIT)["best_x"], float)
    # A: the fit's own evaluation (verbatim _eval_one)
    m = G.make_model(shp, G.unpack(x))
    eA = np.asarray(rollout_errs(m, shp, fps, per_cable=True))            # (T, 4) mm
    # B: the generator's replay of rollout 0
    m = G.make_model({k: F[:, k] for k in range(4)}, G.unpack(x))
    tabs = Y.mujoco_contact_tables(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    cav = [(bid(f"hub_cav{k}"), bid(f"plug{k}_cav0")) for k in range(4)]
    E = SW.end_frames(F, np.repeat(np.arange(T, dtype=float)[:, None], 5, 1))
    seq = Y.Drive.pose_seq([[E[t][k] for k in range(4)] for t in range(T)])
    Vs, gaps, cs = Y._mujoco_speed_run(m, seq, sub, tabs, cav)
    V = np.stack(Vs)                                                       # (T, 4, 16, 2)
    eB = 1000 * np.linalg.norm(V - F[:len(V)], axis=-1).mean(-1)          # (T, 4)
    Path(out_tracks).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_tracks) / f"{clip}.npz", fps=fps, t0=z["t0"], t1=z["t1"],
             **{f"c{k}": V[:, k].astype(np.float32) for k in range(4)})
    res = {"fit_eval": float(eA.mean()), "generator_replay": float(eB.mean()), "n_frames": len(V), "T": T,
           "max_frame_diff_mm": float(np.abs(eA[:len(eB)] - eB).max()),
           "contacts_real_timing": int(cs["pairs"].sum())}
    if render_dir:
        res["render"] = _render(m, seq, sub, F, V, clip, Path(render_dir))
    res["sec"] = time.time() - t0
    return clip, res


def _render(m, seq, sub, F, Vref, clip, out):
    """re-run the twin with MuJoCo's renderer; real nodes as green spheres"""
    import imageio.v2 as imageio
    import mujoco
    import yang_make_dataset as Y
    out.mkdir(parents=True, exist_ok=True)
    d = mujoco.MjData(m)
    drv = Y.Drive(m)
    r = mujoco.Renderer(m, 720, 960)
    cam = mujoco.MjvCamera()
    P = F.reshape(-1, 2)
    c = 0.5 * (P.min(0) + P.max(0)); ext = float((P.max(0) - P.min(0)).max())
    cam.lookat[:] = [c[0], c[1], 0.0]; cam.distance = 0.9 + 1.3 * ext; cam.elevation = -80; cam.azimuth = 90
    drv.apply(d, seq[0])
    for _ in range(300):
        mujoco.mj_step(m, d)
    frames, dev = [], 0.0
    for t in range(len(seq)):
        if t:
            drv.step_frame(d, seq[t - 1], seq[t], sub)
        V = Y.vertices(m, d)[:, :, :2]
        if t < len(Vref):
            dev = max(dev, float(1000 * np.abs(V - Vref[t]).max()))
        r.update_scene(d, camera=cam)
        sc = r.scene
        for k in range(4):
            for p in F[t, k]:
                if sc.ngeom >= sc.maxgeom:
                    break
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]),
                                    np.array([p[0], p[1], 0.006]), np.eye(3).ravel(), np.array([0.1, 0.8, 0.1, 1.0], np.float32))
                sc.ngeom += 1
        frames.append(r.render())
    imageio.mimwrite(out / f"{clip}_twin.mp4", frames, fps=8, macro_block_size=1)
    pick = np.linspace(0, len(frames) - 1, 8).astype(int)
    sheet = np.concatenate([np.concatenate([frames[i][::2, ::2] for i in pick[:4]], 1),
                            np.concatenate([frames[i][::2, ::2] for i in pick[4:]], 1)], 0)
    imageio.imwrite(out / f"{clip}_twin_sheet.png", sheet)
    r.close()
    return {"frames": len(frames), "sheet_frames": pick.tolist(), "max_dev_from_twin_mm": dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tracks", required=True)
    ap.add_argument("--render", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f"{out} exists -- never overwrite"
    if a.render:
        assert not (ROOT / a.render).exists(), f"{a.render} exists -- renders always go to a NEW folder"
    clips = sorted(CLIP_TRIMS)
    assert len(clips) == 14
    ref = json.loads((ROOT / "data/mujoco_general_fit3_validate.json").read_text())
    ref = ref.get("clips", ref) if isinstance(ref, dict) else ref
    jobs = [(c, str(ROOT / a.out_tracks), str(ROOT / a.render) if a.render else "") for c in clips]
    res = {}
    with mp.get_context("fork").Pool(min(a.workers, 14)) as pool:
        for clip, r in pool.imap_unordered(_one, jobs):
            r["row"] = ref.get(clip)
            res[clip] = r
            print(f"{clip}: fit eval {r['fit_eval']:.3f}  generator replay {r['generator_replay']:.3f}  "
                  f"(validate.json {r['row']:.3f}; max per-frame diff {r['max_frame_diff_mm']:.3f} mm; "
                  f"{r['n_frames']}/{r['T']} frames; render dev {r.get('render', {}).get('max_dev_from_twin_mm', float('nan')):.3f} mm; "
                  f"{r['sec']:.0f}s)", flush=True)
    res = dict(sorted(res.items()))
    A = np.array([r["fit_eval"] for r in res.values()]); B = np.array([r["generator_replay"] for r in res.values()])
    R = np.array([r["row"] for r in res.values()])
    summ = {"fit_eval_mean": float(A.mean()), "fit_eval_median": float(np.median(A)),
            "generator_mean": float(B.mean()), "generator_median": float(np.median(B)),
            "validate_json_mean": float(R.mean()), "validate_json_median": float(np.median(R)),
            "max_abs_diff_generator_vs_fit": float(np.abs(A - B).max())}
    out.write_text(json.dumps({"clips": res, "summary": summ, "fit": str(FIT.relative_to(ROOT))}, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
