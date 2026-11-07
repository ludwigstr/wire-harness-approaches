"""Prediction overlay videos of the Newton fit (run tinygpu_2080ti) on the 36 clips.

  rollout: one fresh process per clip, same evaluation as newton_validate_tracks.py, keeping the predicted nodes.
  render:  dashed prediction drawn on clips36_final_v1/check_videos (raw camera images are not on this machine).

The check videos have the measured skeleton burned in, so measured = their own lines. The board-to-pixel map is fitted
per clip against those burned-in lines (similarity + y stretch from the 1064 -> 1072 px resize) and logged.
"""
import argparse, hashlib, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("WIRE_MASS_PIN", "0.044")
NT = Path("/home/hpc/ws32/ws32101h/Python/Cable_model/wire-harness-approaches/newton_fit_t36")
sys.path.insert(0, str(NT)); sys.path.insert(0, str(NT / "scripts"))
PARAMS = NT / "data/newton_fit_everything_pinned_t36.npz"
VALID = NT / "data/newton_fit_everything_pinned_t36_validate.json"
COL = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (200, 0, 200)]     # BGR, as in track_replay_folder.py
_dev = {}


def _init(q):
    _dev["device"] = f"cuda:{q.get()}"


def one(job):
    c, out = job
    t = time.time()
    import newton_fit_everything_cmaes as E
    import newton_full_scene as N
    kw = E.unpack(np.load(PARAMS, allow_pickle=True)["best"])
    z = np.load(NT / "data/tracks36" / f"{c}.npz")
    shp, fps = {k: np.asarray(z[f"c{k}"], float) for k in range(4)}, float(z["fps"])
    nodes = []
    errs = N.eval_clip(c, shp_fps=(shp, fps), solver_iterations=10, device=_dev.get("device"),
                       nodes_xyz_out=nodes, **kw)
    sub = int(kw.get("subdiv", 1))
    pred = np.array([[fr[k][::sub, :2] for k in range(4)] for fr in nodes])     # (T, 4, n, 2)
    n = pred.shape[2]
    e2 = np.mean([np.linalg.norm(pred[:, k] - shp[k][:, :n], axis=-1).mean(1) for k in range(4)], 0) * 1000
    np.savez(Path(out) / f"{c}.npz", pred=pred, errs=np.asarray(errs), errs_recomputed=e2, seconds=time.time() - t)
    return c, float(np.mean(errs)), float(np.abs(e2 - np.asarray(errs)).max()), len(errs), n, time.time() - t


def rollout(a):
    out = Path(a.out) / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(q.stem for q in (NT / "data/tracks36").glob("*.npz"))
    assert len(clips) == 36
    todo = [c for c in clips if not (out / f"{c}.npz").exists()]
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    for i in range(a.workers):
        q.put(gpus[i % len(gpus)])
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init, initargs=(q,)) as pool:
        for i, (c, e, de, T, n, dt) in enumerate(pool.imap_unordered(one, [(c, str(out)) for c in todo]), 1):
            print(f"[{i}/{len(todo)}] {c}: {e:.3f} mm/node, {T} frames, {n} nodes, recompute diff {de:.2e}, {dt:.0f}s "
                  f"(total {time.time() - t0:.0f}s)", flush=True)


def proj(p, xy):
    s, th, tx, ty, sy = p
    c, si = np.cos(th), np.sin(th)
    x = s * (c * xy[..., 0] - si * xy[..., 1]) + tx
    y = -s * (si * xy[..., 0] + c * xy[..., 1]) + ty
    return np.stack([x, y * sy], -1)


def fit_map(frames, z, ids):
    import cv2
    from scipy.optimize import minimize
    dts = []
    for f in ids:
        g = frames[f].astype(int)
        pack = []
        for k in range(4):
            m = (np.abs(g - np.array(COL[k])).sum(2) < 110).astype(np.uint8)
            m[:150] = 0; m[1030:] = 0
            pack.append(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5))
        dts.append(pack)

    def cost(q):
        e = []
        for i, f in enumerate(ids):
            for k in range(4):
                P = proj(q, z[f"c{k}"][f]).round().astype(int)
                P[:, 0] = P[:, 0].clip(0, 799); P[:, 1] = P[:, 1].clip(0, 1071)
                e.append(dts[i][k][P[:, 1], P[:, 0]])
        return float(np.mean(np.concatenate(e)))
    best = None
    for ty in (1050, 1061, 1072):
        for tx in (18, 24, 30):
            r = minimize(cost, [603.0, -0.015, tx, ty, 1.012], method="Powell",
                         options=dict(xtol=1e-4, ftol=1e-4, maxiter=40))
            if best is None or r.fun < best.fun:
                best = r
    return best.x, float(best.fun)


def dashed(img, pts, col):
    import cv2
    p = np.round(pts).astype(int)
    for i in range(0, len(p) - 1, 2):      # every other segment, as draw_poly(dashed=True) in render_neural_video.py
        pa, pb = (int(p[i][0]), int(p[i][1])), (int(p[i + 1][0]), int(p[i + 1][1]))
        cv2.line(img, pa, pb, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.line(img, pa, pb, col, 2, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, (int(q[0]), int(q[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(q[0]), int(q[1])), 3, col, -1, cv2.LINE_AA)


def band(img, l1, l2, l3):
    import cv2
    h = 136
    img[:h] = (img[:h] * 0.12 + 255 * 0.88).astype(np.uint8)
    for y, t, sc, th in ((32, l1, .62, 2), (72, l2, .72, 2), (112, l3, .62, 2)):
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (20, 20, 20), th, cv2.LINE_AA)


def render_clip(job):
    import cv2
    cv2.setNumThreads(1)
    c, a_check, roll, out, raw, title, fake = job
    z = np.load(NT / "data/tracks36" / f"{c}.npz")
    fps = float(z["fps"])
    cap = cv2.VideoCapture(f"{a_check}/skeleton_check_{c}.mp4")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    T = len(z["c0"])
    assert len(frames) == T, (c, len(frames), T)
    assert frames[0].shape == (1072, 800, 3), frames[0].shape
    if fake:
        pred = np.stack([z[f"c{k}"][:, :15] + [0.03, -0.02] for k in range(4)], 1); errs = np.full(T, 36.0)
    else:
        r = np.load(f"{roll}/{c}.npz"); pred, errs = r["pred"], r["errs"]
    assert len(pred) == T and len(errs) == T
    ids = [int(T * x) for x in (0.0, .2, .4, .6, .8, .98)]
    p, cost = fit_map(frames, z, ids)
    mean = float(np.mean(errs))
    worst, mid = int(np.argmax(errs)), T // 2
    Path(raw).mkdir(exist_ok=True)
    enc = cv2.VideoWriter(f"{raw}/newton_{c}.avi", cv2.VideoWriter_fourcc(*"FFV1"), fps, (800, 1072))
    assert enc.isOpened(), "FFV1 writer unavailable"
    for f in range(T):
        img = frames[f].copy()
        for k in range(4):
            dashed(img, proj(p, pred[f, k]), COL[k])
        band(img, title, f"{c}  t={f / fps:.1f}s  error {errs[f]:.1f} mm/node",
             f"clip mean {mean:.2f} mm/node; solid = measured, dashed = Newton")
        enc.write(img)
        if f == mid:
            cv2.imwrite(f"{out}/newton_{c}_frame{f:03d}_middle.png", img)
        if f == worst:
            cv2.imwrite(f"{out}/newton_{c}_frame{f:03d}_worst.png", img)
    enc.release()
    return c, mean, T, float(cost), [float(x) for x in p]


def render(a):
    out = Path(a.out) / a.folder
    assert a.test or not out.exists(), f"{out} exists -- never overwrite"
    out.mkdir(parents=True, exist_ok=True)
    title = "Newton per-cable fit, 36 clips (30 generations, mass pinned)"
    clips = a.test or sorted(q.stem for q in (NT / "data/tracks36").glob("*.npz"))
    jobs = [(c, a.check, str(Path(a.out) / "rollouts"), str(out), str(Path(a.out) / "raw_avi"), title, bool(a.test)) for c in clips]
    if a.test:
        print(render_clip(jobs[0])); return
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = {}
        for i, (c, m, T, cost, p) in enumerate(pool.imap_unordered(render_clip, jobs), 1):
            res[c] = dict(reproduced=m, n_frames=T, map_cost_px=cost, map=p)
            print(f"render [{i}/{len(jobs)}] {c}: mean {m:.3f}, map cost {cost:.2f} px", flush=True)
    v = json.load(open(VALID))
    rep = {c: x["mittel"] for c, x in v["clips"].items()}
    for c in res:
        res[c]["reported"] = rep[c]; res[c]["diff"] = res[c]["reproduced"] - rep[c]
    tol = 0.5
    over = {c: res[c]["diff"] for c in res if abs(res[c]["diff"]) > tol}
    vals = np.array([res[c]["reproduced"] for c in sorted(res)])
    doc = dict(approach="Newton per-cable fit, 36 clips, mass pinned (run tinygpu_2080ti, 30 generations)",
               params="newton_fit_t36/results/tinygpu_2080ti/newton_fit_everything_pinned_t36.npz",
               params_sha256=hashlib.sha256(PARAMS.read_bytes()).hexdigest(),
               device="4 x RTX 2080 Ti job, one worker process per clip",
               reported_source="newton_fit_t36/results/tinygpu_2080ti/newton_fit_everything_pinned_t36_validate.json",
               base_images="clips36_final_v1/check_videos (measured skeleton burned in, raw images not available)",
               metric=v["protocol"], n_clips=len(res), mean_reproduced=float(vals.mean()),
               median_reproduced=float(np.median(vals)), mean_reported=v["mean"], median_reported=v["median"],
               max_abs_clip_diff=max(abs(res[c]["diff"]) for c in res), tolerance_mm=tol,
               clips_over_tolerance=over, ok=not over, clips=dict(sorted(res.items())))
    (out / "results.json").write_text(json.dumps(doc, indent=1))
    print({k: doc[k] for k in ("mean_reproduced", "median_reproduced", "mean_reported", "max_abs_clip_diff",
                               "clips_over_tolerance", "ok")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rollout", "render"])
    ap.add_argument("--out", default="/home/woody/ws32/ws32101h/newton_t36_videos/run1")
    ap.add_argument("--folder", default="newton_7.6")
    ap.add_argument("--check", default="/home/woody/ws32/ws32101h/newton_t36_videos/check")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--test", nargs="*")
    a = ap.parse_args()
    rollout(a) if a.cmd == "rollout" else render(a)
