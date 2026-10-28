"""Coordinate-descent fit of a per-cable, per-joint weld stiffness (see the
`weld_solref` docstring in mujoco_scene.build_xml): 4 cables x 14 consecutive-
segment joints = 56 independent MuJoCo solref timeconst values, each pulling
its joint back toward the LOCAL BEND ANGLE the real cable had at frame_idx.

A quick sanity sweep (single clip, one uniform value for all 56 joints) found
a real optimum around solref=0.007 (14.8 -> 11.1 mm/node on that clip) but
also that the 4 cables disagree sharply: M0/M1 want a stiff weld, M2/M3 get
WORSE as the weld stiffens. That is exactly the per-position, per-cable
variation this script searches for instead of assuming.

Training uses a small, FAST subset of clips (short ones from
clip-hand-interventions' trim list) so each of the ~224 candidate evaluations
(56 joints x 4 candidate levels) stays cheap; each coordinate is scored using
ONLY that coordinate's own cable's error (the other 3 cables' errors from the
same rollout are ignored for that decision, since the weld at cable c's joint
j has no direct physical effect on a different cable). The final fitted 4x14
array is then validated on the full "honest" 16-clip set (all 19 minus the
excluded-clips three) for a trustworthy number, independent of the training
subset.

Progress is checkpointed after every coordinate to `--out` (numpy .npz: the
running best array plus a log of per-coordinate scores) so the run can be
inspected or resumed without losing hours of work if interrupted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, build_xml, DIR as CLIP_DIR
from mujoco_replay import replay

ROOT = Path(__file__).resolve().parents[1]
N_JOINTS = 14

# clip -> trim cutoff in seconds (None = full clip), from clip-hand-interventions
# and excluded-clips memory. The 3 excluded clips are simply absent here.
# The canonical 14 clips, imported instead of hardcoded. This file used to
# carry its own list of 16, which included 110528 and 172926 -- the two
# clips whose tracking is broken and which clip_set excludes. That second
# list is what made the table's 13.6 look like it contradicted a 14.41.
from clip_set import CLIP_TRIMS  # noqa: E402
TRAIN_CLIPS_FAST = ["20260602_170108", "20260602_121006", "20260602_162650",
                    "20260602_193953", "20260602_165113"]
# Ball-joint spring stiffness, not weld solref. The uniform sweep over all 14
# clips found 20.26 -> 14.83 mm between 0 and 0.1 and was still falling, so the
# candidate set brackets that optimum generously in both directions.
# Widened 2026-10-20: the first run put 25 of 50 joints on the TOP candidate
# with the score still falling there, which is a rim optimum, not a result.
CANDIDATES = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]


def clip_frame_t1(clip, cutoff_sec):
    if cutoff_sec is None:
        return None
    import cv2
    cap = cv2.VideoCapture(f"{CLIP_DIR}/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    cap.release()
    return int(round(cutoff_sec * fps))


def load_clips(clip_names, log):
    cache = {}
    for c in clip_names:
        t1 = clip_frame_t1(c, CLIP_TRIMS[c])
        t_start = time.time()
        shp, t0, t1_, fps = track_clip(c, t0=8, t1=t1)
        cache[c] = (shp, fps)
        log(f"  getrackt: {c} ({len(shp[0])} Frames, {time.time()-t_start:.0f}s)")
    return cache


def eval_clips(cache, clip_names, weld_solref):
    """Returns dict clip -> (4,) per-cable end error in mm."""
    out = {}
    for c in clip_names:
        shp, fps = cache[c]
        n = len(shp[0])
        xml = build_xml(shp, weld_solref=weld_solref)
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
        traj = replay(m, d, shp, n, sub)
        # MEAN OVER ALL FRAMES, not the last one. The fit used to score only
        # traj[-1] against shp[k][-1] -- one frame out of ~45 -- while every number
        # this project reports is the mean over the whole rollout. Optimising one
        # objective and reporting another is the same mistake as fitting at 10
        # solver iterations and validating at 20. `replay` already returns every
        # frame, so this costs nothing.
        err = np.array([1000.0 * float(np.mean([
            np.linalg.norm(traj[f][k][:, :2] - shp[k][f], axis=1).mean()
            for f in range(len(traj))])) for k in range(4)])
        out[c] = err
    return out


_CACHE = {}          # clip -> (shp, fps); filled before the pool forks


def _eval_one(job):
    """One (candidate index, clip) simulation. Reads the tracked clip from the
    inherited module global instead of pickling it per job -- tracking costs
    45s per clip and the arrays are large."""
    cand_i, clip, stiff = job
    shp, fps = _CACHE[clip]
    import mujoco
    from mujoco_scene import set_per_joint_stiffness
    from mujoco_full_loco import rollout_errs
    m = mujoco.MjModel.from_xml_string(build_xml(shp))
    if stiff is not None:
        set_per_joint_stiffness(m, stiff)
    # the SAME rollout and the SAME error the table reports, per cable
    return cand_i, clip, rollout_errs(m, shp, fps, per_cable=True).mean(0)


def eval_clips_par(pool, clip_names, stiff):
    """One stiffness array over all clips, spread across the pool. `stiff=None`
    is the no-spring baseline."""
    jobs = [(0, c, stiff) for c in clip_names]
    return {c: e for _, c, e in pool.map(_eval_one, jobs)}


def eval_candidates_par(pool, clip_names, trials):
    """All candidates x all clips in ONE map. Per coordinate that is
    4 x 14 = 56 independent simulations instead of 4 sequential sweeps over
    the clip list -- the whole reason this variant exists."""
    jobs = [(i, c, t) for i, t in enumerate(trials) for c in clip_names]
    out = [{} for _ in trials]
    for i, c, e in pool.map(_eval_one, jobs):
        out[i][c] = e
    return out


def mean_cable_err(results, cable):
    return float(np.mean([e[cable] for e in results.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jnt_stiffness_fit.npz")
    ap.add_argument("--log", default="data/jnt_stiffness_fit_log.jsonl")
    ap.add_argument("--train-all", action="store_true",
                    help="Auf allen 14 Clips aus clip_set trainieren statt "
                         "nur den 5 kurzen (viel langsamer, aber dieselbe "
                         "Menge, ueber die berichtet wird)")
    ap.add_argument("--clips", default=None,
                    help="Komma-getrennte Clip-Liste als Trainingsmenge "
                         "(ueberschreibt --train-all/die 5 schnellen Clips) "
                         "-- fuer gezieltes Nachfitten auf den aktuell "
                         "schlechtesten Clips")
    ap.add_argument("--init", default=None,
                    help="Bestehende .npz, deren 'best'-Matrix als "
                         "Startpunkt dient statt bei 'aus' (5.0) fuer alle "
                         "56 Koordinaten zu beginnen -- fuer eine gezielte "
                         "zweite Runde auf Basis eines bereits gefitteten "
                         "Ergebnisses")
    ap.add_argument("--init-value", type=float, default=0.1,
                    help="uniform starting stiffness for all 56 joints")
    ap.add_argument("--candidates", default=None,
                    help="Comma-separated candidate set, overriding CANDIDATES. "
                         "Round 3 (2026-10-21): the widened run still put 19 of "
                         "56 joints on its top value 1.6, so the set is extended "
                         "upward and the fit restarted from that result.")
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel simulations; 4*len(train_clips) is the "
                         "useful ceiling per coordinate")
    ap.add_argument("--resume", action="store_true",
                    help="Bei vorhandenem --out Checkpoint dort weitermachen "
                         "statt neu zu starten (die Hintergrundlaeufe hier "
                         "werden unvorhersehbar vom Environment beendet, "
                         "typischerweise nach ein paar Stunden -- ohne "
                         "Resume wuerde jeder Abbruch den Fortschritt bis "
                         "dahin wegwerfen)")
    args = ap.parse_args()
    global CANDIDATES
    if args.candidates:
        CANDIDATES = [float(x) for x in args.candidates.split(",")]

    all_clip_names = list(CLIP_TRIMS.keys())
    if args.clips is not None:
        train_clips = args.clips.split(",")
    else:
        train_clips = all_clip_names if args.train_all else TRAIN_CLIPS_FAST

    log_path = ROOT / args.log
    log_f = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    out_path = ROOT / args.out
    # start at the uniform optimum, so coordinate 1 can only improve on the
    # best single-value result rather than rediscovering it 56 times
    best = np.full((4, N_JOINTS), float(args.init_value))
    history = []
    base_avg = None
    base_per_clip = None
    if args.resume and out_path.exists():
        ckpt = np.load(out_path, allow_pickle=True)
        best = ckpt["best"]
        history = json.loads(str(ckpt["history"]))
        base_avg = float(ckpt["base_avg"])
        if "base_per_clip" in ckpt:
            base_per_clip = json.loads(str(ckpt["base_per_clip"]))
        log(f"=== Gelenkfeder-Fit FORTGESETZT ({len(history)}/{4*N_JOINTS} "
            f"Koordinaten schon fertig) ===")
    else:
        log("=== Gelenkfeder-Fit gestartet ===")
        if args.init is not None:
            init_ckpt = np.load(ROOT / args.init, allow_pickle=True)
            best = init_ckpt["best"]
            log(f"Startpunkt aus {args.init} geladen (history NICHT "
                f"uebernommen -- alle 56 Koordinaten werden mit diesem "
                f"Startwert neu bewertet, nicht uebersprungen)")
    log(f"Trainings-Clips ({len(train_clips)}): {train_clips}")
    log(f"Kandidaten pro Koordinate: {CANDIDATES}")
    log("Tracke alle Clips einmal (Cache fuer den ganzen Lauf)...")
    cache = load_clips(all_clip_names, log)
    global _CACHE
    _CACHE = cache
    import multiprocessing as mp
    # fork, so the tracked clips are inherited copy-on-write instead of
    # pickled into every job. MuJoCo here is CPU-only, nothing CUDA-bound
    # that forking would break.
    pool = mp.get_context("fork").Pool(args.workers)
    log(f"Pool: {args.workers} parallele Simulationen "
        f"({4 * len(train_clips)} je Koordinate)")

    if base_avg is None or base_per_clip is None:
        # baseline (no welds at all) on the full honest set, for reference.
        # Also hit when resuming from an OLDER checkpoint that has base_avg
        # but predates base_per_clip -- recompute both together so they can
        # never end up inconsistent/missing again.
        log("Baseline (keine Gelenkfeder) auf allen Clips...")
        base = eval_clips_par(pool, all_clip_names, None)
        base_avg = float(np.mean([e.mean() for e in base.values()]))
        base_per_clip = {c: float(e.mean()) for c, e in base.items()}
        log(f"Baseline Mittel (Endfehler): {base_avg:.2f} mm/Knoten")
    else:
        log(f"Baseline aus Checkpoint: {base_avg:.2f} mm/Knoten")

    done_coords = {(h["cable"], h["joint"]) for h in history}
    n_done_before = len(history)
    n_done_session = 0
    t_run_start = time.time()
    total_coords = 4 * N_JOINTS
    for c in range(4):
        for j in range(N_JOINTS):
            if (c, j) in done_coords:
                continue
            t_c = time.time()
            trials = []
            for cand in CANDIDATES:
                trial = best.copy()
                trial[c, j] = cand
                trials.append(trial)
            results = eval_candidates_par(pool, train_clips, trials)
            scores = [mean_cable_err(r, c) for r in results]
            best_idx = int(np.argmin(scores))
            best[c, j] = CANDIDATES[best_idx]
            history.append({"cable": c, "joint": j, "scores": scores,
                            "chosen": CANDIDATES[best_idx]})
            np.savez(ROOT / args.out, best=best, history=json.dumps(history),
                    base_avg=base_avg, base_per_clip=json.dumps(base_per_clip))
            n_done_session += 1
            coord_i = n_done_before + n_done_session
            elapsed = time.time() - t_run_start
            eta = elapsed / n_done_session * (total_coords - coord_i)
            log(f"[{coord_i}/{total_coords}] M{c} Gelenk{j}: "
                f"scores={[round(s,1) for s in scores]} -> {CANDIDATES[best_idx]:g} "
                f"({time.time()-t_c:.0f}s, ETA {eta/60:.0f} min)")

    if args.resume and out_path.exists() and "final_avg" in np.load(out_path, allow_pickle=True):
        log("=== Endergebnis schon im Checkpoint vorhanden, ueberspringe Validierung ===")
        ckpt = np.load(out_path, allow_pickle=True)
        final_avg = float(ckpt["final_avg"])
        per_clip = json.loads(str(ckpt["per_clip"]))
    else:
        log("=== Pass 1 fertig, Endvalidierung ===")
        final = eval_clips_par(pool, all_clip_names, best)
        final_avg = float(np.mean([e.mean() for e in final.values()]))
        per_clip = {c: float(e.mean()) for c, e in final.items()}
        # Save BEFORE printing/logging the breakdown: a crash mid-print (this
        # exact spot killed 3 runs in a row before base_per_clip existed) must
        # not cost the already-computed final result.
        np.savez(ROOT / args.out, best=best, history=json.dumps(history),
                base_avg=base_avg, base_per_clip=json.dumps(base_per_clip),
                final_avg=final_avg, per_clip=json.dumps(per_clip))
    log(f"Baseline (kein Weld):     {base_avg:.2f} mm/Knoten")
    log(f"Gefittete Gelenkfeder:   {final_avg:.2f} mm/Knoten")
    for c, v in sorted(per_clip.items(), key=lambda kv: -kv[1]):
        log(f"  {c:<20} {v:6.1f}  (Baseline {base_per_clip[c]:6.1f})")
    log(f"-> {ROOT / args.out}")
    pool.close()
    pool.join()
    log("=== FERTIG ===")


if __name__ == "__main__":
    main()
