"""Tracking-noise augmentation for RopeDreamer's simulated training data
(user, 2026-10-28: "Das mit dem Zittern gerne mit rein").

The real tracks of the 14 clips differ from their exact MuJoCo twins
(data/wm/ropedreamer_twins14_twin_tracks) in ways the tracker causes and the
simulation does not have; measured per cable, mean over clips:
    frame-to-frame change of the interior turning angles  2.61 deg  (twin 0.64)
    frame-to-frame change of the interior link lengths    0.55 mm   (twin 0.02)
    first / last link length                               51.0 mm   (twin 47.9, interior 48.1 both)
    first / last turning angle                             3.8 / 4.1 deg (twin 10.7 / 5.5)
    frame-to-frame change of the end-node motion (|d dp|)  6.5 / 5.0 mm (twin 5.8 / 4.5)
The model below perturbs a clean (T, 16, 2) node sequence the same way:
  1. along-cable resampling jitter: every interior node slides along the
     polyline by an independent N(0, sigma_shift) per frame (the tracker's
     resampling does not keep the node spacing);
  2. per-node position noise, temporally independent, interior nodes, split
     into the cable's local NORMAL N(0, sigma_perp) and TANGENT N(0, sigma_tan)
     directions (an isotropic sigma_node is kept for completeness): the real
     tracks jitter in angle (4x the twin) far more than in link length, which
     isotropic noise cannot reproduce -- calibrated, isotropic noise gave the
     right length jitter at a third too little angle jitter;
  3. end straightening + extension: the tracker's end links are longer and
     nearly straight -- node 0 / 15 are pushed outward along the chord
     node 2 -> node 0 (node 13 -> node 15) by `end_ext` (constant per
     trajectory, N(end_ext, end_ext_sd)), and node 1 / 14 are pulled towards
     that chord by the fraction `straighten`;
  4. end-node jitter N(0, sigma_end) per frame on node 0 / 15 (enters the
     end actions dp the model is driven with).
  5. SHAPE gap ("model error"): a smooth displacement normal to the cable,
     d(s, t) = sum_m a_m(t) sin(m pi s/L), m = 1..3 (weights 1, 0.6, 0.4),
     a_m(t) = static part N(0, sigma_shape_static) per trajectory + an AR(1)
     part with stationary std sigma_shape_dyn and per-frame correlation
     rho_shape; plus a static offset of node 0 / node 15, N(0, end_off0 /
     end_off15) per trajectory. Reason (diagnostic 2026-10-28): the real
     tracks differ from their exact twins by ~8 mm per node (the physics
     model's error, half static half slowly varying, mostly normal to the
     cable, largest mid-cable, 7.8 / 4.2 mm at node 0 / 15), and THAT -- not
     the frame-to-frame jitter (items 1-4) -- is what the sim-trained model
     cannot take: the old checkpoint scores 93 mm on the twins, 90 mm on
     twins + jitter, 136 mm on twins with the real interior shape, 162 mm on
     the real tracks.
  6. SPATIAL SMOOTHING: the real tracks are smoother along the cable than the
     simulated chain (mean |turning angle| 11.7 vs 13.1 deg). One Laplacian
     smoothing pass of the interior nodes, Q_i += w (0.5 (Q_{i-1} + Q_{i+1}) -
     Q_i), with w ~ U(0, smooth_max) per trajectory, applied first. Diagnostic:
     the old checkpoint goes from 93 mm (twins) to 125 mm (twins smoothed once,
     w 0.3, turning angle 10.6 deg) and 155 mm (w 0.5, twice) -- the real
     tracks' 162 mm is reproduced by smoothing, not by jitter or a random shape
     gap of the right size (90 / 95 mm).
Parameters are calibrated with scripts/ropedreamer_noise_calibrate.py so the
augmented twins reproduce the real statistics (see its output).
"""
import numpy as np

DEFAULT = {"sigma_shift": 0.0, "sigma_node": 0.0, "sigma_perp": 0.0, "sigma_tan": 0.0, "end_ext": 0.0, "end_ext_sd": 0.0,
           "straighten": 0.0, "sigma_end": 0.0, "sigma_shape_static": 0.0, "sigma_shape_dyn": 0.0, "rho_shape": 0.9,
           "end_off0": 0.0, "end_off15": 0.0, "smooth_max": 0.0}


def _resample(P, s_new):
    """points at arc positions s_new (L,) along the polyline P (L, 2)"""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    x = np.interp(s_new, d, P[:, 0]); y = np.interp(s_new, d, P[:, 1])
    return np.stack([x, y], 1)


def track_noise(P, rng, p):
    """P (T, L, 2) clean nodes -> noisy copy (the observation a tracker would give)"""
    P = np.asarray(P, float)
    T, L, _ = P.shape
    if p.get("smooth_max", 0) > 0:
        w = rng.uniform(0, p["smooth_max"])
        P = P.copy()
        P[:, 1:-1] = P[:, 1:-1] + w * (0.5 * (P[:, :-2] + P[:, 2:]) - P[:, 1:-1])
    Q = P.copy()
    if p.get("sigma_shift", 0) > 0:
        for t in range(T):
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[t], axis=0), axis=1))]
            s = d.copy()
            s[1:-1] += rng.normal(0, p["sigma_shift"], L - 2)
            s = np.clip(s, 0, d[-1])
            s[1:-1] = np.sort(s[1:-1])
            Q[t] = _resample(P[t], s)
    if p.get("end_ext", 0) > 0 or p.get("straighten", 0) > 0:
        ext = rng.normal(p.get("end_ext", 0), p.get("end_ext_sd", 0), 2)
        for (e, n1, n2), x in zip(((0, 1, 2), (L - 1, L - 2, L - 3)), ext):
            u = Q[:, e] - Q[:, n2]
            u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
            Q[:, e] = Q[:, e] + x * u
            # pull node 1 towards the straight chord node 2 -> node 0
            a = Q[:, n2] + ((Q[:, n1] - Q[:, n2]) * u).sum(-1, keepdims=True) * u
            Q[:, n1] = Q[:, n1] + p.get("straighten", 0) * (a - Q[:, n1])
    if p.get("sigma_shape_static", 0) > 0 or p.get("sigma_shape_dyn", 0) > 0:
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[0], axis=0), axis=1))]
        u = d / max(d[-1], 1e-9)                                       # arc fraction of every node
        w = np.array([1.0, 0.6, 0.4])
        B = np.stack([np.sin((m + 1) * np.pi * u) for m in range(3)], 1) * w     # (L, 3)
        a_st = rng.normal(0, p.get("sigma_shape_static", 0), 3)
        rho = p.get("rho_shape", 0.9); sd = p.get("sigma_shape_dyn", 0)
        a_dy = np.zeros((T, 3)); a_dy[0] = rng.normal(0, sd, 3)
        for t in range(1, T):
            a_dy[t] = rho * a_dy[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sd, 3)
        amp = (a_st[None] + a_dy) @ B.T                                # (T, L)
        tng = np.gradient(P, axis=1)
        tng = tng / np.maximum(np.linalg.norm(tng, axis=-1, keepdims=True), 1e-9)
        nrm = np.stack([-tng[..., 1], tng[..., 0]], -1)
        Q = Q + amp[..., None] * nrm
    if p.get("end_off0", 0) > 0:
        Q[:, 0] += rng.normal(0, p["end_off0"], 2)
    if p.get("end_off15", 0) > 0:
        Q[:, L - 1] += rng.normal(0, p["end_off15"], 2)
    if p.get("sigma_node", 0) > 0:
        Q[:, 1:-1] += rng.normal(0, p["sigma_node"], (T, L - 2, 2))
    if p.get("sigma_perp", 0) > 0 or p.get("sigma_tan", 0) > 0:
        tng = Q[:, 2:] - Q[:, :-2]                                     # central-difference tangent
        tng = tng / np.maximum(np.linalg.norm(tng, axis=-1, keepdims=True), 1e-9)
        nrm = np.stack([-tng[..., 1], tng[..., 0]], -1)
        Q[:, 1:-1] += (rng.normal(0, p.get("sigma_perp", 0), (T, L - 2, 1)) * nrm
                       + rng.normal(0, p.get("sigma_tan", 0), (T, L - 2, 1)) * tng)
    if p.get("sigma_end", 0) > 0:
        Q[:, [0, L - 1]] += rng.normal(0, p["sigma_end"], (T, 2, 2))
    return Q


def stats(P):
    """the statistics the calibration matches (per cable trajectory)"""
    d = np.diff(P, axis=1); seg = np.linalg.norm(d, axis=-1)
    ang = np.arctan2(d[..., 1], d[..., 0]); turn = np.angle(np.exp(1j * np.diff(ang, axis=1)))
    dp = np.diff(P[:, [0, -1]], axis=0); ddp = np.diff(dp, axis=0)
    return {"turn_dt_mid_deg": np.degrees(np.abs(np.angle(np.exp(1j * np.diff(turn[:, 1:-1], axis=0))))).mean(),
            "seg_dt_mid_mm": 1000 * np.abs(np.diff(seg[:, 1:-1], axis=0)).mean(),
            "seg_end_mm": 1000 * seg[:, [0, -1]].mean(), "seg_mid_mm": 1000 * seg[:, 1:-1].mean(),
            "turn_end_deg": np.degrees(np.abs(turn[:, [0, -1]])).mean(), "turn_mid_deg": np.degrees(np.abs(turn[:, 1:-1])).mean(),
            "ddp_mm": 1000 * np.linalg.norm(ddp, axis=-1).mean()}
