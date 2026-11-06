"""Build project-page/static/js/param_grids.js: fitted per-joint / per-spring values for the approach pages.

Sources (all in the repository):
  newton       newton_fit_t36/data/newton_fit_everything_pinned.npz  'best' (231 values, mass pinned)
               layout: ke56 kd56 kt56 | eps tau | K | mu4 | rest56, each 56 = 4 cables x 14 joints
  mujoco       ropedreamer_twins14/data/mujoco_general_fit3.npz      k, rest_deg, damp (4 x 14)
  ropedreamer  the same MuJoCo fit: the twins the training data is randomised around
  genesis      data/genesis_perjoint_sysid.json                      stiffness, damping (15 joints, shared by the cables)
  phystwin     checkpoints/phystwin_twins/<clip>.pt                  spring stiffness k (4 x 16 x 16); neighbours (i, i+1)
                                                                    and next-but-one (i, i+2), geometric mean over the clips

Run from the repository root:  python project-page/tools/build_param_grids.py .
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def r(a, n=4):
    return [[float(f"{v:.{n}g}") for v in row] for row in np.asarray(a, float)]


out = {}

# ---------------------------------------------------------------- Newton
b = np.load(root / "newton_fit_t36/data/newton_fit_everything_pinned.npz")["best"]
assert b.shape == (231,), b.shape
ke, kd, kt = b[0:56].reshape(4, 14), b[56:112].reshape(4, 14), b[112:168].reshape(4, 14)
rest = np.degrees(b[175:231]).reshape(4, 14)
out["newton"] = {
    "source": "newton_fit_t36/data/newton_fit_everything_pinned.npz",
    "note": "The finished 14-clip optimum (mass pinned at 0.044 kg/m). It is the warm start of the running 36-clip fit.",
    "axis": "joint", "q": [
        {"key": "ke", "label": "Bend stiffness", "unit": "N·m/rad", "kind": "log", "values": r(ke),
         "what": "How strongly each joint resists bending (ke). Joint 1 is next to the hub."},
        {"key": "kt", "label": "Twist stiffness", "unit": "N·m/rad", "kind": "log", "values": r(kt),
         "what": "How strongly each joint resists twisting (kt), fitted separately from bending."},
        {"key": "kd", "label": "Bend damping", "unit": "", "kind": "log", "values": r(kd),
         "what": "How quickly a bending motion dies out at each joint (kd)."},
        {"key": "rest", "label": "Rest angle", "unit": "°", "kind": "signed", "values": r(rest, 3),
         "what": "The bend each joint keeps when nothing pushes it: the cable’s plastic set."},
    ]}

# ---------------------------------------------------------------- MuJoCo (and the RopeDreamer twins)
z = np.load(root / "ropedreamer_twins14/data/mujoco_general_fit3.npz")
mj = [
    {"key": "k", "label": "Bend stiffness", "unit": "N·m/rad", "kind": "log", "values": r(z["k"]),
     "what": "Stiffness of the ball-joint spring at each joint: how strongly it resists bending. Joint 1 is next to the hub."},
    {"key": "rest", "label": "Rest angle", "unit": "°", "kind": "signed", "values": r(z["rest_deg"], 3),
     "what": "The angle each joint’s spring pulls back to, about the board normal: the cable’s plastic set."},
    {"key": "damp", "label": "Damping", "unit": "", "kind": "log", "values": r(z["damp"]),
     "what": "How quickly a motion at each joint dies out."},
]
out["mujoco"] = {"source": "ropedreamer_twins14/data/mujoco_general_fit3.npz",
                 "note": "The 14-clip general fit (181 values). The 36-clip fit is still running and its values are not in the repository.",
                 "axis": "joint", "q": mj}
out["ropedreamer"] = {"source": "ropedreamer_twins14/data/mujoco_general_fit3.npz",
                      "note": "RopeDreamer itself has no physical parameters. Shown is the MuJoCo fit its twins are built from; every training "
                              "rollout scales these values at random (stiffness ×0.5–2, rest angles ×0.8–1.2).",
                      "axis": "joint", "q": mj[:2]}

# ---------------------------------------------------------------- Genesis
g = json.loads((root / "data/genesis_perjoint_sysid.json").read_text(encoding="utf-8"))
out["genesis"] = {"source": "data/genesis_perjoint_sysid.json",
                  "note": "The 14-clip per-joint row: 15 stiffnesses and 15 dampings shared by the four cables. "
                          "The 36-clip general fit (185 values) is not in the repository.",
                  "axis": "joint", "rows": ["All 4 cables"], "q": [
                      {"key": "stiff", "label": "Joint stiffness", "unit": "", "kind": "log", "values": r([g["stiffness"]]),
                       "what": "How strongly each hinge joint resists bending. Joint 1 is next to the hub."},
                      {"key": "damp", "label": "Joint damping", "unit": "", "kind": "log", "values": r([g["damping"]]),
                       "what": "How quickly a motion at each joint dies out."},
                  ]}

# ---------------------------------------------------------------- PhysTwin
try:
    import torch
    ks1, ks2, n = [], [], 0
    for f in sorted(glob.glob(str(root / "checkpoints/phystwin_twins/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        K = d["k"].numpy()
        ks1.append([[K[c, i, i + 1] for i in range(15)] for c in range(4)])
        ks2.append([[K[c, i, i + 2] for i in range(14)] for c in range(4)])
        n += 1
    gm = lambda a: np.exp(np.log(np.clip(np.asarray(a, float), 1e-9, None)).mean(0))
    out["phystwin"] = {"source": "checkpoints/phystwin_twins/<clip>.pt",
                       "note": "Geometric mean over the %d twins of the 14-clip run; the 36-clip twins are not in the repository. "
                               "Each value is bounded to 1–1500 N/m." % n,
                       "axis": "spring", "q": [
                           {"key": "s1", "label": "Stretch springs", "unit": "N/m", "kind": "log", "values": r(gm(ks1)),
                            "what": "Springs between neighbouring nodes: they keep the cable’s length. Spring 1 starts at the hub."},
                           {"key": "s2", "label": "Bend springs", "unit": "N/m", "kind": "log", "values": r(gm(ks2)),
                            "what": "Springs between nodes two apart: a spring-mass model has no bending angle, so these springs resist bending."},
                       ]}
except Exception as e:  # torch missing: skip PhysTwin, keep the rest
    print("PhysTwin skipped:", e)

dst = root / "project-page/static/js/param_grids.js"
dst.write_text("// Generated by tools/build_param_grids.py. Do not edit by hand.\n"
               "window.PARAM_GRIDS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("wrote", dst, sorted(out))
