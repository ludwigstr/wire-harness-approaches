"""Which DLO-Lab checkout the rod scripts import.

Defaults to the PATCHED copy vendored in this project, which adds per-element
stiffness scales to the rod solver (edges_stretching_scale,
hinges_bending_scale, hinges_twisting_scale -- see rod_solver.init_edge_fields).
Set it up with ./setup_dlolab.sh (clones UMass-Embodied-AGI/DLO-Lab at c5026a9 into
extern/DLO-Lab-perjoint and applies patches/dlo-lab-perjoint.patch); WIRE_DLOLAB overrides.

The patch is behaviour-neutral by construction: every scale starts at 1.0, so
an unmodified run reproduces upstream numbers exactly (verified, see
data/genesis_rod_perjoint_parity.json).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHED = ROOT / "extern/DLO-Lab-perjoint"

DLO_LAB = Path(os.environ.get("WIRE_DLOLAB", PATCHED))
