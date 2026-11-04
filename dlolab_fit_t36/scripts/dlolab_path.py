"""Which DLO-Lab checkout the rod scripts import.

Defaults to the PATCHED copy vendored in this project, which adds per-element
stiffness scales to the rod solver (edges_stretching_scale,
hinges_bending_scale, hinges_twisting_scale -- see rod_solver.init_edge_fields).
The upstream checkout under Wire_parameter/extern stays untouched because
another project uses it; set WIRE_DLOLAB to point back at it for a comparison.

The patch is behaviour-neutral by construction: every scale starts at 1.0, so
an unmodified run reproduces upstream numbers exactly (verified, see
data/genesis_rod_perjoint_parity.json).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("extern/DLO-Lab")
PATCHED = ROOT / "extern/DLO-Lab-perjoint"

DLO_LAB = Path(os.environ.get("WIRE_DLOLAB", PATCHED))
