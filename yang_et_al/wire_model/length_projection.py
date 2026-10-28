"""Enforce inextensibility on a predicted shape (PBD distance constraint).

Gu et al. (Neurocomputing 696, 2026), Table 4, contrasts mass-spring-damper
models with position-based dynamics: MSD encodes inextensibility as a stiff
spring, PBD as a hard constraint solved by projection. The surrogate here is
trained on MSD data, so nothing forces its OUTPUT to preserve arc length -- and
measured on the 84 situations, cables whose projected arc length changes by
more than 10% are predicted at 45 mm/node against 14 mm/node for those that
hold their length (r = +0.53, the strongest correlation found in this pipeline).

This applies the PBD half of that contrast as a post-step: project the
predicted polyline onto the nearest one with the observed segment lengths,
holding both plug positions fixed because they are measured, not predicted.

Gauss-Seidel over the distance constraints, as in Muller et al. -- each sweep
moves the two nodes of a segment symmetrically along their connecting line, and
repeated sweeps converge because the constraints are locally decoupled. Fixed
nodes take no share of the correction, so their neighbours absorb it.
"""

from __future__ import annotations

import numpy as np


def project_lengths(shape, target_lengths, iters=20, fix_ends=True):
    """Nearest polyline with the given segment lengths. (N,2) -> (N,2).

    shape           predicted polyline
    target_lengths  (N-1,) desired segment lengths, e.g. measured at t0
    fix_ends        hold nodes 0 and N-1 (the plug centres, which are tracked
                    rather than predicted, so they carry no model error)
    """
    p = np.asarray(shape, float).copy()
    n = len(p)
    if n < 3:
        return p
    w = np.ones(n)
    if fix_ends:
        w[0] = w[-1] = 0.0

    for _ in range(iters):
        for i in range(n - 1):
            d = p[i + 1] - p[i]
            L = float(np.linalg.norm(d))
            if L < 1e-12:
                continue
            corr = (L - target_lengths[i]) * d / L
            wsum = w[i] + w[i + 1]
            if wsum < 1e-12:
                continue
            p[i] += (w[i] / wsum) * corr
            p[i + 1] -= (w[i + 1] / wsum) * corr
    return p


def segment_lengths(shape):
    return np.linalg.norm(np.diff(np.asarray(shape, float), axis=0), axis=1)
