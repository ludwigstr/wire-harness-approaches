"""PhysTwin (Jiang, Hsu, Zhang, Yu, Wang, Li; arXiv 2503.17973, 2025) for the
harness cables: a spring-mass digital twin per cable whose topology and
physical parameters are fitted to the tracked video in two stages (zero-order
on the homogeneous parameters, first-order on the dense per-spring
stiffness). Rendering (Gaussian splats) is not reproduced; we score node
positions. See sim.py and fit.py."""
from .sim import SpringMassTwin
