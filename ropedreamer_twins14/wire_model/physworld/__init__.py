"""PhysWorld (Yang et al., arXiv 2510.21447, 2025) for the harness cables.
Paper: (1) physics-consistent digital twin in an MPM simulator fitted to the
real video, (2) synthesis of many demonstrations from it with varied motion
patterns (Bezier end trajectories, three-phase velocity) and part-aware
perturbations of the per-particle physical properties, (3) a GNN world model
X_{t+1} = f(X_{t-h:t}, a_t, Phi) with the properties Phi as vertex features,
(4) fine-tuning of Phi on the real frames with the network frozen.
Here the digital twin is the PhysTwin spring-mass twin fitted per clip
(wire_model/phystwin) instead of MPM -- a cable is a 1-D object and the
spring-mass twin is what the project fits from the tracks; everything after
the twin follows the paper. See synth.py and scripts/physworld_run.py."""
