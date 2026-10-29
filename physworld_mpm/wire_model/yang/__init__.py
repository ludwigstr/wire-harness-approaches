"""Yang, Stork, Stoyanov, "Learning differentiable dynamics models for shape
control of deformable linear objects", RAS 158 (2022) 104258, for the harness
cables: (1) global simulator parameters identified by differential evolution
on the one-step error, Eq. (1) -- MuJoCo's elastic-rod cable in place of AGX
Dynamics (sim.py); (2) synthetic data from the identified simulator; (3) an
interaction network with a bi-directional LSTM along the chain (model.py)."""
