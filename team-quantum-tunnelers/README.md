# Team Quantum Tunnelers

This folder contains our submission for the Alice & Bob x YQuantum 2026 challenge. We focused on online optimization for cat-qubit stabilization with CMA-ES, using a `dynamiqs` simulation backend and JAX-based batched evaluations.

## Contents

- `notebook.ipynb`
  Final Jupyter notebook for the cat-qubit optimization study.
- `code/build_cat_qubit_optimizer_notebook.py`
  Script that generates the notebook.
- `code/run_cat_qubit_loss_experiments.py`
  Script used to run the final benchmark sweep and export artifacts.
- `results/metrics.json`
  Machine-readable summary of the final run.
- `results/metrics.csv`
  Flat metrics table for quick inspection.
- `results/no_drift_reward_bias_comparison.png`
  Reward and bias convergence for the no-drift run, with standard deviation shading.
- `results/no_drift_lifetime_comparison.png`
  `T_Z` and `T_X` trajectories for the no-drift run, with standard deviation shading.
- `results/drift_tracking_n_200.png`
  Drift benchmark plot showing reward and real-valued compensation tracks for `Re(g_2)` and `Re(eps_d)`.
- `results/cma_best_tz_linear_fit.png`
  Early-time linear lifetime surrogate fit for the best no-drift `T_Z` candidate.
- `results/cma_best_tx_linear_fit.png`
  Early-time linear lifetime surrogate fit for the best no-drift `T_X` candidate.
- `results/cma_epoch_wigner.gif`
  Animated Wigner-function visualization across optimizer epochs for the no-drift CMA-ES run.

## Method Summary

- Optimizer: CMA-ES (`SepCMA`)
- Tuned controls: `Re(g_2)`, `Im(g_2)`, `Re(eps_d)`, `Im(eps_d)`
- Objective:
  `(T_Z / T_X - eta_target)^2 - (T_Z + eta_target * T_X)` with `eta_target = 200`
- Lifetime estimation:
  Early-time linear surrogate fits for `T_Z` and `T_X`
- Final benchmark protocol:
  `30` epochs without drift and `30` epochs with drift
- Drift stress test:
  A single additive hardware jump applied at epoch `20`

## Final Benchmark Highlights

### No Drift

- Best reward: `143.90`
- Best epoch: `12`
- Best `T_Z`: `71.95 us`
- Best `T_X`: `0.36045 us`
- Best bias `T_Z / T_X`: `199.62`
- Closest bias to target: `200.022`

### Drift (Single Jump at Epoch 20)

- Best reward: `143.34`
- Best epoch: `17`
- Best `T_Z`: `71.78 us`
- Best `T_X`: `0.35832 us`
- Best bias `T_Z / T_X`: `200.33`
- Final reward after the post-jump adaptation window: `-962.86`

The drift run is intentionally a hard step-change stress test. It shows that the optimizer performs well before the jump and then partially adapts after the hardware shift, while the no-drift run demonstrates stable convergence near the target bias.

## Reproducibility

The final exported artifacts in `results/` were generated from `code/run_cat_qubit_loss_experiments.py`, which uses the notebook-generated functions and writes the plots, metrics, and Wigner GIF for the final run.
