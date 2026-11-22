# Rebalancing control: results

Same model, hyperparameters, images and splits as the main study. Only the training
loss changes: class-weighted cross-entropy (w_c = n / (K n_c), Q capped) or
oversampling with WeightedRandomSampler. Mean ± SD over seeds (B: 5 seeds; A: 3 seeds).
Run with rebalance_control.py; summarised with summarise_rebalance.py.

| Protocol | Training | Seeds | Accuracy % | Macro-F1 (NSVF) | Se N | Se S | Se V | Se F | PPV S | PPV V | PPV F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A beat-level | none (original) | 5 | 97.18 ± 0.28 | 0.806 ± 0.029 | 0.996 | 0.563 | 0.878 | 0.483 | 0.907 | 0.950 | 0.942 |
| A beat-level | class weighting | 3 | 92.86 ± 0.10 | 0.726 ± 0.005 | 0.932 | 0.829 | 0.944 | 0.823 | 0.387 | 0.761 | 0.442 |
| A beat-level | oversampling | 3 | 91.92 ± 1.61 | 0.700 ± 0.025 | 0.920 | 0.859 | 0.943 | 0.873 | 0.369 | 0.762 | 0.340 |
| B inter-patient | none (original) | 5 | 91.26 ± 0.54 | 0.422 ± 0.008 | 0.971 | 0.011 | 0.734 | 0.003 | 0.104 | 0.696 | 0.004 |
| B inter-patient | class weighting | 5 | 59.67 ± 6.95 | 0.342 ± 0.029 | 0.598 | 0.359 | 0.780 | 0.056 | 0.070 | 0.381 | 0.004 |
| B inter-patient | oversampling | 5 | 59.18 ± 5.30 | 0.354 ± 0.022 | 0.588 | 0.488 | 0.775 | 0.045 | 0.087 | 0.409 | 0.003 |

Full per-seed outputs, including confusion matrices: ctrl_<protocol>_<balance>_<seed>.json.
Sanity check: protocol B seed 0 was re-run from a fresh dataset build (acc 0.9089,
macro-F1 0.414) against the stored result (0.9087, 0.414).
