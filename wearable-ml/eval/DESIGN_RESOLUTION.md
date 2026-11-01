# Design: model x HR resolution (Stage 2D, Part 1)

Code only so far. No model has been run on R2. Draft protocol: `DRAFT_AMENDMENT_5.md` (not applied).

| file | role |
|---|---|
| `resolution.py` | rebuild R1 (project code, per-beat slope, parity only), R1s (same beats, slope per second) and R2 (3.0 s bin mean of the same beats, slope per second) from raw WESAD, one pickle in memory at a time; parity check against `results/features.csv`; slope comparison R1s vs R1; writes `out/resolution/windows_R1s.csv`, `windows_R2.csv`, `parity.json` |
| `cross.py` | Amendment 4's nested LOSO, with the fitted model scoring several row-aligned test tables (cross-resolution for free) |
| `interaction.py` | `plan`, `parity`, `run --go`; interaction / resolution / transfer contrasts, rankings, heatmap, CSVs, EXPERIMENTS.md and manifest |
| `tests/test_resolution.py` | resampling, slope definition, parity, leakage, determinism, no local paths |
| `README.md` | how to point the code at the raw WESAD folder |

Conditions: R1s, R2, R1stoR2 (R1s and R1stoR2 share fits). Common window set: 1012 windows (3 of 1015 have < 10 bins at R2).
Beat time = midpoint of the inter-beat interval. Bins: floor(t / 3 s), 20 per window; empty bins skipped.

HR_slope: bpm per second, least-squares line on the actual timestamps (beat times at R1s, bin-centre times at R2). The per-beat slope of `features.csv` is used only for parity and for the descriptive R1s vs R1 comparison (Pearson r 0.967 on the 1012 shared windows). The other four features are the same at R1 and R1s.

Limits: downsampling is not Fitbit smoothing or motion handling, so a device gap remains; 15 subjects give the interaction test low power and wide CIs; the cross-resolution case also carries a shift in feature distributions (R2 mean about 1.3 bpm lower, smaller SD).

Run time: unchanged by the slope change, about 68 minutes at n_jobs=2 (`python -m eval.interaction plan`).
