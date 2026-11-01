# Error analysis (EXPLORATORY)

Predictions: svm (3-class), default hyperparameters, one run, from cv_predictions_hr_only.csv. 1015 windows, 15 subjects. Context (native HR, wrist motion) joined from results/features.csv; the HR-only models never see motion. Not pre-declared; 15 subjects, so correlations are fragile.

Overall window error rate: 0.451.

## 1. Which subjects fail

![](subject_response_vs_f1.png)

| subject_id | windows | error_rate | macro_f1 | resting_hr | stress_response | amusement_response | motion_median |
|---|---|---|---|---|---|---|---|
| S15 | 69 | 0.710 | 0.267 | 77.489 | 1.127 | 0.563 | 2.011 |
| S10 | 69 | 0.638 | 0.287 | 91.548 | 5.626 | -14.146 | 1.930 |
| S13 | 69 | 0.609 | 0.332 | 83.382 | 3.477 | -2.871 | 3.225 |
| S6 | 67 | 0.507 | 0.403 | 71.430 | 0.862 | -2.633 | 1.638 |
| S3 | 67 | 0.552 | 0.412 | 61.851 | 20.983 | 2.183 | 1.775 |
| S4 | 66 | 0.545 | 0.418 | 61.871 | 18.122 | 2.039 | 1.014 |
| S7 | 67 | 0.463 | 0.501 | 72.855 | 8.747 | -14.099 | 0.660 |
| S8 | 68 | 0.412 | 0.506 | 73.069 | 3.725 | -1.447 | 1.626 |
| S9 | 66 | 0.485 | 0.507 | 76.702 | 2.278 | 9.338 | 1.431 |
| S5 | 67 | 0.388 | 0.512 | 66.212 | 18.161 | 3.891 | 1.001 |
| S2 | 65 | 0.369 | 0.544 | 73.908 | 1.221 | -1.490 | 1.846 |
| S16 | 67 | 0.418 | 0.549 | 68.578 | 31.993 | 12.274 | 1.916 |
| S17 | 69 | 0.275 | 0.566 | 66.926 | 22.694 | 5.323 | 1.276 |
| S11 | 70 | 0.286 | 0.583 | 75.179 | 4.544 | -0.991 | 2.908 |
| S14 | 69 | 0.116 | 0.867 | 73.888 | 36.322 | 8.445 | 1.726 |

Spearman correlation, stress response vs per-subject macro-F1: rho = +0.52 (p = 0.044, n = 15).

## 2. Which conditions fail

| true \ predicted | amusement | baseline | stress |
|---|---|---|---|
| amusement | 46 | 81 | 29 |
| baseline | 161 | 290 | 106 |
| stress | 36 | 45 | 221 |

| condition | error_rate | windows |
|---|---|---|
| amusement | 0.705 | 156 |
| baseline | 0.479 | 557 |
| stress | 0.268 | 302 |

## 3. Where inside a condition block

| condition | position | error_rate | windows |
|---|---|---|---|
| amusement | windows 1-2 | 0.767 | 30 |
| amusement | windows 3-5 | 0.622 | 45 |
| amusement | window 6+ | 0.728 | 81 |
| baseline | windows 1-2 | 0.567 | 30 |
| baseline | windows 3-5 | 0.400 | 45 |
| baseline | window 6+ | 0.481 | 482 |
| stress | windows 1-2 | 0.100 | 30 |
| stress | windows 3-5 | 0.178 | 45 |
| stress | window 6+ | 0.308 | 227 |

## 4. Wrist motion (not a model input)

![](error_by_motion.png)

| condition | motion tertile | error_rate | windows |
|---|---|---|---|
| amusement | low | 0.676 | 108 |
| amusement | mid | 0.679 | 28 |
| amusement | high | 0.900 | 20 |
| baseline | low | 0.321 | 162 |
| baseline | mid | 0.494 | 239 |
| baseline | high | 0.622 | 156 |
| stress | low | 0.275 | 69 |
| stress | mid | 0.386 | 70 |
| stress | high | 0.215 | 163 |

Median ACC_mag_std by condition: amusement 0.70, baseline 1.72, stress 2.66.

Non-stress windows called stress (n = 135) had median motion 2.48 vs 1.26 for correct non-stress windows (n = 336); Mann-Whitney p = 2.29e-17 (windows overlap and are not independent, so p is optimistic).

## 5. Absolute vs person-relative heart rate

Error rate by the subject's resting-HR tertile:

| resting HR | amusement | baseline | stress |
|---|---|---|---|
| low | 0.784 | 0.454 | 0.220 |
| mid | 0.673 | 0.319 | 0.313 |
| high | 0.660 | 0.663 | 0.272 |

Diagnostic (uses each subject's own baseline, so NOT a model result): AUROC of HR_mean for stress vs baseline is 0.797 pooled across people and 0.803 after subtracting each subject's resting HR.

## 6. Examples

Examples: for each error type (true condition -> predicted class), the 2 windows closest to that type's median HR_mean (typical, not extreme, errors).

| subject_id | window_start_s | condition | predicted | position | HR_mean | HR_std | ACC_mag_std |
|---|---|---|---|---|---|---|---|
| S2 | 4950.000 | amusement | baseline | 7 | 71.370 | 10.910 | 0.470 |
| S8 | 1890.000 | amusement | baseline | 8 | 71.280 | 7.820 | 0.820 |
| S5 | 2100.000 | amusement | stress | 8 | 74.570 | 15.330 | 0.410 |
| S13 | 1680.000 | amusement | stress | 1 | 74.790 | 15.270 | 2.820 |
| S6 | 690.000 | baseline | amusement | 2 | 74.210 | 12.610 | 1.520 |
| S9 | 390.000 | baseline | amusement | 11 | 74.190 | 10.410 | 1.350 |
| S13 | 690.000 | baseline | stress | 21 | 79.850 | 10.720 | 1.570 |
| S9 | 750.000 | baseline | stress | 23 | 79.730 | 12.000 | 1.800 |
| S9 | 2310.000 | stress | amusement | 15 | 78.030 | 5.930 | 0.190 |
| S9 | 2250.000 | stress | amusement | 13 | 77.440 | 9.880 | 0.410 |
| S4 | 3840.000 | stress | baseline | 8 | 75.290 | 8.800 | 3.100 |
| S4 | 3870.000 | stress | baseline | 9 | 75.230 | 10.010 | 2.330 |

![](timelines.png)

Timelines: the two lowest-F1 subjects and the highest. Circles = misclassified.
