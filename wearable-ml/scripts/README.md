# scripts/ - data preparation and the first (default-hyperparameter) pipeline

Run from `wearable-ml/` with the venv active. Settings live in `project_config.py`
(`models/project_config.py` is an identical copy used by the demo and `eval/`).

| step | command | what it does | output |
|---|---|---|---|
| 1 | `python scripts/01_load_and_inspect.py [S2]` | inspects one subject's `.pkl` (structure, rates, label timeline) | console only |
| 2 | `python scripts/02_feature_extraction.py` | 60 s windows, 30 s step, 100% label purity, 40 features per window (needs raw WESAD in `data/WESAD/`) | `results/features.csv` (overwrites the tracked file) |
| 3 | `python scripts/03_train_model.py` | RF + SVM, LOSO CV, full and HR-only feature sets (`random_state=42`) | `results/cv_predictions_*.csv`, `results/scores_*.json`, `models/*.joblib` |
| 4 | `python scripts/04_evaluate.py` | confusion matrices, per-subject F1, classification reports | `results/*.png`, `results/*.txt`, `RESULTS.md` |
| 5 | `python scripts/05_apply_to_personal_data.py inspect` then `predict` | optional: scores your own Google Health heart rate (needs the sibling dashboard project); a pipeline demo, not a validated detector | `results/personal/` (git-ignored) |

`features.py` holds the feature and heart-rate-from-BVP code (band-pass 0.7-3.7 Hz, peak detection,
inter-beat-interval median filter of width 5, 40-180 bpm). The tuned, pre-declared evaluation that
produces the headline results is in `eval/`, not here.
