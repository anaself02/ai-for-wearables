# models/ - shared feature definitions and the saved SVMs

| file | what it is |
|---|---|
| `features.py`, `project_config.py` | identical copies of `scripts/features.py` and `scripts/project_config.py` (feature and heart-rate code, constants), importable by the demo and `eval/` |
| `feature_config.json`, `feature_config_hr_only.json` | feature names of the full (40) and HR-only (5) sets |
| `svm_model.joblib`, `svm_hr_only.joblib` | SVMs trained on all 15 WESAD subjects by `scripts/03_train_model.py` (default hyperparameters, `random_state=42`); need scikit-learn 1.5.2 (`requirements.txt`) |
| `hr_training_distribution.json` | mean, SD, min, median, max of each HR feature over the training windows (used for the out-of-distribution check) |
| `model_card.json` | what the HR-only model was trained on, its classes and limits |

These files are outputs of `scripts/03_train_model.py` and are tracked so the demo runs without retraining.
The tuned models of `eval/` are fitted inside each cross-validation fold and are not saved.
