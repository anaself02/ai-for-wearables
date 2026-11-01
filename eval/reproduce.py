"""Reproduce the existing Table I number: SVM, full 40 features, 3-class,
leave-one-subject-out, default hyperparameters, pooled out-of-fold macro-F1
(results/scores_full.json: 0.716951732295367).

Settings copied from scripts/03_train_model.py build_models()["svm"]:
StandardScaler + SVC(kernel="rbf", C=1.0, gamma="scale",
class_weight="balanced", random_state=42), LeaveOneGroupOut by subject_id,
macro-F1 over the pooled predictions of all 15 folds (not a per-subject mean).
The table is read in file order, as that script reads it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .data import check_path

EXPECTED_FULL_SVM_MACRO_F1 = 0.716951732295367
NON_FEATURES = ("subject_id", "label", "label_name", "window_start_s")


def full_svm_loso(path=config.FEATURES):
    table = pd.read_csv(check_path(path))
    features = [c for c in table.columns if c not in NON_FEATURES]
    X, y, groups = table[features], table["label"].astype(int), table["subject_id"]
    predictions = np.empty(len(y), dtype=int)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        model = Pipeline([("scaler", StandardScaler()),
                          ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                                             class_weight="balanced", random_state=42))])
        model.fit(X.iloc[train], y.iloc[train])
        predictions[test] = model.predict(X.iloc[test])
    return float(f1_score(y, predictions, average="macro")), len(features)
