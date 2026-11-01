"""The binary stress / non-stress models, trained in memory on WESAD only.

THE ONLY MODULE IN THE DASHBOARD THAT FITS ANYTHING, AND ONLY ON WESAD
---------------------------------------------------------------------
`train()` takes no data argument. It reads the WESAD feature table that the
wearable-ml project built (``results/features.csv``, 15 lab subjects) and
nothing else, so there is no way to hand it a person's readings or a session
label. Nothing is saved: the models live in this process's memory only, and
no file in wearable-ml is written.

What it builds, as in the wearable-ml demo (Stage 1, default hyperparameters,
no tuning):

  majority  the more frequent binary class in the WESAD table (non-stress)
  rule      one HR_mean threshold maximising balanced accuracy on WESAD
  svm       StandardScaler + SVC(rbf, C=1.0, gamma="scale",
            class_weight="balanced"), the repo's default from
            scripts/03_train_model.py; test_sessions pins it to that file

Session labels never reach this module. They are only compared with its
predictions, in people/sessions.py, which contains no fit call (a test scans
it and runs it with every fit patched to raise).
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings

FEATURE_NAMES = ["HR_mean", "HR_std", "HR_min", "HR_max", "HR_slope"]
STRESS = 2               # WESAD's stress label; binary 1 = stress, 0 = non-stress
RANDOM_STATE = 42        # scripts/project_config.py RANDOM_STATE

MODEL_LABEL = ("Binary SVM (stress vs non-stress), trained in memory on WESAD "
               "hr_only features with default hyperparameters (no tuning). "
               "Never saved.")


class WesadUnavailable(Exception):
    """The WESAD feature table is missing or unusable."""


@dataclass(frozen=True)
class BinaryModels:
    svm: object
    threshold: float
    majority: int
    distribution: dict        # {feature: {min, max, mean, std}} of the WESAD windows
    n_windows: int
    n_subjects: int


def features_path():
    return os.path.join(settings.WEARABLE_ML_DIR, "results", "features.csv")


def best_threshold(x, positive):
    """Cut-off on x that maximises balanced accuracy for 'x > cut = positive'.

    Same rule as best_threshold in wearable-ml/demo/demo_wesad.py; a test
    checks the two give the same cut on the WESAD table.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = len(x) - n_pos
    if n_pos == 0 or n_neg == 0:
        return -np.inf
    order = np.argsort(x, kind="stable")
    xs, ps = x[order], positive[order]
    pos_below, neg_below = np.cumsum(ps), np.cumsum(~ps)
    best_score, best_cut = -1.0, -np.inf
    for i in range(len(xs) - 1):
        if xs[i] == xs[i + 1]:
            continue
        score = 0.5 * ((n_pos - pos_below[i]) / n_pos + neg_below[i] / n_neg)
        if score > best_score:
            best_score, best_cut = score, 0.5 * (xs[i] + xs[i + 1])
    return float(best_cut)


def svm_template():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                           class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def train():
    """The binary models, fitted on the WESAD table only. Cached per file version."""
    path = features_path()
    if not os.path.exists(path):
        raise WesadUnavailable(
            "The WESAD feature table was not found at {}. The binary model "
            "cannot be trained, so nothing is scored.".format(path))
    return _train(path, os.path.getmtime(path))


@lru_cache(maxsize=1)
def _train(path, _mtime):
    import pandas as pd

    try:
        table = pd.read_csv(path, usecols=["subject_id", "label"] + FEATURE_NAMES)
    except (ValueError, OSError) as exc:
        raise WesadUnavailable("The WESAD feature table could not be read: {}".format(exc))
    table = table.dropna()
    if not table["subject_id"].astype(str).str.fullmatch(r"S\d+").all():
        raise WesadUnavailable("The feature table has rows that are not WESAD subjects.")

    y = (table["label"] == STRESS).astype(int)
    X = table[FEATURE_NAMES]
    svm = svm_template()
    svm.fit(X, y)       # WESAD windows only; never a person's data
    return BinaryModels(
        svm=svm,
        threshold=best_threshold(X["HR_mean"], y == 1),
        majority=int(y.value_counts().idxmax()),
        distribution={name: {"min": float(X[name].min()), "max": float(X[name].max()),
                             "mean": float(X[name].mean()), "std": float(X[name].std())}
                      for name in FEATURE_NAMES},
        n_windows=int(len(table)),
        n_subjects=int(table["subject_id"].nunique()),
    )
