"""
demo_wesad.py
-------------
The WESAD-only half of the demo: leave-one-subject-out evaluation on the cached
feature table (results/features.csv), for a 3-class and a binary task.

Everything here learns from WESAD features only. Nothing is written to disk.

Hyperparameters are the repo's defaults, taken from scripts/03_train_model.py
so there is a single definition. They are NOT tuned here; tuning is Stage 2.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import LeaveOneGroupOut

STRESS = 2

TASKS = [
    {"key": "binary",
     "title": "BINARY: stress vs non-stress  (primary task for personal-data transfer)",
     "labels": [0, 1],
     "names": {0: "non-stress", 1: "stress"}},
    {"key": "three",
     "title": "3-CLASS: baseline / stress / amusement  (secondary, WESAD only)",
     "labels": [1, 2, 3],
     "names": {1: "baseline", 2: "stress", 3: "amusement"}},
]

SAVED_KEYS = {"SVM (RBF, default)": "svm",
              "random forest (default)": "random_forest"}


def task_labels(table, key):
    """The label vector for a task: binary merges baseline + amusement."""
    if key == "binary":
        return (table["label"] == STRESS).astype(int)
    return table["label"].astype(int)


# ---------------------------------------------------------------------------
# the rule baseline
# ---------------------------------------------------------------------------

def best_threshold(x, positive):
    """Cut-off on x that maximises balanced accuracy for 'x > cut = positive'."""
    x = np.asarray(x, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = len(x) - n_pos
    if n_pos == 0 or n_neg == 0:
        return -np.inf

    order = np.argsort(x, kind="stable")
    xs, ps = x[order], positive[order]
    pos_below = np.cumsum(ps)
    neg_below = np.cumsum(~ps)

    best_score, best_cut = -1.0, -np.inf
    for i in range(len(xs) - 1):
        if xs[i] == xs[i + 1]:
            continue
        tp = n_pos - pos_below[i]
        tn = neg_below[i]
        score = 0.5 * (tp / n_pos + tn / n_neg)
        if score > best_score:
            best_score, best_cut = score, 0.5 * (xs[i] + xs[i + 1])
    return best_cut


class HRMeanThreshold(ClassifierMixin, BaseEstimator):
    """Single-threshold rule on HR_mean, fitted on the training folds only.

    Binary: predicts 1 (stress) when HR_mean is above the threshold.
    3-class: predicts stress (2) above it and baseline (1) below it, and NEVER
    predicts amusement, so its amusement recall is 0 by construction.
    """

    def __init__(self, three_class=False, feature="HR_mean"):
        self.three_class = three_class
        self.feature = feature

    def fit(self, X, y):
        y = np.asarray(y)
        positive = (y == STRESS) if self.three_class else (y == 1)
        self.threshold_ = best_threshold(np.asarray(X[self.feature]), positive)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        above = np.asarray(X[self.feature]) > self.threshold_
        if self.three_class:
            return np.where(above, STRESS, 1)
        return above.astype(int)


# ---------------------------------------------------------------------------
# models and cross-validation
# ---------------------------------------------------------------------------

def load_train_module():
    """scripts/03_train_model.py, imported only to reuse build_models()."""
    return importlib.import_module("03_train_model")


def model_templates(task_key, with_rf=False):
    built = load_train_module().build_models()
    templates = {
        "majority class": DummyClassifier(strategy="most_frequent"),
        "rule: HR_mean threshold": HRMeanThreshold(three_class=(task_key == "three")),
        "SVM (RBF, default)": built["svm"],
    }
    if with_rf:
        templates["random forest (default)"] = built["random_forest"]
    return templates


def loso_predict(template, X, y, groups):
    """Out-of-fold predictions: every window is predicted by a model that never
    saw that window's subject."""
    predictions = np.empty(len(y), dtype=np.asarray(y).dtype)
    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        model = clone(template)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        predictions[test_index] = model.predict(X.iloc[test_index])
    return predictions


def summarise(y, predictions, groups, labels):
    """Pooled and per-subject metrics for one set of out-of-fold predictions."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=labels, zero_division=0)

    per_subject = [
        f1_score(y[groups == s], predictions[groups == s],
                 average="macro", zero_division=0)
        for s in sorted(set(groups))
    ]
    matrix = pd.crosstab(pd.Series(y, name="actual"),
                         pd.Series(predictions, name="predicted")
                         ).reindex(index=labels, columns=labels, fill_value=0)

    return {
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": dict(zip(labels, precision)),
        "recall": dict(zip(labels, recall)),
        "support": dict(zip(labels, support)),
        "per_subject_mean": float(np.mean(per_subject)),
        "per_subject_sd": float(np.std(per_subject, ddof=1)),
        "confusion": matrix,
    }


def shuffled_reference(y, n=200, seed=0):
    """Macro-F1 of the true labels against randomly permuted copies of them."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    values = [f1_score(y, rng.permutation(y), average="macro", zero_division=0)
              for _ in range(n)]
    return float(np.mean(values)), float(np.std(values))


def evaluate_task(table, task, feature_sets, with_rf=False):
    """{feature set: {model name: summary}} for one task, leave-one-subject-out."""
    y = task_labels(table, task["key"])
    groups = table["subject_id"]
    out = {}
    for fs_name, columns in feature_sets.items():
        X = table[columns]
        out[fs_name] = {}
        for name, template in model_templates(task["key"], with_rf).items():
            predictions = loso_predict(template, X, y, groups)
            out[fs_name][name] = summarise(y, predictions, groups, task["labels"])
    return out


def fit_binary_hr_model(table, feature_names):
    """Fit the binary stress/non-stress SVM on WESAD heart-rate features.

    In memory only: it is never saved, so no model file in models/ is touched.
    Personal data is never given to this function.
    """
    template = load_train_module().build_models()["svm"]
    model = clone(template)
    model.fit(table[feature_names], task_labels(table, "binary"))
    return model
