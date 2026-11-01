"""Estimators. Every scaled model is a Pipeline whose StandardScaler is fitted
inside `fit`, so only on the rows it is fitted on (outer: 14 subjects; inner:
the inner training subjects)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def best_threshold(x, positive):
    """Cut on x maximising balanced accuracy for "x > cut means positive".

    Same rule as best_threshold in demo/demo_wesad.py; a test pins the two.
    """
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


class HRMeanThreshold(ClassifierMixin, BaseEstimator):
    """Baseline: one HR_mean cut fitted on the training subjects only.

    Binary: stress above the cut. 3-class: stress above, baseline below, never
    amusement (its recall is 0 by construction). Uses the raw HR_mean column
    (column 0 in config.HR_FEATURES order), unscaled.
    """

    def __init__(self, three_class=False, column=0):
        self.three_class = three_class
        self.column = column

    def fit(self, X, y):
        y = np.asarray(y)
        x = np.asarray(X)[:, self.column]
        self.threshold_ = best_threshold(x, y == (config.STRESS if self.three_class else 1))
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, self.column] - self.threshold_

    def predict(self, X):
        above = self.decision_function(X) > 0
        return np.where(above, config.STRESS, 1) if self.three_class else above.astype(int)


class OversampledMLP(ClassifierMixin, BaseEstimator):
    """MLPClassifier with early stopping, trained after randomly oversampling
    every minority class up to the majority count.

    scikit-learn 1.5.2's MLPClassifier.fit accepts no sample weights, so
    oversampling is the balancing method. It happens inside fit, i.e. inside
    the training fold only, and after the pipeline's scaler (which therefore
    sees each training row once). Rows passed to predict are never resampled.
    """

    def __init__(self, hidden_layer_sizes=(32,), alpha=1e-4, random_state=0,
                 max_iter=config.MLP_MAX_ITER):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        index = [np.flatnonzero(y == c) for c in classes]
        extra = [rng.choice(ix, counts.max() - len(ix), replace=True) for ix in index]
        rows = np.concatenate(index + extra)
        rows = rows[rng.permutation(len(rows))]
        self.n_oversampled_ = int(len(rows) - len(y))
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                  early_stopping=True, max_iter=self.max_iter,
                                  random_state=self.random_state)
        self.mlp_.fit(X[rows], y[rows])
        self.classes_ = self.mlp_.classes_
        return self

    def predict(self, X):
        return self.mlp_.predict(X)

    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)


def make(name, task, seed):
    """An unfitted estimator for one model or baseline."""
    three = task == "three"
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "threshold":
        return HRMeanThreshold(three_class=three)
    if name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=config.RF_TREES, class_weight="balanced",
                                     random_state=seed, n_jobs=config.RF_JOBS)
    elif name == "mlp":
        clf = OversampledMLP(random_state=seed)
    else:
        raise ValueError("unknown model {}".format(name))
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def positive_score(estimator, X):
    """A continuous score for the positive (stress) class, for binary AUROC."""
    try:
        return np.asarray(estimator.decision_function(X), dtype=float)
    except AttributeError:
        pass
    try:
        proba = estimator.predict_proba(X)
        return np.asarray(proba[:, -1], dtype=float)
    except AttributeError:
        return np.zeros(len(X))
