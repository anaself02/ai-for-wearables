"""Estimators. Every scaled model is a Pipeline whose StandardScaler is fitted
inside `fit`, so only on the rows it is fitted on (outer: 14 subjects; inner:
the inner training subjects)."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def best_threshold(x, y, low, high, labels):
    """Cut on x that maximises macro-F1 of the rule "x > cut -> high, else low".

    `y` are the true labels of the rows `x` came from (the TRAINING subjects
    only; the caller never passes the test subject), `labels` the task's class
    labels over which macro-F1 is averaged. A class the rule never predicts
    (3-class amusement) has F1 0. Ties go to the lowest cut. Returns -inf
    when x has fewer than two distinct values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    cut_after = np.flatnonzero(xs[:-1] != xs[1:])           # a cut sits after these rows
    if len(cut_after) == 0:
        return -np.inf
    f1 = np.zeros(len(cut_after))
    for label in labels:
        is_label = (ys == label).astype(float)
        below = np.cumsum(is_label)[cut_after]               # rows of this class below the cut
        support = is_label.sum()
        above = support - below
        n_below, n_above = cut_after + 1.0, len(xs) - cut_after - 1.0
        if label == low:
            tp, predicted = below, n_below
        elif label == high:
            tp, predicted = above, n_above
        else:
            tp, predicted = np.zeros(len(cut_after)), np.zeros(len(cut_after))
        denominator = predicted + support
        f1 += np.divide(2.0 * tp, denominator, out=np.zeros(len(cut_after)),
                        where=denominator > 0)
    best = int(np.argmax(f1))                                # first maximum
    return float(0.5 * (xs[cut_after[best]] + xs[cut_after[best] + 1]))


class HRMeanThreshold(ClassifierMixin, BaseEstimator):
    """Baseline: one HR_mean cut chosen to maximise macro-F1 on the training
    subjects only (PRE_DECLARED.md Amendment 4).

    Binary: stress above the cut, non-stress below. 3-class: stress above,
    baseline below, never amusement (its recall is 0 by construction). Uses the
    raw HR_mean column (column 0 in config.HR_FEATURES order), unscaled.
    """

    def __init__(self, three_class=False, column=0):
        self.three_class = three_class
        self.column = column

    def fit(self, X, y):
        y = np.asarray(y)
        x = np.asarray(X)[:, self.column]
        if self.three_class:
            low, high, labels = 1, config.STRESS, config.TASKS["three"]["labels"]
        else:
            low, high, labels = 0, 1, config.TASKS["binary"]["labels"]
        self.low_, self.high_ = low, high
        self.threshold_ = best_threshold(x, y, low, high, labels)
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, self.column] - self.threshold_

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.high_, self.low_)


class OversampledMLP(ClassifierMixin, BaseEstimator):
    """MLPClassifier trained for a fixed max_iter (300) with NO early stopping
    and no validation split inside fit, after randomly oversampling every
    minority class up to the majority count.

    scikit-learn 1.5.2's MLPClassifier.fit accepts no sample weights, so
    oversampling is the balancing method. It happens inside fit on the rows
    fit receives: in the inner tuning loop those are the inner-TRAINING rows
    only (inner-validation rows are only ever passed to predict), and in the
    outer refit the 14 training subjects. Rows passed to predict are never
    resampled. The network stops early only if the training loss stops
    improving (tol); `converged_` is False when it ran to max_iter, which is
    exactly when scikit-learn emits a ConvergenceWarning.
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
                                  early_stopping=False, max_iter=self.max_iter,
                                  random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)   # counted via converged_
            self.mlp_.fit(X[rows], y[rows])
        self.converged_ = bool(self.mlp_.n_iter_ < self.max_iter)
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


def converged(estimator, X=None, y=None):
    """1.0 if the fitted model converged (or has no iteration limit), else 0.0.

    Signature (estimator, X, y) so it can be a GridSearchCV scorer: it is then
    evaluated inside every inner fit, also in worker processes.
    """
    clf = getattr(estimator, "named_steps", {}).get("clf", estimator)
    return float(getattr(clf, "converged_", True))


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
