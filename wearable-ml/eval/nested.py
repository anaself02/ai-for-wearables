"""Nested leave-one-subject-out for one (task, model, seed).

Outer: each of the 15 subjects is the test subject once. Inner (models with a
grid): the 14 training subjects are split into config.INNER_FOLDS subject
groups (shuffled by the seed); every grid point is fitted on the other groups
and scored by macro-F1 on the held-out group; the best mean inner macro-F1 wins
(ties: first in grid order), is refitted on all 14 subjects and predicts the
test subject once. Baselines are fitted on the 14 subjects without tuning.
Nothing about the test subject is used before its prediction.
"""

import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from . import config, folds, metrics, models
from .data import labels


def run(table, task, model, seed, grid=None, subjects=None, feature_names=None,
        inner_k=config.INNER_FOLDS, on_fitted=None):
    """-> {"predictions": [...], "folds": [...], "failures": [...], "seconds": float}

    on_fitted(test_subject, fitted_estimator, train_index, test_index), if given,
    is called after each outer refit (tests use it to inspect what was fitted).
    """
    feature_names = feature_names or config.HR_FEATURES
    X_all = table[feature_names].to_numpy(dtype=float)
    y_all = labels(table, task)
    groups = table["subject_id"].to_numpy()
    subjects = subjects or sorted(set(groups), key=folds.subject_key)
    task_labels = config.TASKS[task]["labels"]
    grid = config.GRIDS.get(model) if grid is None else grid

    started = time.perf_counter()
    predictions, fold_rows, failures = [], [], []
    for test_subject, train_subjects in folds.outer_folds(subjects):
        train_index = np.flatnonzero(np.isin(groups, train_subjects))
        test_index = np.flatnonzero(groups == test_subject)
        X_train, y_train = X_all[train_index], y_all[train_index]
        chosen, inner_score = None, None
        fold_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                estimator = models.make(model, task, seed)
                if grid:
                    inner = folds.inner_folds(train_subjects, seed, inner_k)
                    search = GridSearchCV(clone(estimator), grid, scoring="f1_macro",
                                          cv=folds.inner_splits(groups, train_index, inner),
                                          refit=True, n_jobs=1, error_score=np.nan)
                    search.fit(X_train, y_train)
                    if not np.isfinite(search.best_score_):
                        raise RuntimeError("every grid point failed in the inner folds")
                    fitted = search.best_estimator_
                    chosen = {k: _plain(v) for k, v in search.best_params_.items()}
                    inner_score = float(search.best_score_)
                    n_failed = int(np.sum(~np.isfinite(search.cv_results_["mean_test_score"])))
                    if n_failed:
                        failures.append({"subject": test_subject, "kind": "grid points failed",
                                         "count": n_failed})
                else:
                    fitted = clone(estimator).fit(X_train, y_train)
                if on_fitted is not None:
                    on_fitted(test_subject, fitted, train_index, test_index)
                y_pred = fitted.predict(X_all[test_index])
                score = (models.positive_score(fitted, X_all[test_index])
                         if task == "binary" else None)
            except Exception as exc:                     # recorded, never hidden
                failures.append({"subject": test_subject, "kind": "fold failed",
                                 "error": "{}: {}".format(type(exc).__name__, exc)})
                continue
        n_convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        if n_convergence:
            failures.append({"subject": test_subject, "kind": "convergence warnings",
                             "count": n_convergence})
        m = metrics.fold_metrics(y_all[test_index], y_pred, score, task_labels)
        fold_rows.append({"task": task, "model": model, "seed": seed, "subject": test_subject,
                          "chosen": chosen, "inner_macro_f1": inner_score,
                          "seconds": time.perf_counter() - fold_started, **m})
        for i, row in enumerate(test_index):
            predictions.append({
                "task": task, "model": model, "seed": seed, "fold": test_subject,
                "subject": test_subject, "condition": table["label_name"].iat[row],
                "window_start_s": float(table["window_start_s"].iat[row]),
                "y_true": int(y_all[row]), "y_pred": int(y_pred[i]),
                "score": None if score is None else float(score[i])})
    return {"predictions": predictions, "folds": fold_rows, "failures": failures,
            "seconds": time.perf_counter() - started}


def _plain(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
