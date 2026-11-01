"""Nested leave-one-subject-out where one fit is scored on several test tables.

Same procedure as nested.run (Amendment 4: outer LOSO, inner subject-grouped
folds shuffled by the seed, best mean inner macro-F1, ties to the first grid point,
refit on the 14 training subjects, test subject predicted once per table), with one
addition: the training data comes from `train_table`, and the fitted model predicts
the test subject's rows in every table of `eval_tables` (row-aligned with it).

With eval_tables = {"R1s": r1, "R1stoR2": r2} and train_table = r1, one set of fits
gives both the same-resolution result and the cross-resolution (train R1s, test R2)
result, so the cross-resolution diagnostic costs no extra fits. Tuning only ever
sees the training table's training subjects.
"""

import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from . import config, folds, metrics, models
from .data import labels
from .nested import _plain


def _aligned(a, b):
    return (len(a) == len(b)
            and (a["subject_id"].to_numpy() == b["subject_id"].to_numpy()).all()
            and np.allclose(a["window_start_s"].to_numpy(), b["window_start_s"].to_numpy())
            and (a["label"].to_numpy() == b["label"].to_numpy()).all())


def run(train_table, eval_tables, model, seed, task="binary", grid=None, subjects=None,
        inner_k=config.INNER_FOLDS, on_fitted=None, n_jobs=config.GRID_JOBS):
    """-> {"results": {name: {"predictions": [...], "folds": [...]}}, "failures": [...],
    "seconds": float}.

    on_fitted(test_subject, fitted, train_index, test_index) is called after each outer
    refit (tests use it to check which rows were fitted)."""
    for name, table in eval_tables.items():
        if not _aligned(train_table, table):
            raise ValueError("table {!r} is not row-aligned with the training table".format(name))
    features = config.HR_FEATURES
    X_train_all = train_table[features].to_numpy(dtype=float)
    X_eval = {n: t[features].to_numpy(dtype=float) for n, t in eval_tables.items()}
    y_all = labels(train_table, task)
    groups = train_table["subject_id"].to_numpy()
    subjects = subjects or sorted(set(groups), key=folds.subject_key)
    task_labels = config.TASKS[task]["labels"]
    grid = config.GRIDS.get(model) if grid is None else grid

    started = time.perf_counter()
    results = {n: {"predictions": [], "folds": []} for n in eval_tables}
    failures = []
    for test_subject, train_subjects in folds.outer_folds(subjects):
        train_index = np.flatnonzero(np.isin(groups, train_subjects))
        test_index = np.flatnonzero(groups == test_subject)
        X_train, y_train = X_train_all[train_index], y_all[train_index]
        chosen, inner_score = None, None
        fold_started = time.perf_counter()
        try:
            estimator = models.make(model, task, seed)
            if grid:
                inner = folds.inner_folds(train_subjects, seed, inner_k)
                search = GridSearchCV(
                    clone(estimator), grid,
                    scoring={"f1_macro": "f1_macro", "converged": models.converged},
                    refit="f1_macro", cv=folds.inner_splits(groups, train_index, inner),
                    n_jobs=n_jobs, error_score=np.nan)
                search.fit(X_train, y_train)
                if not np.isfinite(search.best_score_):
                    raise RuntimeError("every grid point failed in the inner folds")
                fitted = search.best_estimator_
                chosen = {k: _plain(v) for k, v in search.best_params_.items()}
                inner_score = float(search.best_score_)
                n_failed = int(np.sum(~np.isfinite(search.cv_results_["mean_test_f1_macro"])))
                if n_failed:
                    failures.append({"subject": test_subject, "kind": "grid points failed",
                                     "count": n_failed})
                per_split = np.array([search.cv_results_["split{}_test_converged".format(i)]
                                      for i in range(len(inner))])
                n_inner_fits, n_not_converged = int(per_split.size), int(np.sum(per_split < 0.5))
            else:
                fitted = clone(estimator).fit(X_train, y_train)
                n_inner_fits = n_not_converged = 0
            refit_converged = bool(models.converged(fitted))
            if on_fitted is not None:
                on_fitted(test_subject, fitted, train_index, test_index)
            predicted = {}
            for name, X in X_eval.items():
                predicted[name] = (fitted.predict(X[test_index]),
                                   models.positive_score(fitted, X[test_index])
                                   if task == "binary" else None)
        except Exception as exc:                         # recorded, never hidden
            failures.append({"subject": test_subject, "kind": "fold failed",
                             "error": "{}: {}".format(type(exc).__name__, exc)})
            continue
        if n_not_converged or not refit_converged:
            failures.append({"subject": test_subject, "kind": "convergence warnings",
                             "inner_fits_not_converged": n_not_converged,
                             "inner_fits": n_inner_fits,
                             "outer_refit_converged": refit_converged})
        seconds = time.perf_counter() - fold_started
        for name, (y_pred, score) in predicted.items():
            m = metrics.fold_metrics(y_all[test_index], y_pred, score, task_labels)
            results[name]["folds"].append({
                "task": task, "model": model, "seed": seed, "subject": test_subject,
                "condition": name, "chosen": chosen, "inner_macro_f1": inner_score,
                "seconds": seconds, **m})
            table = eval_tables[name]
            for i, row in enumerate(test_index):
                results[name]["predictions"].append({
                    "task": task, "model": model, "seed": seed, "condition": name,
                    "fold": test_subject, "subject": test_subject,
                    "label_name": table["label_name"].iat[row],
                    "window_start_s": float(table["window_start_s"].iat[row]),
                    "y_true": int(y_all[row]), "y_pred": int(y_pred[i]),
                    "score": None if score is None else float(score[i])})
    return {"results": results, "failures": failures, "seconds": time.perf_counter() - started}
