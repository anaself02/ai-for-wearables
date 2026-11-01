"""Per-fold metrics, subject-level summaries and the bootstrap over subjects."""

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from . import config


def fold_metrics(y_true, y_pred, score, labels):
    """Metrics for one held-out subject (one outer fold)."""
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": {int(c): float(p) for c, p in zip(labels, precision)},
        "recall": {int(c): float(r) for c, r in zip(labels, recall)},
        "support": {int(c): int(s) for c, s in zip(labels, support)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "auroc": None,
    }
    if len(labels) == 2 and score is not None and len(set(np.asarray(y_true))) == 2:
        # A constant score ranks nothing: AUROC 0.5 by definition.
        out["auroc"] = (float(roc_auc_score(y_true, score)) if np.ptp(score) > 0 else 0.5)
    return out


def bootstrap_ci(values, n=config.N_BOOT, seed=config.BOOT_SEED, alpha=0.05):
    """(mean, low, high): percentile CI of the mean, resampling subjects with replacement."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return (float(values.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def subject_table(fold_rows, metric):
    """{model: {subject: value averaged over seeds}}; seeds are averaged first."""
    acc = {}
    for row in fold_rows:
        if row.get(metric) is None:
            continue
        acc.setdefault(row["model"], {}).setdefault(row["subject"], []).append(row[metric])
    return {m: {s: float(np.mean(v)) for s, v in subjects.items()}
            for m, subjects in acc.items()}


def summarise(fold_rows, metric):
    """Per model: mean and SD over subjects, bootstrap 95% CI of the mean."""
    out = {}
    for model, per_subject in subject_table(fold_rows, metric).items():
        values = [per_subject[s] for s in sorted(per_subject)]
        mean, lo, hi = bootstrap_ci(values)
        out[model] = {"mean": mean,
                      "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                      "ci_low": lo, "ci_high": hi, "n_subjects": len(values)}
    return out


def paired_differences(fold_rows, metric, pairs=config.PAIRS):
    """first minus second, per subject, then a paired bootstrap over subjects."""
    table = subject_table(fold_rows, metric)
    out = {}
    for a, b in pairs:
        if a not in table or b not in table:
            continue
        common = sorted(set(table[a]) & set(table[b]))
        mean, lo, hi = bootstrap_ci([table[a][s] - table[b][s] for s in common])
        out["{} - {}".format(a, b)] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                                       "excludes_zero": bool(lo > 0 or hi < 0),
                                       "n_subjects": len(common)}
    return out
