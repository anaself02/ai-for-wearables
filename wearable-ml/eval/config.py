"""Every frozen choice of the evaluation, in one place (PRE_DECLARED.md Amendment 4).

Nothing here may change after test results have been seen.
"""

import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(EVAL_DIR)
FEATURES = os.path.join(ML_DIR, "results", "features.csv")   # the single input
OUT_DIR = os.path.join(EVAL_DIR, "out")
FOLDS_FILE = os.path.join(EVAL_DIR, "folds.json")
EXPERIMENTS = os.path.join(EVAL_DIR, "EXPERIMENTS.md")
MANIFEST = os.path.join(EVAL_DIR, "manifest.json")

#: sha256 of results/features.csv with every CRLF replaced by LF (Amendment 6), so the check
#: does not depend on the line endings of the checkout. Same data as the hash frozen on
#: 2026-11-01 (Amendment 4), which was taken on the Windows copy with CRLF line endings:
#: d0d72f86b8a66c09ff0d3258bbba1fbf480f3f92cd85e606af0289d0aaaf2c36
FEATURES_SHA256 = "167f323da96a6e3c034c32f8cf05c8d506084c0dbf9518c36cd7a7341fab207a"

HR_FEATURES = ["HR_mean", "HR_std", "HR_min", "HR_max", "HR_slope"]
STRESS = 2

TASKS = {
    # primary: stress (2) vs non-stress (baseline 1 + amusement 3)
    "binary": {"labels": [0, 1], "names": {0: "non-stress", 1: "stress"}},
    # secondary, WESAD only
    "three": {"labels": [1, 2, 3], "names": {1: "baseline", 2: "stress", 3: "amusement"}},
}

SEEDS = [0, 1, 2, 3, 4]
INNER_FOLDS = 7                 # 14 training subjects -> 7 folds of 2 subjects
N_BOOT = 10000
BOOT_SEED = 6322
RF_TREES = 300
RF_JOBS = 4                     # threads per forest
GRID_JOBS = 2                   # parallel grid-search fits (processes); results do not depend on it
MLP_MAX_ITER = 300              # fixed; no early stopping, no validation split inside fit

GRIDS = {
    "svm": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    "rf": {"clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 5]},
    "mlp": {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2]},
}
MODELS = ["svm", "rf", "mlp"]
BASELINES = ["majority", "threshold"]

#: pairwise differences reported with bootstrap CIs (first minus second)
PAIRS = [("mlp", "svm"), ("mlp", "rf"), ("rf", "svm")] + [
    (m, b) for m in MODELS for b in BASELINES]

#: paths this package must never read (personal data)
FORBIDDEN = ("data-samples", "takeout", "heart_rate_", "health dashboard")
