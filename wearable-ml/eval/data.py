"""The single input: wearable-ml/results/features.csv, read as-is (never rebuilt).

Audit (2026-11-01, read-only; scripts/02_feature_extraction.py, features.py,
project_config.py): one row per 60 s window, 30 s step, per subject S2-S17
(S1, S12 absent), only windows 100% inside one condition. Labels 1 baseline,
2 stress, 3 amusement (557 / 302 / 156 windows, 1015 total). HR comes from the
wrist BVP (64 Hz): 0.7-3.7 Hz Butterworth, peak detection (min 0.4 s apart),
inter-beat intervals median-filtered over 5 beats, 40-180 bpm kept; one value
per beat. HR_slope is per beat (fs=1). No missing values.
"""

import hashlib
import os

from . import config


class ForbiddenPath(ValueError):
    """Raised when a path points at personal data (see config.FORBIDDEN)."""
    pass


def check_path(path):
    """Refuse personal-data paths.

    Input: a file path. Output: the same path. Raises ForbiddenPath if the
    absolute, lower-cased path contains a fragment of config.FORBIDDEN.
    Randomness: none.
    """
    lowered = os.path.abspath(path).lower()
    for part in config.FORBIDDEN:
        if part in lowered:
            raise ForbiddenPath("eval/ never reads personal data: {}".format(path))
    return path


def sha256(path):
    """sha256 hex digest of a file's raw bytes (line endings kept).

    Input: a path. Output: 64 hex characters. Used for the code hashes in the
    manifests; the feature table uses features_sha256 instead. Randomness: none.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def features_sha256(path=config.FEATURES):
    """sha256 of the feature table after replacing CRLF by LF (Amendment 6)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def load_table(path=config.FEATURES):
    """Read the feature table (results/features.csv by default).

    Input: a path (checked by check_path). Output: a DataFrame sorted by
    subject_id then window_start_s (stable), index reset. Raises ValueError if any
    value is missing. Randomness: none.
    """
    import pandas as pd

    check_path(path)
    table = pd.read_csv(path)
    if table.isna().any().any():
        raise ValueError("features table has missing values")
    return table.sort_values(["subject_id", "window_start_s"], kind="stable").reset_index(drop=True)


def labels(table, task):
    """Class labels of a table for a task.

    Input: the table and task 'binary' or 'three'. Output: an int numpy array;
    binary: 1 for stress (label 2), 0 otherwise; three: the labels 1, 2, 3.
    Randomness: none.
    """
    if task == "binary":
        return (table["label"] == config.STRESS).astype(int).to_numpy()
    return table["label"].astype(int).to_numpy()
