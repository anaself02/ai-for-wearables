"""Outer leave-one-subject-out folds and subject-grouped inner folds."""

import json
import random

from . import config


def subject_key(subject):
    """Sort key of a subject id: 'S12' -> 12. Randomness: none."""
    return int(subject[1:])


def outer_folds(subjects):
    """[(test_subject, [train subjects])], one fold per subject, in subject order."""
    subjects = sorted(subjects, key=subject_key)
    return [(s, [t for t in subjects if t != s]) for s in subjects]


def inner_folds(train_subjects, seed, k=config.INNER_FOLDS):
    """k groups of validation subjects (shuffled by seed); every subject in exactly one."""
    shuffled = sorted(train_subjects, key=subject_key)
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[i::k], key=subject_key) for i in range(k)]


def inner_splits(groups, train_index, val_groups):
    """(fit rows, validation rows) per inner fold, as positions within train_index."""
    import numpy as np

    train_groups = np.asarray(groups)[train_index]
    out = []
    for held in val_groups:
        val = np.flatnonzero(np.isin(train_groups, held))
        fit = np.flatnonzero(~np.isin(train_groups, held))
        out.append((fit, val))
    return out


def all_folds(subjects, seeds=config.SEEDS, k=config.INNER_FOLDS):
    """Every split of the evaluation as one dict.

    Inputs: subject ids, seeds (default config.SEEDS 0-4), number of inner folds
    (default 7). Output: {'outer': [{'test', 'train'}], 'inner': {seed: {test
    subject: [validation groups]}}, 'inner_k': k}. Seeds: each seed drives
    inner_folds (random.Random(seed)); the outer folds are not random.
    """
    return {
        "outer": [{"test": test, "train": train} for test, train in outer_folds(subjects)],
        "inner": {str(seed): {test: inner_folds(train, seed, k)
                              for test, train in outer_folds(subjects)} for seed in seeds},
        "inner_k": k,
    }


def write_folds(subjects, path=config.FOLDS_FILE, seeds=config.SEEDS):
    """Write all_folds(subjects, seeds) as JSON (eval/folds.json by default).

    Inputs: subject ids, output path, seeds. Output: the path. Seeds: as all_folds.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(all_folds(subjects, seeds), handle, indent=1)
    return path
