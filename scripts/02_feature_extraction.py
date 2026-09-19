"""
02_feature_extraction.py
------------------------
STEP 3 of the project plan: turn raw wrist signals into a table of engineered
features, one row per 60-second window.

For every subject S2..S17 (S1 and S12 do not exist in WESAD):
  1. load the synchronised S{n}.pkl,
  2. slide a 60 s window across the recording in 30 s steps (50% overlap),
  3. for each window, look at the label samples it covers:
        - if the window is not pure enough (see LABEL_PURITY_THRESHOLD in
          config.py) it is discarded,
        - if its label is not baseline/stress/amusement it is discarded,
  4. compute the 40 engineered features for the window (features.py),
  5. record the features plus subject_id, label and window start time.

Everything is concatenated into one table and written to results/features.csv.

The subject_id column is essential: in 03_train_model.py we split train/test
BY SUBJECT. Without that column we could not do it.

Usage:
    python 02_feature_extraction.py
"""

import gc
import os
import pickle
import time

import numpy as np
import pandas as pd

import project_config as config
import features


def load_subject(subject):
    """Load one subject's synchronised .pkl (Python-2 pickle -> latin1)."""
    path = os.path.join(config.WESAD_DIR, subject, subject + ".pkl")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def window_label(label_segment):
    """
    Decide the label for one window.

    `label_segment` is the slice of the 700 Hz label array covering the
    window's 60 seconds. We take the value that occupies the most samples
    (the majority label) and measure its purity - the fraction of the window
    that carries that value.

    Returns (majority_label, purity).
    """
    values, counts = np.unique(label_segment, return_counts=True)
    winner = int(values[np.argmax(counts)])
    purity = float(np.max(counts)) / float(len(label_segment))
    return winner, purity


def extract_subject(subject):
    """
    Produce a list of feature-row dicts for one subject.

    Also returns a small dict of counters so the script can report how many
    windows were dropped and why - useful to show that nothing silently
    disappeared.
    """
    data = load_subject(subject)
    label = data["label"]
    wrist = data["signal"]["wrist"]

    duration_s = len(label) / float(config.FS_LABEL)

    rows = []
    counters = {"total": 0, "dropped_impure": 0,
                "dropped_wrong_class": 0, "dropped_nan": 0, "kept": 0}

    # Slide the window start from 0 s forwards in STEP_SECONDS increments,
    # stopping when a full WINDOW_SECONDS no longer fits in the recording.
    start_s = 0.0
    while start_s + config.WINDOW_SECONDS <= duration_s:
        end_s = start_s + config.WINDOW_SECONDS
        counters["total"] += 1

        # --- the window's label -------------------------------------------
        # Slice the label array using its own sampling rate (700 Hz).
        label_slice = label[int(start_s * config.FS_LABEL):
                            int(end_s * config.FS_LABEL)]
        win_label, purity = window_label(label_slice)

        if purity < config.LABEL_PURITY_THRESHOLD:
            # The window straddles a transition between two conditions.
            # Rather than guess, throw it away.
            counters["dropped_impure"] += 1
            start_s += config.STEP_SECONDS
            continue

        if win_label not in config.TARGET_LABELS:
            # Transient (0), meditation (4) or an "ignore" class (5/6/7).
            counters["dropped_wrong_class"] += 1
            start_s += config.STEP_SECONDS
            continue

        # --- the window's signals -----------------------------------------
        # Each signal is sliced with ITS OWN sampling rate, so all four slices
        # cover the same 60 seconds of wall-clock time even though they have
        # different numbers of samples.
        wrist_window = {
            "ACC":  wrist["ACC"][int(start_s * config.FS_WRIST_ACC):
                                 int(end_s * config.FS_WRIST_ACC)],
            "BVP":  wrist["BVP"][int(start_s * config.FS_WRIST_BVP):
                                 int(end_s * config.FS_WRIST_BVP)],
            "EDA":  wrist["EDA"][int(start_s * config.FS_WRIST_EDA):
                                 int(end_s * config.FS_WRIST_EDA)],
            "TEMP": wrist["TEMP"][int(start_s * config.FS_WRIST_TEMP):
                                  int(end_s * config.FS_WRIST_TEMP)],
        }

        feats = features.window_features(wrist_window)

        # A NaN here means a feature could not be computed - in practice that
        # only happens when no heartbeats were detected in the window (very
        # poor PPG contact). Drop the window instead of imputing a value.
        if any(np.isnan(v) for v in feats.values()):
            counters["dropped_nan"] += 1
            start_s += config.STEP_SECONDS
            continue

        feats["subject_id"] = subject
        feats["label"] = win_label
        feats["label_name"] = config.LABEL_NAMES[win_label]
        feats["window_start_s"] = start_s

        rows.append(feats)
        counters["kept"] += 1
        start_s += config.STEP_SECONDS

    # These pickles are about 1 GB each. Free the memory before moving to the
    # next subject, otherwise 15 of them will not fit.
    del data, label, wrist
    gc.collect()

    return rows, counters


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("Window length : {} s".format(config.WINDOW_SECONDS))
    print("Window step   : {} s ({:.0f}% overlap)".format(
        config.STEP_SECONDS,
        100.0 * (1.0 - config.STEP_SECONDS / float(config.WINDOW_SECONDS))))
    print("Purity needed : {:.0%}".format(config.LABEL_PURITY_THRESHOLD))
    print("Classes kept  : {}".format(
        ", ".join("{}={}".format(k, v) for k, v in config.LABEL_NAMES.items())))
    print()

    all_rows = []
    all_counters = []

    for subject in config.SUBJECTS:
        t0 = time.time()
        rows, counters = extract_subject(subject)
        all_rows.extend(rows)
        counters["subject_id"] = subject
        all_counters.append(counters)

        print("{:4s}  kept {:4d} windows   "
              "(dropped: {:4d} impure, {:4d} other class, {:2d} NaN)   "
              "[{:.0f}s]".format(
                  subject, counters["kept"], counters["dropped_impure"],
                  counters["dropped_wrong_class"], counters["dropped_nan"],
                  time.time() - t0))

    # Column order: identifiers first, then the 40 features in a fixed order.
    table = pd.DataFrame(all_rows)
    ordered_columns = (["subject_id", "label", "label_name", "window_start_s"]
                       + features.FEATURE_NAMES)
    table = table[ordered_columns]

    table.to_csv(config.FEATURES_CSV, index=False)

    print()
    print("Feature table: {} windows x {} features".format(
        len(table), len(features.FEATURE_NAMES)))
    print("Class balance:")
    for name, count in table["label_name"].value_counts().items():
        print("  {:11s} {:4d} windows ({:5.1f}%)".format(
            name, count, 100.0 * count / len(table)))
    print()
    print("Windows per subject:")
    print(table.groupby("subject_id")["label"].count().to_string())
    print()
    print("Written to " + config.FEATURES_CSV)


if __name__ == "__main__":
    main()
