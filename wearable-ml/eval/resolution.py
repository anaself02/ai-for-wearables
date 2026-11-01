"""Rebuild the five HR features at two input resolutions, on identical windows.

This is the input side of the interaction analysis (model x HR resolution).

    python -m eval.resolution build                  # needs data/WESAD (17 GB)
    python -m eval.resolution build --wesad D:/WESAD

Levels
------
native  one heart-rate value per detected beat, exactly as results/features.csv
        (scripts/features.heart_rate_series: band-pass, peaks, 5-beat IBI median
        filter, 40-180 bpm). `build` checks that this level reproduces the HR
        columns of results/features.csv window by window.
3s      the same beat series averaged in 3 s bins anchored at the window start
        (20 bins per 60 s window, empty bins dropped). This SIMULATES a consumer
        device's ~3 s reading spacing; it is not Fitbit's own (undisclosed)
        smoothing, so a gap to real Fitbit data is expected to remain
        (PRE_DECLARED.md, "Input resolution").

Features at both levels: HR_mean, HR_std (population SD), HR_min, HR_max,
HR_slope. HR_slope is the least-squares slope per READING (index as the time
axis, as scripts/features.py does with fs=1): per beat at native, per 3 s bin
at 3s. Each resolution is trained and tested on its own, so the different unit
never crosses a split.

Pairing rule: a window is kept only if it is valid at BOTH levels (native: at
least one beat, as in 02_feature_extraction.py; 3s: at least MIN_BINNED_READINGS
non-empty bins, the same "at least 10 readings" rule used for personal data).
Both output tables therefore hold exactly the same windows in the same order,
so every model x resolution cell is compared on the same rows and the
interaction contrast can be paired per subject.

Also runs the label-to-sampling alignment check promised in PRE_DECLARED.md
(Update 2): signal lengths against the label length at the documented rates,
and where each condition starts and ends, one subject at a time.
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks

from . import config as eval_config
from .data import check_path

SCRIPTS_DIR = os.path.join(eval_config.ML_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import project_config as pconfig  # noqa: E402  (scripts/ is not a package)

LEVELS = ("native", "3s")
BIN_SECONDS = 3.0
MIN_BINNED_READINGS = 10
ALIGN_TOLERANCE_S = 1.0
OUT_DIR = os.path.join(eval_config.EVAL_DIR, "results", "resolution")
ID_COLUMNS = ["subject_id", "label", "label_name", "window_start_s"]


def table_path(level, out_dir=OUT_DIR):
    return os.path.join(out_dir, "features_{}.csv".format(level))


# ---------------------------------------------------------------------------
# Beat series with times (same steps as scripts/features.heart_rate_series)
# ---------------------------------------------------------------------------

def beat_series(bvp, fs=pconfig.FS_WRIST_BVP):
    """(times_s, bpm) for one BVP segment; times are seconds from segment start.

    Identical processing to scripts/features.heart_rate_series, but keeps the
    time of each value (the later peak of its inter-beat interval) so the
    series can be binned. A test pins `bpm` to heart_rate_series exactly.
    """
    bvp = np.asarray(bvp, dtype=float).ravel()
    empty = (np.array([]), np.array([]))
    nyquist = fs / 2.0
    b, a = butter(pconfig.BVP_FILTER_ORDER,
                  [pconfig.BVP_BANDPASS_LOW_HZ / nyquist, pconfig.BVP_BANDPASS_HIGH_HZ / nyquist],
                  btype="band")
    if len(bvp) < 3 * (pconfig.BVP_FILTER_ORDER + 1):
        return empty
    filtered = filtfilt(b, a, bvp)
    sd = float(np.std(filtered))
    if sd == 0.0:
        return empty
    peaks, _ = find_peaks(filtered, distance=int(round(pconfig.BVP_MIN_PEAK_DISTANCE_S * fs)),
                          prominence=0.5 * sd)
    if len(peaks) < 2:
        return empty
    ibi = np.diff(peaks) / float(fs)
    if pconfig.IBI_MEDIAN_FILTER_BEATS > 1:
        ibi = median_filter(ibi, size=pconfig.IBI_MEDIAN_FILTER_BEATS, mode="nearest")
    bpm = 60.0 / ibi
    times = peaks[1:] / float(fs)
    keep = (bpm >= pconfig.HR_MIN_BPM) & (bpm <= pconfig.HR_MAX_BPM)
    return times[keep], bpm[keep]


def bin_series(times, bpm, window_s, bin_s=BIN_SECONDS):
    """Mean bpm per bin_s bin, bins anchored at 0; empty bins dropped."""
    times, bpm = np.asarray(times, dtype=float), np.asarray(bpm, dtype=float)
    n_bins = int(round(window_s / bin_s))
    index = np.minimum((times // bin_s).astype(int), n_bins - 1)
    sums = np.bincount(index, weights=bpm, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins)
    filled = counts > 0
    return sums[filled] / counts[filled]


def hr_features(values):
    """The five HR features; slope per reading (index axis). NaN dict if empty."""
    x = np.asarray(values, dtype=float).ravel()
    if len(x) == 0:
        return dict.fromkeys(eval_config.HR_FEATURES, np.nan)
    slope = float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0]) if len(x) > 1 else 0.0
    return {"HR_mean": float(np.mean(x)), "HR_std": float(np.std(x)),
            "HR_min": float(np.min(x)), "HR_max": float(np.max(x)), "HR_slope": slope}


def window_levels(bvp_window, window_s):
    """{level: features or None} for one window's BVP."""
    times, bpm = beat_series(bvp_window)
    out = {"native": hr_features(bpm) if len(bpm) else None}
    binned = bin_series(times, bpm, window_s) if len(bpm) else np.array([])
    out["3s"] = hr_features(binned) if len(binned) >= MIN_BINNED_READINGS else None
    return out


# ---------------------------------------------------------------------------
# Windowing (same rule as scripts/02_feature_extraction.py)
# ---------------------------------------------------------------------------

def load_subject(wesad_dir, subject):
    path = check_path(os.path.join(wesad_dir, subject, subject + ".pkl"))
    with open(path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def alignment(label, bvp):
    """Signal length vs label length at the documented rates, and condition spans."""
    label_s = len(label) / float(pconfig.FS_LABEL)
    bvp_s = len(bvp) / float(pconfig.FS_WRIST_BVP)
    spans = {}
    for code, name in pconfig.LABEL_NAMES.items():
        where = np.flatnonzero(np.asarray(label) == code)
        if len(where):
            spans[name] = [round(where[0] / pconfig.FS_LABEL, 1),
                           round((where[-1] + 1) / pconfig.FS_LABEL, 1)]
    return {"label_s": round(label_s, 2), "bvp_s": round(bvp_s, 2),
            "diff_s": round(bvp_s - label_s, 2),
            "aligned": bool(abs(bvp_s - label_s) <= ALIGN_TOLERANCE_S), "spans_s": spans}


def extract_subject(data, subject, window_s=pconfig.WINDOW_SECONDS, step_s=pconfig.STEP_SECONDS):
    """-> ({level: [rows]}, counters, alignment). Rows are paired across levels."""
    label = np.asarray(data["label"]).ravel()
    bvp = np.asarray(data["signal"]["wrist"]["BVP"]).ravel()
    check = alignment(label, bvp)
    duration_s = len(label) / float(pconfig.FS_LABEL)
    rows = {level: [] for level in LEVELS}
    counters = {"total": 0, "impure_or_other_class": 0, "no_beats": 0,
                "too_few_binned": 0, "kept": 0}
    start = 0.0
    while start + window_s <= duration_s:
        end = start + window_s
        counters["total"] += 1
        segment = label[int(start * pconfig.FS_LABEL):int(end * pconfig.FS_LABEL)]
        values, counts = np.unique(segment, return_counts=True)
        winner = int(values[np.argmax(counts)])
        purity = counts.max() / float(len(segment))
        if purity < pconfig.LABEL_PURITY_THRESHOLD or winner not in pconfig.TARGET_LABELS:
            counters["impure_or_other_class"] += 1
            start += step_s
            continue
        levels = window_levels(
            bvp[int(start * pconfig.FS_WRIST_BVP):int(end * pconfig.FS_WRIST_BVP)], window_s)
        if levels["native"] is None:
            counters["no_beats"] += 1
        elif levels["3s"] is None:
            counters["too_few_binned"] += 1
        else:
            ident = {"subject_id": subject, "label": winner,
                     "label_name": pconfig.LABEL_NAMES[winner], "window_start_s": start}
            for level in LEVELS:
                rows[level].append({**ident, **levels[level]})
            counters["kept"] += 1
        start += step_s
    return rows, counters, check


# ---------------------------------------------------------------------------
# Build and checks
# ---------------------------------------------------------------------------

def compare_with_frozen(native, frozen):
    """Max |difference| of the HR columns on windows present in both tables."""
    import pandas as pd
    keys = ["subject_id", "window_start_s"]
    merged = pd.merge(native, frozen[keys + ["label"] + eval_config.HR_FEATURES], on=keys,
                      suffixes=("", "_frozen"))
    out = {"windows_native": int(len(native)), "windows_frozen": int(len(frozen)),
           "windows_matched": int(len(merged)),
           "label_mismatches": int((merged["label"] != merged["label_frozen"]).sum())}
    for col in eval_config.HR_FEATURES:
        diff = (merged[col] - merged[col + "_frozen"]).abs()
        out["max_abs_diff_" + col] = float(diff.max()) if len(diff) else None
    return out


def build(wesad_dir, subjects=None, out_dir=OUT_DIR, frozen_path=eval_config.FEATURES, log=print):
    import pandas as pd
    subjects = subjects or pconfig.SUBJECTS
    os.makedirs(out_dir, exist_ok=True)
    all_rows = {level: [] for level in LEVELS}
    report = {"bin_seconds": BIN_SECONDS, "min_binned_readings": MIN_BINNED_READINGS,
              "window_s": pconfig.WINDOW_SECONDS, "step_s": pconfig.STEP_SECONDS,
              "subjects": {}}
    for subject in subjects:
        started = time.time()
        data = load_subject(wesad_dir, subject)
        rows, counters, check = extract_subject(data, subject)
        del data
        gc.collect()
        for level in LEVELS:
            all_rows[level] += rows[level]
        report["subjects"][subject] = {"counters": counters, "alignment": check}
        log("{:4s} kept {:4d}  (impure/other {:4d}, no beats {:2d}, <{} bins {:2d})  "
            "bvp-label {:+.2f} s {}  [{:.0f} s]".format(
                subject, counters["kept"], counters["impure_or_other_class"],
                counters["no_beats"], MIN_BINNED_READINGS, counters["too_few_binned"],
                check["diff_s"], "ok" if check["aligned"] else "MISALIGNED",
                time.time() - started))
    tables = {}
    for level in LEVELS:
        table = pd.DataFrame(all_rows[level], columns=ID_COLUMNS + eval_config.HR_FEATURES)
        table.to_csv(table_path(level, out_dir), index=False)
        tables[level] = table
    assert tables["native"][ID_COLUMNS].equals(tables["3s"][ID_COLUMNS]), "levels not paired"
    report["windows"] = int(len(tables["native"]))
    report["class_counts"] = {k: int(v) for k, v in
                              tables["native"]["label_name"].value_counts().items()}
    report["feature_means"] = {level: {c: float(tables[level][c].mean())
                                       for c in eval_config.HR_FEATURES} for level in LEVELS}
    if frozen_path and os.path.exists(frozen_path):
        report["vs_results_features_csv"] = compare_with_frozen(
            tables["native"], pd.read_csv(check_path(frozen_path)))
    report["all_aligned"] = all(s["alignment"]["aligned"] for s in report["subjects"].values())
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    return tables, report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.resolution")
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--wesad", default=pconfig.WESAD_DIR)
    parser.add_argument("--subjects", default=None, help="comma list, default all 15")
    args = parser.parse_args(argv)
    subjects = args.subjects.split(",") if args.subjects else None
    tables, report = build(args.wesad, subjects)
    print("\n{} paired windows per level; classes {}".format(report["windows"],
                                                           report["class_counts"]))
    print("all subjects aligned:", report["all_aligned"])
    cmp = report.get("vs_results_features_csv")
    if cmp:
        print("native vs results/features.csv: {windows_matched} of {windows_frozen} frozen "
              "windows matched, {label_mismatches} label mismatches".format(**cmp))
        for col in eval_config.HR_FEATURES:
            print("  max |diff| {:9s} {:.3g}".format(col, cmp["max_abs_diff_" + col]))
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
