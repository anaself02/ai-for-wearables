"""Heart-rate input resolution (Stage 2D, Amendment 5 of PRE_DECLARED.md).

R1  = the native beat-level HR series, features exactly as in results/features.csv
      (HR_slope is bpm per BEAT). Used only for the parity check.
R1s = the same beat series and the same mean/std/min/max, but HR_slope is bpm per
      SECOND (least-squares fit on the beat timestamps). This is the native
      resolution used in the interaction analysis.
R2  = the same beat series averaged into 3.0 s bins (bin mean); the same five
      statistics on the bin means, HR_slope in bpm per second (least squares on the
      bin-centre timestamps). R1s and R2 therefore define every feature identically.

Both are rebuilt here from the raw WESAD wrist BVP, one subject's pickle at a time.
R1 is computed by the project's own `heart_rate_series`/`basic_features`
(scripts/features.py), so parity with results/features.csv is by construction and
is checked, not assumed. The beat times needed for R2 come from `beat_series`, a
copy of that function's pipeline that also returns when each beat happened; a test
asserts its bpm values equal the original's.

    python -m eval.resolution build  [--wesad-dir DIR]   rebuild, write out/resolution/*.csv
    python -m eval.resolution parity [--wesad-dir DIR]   same, then compare R1 with features.csv

Only WESAD (S*/S*.pkl). Location: --wesad-dir, else the environment variable
WESAD_DIR, else data/WESAD inside wearable-ml. No personal data. See eval/README.md.
"""

import gc
import json
import os
import pickle
import sys

import numpy as np

from . import config
from .data import check_path

BIN_SECONDS = 3.0
MIN_READINGS = 10
WINDOW_SECONDS = 60
STEP_SECONDS = 30
WESAD_DIR = os.environ.get("WESAD_DIR") or os.path.join(config.ML_DIR, "data", "WESAD")
RES_DIR = os.path.join(config.OUT_DIR, "resolution")
R1S_CSV = os.path.join(RES_DIR, "windows_R1s.csv")
R2_CSV = os.path.join(RES_DIR, "windows_R2.csv")
PARITY_JSON = os.path.join(RES_DIR, "parity.json")
#: R1 must match results/features.csv to this tolerance (same code, so it is tiny)
PARITY_ATOL = 1e-9
SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
            "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
STATS = ["mean", "std", "min", "max", "slope"]


def legacy():
    """The project's own feature code (scripts/features.py, scripts/project_config.py)."""
    scripts = os.path.join(config.ML_DIR, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import features
    import project_config
    return features, project_config


def beat_series(bvp, fs=64):
    """-> (t, bpm): one value per accepted beat interval, t = interval midpoint in
    seconds from the start of the segment. Same pipeline as
    scripts/features.py:heart_rate_series (same filter, peaks, 5-beat IBI median
    filter, 40-180 bpm); that function drops the times, this keeps them."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, filtfilt, find_peaks

    _, pc = legacy()
    empty = (np.array([]), np.array([]))
    nyquist = fs / 2.0
    b, a = butter(pc.BVP_FILTER_ORDER,
                  [pc.BVP_BANDPASS_LOW_HZ / nyquist, pc.BVP_BANDPASS_HIGH_HZ / nyquist],
                  btype="band")
    if len(bvp) < 3 * (pc.BVP_FILTER_ORDER + 1):
        return empty
    filtered = filtfilt(b, a, bvp)
    sd = float(np.std(filtered))
    if sd == 0.0:
        return empty
    peaks, _ = find_peaks(filtered, distance=int(round(pc.BVP_MIN_PEAK_DISTANCE_S * fs)),
                          prominence=0.5 * sd)
    if len(peaks) < 2:
        return empty
    ibi = np.diff(peaks) / float(fs)
    if pc.IBI_MEDIAN_FILTER_BEATS > 1:
        ibi = median_filter(ibi, size=pc.IBI_MEDIAN_FILTER_BEATS, mode="nearest")
    bpm = 60.0 / ibi
    t = 0.5 * (peaks[:-1] + peaks[1:]) / float(fs)
    keep = (bpm >= pc.HR_MIN_BPM) & (bpm <= pc.HR_MAX_BPM)
    return t[keep], bpm[keep]


def resample_bin_mean(t, bpm, length=WINDOW_SECONDS, bin_s=BIN_SECONDS):
    """Average the beat series into consecutive bins of `bin_s` seconds starting at
    t=0. -> (centres, means) for the NON-EMPTY bins only; empty bins are skipped,
    never interpolated. A beat at time t falls in bin floor(t / bin_s)."""
    t = np.asarray(t, dtype=float)
    bpm = np.asarray(bpm, dtype=float)
    n_bins = int(round(length / bin_s))
    if len(t) == 0:
        return np.array([]), np.array([])
    index = np.clip(np.floor(t / bin_s).astype(int), 0, n_bins - 1)
    centres, means = [], []
    for k in range(n_bins):
        in_bin = index == k
        if in_bin.any():
            centres.append((k + 0.5) * bin_s)
            means.append(float(bpm[in_bin].mean()))
    return np.array(centres), np.array(means)


def slope_per_second(times, values):
    """Least-squares slope of values against their timestamps in seconds (bpm per
    second). Fewer than 2 points -> 0.0, as the original basic_features does."""
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(times, dtype=float), np.asarray(values, dtype=float), 1)[0])


def per_second_features(times, values):
    """The five statistics with HR_slope in bpm per second on the actual timestamps.
    Used for R1s (times = beat times, values = bpm) and R2 (bin centres, bin means)."""
    values = np.asarray(values, dtype=float)
    return {"HR_mean": float(np.mean(values)), "HR_std": float(np.std(values)),
            "HR_min": float(np.min(values)), "HR_max": float(np.max(values)),
            "HR_slope": slope_per_second(times, values)}


def r2_features(centres, means, min_readings=MIN_READINGS):
    """Five statistics of the binned series, or None if fewer than `min_readings`
    bins hold a beat. HR_slope: bpm per second on the bin-centre timestamps."""
    if len(means) < min_readings:
        return None
    return per_second_features(centres, means)


def _label_of(segment):
    values, counts = np.unique(segment, return_counts=True)
    return int(values[np.argmax(counts)]), float(counts.max()) / float(len(segment))


def build_subject(subject, wesad_dir=WESAD_DIR):
    """-> list of dicts, one per window the original extraction keeps (same loop,
    same purity rule, same NaN rule), with R1 features (original code) and R2
    features (None if < MIN_READINGS bins)."""
    features, pc = legacy()
    path = check_path(os.path.join(wesad_dir, subject, subject + ".pkl"))
    with open(path, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    label, bvp_all = data["label"], data["signal"]["wrist"]["BVP"]
    duration = len(label) / float(pc.FS_LABEL)
    rows, start = [], 0.0
    while start + WINDOW_SECONDS <= duration:
        end = start + WINDOW_SECONDS
        win_label, purity = _label_of(label[int(start * pc.FS_LABEL):int(end * pc.FS_LABEL)])
        if purity >= pc.LABEL_PURITY_THRESHOLD and win_label in pc.TARGET_LABELS:
            bvp = bvp_all[int(start * pc.FS_WRIST_BVP):int(end * pc.FS_WRIST_BVP)]
            hr = features.heart_rate_series(bvp)
            r1 = features.basic_features(hr, fs=1.0)
            if not any(np.isnan(v) for v in r1.values()):
                t, bpm = beat_series(np.asarray(bvp, dtype=float).ravel())
                if not np.array_equal(bpm, hr):
                    raise RuntimeError("beat_series differs from heart_rate_series in {} "
                                       "at {} s".format(subject, start))
                centres, means = resample_bin_mean(t, bpm)
                rows.append({"subject_id": subject, "label": win_label,
                             "label_name": pc.LABEL_NAMES[win_label], "window_start_s": start,
                             "r1": {"HR_" + k: v for k, v in r1.items()},
                             "r1s": per_second_features(t, bpm),
                             "r2": r2_features(centres, means), "n_beats": int(len(bpm)),
                             "n_bins": int(len(means))})
        start += STEP_SECONDS
    del data, label, bvp_all
    gc.collect()
    return rows


def build_all(subjects=SUBJECTS, wesad_dir=WESAD_DIR, say=print):
    """One subject's pickle in memory at a time."""
    rows = []
    for subject in subjects:
        got = build_subject(subject, wesad_dir)
        rows += got
        say("{}: {} windows, {} with >= {} bins".format(
            subject, len(got), sum(r["r2"] is not None for r in got), MIN_READINGS))
    return rows


def tables(rows):
    """-> (R1s table, R2 table, dropped): row-aligned DataFrames over the COMMON window
    set (windows valid at R2), the same HR columns as results/features.csv. The R1s
    table also carries `HR_slope_perbeat`, the original R1 slope, for description only.
    `dropped` lists windows that failed the >= MIN_READINGS rule."""
    import pandas as pd

    keep = [r for r in rows if r["r2"] is not None]
    dropped = [{"subject_id": r["subject_id"], "window_start_s": r["window_start_s"],
                "label": r["label"], "n_bins": r["n_bins"]} for r in rows if r["r2"] is None]
    meta = ["subject_id", "label", "label_name", "window_start_s"]

    def frame(key):
        data = [{**{m: r[m] for m in meta}, **r[key], "n_bins": r["n_bins"], "n_beats": r["n_beats"],
                 **({"HR_slope_perbeat": r["r1"]["HR_slope"]} if key == "r1s" else {})}
                for r in keep]
        return pd.DataFrame(data).sort_values(["subject_id", "window_start_s"],
                                              kind="stable").reset_index(drop=True)
    return frame("r1s"), frame("r2"), dropped


def slope_comparison(r1s_table):
    """Descriptive: how the per-second slope (R1s) differs from the per-beat slope of
    Amendment 4's R1, on the same windows. No model is involved."""
    import pandas as pd

    s = r1s_table["HR_slope"].to_numpy()
    b = r1s_table["HR_slope_perbeat"].to_numpy()
    return {"n_windows": int(len(s)),
            "pearson_r": float(np.corrcoef(s, b)[0, 1]),
            "spearman_rho": float(pd.Series(s).corr(pd.Series(b), method="spearman")),
            "sd_per_second": float(np.std(s, ddof=1)), "sd_per_beat": float(np.std(b, ddof=1)),
            "ratio_of_sd": float(np.std(s, ddof=1) / np.std(b, ddof=1)),
            "ratio_of_mean_abs": float(np.mean(np.abs(s)) / np.mean(np.abs(b))),
            "ols_coefficient_s_on_b": float(np.polyfit(b, s, 1)[0]),
            "windows_with_opposite_sign": int(np.sum(np.sign(s) * np.sign(b) < 0)),
            "median_beats_per_second": float(np.median(r1s_table["n_beats"] / 60.0))}


def parity(rows, features_csv=config.FEATURES, atol=PARITY_ATOL):
    """Compare ALL rebuilt R1 windows (before the R2 filter) with results/features.csv.
    -> dict; ok is True only if the window sets are identical and every HR value is
    within `atol`."""
    import pandas as pd

    ref = pd.read_csv(check_path(features_csv))
    meta = ["subject_id", "label", "window_start_s"]
    mine = pd.DataFrame([{**{m: r[m] for m in meta}, **r["r1"]} for r in rows])
    key = lambda df: set(zip(df["subject_id"], df["window_start_s"].round(6)))
    only_ref, only_mine = key(ref) - key(mine), key(mine) - key(ref)
    a = ref.sort_values(["subject_id", "window_start_s"]).reset_index(drop=True)
    b = mine.sort_values(["subject_id", "window_start_s"]).reset_index(drop=True)
    same_set = not only_ref and not only_mine and len(a) == len(b)
    out = {"n_reference": int(len(ref)), "n_rebuilt": int(len(mine)),
           "windows_only_in_features_csv": len(only_ref), "windows_only_rebuilt": len(only_mine),
           "same_window_set": bool(same_set), "atol": atol}
    if same_set:
        out["labels_equal"] = bool((a["label"].to_numpy() == b["label"].to_numpy()).all())
        diffs = {f: float(np.max(np.abs(a[f].to_numpy() - b[f].to_numpy())))
                 for f in config.HR_FEATURES}
        out["max_abs_difference"] = diffs
        out["ok"] = bool(out["labels_equal"] and max(diffs.values()) <= atol)
    else:
        out["ok"] = False
    return out


def write_all(say=print, wesad_dir=None):
    """Rebuild R1, R1s and R2 from raw WESAD, check parity and write the tables.

    Inputs: a progress callback and the WESAD folder. Output: the parity dict;
    files out/resolution/windows_R1s.csv, windows_R2.csv and parity.json.
    Randomness: none.
    """
    rows = build_all(wesad_dir=wesad_dir or WESAD_DIR, say=say)
    r1s, r2, dropped = tables(rows)
    result = parity(rows)
    os.makedirs(RES_DIR, exist_ok=True)
    r1s.to_csv(R1S_CSV, index=False)
    r2.to_csv(R2_CSV, index=False)
    result.update({"n_windows_valid_at_R2": int(len(r1s)), "n_dropped_at_R2": dropped,
                   "min_readings": MIN_READINGS, "bin_seconds": BIN_SECONDS,
                   "slope_comparison_R1s_vs_R1": slope_comparison(r1s)})
    with open(PARITY_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    return result


def load_tables():
    """The cached, row-aligned R1s/R2 tables (refuses if R1 parity was not proven)."""
    import pandas as pd

    with open(PARITY_JSON, encoding="utf-8") as handle:
        if not json.load(handle)["ok"]:
            raise SystemExit("R1 does not match results/features.csv: stop (out/resolution/parity.json).")
    r1, r2 = pd.read_csv(R1S_CSV), pd.read_csv(R2_CSV)
    assert (r1[["subject_id", "window_start_s"]].values == r2[["subject_id", "window_start_s"]].values).all()
    return r1, r2


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m eval.resolution")
    ap.add_argument("action", choices=["build", "parity"])
    ap.add_argument("--wesad-dir", default=None,
                    help="folder holding S2/S2.pkl ...; default: $WESAD_DIR, else data/WESAD")
    summary = write_all(say=lambda m: print(m, flush=True), wesad_dir=ap.parse_args().wesad_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "n_dropped_at_R2"}, indent=1))
    print("dropped at R2:", len(summary["n_dropped_at_R2"]))
    sys.exit(0 if summary["ok"] else 1)
