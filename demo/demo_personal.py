"""
demo_personal.py
----------------
The personal-data half of the demo: read one local day of Fitbit heart rate
from a Google Takeout folder, window it exactly like script 05 does, measure
how far it sits from the WESAD training distribution, and report day-level
statistics.

Hard rules, enforced in code and by tests:

  * This module imports NO scikit-learn and never learns anything. The caller
    passes in already-trained models and this module only asks them to predict.
  * The only Takeout files ever opened are named heart_rate_YYYY-MM-DD.csv
    (HR_FILE_RE). The profile, device, location and security files in the same
    export are never read.
  * Nothing is written to disk.
  * Only day-level statistics are meant to be shown. Callers must not print raw
    rows or per-reading times.
"""

import os
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import features  # scripts/features.py: the same feature code as training

WINDOW_SECONDS = 60
STEP_SECONDS = 30
MIN_SAMPLES_PER_WINDOW = 10
WAKING_START_HOUR = 8
WAKING_END_HOUR = 23

#: Pre-declared display rules (see PRE_DECLARED.md).
OOD_FLAG_SHARE = 0.25
DOMINANT_CLASS_SHARE = 0.60

HR_FILE_RE = re.compile(r"^heart_rate_(\d{4}-\d{2}-\d{2})\.csv$")
EXPECTED_COLUMNS = ["timestamp", "beats per minute", "data source"]


class TakeoutError(Exception):
    """Something needed from the Takeout folder is missing or unreadable."""


# ---------------------------------------------------------------------------
# finding and reading the heart-rate files
# ---------------------------------------------------------------------------

def list_hr_files(folder):
    """{date: path} for heart_rate_YYYY-MM-DD.csv files. Opens nothing."""
    if not os.path.isdir(folder):
        raise TakeoutError("Takeout heart-rate folder not found: " + str(folder))
    found = {}
    with os.scandir(folder) as entries:
        for entry in entries:
            match = HR_FILE_RE.match(entry.name)
            if match and entry.is_file():
                found[date.fromisoformat(match.group(1))] = entry.path
    return found


def read_hr_file(path):
    """Read one heart-rate CSV into (epoch_seconds, bpm, source) columns."""
    try:
        table = pd.read_csv(path, usecols=EXPECTED_COLUMNS)
    except ValueError as exc:
        raise TakeoutError(
            "{} does not have the expected columns {}: {}".format(
                os.path.basename(path), EXPECTED_COLUMNS, exc))

    stamps = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    keep = stamps.notna() & table["beats per minute"].notna()
    stamps = stamps[keep]
    epoch = (stamps.dt.tz_convert("UTC").dt.tz_localize(None)
             .values.astype("datetime64[s]").astype("int64"))
    return pd.DataFrame({
        "epoch": epoch,
        "bpm": table.loc[keep, "beats per minute"].astype(float).values,
        "source": table.loc[keep, "data source"].astype(str).values,
    })


def local_day_bounds(day, tz_name):
    """(start_epoch, end_epoch) in UTC seconds for one LOCAL civil day.

    Computed from real midnights, so a daylight-saving day is 23 or 25 hours
    long instead of being assumed to be 24.
    """
    tz = ZoneInfo(tz_name)
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    nxt = day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz)
    return int(start.timestamp()), int(end.timestamp())


def needed_utc_dates(day, tz_name):
    """The UTC dates whose files can contain readings of this local day."""
    start, end = local_day_bounds(day, tz_name)
    first = datetime.fromtimestamp(start, timezone.utc).date()
    last = datetime.fromtimestamp(end - 1, timezone.utc).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def load_local_day(folder, day, tz_name):
    """All readings of one local day, from the one or two UTC files holding it.

    Returns (samples DataFrame, number of files opened).
    """
    files = list_hr_files(folder)
    needed = needed_utc_dates(day, tz_name)
    missing = [d for d in needed if d not in files]
    if missing:
        raise TakeoutError(
            "Local day {} needs {} heart-rate file(s) that are not in the "
            "folder ({} missing).".format(day, len(needed), len(missing)))

    frames = [read_hr_file(files[d]) for d in needed]
    samples = pd.concat(frames, ignore_index=True)
    start, end = local_day_bounds(day, tz_name)
    samples = samples[(samples["epoch"] >= start) & (samples["epoch"] < end)]
    samples = (samples.drop_duplicates("epoch")
               .sort_values("epoch").reset_index(drop=True))
    return samples, len(needed)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def window_grid(day, tz_name):
    """Window start times (UTC epoch s), anchored to local midnight."""
    start, end = local_day_bounds(day, tz_name)
    return np.arange(start, end - WINDOW_SECONDS + 1, STEP_SECONDS)


def build_windows(samples, day, tz_name):
    """HR_* features for every 60 s window with at least 10 readings.

    Same grid and the same feature function as scripts/05 (windows anchored to
    local midnight, 30 s step, features.basic_features with fs=1.0).
    """
    epochs = samples["epoch"].values
    bpm = samples["bpm"].values
    starts = window_grid(day, tz_name)
    lo = np.searchsorted(epochs, starts, side="left")
    hi = np.searchsorted(epochs, starts + WINDOW_SECONDS, side="left")

    rows = []
    for start, a, b in zip(starts, lo, hi):
        if b - a < MIN_SAMPLES_PER_WINDOW:
            continue
        stats = features.basic_features(bpm[a:b], fs=1.0)
        row = {"HR_" + name: value for name, value in stats.items()}
        row["start_epoch"] = int(start)
        row["n_samples"] = int(b - a)
        rows.append(row)

    table = pd.DataFrame(rows)
    if len(table):
        hours = (pd.to_datetime(table["start_epoch"], unit="s", utc=True)
                 .dt.tz_convert(tz_name).dt.hour)
        table["waking"] = (hours >= WAKING_START_HOUR) & (hours < WAKING_END_HOUR)
    return table


def count_windows(samples, day, tz_name):
    """How many windows of this day would be scorable."""
    epochs = samples["epoch"].values
    starts = window_grid(day, tz_name)
    lo = np.searchsorted(epochs, starts, side="left")
    hi = np.searchsorted(epochs, starts + WINDOW_SECONDS, side="left")
    return int(np.sum((hi - lo) >= MIN_SAMPLES_PER_WINDOW))


def choose_day(folder, tz_name, min_windows=1000, max_tries=40):
    """Newest local day whose files all exist and that has enough windows.

    Returns (day, samples, files_opened) or raises TakeoutError.
    """
    files = list_hr_files(folder)
    if not files:
        raise TakeoutError("No heart_rate_YYYY-MM-DD.csv files in " + str(folder))

    tries = 0
    for candidate in sorted(files, reverse=True):
        if tries >= max_tries:
            break
        if any(d not in files for d in needed_utc_dates(candidate, tz_name)):
            continue
        tries += 1
        samples, opened = load_local_day(folder, candidate, tz_name)
        if count_windows(samples, candidate, tz_name) >= min_windows:
            return candidate, samples, opened
    raise TakeoutError(
        "No local day with at least {} scorable windows found.".format(min_windows))


# ---------------------------------------------------------------------------
# day-level statistics
# ---------------------------------------------------------------------------

def reading_spacing(samples):
    """Seconds between consecutive readings: median, p05, p95, gaps > 5 min."""
    gaps = np.diff(samples["epoch"].values)
    if len(gaps) == 0:
        return {"median": float("nan"), "p05": float("nan"),
                "p95": float("nan"), "gaps_over_5min": 0}
    return {
        "median": float(np.median(gaps)),
        "p05": float(np.percentile(gaps, 5)),
        "p95": float(np.percentile(gaps, 95)),
        "gaps_over_5min": int(np.sum(gaps > 300)),
    }


def device_names(samples):
    """{data source: reading count}, as written in the file's own column."""
    return {str(k): int(v) for k, v in samples["source"].value_counts().items()}


def ood_report(windows, distribution, feature_names):
    """Share of windows outside the WESAD training range, per feature.

    Same two measures as script 05 and the dashboard's arousal card, against
    the training statistics saved next to the model.
    """
    stats = distribution["features"]
    report = []
    for name in feature_names:
        train = stats[name]
        values = windows[name].values
        outside = float(np.mean((values < train["min"]) | (values > train["max"])))
        shift = (float((np.median(values) - train["mean"]) / train["std"])
                 if train["std"] else 0.0)
        report.append({"feature": name, "share_outside": outside,
                       "median_shift_sd": shift})
    return report


def class_shares(predictions, names, mask=None):
    """{class name: share of windows}."""
    predictions = np.asarray(predictions)
    if mask is not None:
        predictions = predictions[np.asarray(mask)]
    total = len(predictions)
    if total == 0:
        return {name: float("nan") for name in names.values()}
    return {name: float(np.mean(predictions == label))
            for label, name in names.items()}


def predict_only(model, windows, feature_names):
    """Ask an already-trained model for predictions. Learns nothing."""
    return model.predict(windows[feature_names])


def notes_for(shares, worst_ood_share, dominance=True):
    """Plain-language notes the demo prints about a scored day.

    Thresholds are the pre-declared ones in PRE_DECLARED.md.

    dominance=False drops the "one class takes >= 60%" flag. That is the rule
    for the binary task (Amendment 1): non-stress is 70% of the training prior,
    so class dominance says nothing there. The out-of-distribution flag, which
    depends on the input features and not on the predictions, always applies.
    """
    notes = []
    ood = worst_ood_share >= OOD_FLAG_SHARE
    top_name, top_share = max(shares.items(), key=lambda kv: kv[1])
    degenerate = dominance and top_share >= DOMINANT_CLASS_SHARE

    if ood:
        notes.append(
            "This day is OUT OF DISTRIBUTION: at least {:.0%} of its windows "
            "fall outside the WESAD training range on one heart-rate feature."
            .format(OOD_FLAG_SHARE))
    if degenerate:
        notes.append(
            "Predictions are DEGENERATE: one class ({}) takes {:.0%} of "
            "windows.".format(top_name, top_share))
        if top_name == "amusement":
            notes.append(
                "The model cannot separate amusement from baseline using "
                "heart rate alone, and this day is out of distribution.")
    if ood or degenerate:
        notes.append("The class shares are therefore NOT interpretable.")
    return notes
