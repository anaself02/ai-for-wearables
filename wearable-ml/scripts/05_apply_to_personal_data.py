"""
05_apply_to_personal_data.py
----------------------------
PHASE 3: apply the heart-rate-only model to real heart-rate data from my own
consumer wearable (Fitbit Air, read through the Google Health API).

READ THIS BEFORE READING ANY OUTPUT OF THIS SCRIPT
--------------------------------------------------
The model used here was trained on WESAD: fifteen people in a laboratory,
where "stress" means the Trier Social Stress Test (public speaking plus mental
arithmetic in front of a panel) and "amusement" means watching funny video
clips, each lasting a few minutes while sitting still.

Running it on ordinary daily heart-rate data is a DEMONSTRATION THAT THE
PIPELINE WORKS END TO END. It is not a validated real-world stress detector,
and its output is not evidence that I was or was not stressed on any given
day. Specifically:

  * The model has no "none of the above" class. Every window it is given must
    come back as baseline, stress or amusement, whether or not any of those
    describe what was actually happening.
  * Nothing here is ground-truthed. There are no labels for my own days, so
    there is no accuracy number to report and none is claimed.
  * Real-world heart rate moves for reasons WESAD never contained: climbing
    stairs, coffee, a cold room, illness, poor sleep. The model has no way to
    tell those apart from psychological stress.
  * The sensors are not the same. WESAD heart rate is beat-to-beat, from
    our own peak detection on a 64 Hz PPG signal; the watch reports an
    already-smoothed bpm every ~3 s from its own undisclosed algorithm.
    Pass 2 measures how far apart the two distributions end up.

Any claim stronger than "the pipeline runs and produces plausible-looking
output" is not supported by this script.


THE BRIDGE TO THE DJANGO DASHBOARD
----------------------------------
My Google OAuth credentials, token-refresh logic and Google Health API client
already exist in the separate Django project at config.DASHBOARD_DIR. Rather
than copy any of that here - which would mean two copies of client-secret
handling to keep in sync - this script adds that project to sys.path and calls
django.setup(). From then on we can import and use its code directly:

    accounts.models.GoogleCredential   the stored, encrypted tokens
    health.client.GoogleHealthClient   the API client (refreshes tokens itself)
    health.datatypes                   the data-type registry
    health.cache.fetch_intraday        a cached single-day fetch

No client id, client secret or token is read, written or printed by this file.
Importing the dashboard's settings module also runs its load_dotenv() and its
force_ipv4() call, so we inherit its configuration and its IPv6 workaround.

Side effect to be aware of: when the API client refreshes an access token it
writes the new token back into the dashboard's db.sqlite3. Stop the
dashboard's runserver before running this, or SQLite may report
"database is locked".


PASSES
------
Pass 1 (`inspect`) - implemented here. Fetches raw intraday heart-rate samples
and reports what is actually in them: how many points per day, how far apart
they really are, where the gaps are. No modelling, no predictions. The window
size for Pass 2 is chosen from these findings rather than assumed.

Pass 2 (`predict`) - windows the samples Pass 1 saved into 60 s windows,
compares their HR features against the WESAD training distribution, and only
then runs the heart-rate-only model. Reads no API: it works from the CSV Pass
1 wrote, so the prediction step is reproducible offline.

Usage:
    python 05_apply_to_personal_data.py inspect
    python 05_apply_to_personal_data.py inspect --days 21
    python 05_apply_to_personal_data.py inspect --user myusername
    python 05_apply_to_personal_data.py predict
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import project_config as config


# How many pages of samples to request per civil day.
#
# This number was NOT guessed - it was measured, and the first guess was
# wrong. The data type declares max_page_size = 10000, so 4 pages looked like
# 40000 samples of headroom, far more than a day could hold. In reality the
# API returns about 5000 points per page, so 4 pages capped each day at
# ~20000 samples. Because the API returns the NEWEST samples first, that cap
# silently deleted the EARLIEST part of every day: full days came back
# starting at 10:45 or 11:41 instead of 00:00, and the truncation looked
# exactly like a person who simply does not wear the watch in the morning.
#
# At the observed ~3 s spacing a full 24 h day is roughly 28000-36000
# samples, i.e. about 7 pages. 16 leaves room for a denser day without being
# so large that a mistake costs many wasted requests.
DEFAULT_MAX_PAGES = 16

# A completed day whose first sample is later than this is reported as
# suspicious: either the watch was genuinely off the wrist, or we truncated
# again. Either way it must be visible, not silent.
EXPECTED_DAY_START_MINUTE = 5

# Reference point for turning epoch seconds back into a naive local datetime.
EPOCH = datetime(1970, 1, 1)


# --- Pass 2 settings -------------------------------------------------------

# Window geometry. Deliberately identical to the WESAD training windows
# (project_config.WINDOW_SECONDS / STEP_SECONDS) so that "a window" means the
# same span of time in training and in application. Pass 1 measured the real
# sample spacing at ~3 s, which puts about 20 readings in a 60 s window -
# enough to compute mean/std/min/max/slope over.
PERSONAL_WINDOW_SECONDS = config.WINDOW_SECONDS
PERSONAL_STEP_SECONDS = config.STEP_SECONDS

# A window with fewer readings than this is dropped rather than described by
# statistics computed from almost nothing. 10 is half of the ~20 expected at
# the observed 3 s spacing.
MIN_SAMPLES_PER_WINDOW = 10

# Predictions made while asleep are not meaningful - the model has no sleep
# class, and WESAD contains no sleep at all. We still compute them (dropping
# them silently would hide how much of the day they are), but the per-day
# table reports waking hours separately.
WAKING_START_HOUR = 8
WAKING_END_HOUR = 23


# ---------------------------------------------------------------------------
# Booting the Django project
# ---------------------------------------------------------------------------

def bootstrap_dashboard():
    """
    Make the Django health-dashboard project importable, and initialise it.

    django.setup() must run before any of its models are imported, because
    importing a model requires the app registry to be populated first.

    Its settings module computes BASE_DIR from its own file location, so it
    finds its own .env and its own db.sqlite3 regardless of where this script
    is run from.
    """
    if not os.path.isdir(config.DASHBOARD_DIR):
        raise SystemExit(
            "Cannot find the Django dashboard project at:\n  {}\n"
            "Edit DASHBOARD_DIR in project_config.py if it has moved."
            .format(config.DASHBOARD_DIR))

    sys.path.insert(0, config.DASHBOARD_DIR)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

    import django
    django.setup()


def get_credential(username=None):
    """
    Fetch the stored Google credential from the dashboard's database.

    We never touch the token values themselves - we hand the whole record to
    the dashboard's own client, which knows how to use and refresh it.
    """
    from accounts.models import GoogleCredential

    records = GoogleCredential.objects.select_related("user")
    if username:
        records = records.filter(user__username=username)

    record = records.first()
    if record is None:
        raise SystemExit(
            "No stored Google credentials found in the dashboard's database.\n"
            "Sign in through the dashboard app first.")

    if not record.is_usable:
        raise SystemExit(
            "The stored Google credential is marked as needing reconnection.\n"
            "(Google expires refresh tokens weekly while the OAuth consent "
            "screen is in Testing mode.)\n"
            "Sign in again through the dashboard app, then re-run this.")

    return record


# ---------------------------------------------------------------------------
# Parsing the API's heart-rate samples
# ---------------------------------------------------------------------------

def parse_civil_datetime(civil):
    """
    Turn the API's CivilDateTime into a plain Python datetime.

    Shape returned by the Google Health API:

        {"date": {"year": 2026, "month": 9, "day": 18},
         "time": {"hours": 14, "minutes": 3, "seconds": 27}}

    Fields that happen to be zero are OMITTED, not sent as 0 - midnight
    arrives as an empty time dict. Hence the .get(..., 0) defaults.

    These are CIVIL times: local wall-clock, with no timezone offset attached.
    That is what we want, because the question "was I stressed on Tuesday
    afternoon?" is asked in local time. No conversion is applied.
    """
    civil = civil or {}
    day_part = civil.get("date") or {}
    time_part = civil.get("time") or {}

    if not day_part:
        return None

    return datetime(
        year=day_part["year"],
        month=day_part["month"],
        day=day_part["day"],
        hour=time_part.get("hours", 0),
        minute=time_part.get("minutes", 0),
        second=time_part.get("seconds", 0),
    )


def parse_heart_rate_points(points):
    """
    Turn raw API data points into a list of (timestamp, bpm) rows.

    The payload shape, taken from the dashboard's datatypes registry rather
    than guessed:

        {"heartRate": {"beatsPerMinute": 72,
                       "sampleTime": {"civilTime": {...}}}}
    """
    rows = []
    for point in points:
        body = point.get("heartRate") or {}
        timestamp = parse_civil_datetime((body.get("sampleTime") or {})
                                         .get("civilTime"))
        bpm = body.get("beatsPerMinute")
        if timestamp is not None and bpm is not None:
            rows.append({"timestamp": timestamp, "bpm": float(bpm)})
    return rows


# ---------------------------------------------------------------------------
# Pass 1: inspect
# ---------------------------------------------------------------------------

def fetch_days(credential, n_days, max_pages):
    """
    Fetch raw intraday heart-rate samples for the last n_days civil days.

    Uses the dashboard's cache wrapper, so re-running this does not hammer the
    API: a day that has already finished cannot gain new samples and is cached
    for 24 hours.

    Returns (list of row dicts, list of per-day summary dicts, first raw point).
    """
    from django.utils import timezone
    from health import cache, datatypes
    from health.client import (GoogleHealthClient, GoogleHealthError,
                               GoogleReconnectRequired)

    data_type = datatypes.get("heart-rate")
    client = GoogleHealthClient(credential)

    today = timezone.localdate()
    all_rows = []
    per_day = []
    first_raw_point = None

    print("Fetching '{}' ({}) for the last {} civil days, ending {}."
          .format(data_type.slug, data_type.label, n_days, today))
    print("Page size {}, up to {} pages per day.".format(
        data_type.max_page_size, max_pages))
    print()

    for offset in range(n_days, 0, -1):
        day = today - timedelta(days=offset - 1)
        try:
            fetched = cache.fetch_intraday(client, data_type, day,
                                           user=credential.user,
                                           max_pages=max_pages)
        except GoogleReconnectRequired as exc:
            raise SystemExit(
                "Google rejected the stored refresh token: {}\n"
                "Sign in again through the dashboard app.".format(exc))
        except GoogleHealthError as exc:
            print("  {}  API error: {}".format(day, exc))
            per_day.append({"day": day, "n_points": 0, "error": str(exc)})
            continue

        if first_raw_point is None and fetched.points:
            first_raw_point = fetched.points[0]

        rows = parse_heart_rate_points(fetched.points)
        all_rows.extend(rows)

        per_day.append({
            "day": day,
            "n_points": len(rows),
            "n_raw_points": len(fetched.points),
            "from_cache": fetched.from_cache,
        })
        print("  {}  {:6d} samples{}".format(
            day, len(rows), "  (cached)" if fetched.from_cache else ""))

    return all_rows, per_day, first_raw_point


def describe_gaps(timestamps):
    """
    Summarise the spacing between consecutive samples within one day.

    This is the number that decides Pass 2's window size. A wearable does not
    sample on a fixed schedule - it samples more often during activity and
    drops out entirely when the watch is off the wrist - so a single "sampling
    rate" is the wrong way to describe it. We report the distribution instead.
    """
    if len(timestamps) < 2:
        return None

    ordered = np.sort(np.array(timestamps, dtype="datetime64[s]"))
    gaps = np.diff(ordered).astype(float)      # seconds

    return {
        "n": len(ordered),
        "median_gap_s": float(np.median(gaps)),
        "mean_gap_s": float(np.mean(gaps)),
        "p05_gap_s": float(np.percentile(gaps, 5)),
        "p95_gap_s": float(np.percentile(gaps, 95)),
        "max_gap_s": float(np.max(gaps)),
        "gaps_over_5min": int(np.sum(gaps > 300)),
        "first": str(ordered[0]),
        "last": str(ordered[-1]),
    }


def run_inspect(args):
    """Pass 1: fetch and describe. No modelling of any kind happens here."""
    os.makedirs(config.PERSONAL_DIR, exist_ok=True)

    bootstrap_dashboard()
    credential = get_credential(args.user)

    print("=" * 72)
    print("PASS 1 - INSPECT personal intraday heart rate")
    print("=" * 72)
    print("Dashboard project : " + config.DASHBOARD_DIR)
    print("Credential for    : {} ({})".format(credential.user,
                                               credential.email or "no email"))
    print()

    rows, per_day, first_raw_point = fetch_days(credential, args.days,
                                                args.max_pages)

    if not rows:
        raise SystemExit(
            "\nNo heart-rate samples were returned for this range. Either the "
            "watch has not synced, or the range predates the data.")

    # --- confirm the payload shape rather than trusting the documentation ---
    print()
    print("-" * 72)
    print("Raw shape of one API data point (verify this matches the parser)")
    print("-" * 72)
    print(first_raw_point)
    print()

    table = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    table["day"] = table["timestamp"].dt.date

    # --- per-day summary ----------------------------------------------------
    print("-" * 72)
    print("Per-day coverage and sample spacing")
    print("-" * 72)
    print("{:12s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s} {:>7s}  {:>5s}-{:<5s}".format(
        "day", "samples", "median_s", "p05_s", "p95_s", "maxgap_s",
        "gaps>5m", "first", "last"))

    day_summaries = []
    for day, group in table.groupby("day"):
        stats = describe_gaps(group["timestamp"].values)
        if stats is None:
            print("{:12s} {:>7d}  (too few samples to measure spacing)".format(
                str(day), len(group)))
            continue
        day_summaries.append(dict(day=day, **stats))
        print("{:12s} {:7d} {:9.1f} {:9.1f} {:9.1f} {:9.1f} {:7d}  "
              "{:>5s}-{:<5s}".format(
                  str(day), stats["n"], stats["median_gap_s"],
                  stats["p05_gap_s"], stats["p95_gap_s"], stats["max_gap_s"],
                  stats["gaps_over_5min"],
                  stats["first"][11:16], stats["last"][11:16]))

    # --- truncation guard ---------------------------------------------------
    # The API returns newest samples first, so running out of pages removes
    # the start of the day rather than the end. A completed day that begins
    # late is therefore the symptom of truncation. Raise --max-pages and
    # re-run if this fires; if the count then stays the same, the watch really
    # was off the wrist.
    from django.utils import timezone as dj_timezone
    today = dj_timezone.localdate()

    late_starts = [s for s in day_summaries
                   if s["day"] != today
                   and int(s["first"][11:13]) * 60 + int(s["first"][14:16])
                   > EXPECTED_DAY_START_MINUTE]
    if late_starts:
        print()
        print("  WARNING: {} completed day(s) do not start at midnight:".format(
            len(late_starts)))
        for s in late_starts:
            print("    {} starts at {} ({} samples)".format(
                s["day"], s["first"][11:16], s["n"]))
        print("  Either the watch was off the wrist, or --max-pages "
              "(currently {}) truncated the day.".format(args.max_pages))
        print("  Re-run with a higher --max-pages to tell the two apart.")

    # --- overall ------------------------------------------------------------
    overall = describe_gaps(table["timestamp"].values)
    print()
    print("-" * 72)
    print("Overall")
    print("-" * 72)
    print("  days with data      : {}".format(table["day"].nunique()))
    print("  total samples       : {}".format(len(table)))
    print("  samples per day     : min {:.0f}, median {:.0f}, max {:.0f}".format(
        table.groupby("day").size().min(),
        table.groupby("day").size().median(),
        table.groupby("day").size().max()))
    print("  bpm range           : {:.0f} - {:.0f} (mean {:.1f})".format(
        table["bpm"].min(), table["bpm"].max(), table["bpm"].mean()))

    if day_summaries:
        medians = [s["median_gap_s"] for s in day_summaries]
        print("  median gap per day  : min {:.1f}s, median {:.1f}s, "
              "max {:.1f}s".format(min(medians), float(np.median(medians)),
                                   max(medians)))

    # How many samples would land in a window of each candidate length? This
    # is the question Pass 2 actually needs answered: a window holding one or
    # two samples cannot support std/min/max/slope features.
    print()
    print("  Samples that would fall in one window, at the observed median")
    print("  spacing (per day, using each day's own median gap):")
    for window_minutes in [1, 5, 15, 30, 60]:
        counts = [window_minutes * 60.0 / s["median_gap_s"]
                  for s in day_summaries if s["median_gap_s"] > 0]
        if counts:
            print("    {:3d} min window : median {:6.1f} samples "
                  "(min {:.1f}, max {:.1f})".format(
                      window_minutes, float(np.median(counts)),
                      min(counts), max(counts)))

    # --- save for Pass 2 ----------------------------------------------------
    samples_path = os.path.join(config.PERSONAL_DIR, "hr_samples.csv")
    table[["timestamp", "bpm"]].to_csv(samples_path, index=False)

    summary_path = os.path.join(config.PERSONAL_DIR, "hr_inspection.csv")
    pd.DataFrame(day_summaries).to_csv(summary_path, index=False)

    print()
    print("Raw samples saved to   : " + samples_path)
    print("Per-day summary saved  : " + summary_path)
    print()
    print("Pass 1 complete. No model was loaded and no prediction was made.")


# ---------------------------------------------------------------------------
# Pass 2: window the personal data, compare it to training, predict
# ---------------------------------------------------------------------------

def load_personal_samples():
    """
    Read the heart-rate samples that Pass 1 saved.

    Pass 2 deliberately does NOT call the API again. Running `inspect` first
    and `predict` second means the prediction step is reproducible offline and
    cannot quietly disagree with the data that was inspected.
    """
    path = os.path.join(config.PERSONAL_DIR, "hr_samples.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "No personal samples found at:\n  {}\n"
            "Run Pass 1 first:\n"
            "  python 05_apply_to_personal_data.py inspect".format(path))

    table = pd.read_csv(path, parse_dates=["timestamp"])
    table = table.sort_values("timestamp").reset_index(drop=True)
    return table


def build_personal_windows(samples):
    """
    Slide a 60 s window across each civil day and compute the HR features.

    Windows are anchored to midnight and advance in fixed 30 s steps, rather
    than being anchored to whatever time the first reading happens to arrive.
    That makes a window's start time a real clock time, so a prediction can be
    lined up against "the afternoon of the 16th".

    IMPORTANT - the feature functions are imported from features.py, the same
    module that built the training table. Nothing is reimplemented here, so
    the arithmetic behind HR_mean and friends cannot drift between training
    and application.

    The one thing that CANNOT be made identical is what the numbers are
    computed over:

      training     one value per detected heartbeat (~60-80 per window),
                   from our own peak detection on WESAD's 64 Hz BVP signal
      here         one already-smoothed bpm reading every ~3 s (~20 per
                   window), produced by the watch's own undisclosed algorithm

    So HR_std is measuring the spread of a smoothed series rather than true
    beat-to-beat variability, and HR_slope - which uses the reading index as
    its time axis in both cases, via basic_features(..., fs=1.0) - is "bpm per
    beat" in training but "bpm per 3-second reading" here. Same column names,
    different physical quantities. compare_distributions() below measures how
    far apart they actually end up.
    """
    import features

    # Work in integer epoch seconds: fast to slice with searchsorted, and
    # immune to any timezone handling, since these are already local times.
    timestamps = samples["timestamp"].values.astype("datetime64[s]").astype("int64")
    bpm = samples["bpm"].values.astype(float)

    rows = []
    for day, day_group in samples.groupby(samples["timestamp"].dt.date):
        day_start = np.datetime64(str(day) + "T00:00:00", "s").astype("int64")
        day_end = day_start + 24 * 3600

        window_starts = np.arange(
            day_start, day_end - PERSONAL_WINDOW_SECONDS + 1,
            PERSONAL_STEP_SECONDS)

        for start in window_starts:
            end = start + PERSONAL_WINDOW_SECONDS
            first = np.searchsorted(timestamps, start, side="left")
            last = np.searchsorted(timestamps, end, side="left")

            if last - first < MIN_SAMPLES_PER_WINDOW:
                continue

            window_bpm = bpm[first:last]

            # fs=1.0 means "one sample per unit of the time axis", i.e. the
            # slope is per reading. This is exactly how the training HR
            # features were computed (per beat), which is why the same call
            # is used here - see the docstring above about what that does and
            # does not make comparable.
            stats = features.basic_features(window_bpm, fs=1.0)

            row = {"HR_" + name: value for name, value in stats.items()}
            row["day"] = day
            # Rebuild the local wall-clock time from the epoch seconds we sliced
            # with. These epochs were made from naive local timestamps, so
            # adding them to the epoch date returns the same local time.
            row["window_start"] = EPOCH + timedelta(seconds=int(start))
            row["n_samples"] = int(last - first)
            rows.append(row)

    if not rows:
        raise SystemExit(
            "No windows survived. Either there are too few samples, or "
            "MIN_SAMPLES_PER_WINDOW is set too high.")

    windows = pd.DataFrame(rows)
    windows["hour"] = windows["window_start"].dt.hour
    return windows


def compare_distributions(personal_windows, feature_names):
    """
    Put the WESAD training HR features and the personal HR features side by
    side, BEFORE any prediction is shown.

    This is the honest way to present what follows. A model is only entitled
    to be trusted on inputs that resemble what it was trained on; an RBF-kernel
    SVM in particular judges by distance in feature space, so inputs that sit
    outside the training range do not degrade gracefully - they land wherever
    happens to be nearest.

    Two things are reported per feature:
      * the summary statistics of each distribution, so the shift is visible;
      * the share of personal windows falling outside the training min-max
        range, which is the bluntest possible out-of-distribution measure and
        needs no assumptions about shape.
    """
    # The training distribution is read from the summary saved beside the
    # model by 03_train_model.py, not recomputed from results/features.csv.
    # One definition of "the training distribution", shared with the Django
    # dashboard's arousal card, and it travels with the model.
    import json
    distribution_path = os.path.join(config.MODELS_DIR,
                                     "hr_training_distribution.json")
    with open(distribution_path) as f:
        training_distribution = json.load(f)
    training_stats = training_distribution["features"]

    print("-" * 78)
    print("TRAINING vs PERSONAL feature distributions (before any prediction)")
    print("-" * 78)
    print("WESAD training windows : {}".format(
        training_distribution["n_windows"]))
    print("Personal windows       : {}".format(len(personal_windows)))
    print()
    print("{:10s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "feature", "source", "mean", "std", "min", "median", "max"))

    comparison = []
    for name in feature_names:
        a = training_stats[name]          # a dict of summary statistics
        b = personal_windows[name]        # the actual personal values

        print("{:10s} {:>8s} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f}".format(
            name, "WESAD", a["mean"], a["std"], a["min"], a["median"],
            a["max"]))
        print("{:10s} {:>8s} {:9.3f} {:9.3f} {:9.3f} {:9.3f} {:9.3f}".format(
            "", "personal", b.mean(), b.std(), b.min(), b.median(), b.max()))

        # Share of personal windows outside the training range.
        outside = ((b < a["min"]) | (b > a["max"])).mean()

        # How far the personal centre sits from the training centre, measured
        # in training standard deviations. Roughly: "how many sigma out".
        sigma_shift = (((b.median() - a["mean"]) / a["std"])
                       if a["std"] else np.nan)

        print("{:10s} {:>8s} outside training range: {:5.1%}   "
              "median shift: {:+.2f} training sd".format(
                  "", "->", outside, sigma_shift))
        print()

        comparison.append({
            "feature": name,
            "wesad_mean": a["mean"], "wesad_std": a["std"],
            "wesad_min": a["min"], "wesad_max": a["max"],
            "personal_mean": b.mean(), "personal_std": b.std(),
            "personal_min": b.min(), "personal_max": b.max(),
            "personal_outside_training_range": float(outside),
            "median_shift_in_training_sd": float(sigma_shift),
        })

    return pd.DataFrame(comparison)


def summarise_by_day(predictions):
    """
    Collapse per-window predictions into one row per day.

    Reports the whole day and the waking hours separately. Overnight windows
    are not evidence of anything - the model has no sleep class and WESAD
    contains no sleep - so a day's "stress share" computed across 24 hours is
    diluted by eight hours the model was never meant to see.
    """
    waking = predictions[(predictions["hour"] >= WAKING_START_HOUR)
                         & (predictions["hour"] < WAKING_END_HOUR)]

    rows = []
    for day, group in predictions.groupby("day"):
        day_waking = waking[waking["day"] == day]
        row = {
            "day": day,
            "windows": len(group),
            "windows_waking": len(day_waking),
            "mean_bpm": group["HR_mean"].mean(),
        }
        for label in sorted(config.LABEL_NAMES):
            name = config.LABEL_NAMES[label]
            row["share_" + name] = (group["predicted"] == label).mean()
            row["waking_share_" + name] = (
                (day_waking["predicted"] == label).mean()
                if len(day_waking) else np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def run_predict(args):
    """Pass 2: window the personal data, compare it to training, predict."""
    import joblib
    import json

    os.makedirs(config.PERSONAL_DIR, exist_ok=True)

    model_path = os.path.join(config.MODELS_DIR, "svm_hr_only.joblib")
    config_path = os.path.join(config.MODELS_DIR, "feature_config_hr_only.json")
    if not os.path.exists(model_path):
        raise SystemExit("No heart-rate-only model at {}.\n"
                         "Run 03_train_model.py first.".format(model_path))

    with open(config_path) as f:
        feature_config = json.load(f)
    # The saved config, not a hard-coded list, decides which columns the model
    # gets and in what order. Feeding a scikit-learn pipeline the right values
    # in the wrong order produces no error - only wrong answers.
    feature_names = feature_config["feature_names_in_order"]

    print("=" * 78)
    print("PASS 2 - APPLY the heart-rate-only model to personal wearable data")
    print("=" * 78)
    print()
    print("!! This is a demonstration that the pipeline runs end to end.")
    print("!! The model was trained on laboratory-induced stress (a public")
    print("!! speaking test) and amusement (funny video clips) in 15 people.")
    print("!! It has never seen real-world daily life, it has no 'neither'")
    print("!! class, and there are no ground-truth labels for these days.")
    print("!! Nothing below is evidence that I was or was not stressed.")
    print()

    samples = load_personal_samples()
    print("Loaded {} heart-rate samples, {} to {}.".format(
        len(samples), samples["timestamp"].min(), samples["timestamp"].max()))

    windows = build_personal_windows(samples)
    print("Built {} windows of {} s, stepping {} s, keeping only windows with "
          ">= {} readings.".format(
              len(windows), PERSONAL_WINDOW_SECONDS, PERSONAL_STEP_SECONDS,
              MIN_SAMPLES_PER_WINDOW))
    print("Readings per window: median {:.0f} (min {}, max {}).".format(
        windows["n_samples"].median(), windows["n_samples"].min(),
        windows["n_samples"].max()))
    print()

    # --- the comparison comes BEFORE the predictions ------------------------
    comparison = compare_distributions(windows, feature_names)
    comparison.to_csv(
        os.path.join(config.PERSONAL_DIR, "feature_distribution_comparison.csv"),
        index=False)

    worst = comparison.loc[
        comparison["personal_outside_training_range"].idxmax()]
    print("Largest out-of-distribution feature: {} - {:.1%} of personal "
          "windows fall outside the range the model was trained on.".format(
              worst["feature"], worst["personal_outside_training_range"]))
    print()

    # --- predictions --------------------------------------------------------
    model = joblib.load(model_path)
    windows["predicted"] = model.predict(windows[feature_names])
    windows["predicted_name"] = windows["predicted"].map(config.LABEL_NAMES)

    print("-" * 78)
    print("PREDICTIONS - per window")
    print("-" * 78)
    for label in sorted(config.LABEL_NAMES):
        name = config.LABEL_NAMES[label]
        count = int((windows["predicted"] == label).sum())
        print("  {:11s} {:6d} windows ({:5.1%})".format(
            name, count, count / len(windows)))
    print()

    per_day = summarise_by_day(windows)

    print("-" * 78)
    print("PREDICTIONS - per day  (sanity-check these against days you")
    print("remember as stressful or relaxed)")
    print("-" * 78)
    print("{:12s} {:>8s} {:>9s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
        "day", "windows", "mean bpm", "baseline", "stress",
        "amusement", "stress(wake)"))
    for _, row in per_day.iterrows():
        waking_stress = row["waking_share_stress"]
        print("{:12s} {:8d} {:9.1f} {:9.1%} {:9.1%} {:11.1%} {:11s}".format(
            str(row["day"]), int(row["windows"]), row["mean_bpm"],
            row["share_baseline"], row["share_stress"], row["share_amusement"],
            "n/a" if pd.isna(waking_stress) else "{:.1%}".format(waking_stress)))
    print()

    # --- save ---------------------------------------------------------------
    windows_path = os.path.join(config.PERSONAL_DIR, "predictions_windows.csv")
    windows.to_csv(windows_path, index=False)

    per_day_path = os.path.join(config.PERSONAL_DIR, "predictions_per_day.csv")
    per_day.to_csv(per_day_path, index=False)

    # A compact summary that 04_evaluate.py folds into RESULTS.md, so the
    # report has one source rather than two.
    summary = {
        "n_samples": int(len(samples)),
        "first_sample": str(samples["timestamp"].min()),
        "last_sample": str(samples["timestamp"].max()),
        "n_days": int(windows["day"].nunique()),
        "n_windows": int(len(windows)),
        "window_seconds": PERSONAL_WINDOW_SECONDS,
        "step_seconds": PERSONAL_STEP_SECONDS,
        "min_samples_per_window": MIN_SAMPLES_PER_WINDOW,
        "median_readings_per_window": float(windows["n_samples"].median()),
        "waking_hours": [WAKING_START_HOUR, WAKING_END_HOUR],
        "predicted_share": {
            config.LABEL_NAMES[label]:
                float((windows["predicted"] == label).mean())
            for label in sorted(config.LABEL_NAMES)
        },
        "out_of_distribution": {
            row["feature"]: {
                "share_outside_training_range":
                    row["personal_outside_training_range"],
                "median_shift_in_training_sd":
                    row["median_shift_in_training_sd"],
            }
            for _, row in comparison.iterrows()
        },
    }
    summary_path = os.path.join(config.PERSONAL_DIR, "prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Per-window predictions : " + windows_path)
    print("Per-day summary        : " + per_day_path)
    print("Summary for RESULTS.md : " + summary_path)
    print()
    print("Reminder: no ground truth exists for these days, so no accuracy is")
    print("reported and none can be. The value of this step is that the whole")
    print("pipeline - fetch, window, feature, model - runs on real wearable")
    print("data, not that its output is correct.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help="Pass 1: fetch personal heart rate and report what is in it.")
    inspect.add_argument("--days", type=int, default=14,
                         help="How many civil days back to fetch (default 14).")
    inspect.add_argument("--user", default=None,
                         help="Dashboard username. Defaults to the only user "
                              "with stored credentials.")
    inspect.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                         dest="max_pages",
                         help="Max API pages per day (default {}). See "
                              "DEFAULT_MAX_PAGES for why this matters."
                              .format(DEFAULT_MAX_PAGES))
    inspect.set_defaults(func=run_inspect)

    predict = subparsers.add_parser(
        "predict",
        help="Pass 2: window the samples saved by 'inspect', compare them to "
             "the training distribution, and run the heart-rate-only model.")
    predict.set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
