"""Score one person's day of stored heart rate with the WESAD-trained model.

This is the dashboard's action/feedback layer: it takes a model trained in a
separate project and does something with its output, rather than drawing
another chart of the same numbers.

MODEL TRAINED ON WESAD ONLY; UPLOADED DATA IS ONLY SCORED
---------------------------------------------------------
Nothing in this module trains, fits, tunes or saves a model. It loads a
pipeline that was fitted on the public WESAD dataset in the wearable-ml project
and calls `predict` on a person's windows. A guard test runs this whole path
with every fit and save patched to raise.

WHAT IT IS NOT
--------------
The model was trained on WESAD - fifteen people in a laboratory, where
"stress" is a public-speaking test and "amusement" is a funny video clip. It
has never seen ordinary daily life, it has no "none of these" class, and
there is no ground truth for anybody's real days. Every assessment this
module produces therefore travels with its own caveat, and `Assessment` is
built so that a prediction cannot be constructed without one.

WHOSE DATA
----------
Every assessment is for one named person on one local day, read through
people.repository, which refuses to read without a Person. The device and the
spacing between that device's readings travel with the result, because a
Fitbit's smoothed readings every ~2-3 s and an EmotiBit's are different
measurements and the difference is a confound that must stay visible.

HOW IT REACHES THE MODEL
------------------------
The trained pipeline lives in the wearable-ml project at
``settings.WEARABLE_ML_DIR``. Four files are read, never written:

    models/svm_hr_only.joblib            the scikit-learn pipeline
    models/feature_config_hr_only.json   feature names, in the fitted order
    models/hr_training_distribution.json per-feature training statistics
    models/model_card.json               scores and limitations

Unpickling a scikit-learn pipeline needs scikit-learn importable in *this*
process, which is why the requirements pin the same versions that project
used. Those imports are kept inside the functions that need them: a missing or
broken ML stack then degrades this one page instead of stopping the site.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from django.conf import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings that must match how the model was trained
# ---------------------------------------------------------------------------
# These mirror WINDOW_SECONDS / STEP_SECONDS in wearable-ml's
# project_config.py. A window here has to mean the same span of time it meant
# during training, or the features describe something else.
WINDOW_SECONDS = 60
STEP_SECONDS = 30

# A Fitbit reports roughly one reading every 2-3 s, so a 60 s window holds
# about 20-26. Fewer than half of that and the statistics are not worth
# computing, so the window is dropped.
MIN_SAMPLES_PER_WINDOW = 10

# A day is only offered for scoring when it holds at least this many scorable
# windows (30 minutes of wear). It is a constant the page prints, not a hidden
# cut-off.
MIN_WINDOWS_TO_SCORE = 60

# Predictions made while asleep are not meaningful: the model has no sleep
# class and WESAD contains no sleep. They are still computed - hiding them
# would misrepresent how much of the day they are - but reported separately.
WAKING_START_HOUR = 8
WAKING_END_HOUR = 23

# Display rules, fixed before the first run (wearable-ml/PRE_DECLARED.md).
# A day is flagged OUT OF DISTRIBUTION when this share of its windows falls
# outside the training range on any heart-rate feature. Predictions are flagged
# DEGENERATE when one class takes this share of windows (the 3-class rule).
OOD_FLAG_SHARE = 0.25
DOMINANT_CLASS_SHARE = 0.60

WESAD_ONLY_NOTICE = "Model trained on WESAD only; uploaded data is only scored."

MODEL_FILENAME = "svm_hr_only.joblib"
FEATURE_CONFIG_FILENAME = "feature_config_hr_only.json"
DISTRIBUTION_FILENAME = "hr_training_distribution.json"
MODEL_CARD_FILENAME = "model_card.json"

# The scikit-learn that wrote the pickle. A different version can quietly
# migrate a pickle rather than refuse it, so a mismatch is surfaced instead
# of trusted.
EXPECTED_SKLEARN_VERSION = "1.5.2"

# Plain-language wording for each class the model can emit. The raw class
# names ("stress", "amusement") would overstate what a wrist heart rate can
# support, so the page never shows them as the headline.
CLASS_WORDING = {
    1: ("Settled baseline signal",
        "This day's heart-rate pattern most resembles the calm baseline "
        "condition the model was trained on."),
    2: ("Elevated arousal signal",
        "This day's heart-rate pattern most resembles the stressed condition "
        "the model was trained on. That is not the same as saying the person "
        "was stressed."),
    3: ("Light arousal signal",
        "This day's heart-rate pattern most resembles the amusement condition "
        "the model was trained on - raised, but not in the way the stress "
        "condition was."),
}

# Class names for the notes only (the 3-class model's own labels).
CLASS_NAMES = {1: "baseline", 2: "stress", 3: "amusement"}

# Every assessment is labelled low confidence. Nothing here has been
# validated against real-world stress, so there is no honest basis on which
# to ever call one of these predictions high confidence.
CONFIDENCE = "low confidence"


class ArousalUnavailable(Exception):
    """Something needed for an assessment is missing or unusable."""


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Assessment:
    """One person-day's assessment, or an explanation of why there isn't one.

    Build these through the two classmethods, never directly. `predicted`
    requires the out-of-distribution evidence as an argument, which is what
    makes "a label with no caveat" impossible to construct rather than merely
    discouraged.
    """

    day: str
    available: bool
    reason: str = ""
    headline: str = ""
    explanation: str = ""
    confidence: str = ""
    shares: dict = field(default_factory=dict)
    waking_shares: dict = field(default_factory=dict)
    n_windows: int = 0
    n_waking_windows: int = 0
    mean_bpm: float = 0.0
    ood_rows: list = field(default_factory=list)
    worst_ood_feature: str = ""
    worst_ood_share: float = 0.0
    notes: list = field(default_factory=list)
    device_names: list = field(default_factory=list)
    spacing: dict = field(default_factory=dict)
    n_readings: int = 0
    timezone: str = ""
    median_values_per_window: float = 0.0
    wesad_values_per_window: float = 0.0

    @classmethod
    def unavailable(cls, day, reason):
        """No prediction, and a plain explanation of why."""
        return cls(day=str(day), available=False, reason=reason)

    @classmethod
    def predicted(cls, day, dominant, shares, waking_shares, n_windows,
                  n_waking_windows, mean_bpm, ood_rows, **context):
        """A prediction, which cannot be built without its caveat evidence."""
        if not ood_rows:
            raise ValueError(
                "An assessment may not be created without its "
                "out-of-distribution evidence.")

        headline, explanation = CLASS_WORDING[dominant]
        worst = max(ood_rows, key=lambda row: row["share_outside"])

        return cls(
            day=str(day),
            available=True,
            headline=headline,
            explanation=explanation,
            confidence=CONFIDENCE,
            shares=shares,
            waking_shares=waking_shares,
            n_windows=n_windows,
            n_waking_windows=n_waking_windows,
            mean_bpm=mean_bpm,
            ood_rows=ood_rows,
            worst_ood_feature=worst["feature"],
            worst_ood_share=worst["share_outside"],
            notes=notes_for(shares, worst["share_outside"]),
            **context,
        )

    @property
    def caveat(self):
        """The one-line caveat shown beside the headline. Never empty."""
        if not self.available:
            return ""
        return (
            "Wrist heart rate only, from a model trained on laboratory stress "
            "- not a clinical or validated measure. {:.0%} of this day's "
            "windows fall outside the range it was trained on ({})."
        ).format(self.worst_ood_share, self.worst_ood_feature)


def notes_for(shares, worst_ood_share):
    """Plain-language notes shown with the label, from the pre-declared rules."""
    notes = []
    named = {CLASS_NAMES[label]: share for label, share in shares.items()}
    top_name, top_share = max(named.items(), key=lambda kv: kv[1])
    out_of_distribution = worst_ood_share >= OOD_FLAG_SHARE
    degenerate = top_share >= DOMINANT_CLASS_SHARE

    if out_of_distribution:
        notes.append(
            "This day is OUT OF DISTRIBUTION: at least {:.0%} of its windows "
            "fall outside the WESAD training range on one heart-rate feature."
            .format(OOD_FLAG_SHARE))
    if degenerate:
        notes.append(
            "The predictions are DEGENERATE: one class ({}) takes {:.0%} of "
            "windows.".format(top_name, top_share))
        if top_name == "amusement":
            notes.append(
                "The model cannot separate amusement from baseline using "
                "heart rate alone, and this day is out of distribution.")
    if out_of_distribution or degenerate:
        notes.append("The class shares are therefore NOT interpretable.")
    return notes


# ---------------------------------------------------------------------------
# Reaching the model files
# ---------------------------------------------------------------------------

def models_dir():
    """The wearable-ml models directory, from settings."""
    return os.path.join(settings.WEARABLE_ML_DIR, "models")


def _read_json(filename):
    path = os.path.join(models_dir(), filename)
    if not os.path.exists(path):
        raise ArousalUnavailable(
            "The model file {} was not found. Expected it under {}."
            .format(filename, models_dir()))
    # encoding is explicit: without it Windows reads these as cp1252.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_card():
    """The model's own description of itself, for the methodology page.

    Returns None rather than raising: the page should still render and say
    the card is missing.
    """
    try:
        return _read_json(MODEL_CARD_FILENAME)
    except (ArousalUnavailable, ValueError) as exc:
        logger.warning("Model card unavailable: %s", exc)
        return None


def load_artifacts():
    """Load the pipeline, its feature order and the training distribution.

    Raises ArousalUnavailable with a readable message for every failure the
    caller can do something about - a missing project, a missing model, an
    ML stack that is not installed, or a scikit-learn that did not write
    this pickle.
    """
    model_path = os.path.join(models_dir(), MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise ArousalUnavailable(
            "No trained model at {}. Train one in the wearable-ml project, "
            "or point WEARABLE_ML_DIR at it.".format(model_path))

    feature_config = _read_json(FEATURE_CONFIG_FILENAME)
    distribution = _read_json(DISTRIBUTION_FILENAME)

    # Imported here, not at module scope, so that an ML stack which is
    # missing or broken costs this one page and not the whole site.
    try:
        import joblib
        import sklearn
    except ImportError as exc:
        raise ArousalUnavailable(
            "scikit-learn is not installed in this environment, so the "
            "trained model cannot be loaded ({}). Install the pinned "
            "versions from requirements.txt.".format(exc))

    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        # Loading anyway risks a silently migrated pickle whose predictions
        # no longer match the ones the model was validated on.
        raise ArousalUnavailable(
            "This model was saved by scikit-learn {}, but {} is installed. "
            "Refusing to load it rather than trust a migrated model."
            .format(EXPECTED_SKLEARN_VERSION, sklearn.__version__))

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # a corrupt or incompatible pickle
        raise ArousalUnavailable(
            "The trained model could not be loaded: {}".format(exc))

    return model, feature_config["feature_names_in_order"], distribution


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def basic_features(values):
    """The five engineered features: mean, std, min, max and slope.

    SOURCE OF TRUTH: this is a deliberate reimplementation of
    ``basic_features`` in wearable-ml/scripts/features.py, which is what
    built the training table. Any change there must be mirrored here, and
    test_arousal.py pins the two together with fixed input/output values so
    a drift fails a test rather than silently changing predictions.

    It is reimplemented rather than imported for two reasons. Only these five
    statistics are needed here - the training module also contains BVP peak
    detection, which cannot run on this data because a consumer wearable
    reports a heart rate directly and never exposes the raw pulse waveform.
    And importing it would mean putting that project's scripts directory on
    sys.path, where the top-level module names are ``features`` and
    ``project_config``; the same kind of name collision has already bitten
    this pair of projects once.

    ``slope`` is the least-squares gradient over the reading index, matching
    ``basic_features(..., fs=1.0)`` in training - a change per reading, not
    per second.
    """
    import numpy as np

    values = np.asarray(values, dtype=float).ravel()
    if len(values) == 0:
        raise ValueError("basic_features needs at least one value")

    if len(values) < 2:
        slope = 0.0
    else:
        index = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(index, values, 1)[0])

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "slope": slope,
    }


def window_starts(day, tz_name):
    """Window start times (UTC epoch seconds), anchored to local midnight.

    Built from the real local midnights, so a daylight-saving day is 23 or 25
    hours long and its windows follow.
    """
    import numpy as np

    from people.repository import local_day_bounds

    start, end = local_day_bounds(day, tz_name)
    return np.arange(start, end - WINDOW_SECONDS + 1, STEP_SECONDS)


def count_scorable_windows(epochs, day, tz_name):
    """How many 60 s windows of this day hold at least 10 readings."""
    import numpy as np

    if len(epochs) == 0:
        return 0
    starts = window_starts(day, tz_name)
    lo = np.searchsorted(epochs, starts, side="left")
    hi = np.searchsorted(epochs, starts + WINDOW_SECONDS, side="left")
    return int(np.sum((hi - lo) >= MIN_SAMPLES_PER_WINDOW))


def build_windows(epochs, bpm, day, tz_name, feature_names):
    """Slide a 60 s window across a local day and compute features for each.

    Windows are anchored to local midnight and step every 30 s, so a window's
    start is a real clock time and can be reported as "this afternoon".

    Returns (feature dicts, waking flags, readings per window). Raises
    ArousalUnavailable if the day holds too little data to window at all.
    """
    import numpy as np

    if len(epochs) == 0:
        raise ArousalUnavailable(
            "No heart-rate readings are stored for this person on this day.")

    starts = window_starts(day, tz_name)
    lo = np.searchsorted(epochs, starts, side="left")
    hi = np.searchsorted(epochs, starts + WINDOW_SECONDS, side="left")

    tz = ZoneInfo(tz_name)
    rows, waking, sizes = [], [], []
    for start, first, last in zip(starts, lo, hi):
        if last - first < MIN_SAMPLES_PER_WINDOW:
            continue
        stats = basic_features(bpm[first:last])
        rows.append({"HR_" + name: value for name, value in stats.items()})
        hour = datetime.fromtimestamp(int(start), timezone.utc).astimezone(tz).hour
        waking.append(WAKING_START_HOUR <= hour < WAKING_END_HOUR)
        sizes.append(int(last - first))

    if not rows:
        raise ArousalUnavailable(
            "Not enough heart-rate data for this day: no 60-second window "
            "held at least {} readings.".format(MIN_SAMPLES_PER_WINDOW))

    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise ArousalUnavailable(
            "The model expects features this code does not compute: {}."
            .format(", ".join(missing)))

    return rows, waking, sizes


# ---------------------------------------------------------------------------
# Out-of-distribution check
# ---------------------------------------------------------------------------

def out_of_distribution(rows, feature_names, distribution):
    """Measure how far this day's features sit outside the training data.

    Same two measures the wearable-ml demo uses, against the same training
    statistics file:

      share_outside - the fraction of the day's windows whose value falls
                      outside the min-max range seen in training. Blunt, but
                      it needs no assumption about the shape of either
                      distribution.
      median_shift  - how far the day's median sits from the training mean,
                      counted in training standard deviations.

    This matters more than it would for most models. The classifier is an
    RBF-kernel SVM, which judges by distance in feature space: inputs outside
    the region it was fitted on do not degrade gracefully, they land wherever
    happens to be nearest.
    """
    import numpy as np

    stats = distribution["features"]
    report = []

    for name in feature_names:
        training = stats[name]
        values = np.array([row[name] for row in rows], dtype=float)

        outside = float(np.mean((values < training["min"])
                                | (values > training["max"])))
        shift = (float((np.median(values) - training["mean"]) / training["std"])
                 if training["std"] else 0.0)

        report.append({
            "feature": name,
            "share_outside": outside,
            "median_shift": shift,
            "training_min": training["min"],
            "training_max": training["max"],
            "today_median": float(np.median(values)),
        })

    return report


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def _shares(predictions):
    """Fraction of windows predicted as each class."""
    total = len(predictions)
    if total == 0:
        return {}
    return {
        int(label): float(sum(1 for p in predictions if int(p) == label)) / total
        for label in CLASS_WORDING
    }


def assess(person, day):
    """Score one local day of one person's stored heart rate.

    `person` must be a Person: the readings are read through the repository,
    which refuses anything else. Never raises for an expected problem: a
    missing model, an ML stack that is not installed, or a day with too little
    data all come back as an unavailable Assessment carrying a readable reason.
    """
    from people import repository

    repository.require_person(person)
    try:
        model, feature_names, distribution = load_artifacts()

        epochs, bpm = repository.local_day_readings(person, day)
        rows, waking, sizes = build_windows(
            epochs, bpm, day, person.timezone, feature_names)

        if len(rows) < MIN_WINDOWS_TO_SCORE:
            raise ArousalUnavailable(
                "Too few scorable windows on this day: {} (at least {} are "
                "needed, which is {} minutes of wear).".format(
                    len(rows), MIN_WINDOWS_TO_SCORE,
                    MIN_WINDOWS_TO_SCORE * STEP_SECONDS // 60))

        import numpy as np
        import pandas as pd

        # A DataFrame with named columns, in the order the pipeline was
        # fitted on. Order matters and is silent when wrong: feeding the
        # right five numbers in the wrong order raises nothing, it just
        # returns a different answer.
        frame = pd.DataFrame(rows)[feature_names]
        predictions = [int(value) for value in model.predict(frame)]

        waking_predictions = [p for p, flag in zip(predictions, waking) if flag]
        shares = _shares(predictions)
        dominant = max(shares.items(), key=lambda kv: kv[1])[0]
        mean_hr_training = distribution["features"]["HR_mean"]["mean"]

        return Assessment.predicted(
            day=day,
            dominant=dominant,
            shares=shares,
            waking_shares=_shares(waking_predictions),
            n_windows=len(predictions),
            n_waking_windows=len(waking_predictions),
            mean_bpm=sum(row["HR_mean"] for row in rows) / len(rows),
            ood_rows=out_of_distribution(rows, feature_names, distribution),
            device_names=repository.device_names_for_day(person, day),
            spacing=repository.spacing(epochs),
            n_readings=int(len(epochs)),
            timezone=person.timezone,
            median_values_per_window=float(np.median(sizes)),
            # One value per heartbeat, so about one per beat in a 60 s window.
            wesad_values_per_window=float(mean_hr_training),
        )

    except ArousalUnavailable as exc:
        return Assessment.unavailable(day, str(exc))
    except Exception as exc:  # anything unforeseen: degrade, do not 500
        logger.exception("Arousal assessment failed for %s", day)
        return Assessment.unavailable(
            day, "The assessment could not be computed: {}".format(exc))


def scorable_days(person):
    """Local days with data, newest first, each with its scorable-window count.

    A day is `scorable` when it holds at least MIN_WINDOWS_TO_SCORE windows of
    at least MIN_SAMPLES_PER_WINDOW readings.
    """
    from people import repository

    repository.require_person(person)
    days = []
    for day in sorted(repository.local_dates_with_data(person), reverse=True):
        epochs, _ = repository.local_day_readings(person, day)
        windows = count_scorable_windows(epochs, day, person.timezone)
        days.append({"day": day, "readings": int(len(epochs)),
                     "windows": windows,
                     "scorable": windows >= MIN_WINDOWS_TO_SCORE})
    return days


def default_day(days):
    """The most recent scorable day from `scorable_days`, or None."""
    for entry in days:
        if entry["scorable"]:
            return entry["day"]
    return None
