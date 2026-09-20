"""Run the WESAD-trained heart-rate model against a day of live readings.

This is the dashboard's action/feedback layer: it takes a model trained in a
separate project and does something with its output, rather than drawing
another chart of the same numbers.

WHAT IT IS NOT
--------------
The model was trained on WESAD - fifteen people in a laboratory, where
"stress" is a public-speaking test and "amusement" is a funny video clip. It
has never seen ordinary daily life, it has no "none of these" class, and
there is no ground truth for anybody's real days. Every assessment this
module produces therefore travels with its own caveat, and `Assessment` is
built so that a prediction cannot be constructed without one. See
`templates/health/methodology.html`, which renders the model's own card.

HOW IT REACHES THE MODEL
------------------------
The trained pipeline lives in the wearable-ml project at
``settings.WEARABLE_ML_DIR``. Four files are read, never written:

    models/svm_hr_only.joblib            the scikit-learn pipeline
    models/feature_config_hr_only.json   feature names, in the fitted order
    models/hr_training_distribution.json per-feature training statistics
    models/model_card.json               scores and limitations

Unpickling a scikit-learn pipeline needs scikit-learn importable in *this*
process, which is why the dashboard's requirements.txt pins the same
versions that project used. Those imports are deliberately kept inside the
functions that need them: a missing or broken ML stack then degrades this one
card instead of stopping the whole site from starting.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

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

# The watch reports roughly one reading every 3 s, so a 60 s window holds
# about 20. Fewer than half of that and the statistics are not worth
# computing, so the window is dropped.
MIN_SAMPLES_PER_WINDOW = 10

# Predictions made while asleep are not meaningful: the model has no sleep
# class and WESAD contains no sleep. They are still computed - hiding them
# would misrepresent how much of the day they are - but reported separately.
WAKING_START_HOUR = 8
WAKING_END_HOUR = 23

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
# support, so the card never shows them.
CLASS_WORDING = {
    1: ("Settled baseline signal",
        "Today's heart-rate pattern most resembles the calm baseline "
        "condition the model was trained on."),
    2: ("Elevated arousal signal",
        "Today's heart-rate pattern most resembles the stressed condition "
        "the model was trained on. That is not the same as saying you were "
        "stressed."),
    3: ("Light arousal signal",
        "Today's heart-rate pattern most resembles the amusement condition "
        "the model was trained on - raised, but not in the way the stress "
        "condition was."),
}

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
    """One day's assessment, or an explanation of why there isn't one.

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

    @classmethod
    def unavailable(cls, day, reason):
        """No prediction, and a plain explanation of why."""
        return cls(day=str(day), available=False, reason=reason)

    @classmethod
    def predicted(cls, day, dominant, shares, waking_shares, n_windows,
                  n_waking_windows, mean_bpm, ood_rows):
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
        )

    @property
    def caveat(self):
        """The one-line caveat shown beside the headline. Never empty."""
        if not self.available:
            return ""
        return (
            "Wrist heart rate only, from a model trained on laboratory stress "
            "- not a clinical or validated measure. {:.0%} of today's windows "
            "fall outside the range it was trained on ({})."
        ).format(self.worst_ood_share, self.worst_ood_feature)


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
    # missing or broken costs this one card and not the whole site.
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


def parse_samples(points):
    """Turn raw heart-rate data points into (datetime, bpm) pairs.

    The payload shape, confirmed against the live API rather than assumed:

        {"heartRate": {"beatsPerMinute": "71",
                       "sampleTime": {"civilTime": {"date": {...},
                                                    "time": {...}}}}}

    beatsPerMinute arrives as a STRING. Zero-valued time fields are omitted
    entirely rather than sent as 0, so midnight arrives as an empty dict.
    Civil times are local wall-clock with no offset, which is what we want -
    "this afternoon" is a local-time question.
    """
    samples = []
    for point in points:
        body = point.get("heartRate") or {}
        civil = (body.get("sampleTime") or {}).get("civilTime") or {}
        day_part = civil.get("date") or {}
        time_part = civil.get("time") or {}
        bpm = body.get("beatsPerMinute")

        if not day_part or bpm is None:
            continue

        samples.append((
            datetime(
                year=day_part["year"],
                month=day_part["month"],
                day=day_part["day"],
                hour=time_part.get("hours", 0),
                minute=time_part.get("minutes", 0),
                second=time_part.get("seconds", 0),
            ),
            float(bpm),
        ))

    samples.sort(key=lambda pair: pair[0])
    return samples


def build_windows(samples, feature_names):
    """Slide a 60 s window across the day and compute features for each.

    Windows are anchored to midnight and step every 30 s, so a window's start
    is a real clock time and can be reported as "this afternoon" rather than
    as an offset from whenever the first reading happened to arrive.

    Returns (list of feature dicts, list of window start datetimes). Raises
    ArousalUnavailable if the day holds too little data to window at all.
    """
    if not samples:
        raise ArousalUnavailable(
            "No heart-rate readings were returned for this day.")

    import numpy as np

    times = np.array([sample[0] for sample in samples], dtype="datetime64[s]")
    bpm = np.array([sample[1] for sample in samples], dtype=float)

    midnight = times[0].astype("datetime64[D]").astype("datetime64[s]")
    day_end = midnight + np.timedelta64(24 * 3600, "s")

    rows = []
    starts = []
    start = midnight
    step = np.timedelta64(STEP_SECONDS, "s")
    length = np.timedelta64(WINDOW_SECONDS, "s")

    while start + length <= day_end:
        first = int(np.searchsorted(times, start, side="left"))
        last = int(np.searchsorted(times, start + length, side="left"))

        if last - first >= MIN_SAMPLES_PER_WINDOW:
            stats = basic_features(bpm[first:last])
            rows.append({"HR_" + name: value for name, value in stats.items()})
            starts.append(start.astype(datetime))

        start = start + step

    if not rows:
        raise ArousalUnavailable(
            "Not enough heart-rate data for this day: no 60-second window "
            "held at least {} readings.".format(MIN_SAMPLES_PER_WINDOW))

    missing = [name for name in feature_names if name not in rows[0]]
    if missing:
        raise ArousalUnavailable(
            "The model expects features this code does not compute: {}."
            .format(", ".join(missing)))

    return rows, starts


# ---------------------------------------------------------------------------
# Out-of-distribution check
# ---------------------------------------------------------------------------

def out_of_distribution(rows, feature_names, distribution):
    """Measure how far today's features sit outside the training data.

    Same two measures script 05 uses in the wearable-ml project, against the
    same training statistics file:

      share_outside - the fraction of today's windows whose value falls
                      outside the min-max range seen in training. Blunt, but
                      it needs no assumption about the shape of either
                      distribution.
      median_shift  - how far today's median sits from the training mean,
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


def assess(points, day):
    """Turn one day of raw heart-rate points into an Assessment.

    Never raises for an expected problem: a missing model, an ML stack that
    is not installed, or a day with too little data all come back as an
    unavailable Assessment carrying a readable reason.
    """
    try:
        model, feature_names, distribution = load_artifacts()

        samples = parse_samples(points)
        rows, starts = build_windows(samples, feature_names)

        import pandas as pd

        # A DataFrame with named columns, in the order the pipeline was
        # fitted on. Order matters and is silent when wrong: feeding the
        # right five numbers in the wrong order raises nothing, it just
        # returns a different answer.
        frame = pd.DataFrame(rows)[feature_names]
        predictions = [int(value) for value in model.predict(frame)]

        waking = [
            prediction
            for prediction, start in zip(predictions, starts)
            if WAKING_START_HOUR <= start.hour < WAKING_END_HOUR
        ]

        dominant = max(_shares(predictions).items(), key=lambda kv: kv[1])[0]
        mean_bpm = sum(row["HR_mean"] for row in rows) / len(rows)

        return Assessment.predicted(
            day=day,
            dominant=dominant,
            shares=_shares(predictions),
            waking_shares=_shares(waking),
            n_windows=len(predictions),
            n_waking_windows=len(waking),
            mean_bpm=mean_bpm,
            ood_rows=out_of_distribution(rows, feature_names, distribution),
        )

    except ArousalUnavailable as exc:
        return Assessment.unavailable(day, str(exc))
    except Exception as exc:  # anything unforeseen: degrade, do not 500
        logger.exception("Arousal assessment failed for %s", day)
        return Assessment.unavailable(
            day, "The assessment could not be computed: {}".format(exc))


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_parts(user, day):
    return ("arousal", user.pk, str(day))


def cached_assessment(user, day):
    """Return a previously computed assessment, or None. Never computes."""
    from . import cache as health_cache

    return health_cache.cached_compute(
        _cache_parts(user, day), settings.AROUSAL_CACHE_SECONDS,
        produce=None, only_if_cached=True)


def store_unavailable(user, day, reason):
    """Cache an explanation so the card can show why there is no assessment.

    Used when the failure happens before `assess` is reached - a failed API
    fetch, say. Without this the card would fall back to "not computed yet"
    and give the visitor no idea that anything had gone wrong.
    """
    from . import cache as health_cache

    assessment = Assessment.unavailable(day, reason)
    health_cache.cached_compute(
        _cache_parts(user, day), settings.AROUSAL_CACHE_SECONDS,
        lambda: assessment, force=True)
    return assessment


def compute_assessment(client, user, day, *, force=False):
    """Compute (or reuse) an assessment for one day, and cache it.

    The heart-rate samples come from the same cached intraday fetch the rest
    of the dashboard uses, so asking for an assessment on a day already being
    viewed costs no extra API calls.
    """
    from . import cache as health_cache
    from . import datatypes

    def produce():
        fetched = health_cache.fetch_intraday(
            client, datatypes.get("heart-rate"), day, user=user, force=force)
        return assess(fetched.points, day)

    return health_cache.cached_compute(
        _cache_parts(user, day), settings.AROUSAL_CACHE_SECONDS,
        produce, force=force)
