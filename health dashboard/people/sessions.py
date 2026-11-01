"""Labelled sessions: entering them, their coverage, and grading the WESAD models.

LABELS ONLY GRADE
-----------------
A session is a span a person labelled themselves. Its label is compared with
predictions of models trained on WESAD (health/wesad_binary.py) and is used for
nothing else: nothing here trains, tunes, calibrates, sets a threshold or
adapts. This module makes no fit call at all; a test scans it and runs it with
every fit and save patched to raise. The rule below was written down first, in
docs/SESSIONS_PROTOCOL.md and as Amendment 2 of wearable-ml/PRE_DECLARED.md.

THE LABELLING RULE
------------------
Windows are the shared ones (health.arousal.window_features: 60 s, 30 s step,
anchored to local midnight, at least 10 readings). A window is labelled only if
it lies fully inside one session and starts at least LABEL_LAG_SECONDS after
the session start. Primary score: rest (non-stress, 0) against stress_task
(stress, 1). recovery and natural_stress are reported per session only.

WHOSE DATA
----------
Everything takes a Person and reads through people.repository. A person's
sessions are only ever graded against their own readings. The free-text note
is never logged, exported or returned by anything in this module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from health import arousal
from health.wesad_binary import FEATURE_NAMES

from . import repository
from .models import Session

LABEL_LAG_SECONDS = 60
MIN_WINDOWS_PER_CLASS = 30
COVERAGE_WARNING = 0.80

#: kind -> binary label for the primary score; other kinds are not scored.
PRIMARY_LABELS = {Session.Kind.REST: 0, Session.Kind.STRESS_TASK: 1}
CLASS_NAMES = {0: "non-stress", 1: "stress"}

NOT_ENOUGH = ("Not enough labelled data to score: at least {} labelled windows "
              "are needed in each class.".format(MIN_WINDOWS_PER_CLASS))
NO_INTERVALS = ("No confidence interval is possible with this few sessions, and "
                "data from one person do not generalise to anyone else.")


class SessionError(ValueError):
    """A session that cannot be stored as entered."""


# ---------------------------------------------------------------------------
# Local time in, UTC stored
# ---------------------------------------------------------------------------

def local_to_utc(day, clock, tz_name):
    """A local date and time -> an aware UTC datetime.

    On a daylight-saving day a clock time can not exist (the spring gap) or
    happen twice (the autumn hour). Both are refused rather than guessed.
    """
    tz = ZoneInfo(tz_name)
    naive = datetime.combine(day, clock)
    early = naive.replace(tzinfo=tz, fold=0)
    late = naive.replace(tzinfo=tz, fold=1)
    # A time in the spring gap does not survive a round trip through UTC.
    back = early.astimezone(timezone.utc).astimezone(tz)
    if back.replace(tzinfo=None) != naive:
        raise SessionError(
            "{} on {} does not exist in {} (the clocks go forward).".format(
                clock.strftime("%H:%M"), day, tz_name))
    if early.utcoffset() != late.utcoffset():
        raise SessionError(
            "{} on {} happens twice in {} (the clocks go back). Pick a time "
            "outside that hour.".format(clock.strftime("%H:%M"), day, tz_name))
    return early.astimezone(timezone.utc)


def check_span(person, start_utc, end_utc, exclude_pk=None):
    """End after start, and no overlap with the person's other sessions."""
    repository.require_person(person)
    if end_utc <= start_utc:
        raise SessionError("The end must be after the start.")
    clash = Session.objects.filter(person=person, start_utc__lt=end_utc,
                                   end_utc__gt=start_utc)
    if exclude_pk is not None:
        clash = clash.exclude(pk=exclude_pk)
    if clash.exists():
        raise SessionError("This session overlaps another of your sessions.")


def numbered(person):
    """The person's sessions in time order, each with its number (1..N)."""
    repository.require_person(person)
    return list(enumerate(Session.objects.filter(person=person).order_by("start_utc"), 1))


def _epoch(moment):
    return int(moment.timestamp())


# ---------------------------------------------------------------------------
# Coverage and windows
# ---------------------------------------------------------------------------

def coverage(epochs, start, end, tz_name):
    """Readings in the session, first and last (local), share of minutes with data."""
    tz = ZoneInfo(tz_name)
    minutes = max(1, -(-(end - start) // 60))
    covered = len(set(((epochs - start) // 60).tolist())) if len(epochs) else 0
    share = covered / minutes
    return {
        "readings": int(len(epochs)),
        "first": (datetime.fromtimestamp(int(epochs[0]), timezone.utc).astimezone(tz)
                  if len(epochs) else None),
        "last": (datetime.fromtimestamp(int(epochs[-1]), timezone.utc).astimezone(tz)
                 if len(epochs) else None),
        "share": share,
        "low": share < COVERAGE_WARNING,
    }


def label_window_starts(start, end, tz_name):
    """Starts of the windows that may carry this session's label.

    On the shared grid (local midnight of the session's start day, 30 s step),
    starting at least LABEL_LAG_SECONDS after the session start and ending by
    the session end. A window crossing either boundary is never included.
    """
    local_day = datetime.fromtimestamp(start, timezone.utc).astimezone(ZoneInfo(tz_name)).date()
    anchor = repository.local_day_bounds(local_day, tz_name)[0]
    return arousal.grid_starts(anchor, start + LABEL_LAG_SECONDS,
                               end - arousal.WINDOW_SECONDS)


def session_windows(person):
    """[(number, session, readings, coverage, feature rows)] for the person."""
    out = []
    for number, session in numbered(person):
        start, end = _epoch(session.start_utc), _epoch(session.end_utc)
        epochs, bpm = repository.readings_between(person, start, end)
        rows, _, _ = arousal.window_features(
            epochs, bpm, label_window_starts(start, end, person.timezone))
        out.append((number, session, epochs,
                    coverage(epochs, start, end, person.timezone), rows))
    return out


def coverage_rows(person):
    """Per session coverage only, for the list (no model needed)."""
    out = []
    for number, session in numbered(person):
        start, end = _epoch(session.start_utc), _epoch(session.end_utc)
        epochs, _ = repository.readings_between(person, start, end)
        out.append({"number": number, "session": session,
                    "coverage": coverage(epochs, start, end, person.timezone)})
    return out


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def metrics(truth, predicted):
    """Macro-F1, balanced accuracy, per-class precision/recall, confusion matrix."""
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    precision, recall, _, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1],
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class": [{"name": CLASS_NAMES[k], "precision": float(precision[k]),
                       "recall": float(recall[k]), "support": int(support[k])}
                      for k in (0, 1)],
        # rows = actual non-stress, stress; columns = predicted non-stress, stress
        "confusion": [[int(v) for v in row] for row in matrix],
    }


def predict_all(models, rows):
    """{model name: predictions} for the same windows, from WESAD-fitted models."""
    import numpy as np
    import pandas as pd

    if not rows:
        return {"majority": [], "rule": [], "svm": []}
    frame = pd.DataFrame(rows)[FEATURE_NAMES]
    return {
        "majority": [models.majority] * len(rows),
        "rule": (np.asarray(frame["HR_mean"]) > models.threshold).astype(int).tolist(),
        "svm": [int(v) for v in models.svm.predict(frame)],
    }


MODEL_NAMES = [("majority", "Majority class (WESAD)"),
               ("rule", "Rule: HR_mean threshold (WESAD)"),
               ("svm", "Binary SVM (WESAD)")]


@dataclass(frozen=True)
class SessionScore:
    """A person's graded sessions, or why there is no grade.

    Build through `unavailable` or `graded`; `graded` cannot be called without
    the out-of-distribution evidence, so a score cannot exist without its caveat.
    """

    available: bool
    reason: str = ""
    enough: bool = False
    counts: dict = field(default_factory=dict)
    results: list = field(default_factory=list)       # [(label, metrics)]
    per_session: list = field(default_factory=list)
    ood_rows: list = field(default_factory=list)
    worst_ood_feature: str = ""
    worst_ood_share: float = 0.0
    devices: list = field(default_factory=list)
    spacing: dict = field(default_factory=dict)
    threshold: float = 0.0
    n_wesad_windows: int = 0
    n_wesad_subjects: int = 0

    @classmethod
    def unavailable(cls, reason):
        return cls(available=False, reason=reason)

    @classmethod
    def graded(cls, ood_rows, **values):
        if not ood_rows:
            raise ValueError("A session score may not be created without its "
                             "out-of-distribution evidence.")
        worst = max(ood_rows, key=lambda row: row["share_outside"])
        return cls(available=True, ood_rows=ood_rows, worst_ood_feature=worst["feature"],
                   worst_ood_share=worst["share_outside"], **values)

    @property
    def out_of_distribution(self):
        return self.worst_ood_share >= arousal.OOD_FLAG_SHARE

    @property
    def caveat(self):
        if not self.available:
            return ""
        return ("Wrist heart rate only, graded against your own labels with models "
                "trained on laboratory stress (WESAD) - not a clinical or validated "
                "measure. {:.0%} of the labelled windows fall outside the WESAD "
                "training range ({}).").format(self.worst_ood_share, self.worst_ood_feature)


def score(person):
    """Grade the WESAD binary models against the person's own session labels."""
    from health import wesad_binary

    repository.require_person(person)
    windows = session_windows(person)
    if not windows:
        return SessionScore.unavailable("No sessions recorded yet.")
    try:
        models = wesad_binary.train()
    except wesad_binary.WesadUnavailable as exc:
        return SessionScore.unavailable(str(exc))

    import numpy as np

    truth, primary = [], {name: [] for name, _ in MODEL_NAMES}
    all_rows, per_session, epochs_seen = [], [], []
    for number, session, epochs, cover, rows in windows:
        predictions = predict_all(models, rows)
        label = PRIMARY_LABELS.get(session.kind)
        if label is not None:
            truth.extend([label] * len(rows))
            for name, _ in MODEL_NAMES:
                primary[name].extend(predictions[name])
        all_rows.extend(rows)
        if len(epochs):
            epochs_seen.append(epochs)
        start = _epoch(session.start_utc)
        per_session.append({
            "number": number, "kind": session.kind,
            "kind_label": session.get_kind_display(),
            "primary": label is not None,
            "start": start, "end": _epoch(session.end_utc),
            "rating": session.rating, "coverage": cover, "windows": len(rows),
            "svm_stress": (float(np.mean(predictions["svm"])) if rows else None),
            "rule_stress": (float(np.mean(predictions["rule"])) if rows else None),
        })

    counts = {"non_stress": truth.count(0), "stress": truth.count(1),
              "other": len(all_rows) - len(truth)}
    if not all_rows:
        return SessionScore.unavailable(
            "No labelled window yet: no window inside a session (after the {} s "
            "buffer) held at least {} readings.".format(
                LABEL_LAG_SECONDS, arousal.MIN_SAMPLES_PER_WINDOW))

    enough = min(counts["non_stress"], counts["stress"]) >= MIN_WINDOWS_PER_CLASS
    results = ([(title, metrics(truth, primary[name])) for name, title in MODEL_NAMES]
               if enough else [])
    gaps = np.concatenate([np.diff(e) for e in epochs_seen]) if epochs_seen else np.array([])
    return SessionScore.graded(
        ood_rows=arousal.out_of_distribution(all_rows, FEATURE_NAMES,
                                             {"features": models.distribution}),
        enough=enough, counts=counts, results=results, per_session=per_session,
        devices=sorted({name for _, s, _, _, _ in windows
                        for name in repository.device_names_between(
                            person, _epoch(s.start_utc), _epoch(s.end_utc))}),
        spacing={"median": float(np.median(gaps)) if len(gaps) else None},
        threshold=models.threshold, n_wesad_windows=models.n_windows,
        n_wesad_subjects=models.n_subjects,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_COLUMNS = ["session", "kind", "in_primary_score", "start_minute",
                  "duration_minutes", "readings", "coverage_share",
                  "labelled_windows", "rating", "svm_share_stress",
                  "rule_share_stress", "device"]


def export_rows(result):
    """Session-level rows with no dates and no notes: "Session N", minutes since
    the first session's start, and the graded shares."""
    if not result.available:
        return []
    first = min(row["start"] for row in result.per_session)
    device = "; ".join(result.devices)

    def fmt(value):
        return "" if value is None else "{:.3f}".format(value)

    return [{
        "session": "Session {}".format(row["number"]),
        "kind": row["kind"],
        "in_primary_score": "yes" if row["primary"] else "no",
        "start_minute": (row["start"] - first) // 60,
        "duration_minutes": round((row["end"] - row["start"]) / 60, 1),
        "readings": row["coverage"]["readings"],
        "coverage_share": fmt(row["coverage"]["share"]),
        "labelled_windows": row["windows"],
        "rating": "" if row["rating"] is None else row["rating"],
        "svm_share_stress": fmt(row["svm_stress"]),
        "rule_share_stress": fmt(row["rule_stress"]),
        "device": device,
    } for row in result.per_session]

