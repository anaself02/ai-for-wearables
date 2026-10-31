"""Dashboard presentation layer: API data points -> Chart.js-shaped series.

Every extractor here was written against a real response captured from the live
API on 2026-09-19, not from the reference docs. The shapes are irregular enough
that guessing does not work:

  * int64 fields arrive as JSON **strings** (`countSum`, `beatsPerMinute`,
    `minutesAsleep`) while floats arrive as **numbers** (`beatsPerMinuteAvg`,
    `averagePercentage`). Both have to be coerced.
  * dailyRollUp points carry their date at `civilStartTime.date`; `list` points
    carry it inside the type's own payload at `<payloadKey>.date`.
  * `active-minutes` does not roll up to a scalar - it returns one entry per
    activity level that has to be summed.
  * Responses come back newest-first, so every series is sorted before charting.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from . import datatypes

# Presentation groups.
DAILY = "daily"
EVENT = "event"
#: A live data type exists but returns nothing for this user.
UNAVAILABLE = "unavailable"
#: No live endpoint exists at all - historical import is the only source.
LEGACY = "legacy"

# Where a metric's daily series comes from.
SOURCE_ROLLUP = "rollup"
SOURCE_LIST = "list"


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    number = as_float(value)
    return None if number is None else int(number)


def iso_date(date_obj: dict | None) -> str:
    """{'year': 2026, 'month': 9, 'day': 19} -> '2026-09-19'."""
    if not date_obj:
        return ""
    return "{:04d}-{:02d}-{:02d}".format(
        date_obj.get("year", 0), date_obj.get("month", 0), date_obj.get("day", 0)
    )


def clock_label(civil_time: dict | None) -> str:
    """{'hours': 16, 'minutes': 13} -> '16:13'. Midnight arrives as {}."""
    civil_time = civil_time or {}
    return "{:02d}:{:02d}".format(civil_time.get("hours", 0), civil_time.get("minutes", 0))


def rollup_date(point: dict) -> str:
    return iso_date(point.get("civilStartTime", {}).get("date"))


# ---------------------------------------------------------------------------
# series assembly
# ---------------------------------------------------------------------------


def chart_data(labels, datasets) -> dict:
    return {"labels": labels, "datasets": datasets}


def _single(rows, label, unit=""):
    """rows: iterable of (label, value). Sorted ascending, one dataset out."""
    rows = sorted((r for r in rows if r[0]), key=lambda r: r[0])
    return chart_data(
        [r[0] for r in rows],
        [{"label": label, "data": [r[1] for r in rows], "unit": unit}],
    )


def _daily_scalar(payload_key, value_key, label, unit, cast=as_float):
    """Extractor for a `list` response that is already one point per day."""

    def extract(points):
        rows = []
        for point in points:
            body = point.get(payload_key) or {}
            rows.append((iso_date(body.get("date")), cast(body.get(value_key))))
        return _single(rows, label, unit)

    return extract


# -- rollup extractors ------------------------------------------------------


def steps_series(points):
    rows = [
        (rollup_date(p), as_int((p.get("steps") or {}).get("countSum"))) for p in points
    ]
    return _single(rows, "Steps", "steps")


def distance_series(points):
    rows = []
    for point in points:
        millimetres = as_float((point.get("distance") or {}).get("millimetersSum"))
        kilometres = None if millimetres is None else round(millimetres / 1_000_000, 2)
        rows.append((rollup_date(point), kilometres))
    return _single(rows, "Distance", "km")


def active_minutes_series(points):
    """activeMinutes rolls up per activity level, so the levels get summed."""
    rows = []
    for point in points:
        levels = (point.get("activeMinutes") or {}).get(
            "activeMinutesRollupByActivityLevel", []
        )
        total = sum(as_int(level.get("activeMinutesSum")) or 0 for level in levels)
        rows.append((rollup_date(point), total or None))
    return _single(rows, "Active minutes", "min")


def heart_rate_series(points):
    """heart-rate rolls up to an average plus the day's min and max."""
    ordered = sorted(points, key=rollup_date)
    labels, averages, minimums, maximums = [], [], [], []
    for point in ordered:
        beats = point.get("heartRate") or {}
        labels.append(rollup_date(point))
        average = as_float(beats.get("beatsPerMinuteAvg"))
        averages.append(None if average is None else round(average, 1))
        minimums.append(as_float(beats.get("beatsPerMinuteMin")))
        maximums.append(as_float(beats.get("beatsPerMinuteMax")))
    return chart_data(
        labels,
        [
            {"label": "Average", "data": averages, "unit": "bpm"},
            {"label": "Min", "data": minimums, "unit": "bpm", "faint": True},
            {"label": "Max", "data": maximums, "unit": "bpm", "faint": True},
        ],
    )


# -- sleep ------------------------------------------------------------------

SLEEP_STAGES = ["DEEP", "REM", "LIGHT", "AWAKE"]


def _sleep_night_date(point) -> str:
    """Nights are keyed by the civil date they END on, matching the API filter."""
    interval = (point.get("sleep") or {}).get("interval") or {}
    civil_end = interval.get("civilEndTime")
    if civil_end:
        return iso_date(civil_end.get("date"))
    return (interval.get("endTime") or "")[:10]


def sleep_series(points):
    """One stacked bar per night, split by sleep stage."""
    ordered = sorted(points, key=_sleep_night_date)
    labels = [_sleep_night_date(p) for p in ordered]
    per_stage = {stage: [] for stage in SLEEP_STAGES}

    for point in ordered:
        summary = (point.get("sleep") or {}).get("summary") or {}
        minutes_by_stage = {
            entry.get("type"): as_int(entry.get("minutes"))
            for entry in summary.get("stagesSummary", [])
        }
        for stage in SLEEP_STAGES:
            per_stage[stage].append(minutes_by_stage.get(stage))

    return chart_data(
        labels,
        [
            {"label": stage.title(), "data": per_stage[stage], "unit": "min"}
            for stage in SLEEP_STAGES
        ],
    )


def sleep_rows(points):
    rows = []
    for point in sorted(points, key=_sleep_night_date, reverse=True):
        sleep = point.get("sleep") or {}
        summary = sleep.get("summary") or {}
        asleep = as_int(summary.get("minutesAsleep"))
        in_bed = as_int(summary.get("minutesInSleepPeriod"))
        stages = {
            entry.get("type"): as_int(entry.get("minutes"))
            for entry in summary.get("stagesSummary", [])
        }
        rows.append(
            {
                "date": _sleep_night_date(point),
                "asleep": asleep,
                "asleep_hm": _hours_minutes(asleep),
                "in_bed_hm": _hours_minutes(in_bed),
                "awake": as_int(summary.get("minutesAwake")),
                "to_fall_asleep": as_int(summary.get("minutesToFallAsleep")),
                # The API exposes no sleep score; efficiency is derived.
                "efficiency": (
                    round(asleep / in_bed * 100) if asleep and in_bed else None
                ),
                "deep": stages.get("DEEP"),
                "rem": stages.get("REM"),
                "light": stages.get("LIGHT"),
                "stages_available": bool(stages),
                # Naps come back as ordinary sleep sessions. `nap` is the
                # reliable flag: observed live, `mainSleep` is absent on one nap
                # and true on another, so it cannot be used to tell them apart.
                "is_nap": bool((sleep.get("metadata") or {}).get("nap", False)),
            }
        )
    return rows


# Display order for the hypnogram, shallowest at the top. "Restlessness" is not
# a stage type - it is drawn from `shortAwakenings`, which are nested inside the
# stage segments. Giving it its own band shows that layer without adding its
# minutes to the Awake total, which is what would have double-counted.
HYPNOGRAM_ROWS = [
    ("AWAKE", "awake", "Total awake time"),
    ("__RESTLESS__", "restless", "Restlessness"),
    ("REM", "rem", "REM"),
    ("LIGHT", "light", "Light"),
    ("DEEP", "deep", "Deep"),
]

RESTLESS_ROW = "__RESTLESS__"


def _utc_offset_seconds(raw) -> int:
    """'-14400s' -> -14400. Absent offset means UTC."""
    if not raw:
        return 0
    return int(float(str(raw).rstrip("s")))


def _parse_instant(raw):
    from datetime import datetime

    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def _local_clock(instant_raw, offset_raw) -> str:
    """Wall-clock HH:MM at the recording device, from the UTC stamp + its offset."""
    from datetime import timedelta as _timedelta

    moment = _parse_instant(instant_raw) + _timedelta(
        seconds=_utc_offset_seconds(offset_raw)
    )
    return f"{moment.hour:02d}:{moment.minute:02d}"


def hypnogram(point) -> dict | None:
    """One night's stage segments, laid out for a stacked timeline chart.

    Verified against a live response: `stages` tiles the night exactly (no gaps,
    no overlaps) and its per-stage durations match `summary.stagesSummary`
    minute for minute, so totals are computed from the segments themselves.

    `shortAwakenings` are deliberately kept separate. They are nested *within*
    the stage segments - 21 of them on a night whose AWAKE count is 3 - so
    adding them to the Awake total would double-count.

    Returns None for a night with no stage data (a `classic` sleep record).
    """
    sleep = (point or {}).get("sleep") or {}
    segments = sleep.get("stages") or []
    interval = sleep.get("interval") or {}
    if not segments or not interval.get("startTime"):
        return None

    origin = _parse_instant(interval["startTime"])

    def minutes_from_start(raw):
        return (_parse_instant(raw) - origin).total_seconds() / 60

    total_span = minutes_from_start(interval.get("endTime", interval["startTime"]))
    if total_span <= 0:
        return None

    restless = sleep.get("shortAwakenings") or []

    rows = []
    total_segments = 0

    for stage_type, key, label in HYPNOGRAM_ROWS:
        source = restless if stage_type == RESTLESS_ROW else [
            s for s in segments if s.get("type") == stage_type
        ]

        band = []
        minutes = 0.0
        for segment in source:
            start = minutes_from_start(segment["startTime"])
            end = minutes_from_start(segment["endTime"])
            if end <= start:
                continue
            minutes += end - start
            # Percentages, so the band is laid out in CSS and stays responsive.
            band.append(
                {
                    "left": round(max(start, 0.0) / total_span * 100, 4),
                    "width": round((end - start) / total_span * 100, 4),
                    "from": _local_clock(
                        segment["startTime"], segment.get("startUtcOffset")
                    ),
                    "to": _local_clock(segment["endTime"], segment.get("endUtcOffset")),
                    "minutes": round(end - start, 1),
                }
            )

        band.sort(key=lambda s: s["left"])
        total_segments += len(band)
        rows.append(
            {
                "type": stage_type,
                "key": key,
                "label": label,
                "minutes": round(minutes),
                "duration": _hours_minutes(round(minutes)) or "0m",
                "count": len(band),
                "segments": band,
                # Restlessness sits inside the stages rather than beside them.
                "nested": stage_type == RESTLESS_ROW,
            }
        )

    start_minute = _minute_of_day(
        interval["startTime"], interval.get("startUtcOffset")
    )

    def clock_at(offset_minutes):
        total = int(round(start_minute + offset_minutes)) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"

    return {
        "rows": rows,
        "total_minutes": round(total_span),
        "bedtime": _local_clock(interval["startTime"], interval.get("startUtcOffset")),
        "midpoint": clock_at(total_span / 2),
        "wake_time": _local_clock(
            interval.get("endTime", interval["startTime"]), interval.get("endUtcOffset")
        ),
        "start_minute_of_day": start_minute,
        "segment_count": total_segments,
    }


def _minute_of_day(instant_raw, offset_raw) -> int:
    """Local minutes past midnight, so the axis can print real clock times."""
    from datetime import timedelta as _timedelta

    moment = _parse_instant(instant_raw) + _timedelta(
        seconds=_utc_offset_seconds(offset_raw)
    )
    return moment.hour * 60 + moment.minute


def _hours_minutes(minutes):
    """450 -> '7h 30m', 43 -> '43m'. A leading '0h' just reads as noise."""
    if not minutes:
        return None
    hours, remainder = divmod(int(minutes), 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


# -- exercise ---------------------------------------------------------------


def _duration_seconds(value):
    """'2611.095s' -> 2611.095"""
    if not value:
        return None
    return as_float(str(value).rstrip("s"))


def exercise_rows(points):
    rows = []
    for point in points:
        exercise = point.get("exercise") or {}
        interval = exercise.get("interval") or {}
        metrics = exercise.get("metricsSummary") or {}
        seconds = _duration_seconds(exercise.get("activeDuration"))
        distance_mm = as_float(metrics.get("distanceMillimeters"))
        rows.append(
            {
                "start": (interval.get("startTime") or "")[:16].replace("T", " "),
                "date": (interval.get("startTime") or "")[:10],
                "type": exercise.get("exerciseType", ""),
                "name": exercise.get("displayName", ""),
                "duration_hm": _hours_minutes(int(seconds // 60)) if seconds else None,
                "minutes": int(seconds // 60) if seconds else None,
                "calories": as_int(metrics.get("caloriesKcal")),
                "km": None if distance_mm is None else round(distance_mm / 1_000_000, 2),
                "steps": as_int(metrics.get("steps")),
                "avg_hr": as_int(metrics.get("averageHeartRateBeatsPerMinute")),
                "has_gps": (exercise.get("exerciseMetadata") or {}).get("hasGps", False),
            }
        )
    rows.sort(key=lambda row: row["start"], reverse=True)
    return rows


def exercise_series(points):
    """Minutes of exercise per day, so the card still has a trend to show."""
    per_day: dict[str, int] = {}
    for row in exercise_rows(points):
        if row["date"] and row["minutes"]:
            per_day[row["date"]] = per_day.get(row["date"], 0) + row["minutes"]
    return _single(list(per_day.items()), "Exercise minutes", "min")


# -- intraday drill-down ----------------------------------------------------


def _sample_intraday(payload_key, value_key, label, unit, cast=as_float):
    def extract(points):
        rows = []
        for point in points:
            body = point.get(payload_key) or {}
            sample_time = body.get("sampleTime") or {}
            rows.append(
                (clock_label(sample_time.get("civilTime", {}).get("time")), cast(body.get(value_key)))
            )
        return _single(rows, label, unit)

    return extract


def _interval_intraday(payload_key, value_key, label, unit, transform=as_int):
    """Extractor for interval types, whose time sits on interval.civilStartTime."""

    def extract(points):
        rows = []
        for point in points:
            body = point.get(payload_key) or {}
            interval = body.get("interval") or {}
            rows.append(
                (
                    clock_label((interval.get("civilStartTime") or {}).get("time")),
                    transform(body.get(value_key)),
                )
            )
        return _single(rows, label, unit)

    return extract


def _mm_to_km(value):
    millimetres = as_float(value)
    return None if millimetres is None else round(millimetres / 1_000_000, 3)


INTRADAY_EXTRACTORS = {
    "heart-rate": _sample_intraday("heartRate", "beatsPerMinute", "Heart rate", "bpm", as_int),
    "oxygen-saturation": _sample_intraday(
        "oxygenSaturation", "percentage", "SpO2", "%"
    ),
    "heart-rate-variability": _sample_intraday(
        "heartRateVariability",
        "rootMeanSquareOfSuccessiveDifferencesMilliseconds",
        "HRV (RMSSD)",
        "ms",
    ),
    "steps": _interval_intraday("steps", "count", "Steps", "steps"),
    "distance": _interval_intraday(
        "distance", "millimeters", "Distance", "km", _mm_to_km
    ),
}


# ---------------------------------------------------------------------------
# the metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    slug: str
    label: str
    group: str
    chart: str  # line | bar | stacked-bar | none
    unit: str = ""
    source: str = SOURCE_LIST
    series: Callable[[list[dict]], dict] | None = None
    rows: Callable[[list[dict]], list[dict]] | None = None
    #: data type to fetch for the single-day drill-down, if any
    intraday_slug: str | None = None
    #: HistoricalMetric.metric_type to fall back on when the live API has nothing.
    #: Only ever displayed as clearly-labelled legacy data, never as live.
    historical_metric_type: str | None = None
    summary_label: str = "Latest"
    empty_message: str = ""
    blurb: str = ""

    @property
    def data_type(self) -> datatypes.DataType | None:
        """None for LEGACY metrics, which have no live endpoint."""
        return datatypes.REGISTRY.get(self.slug)

    @property
    def is_api_backed(self) -> bool:
        """False when the card is rendered from stored history alone."""
        return self.group not in (UNAVAILABLE, LEGACY)

    @property
    def intraday_type(self) -> datatypes.DataType | None:
        if not self.intraday_slug:
            return None
        return datatypes.REGISTRY[self.intraday_slug]


METRICS: list[Metric] = [
    Metric(
        slug="steps",
        label="Steps",
        group=DAILY,
        chart="bar",
        unit="steps",
        source=SOURCE_ROLLUP,
        series=steps_series,
        intraday_slug="steps",
        summary_label="Today",
    ),
    Metric(
        slug="sleep",
        label="Sleep",
        group=EVENT,
        chart="stacked-bar",
        unit="min",
        series=sleep_series,
        rows=sleep_rows,
        summary_label="Last night",
        blurb="The API exposes no sleep score - efficiency is computed from minutes asleep over minutes in the sleep period.",
    ),
    Metric(
        slug="sleep-score",
        label="Sleep score",
        group=LEGACY,
        chart="line",
        unit="/100",
        historical_metric_type="sleep-score",
        summary_label="Last scored",
        empty_message=(
            "No sleep score data. The Google Health API has no sleep score "
            "endpoint - import one from a Google Takeout export."
        ),
        blurb=(
            "Google Health exposes sleep stages and minute counts but no score of "
            "any kind, so this can only come from a Takeout export. It is never "
            "refreshed by sync."
        ),
    ),
    Metric(
        slug="daily-resting-heart-rate",
        label="Resting heart rate",
        group=DAILY,
        chart="line",
        unit="bpm",
        series=_daily_scalar(
            "dailyRestingHeartRate", "beatsPerMinute", "Resting HR", "bpm", as_int
        ),
        summary_label="Today",
    ),
    Metric(
        slug="heart-rate",
        label="Heart rate",
        group=DAILY,
        chart="line",
        unit="bpm",
        source=SOURCE_ROLLUP,
        series=heart_rate_series,
        intraday_slug="heart-rate",
        summary_label="Today's average",
        blurb="Daily average with the day's min and max. Raw samples run to thousands of points, so the full-resolution trace lives on the single-day view.",
    ),
    Metric(
        slug="daily-heart-rate-variability",
        label="HRV",
        group=DAILY,
        chart="line",
        unit="ms",
        series=_daily_scalar(
            "dailyHeartRateVariability",
            "averageHeartRateVariabilityMilliseconds",
            "HRV (RMSSD)",
            "ms",
        ),
        intraday_slug="heart-rate-variability",
        summary_label="Last night",
    ),
    Metric(
        slug="daily-oxygen-saturation",
        label="Blood oxygen",
        group=DAILY,
        chart="line",
        unit="%",
        series=_daily_scalar(
            "dailyOxygenSaturation", "averagePercentage", "SpO2", "%"
        ),
        intraday_slug="oxygen-saturation",
        summary_label="Last night",
    ),
    Metric(
        slug="daily-respiratory-rate",
        label="Respiratory rate",
        group=DAILY,
        chart="line",
        unit="br/min",
        series=_daily_scalar(
            "dailyRespiratoryRate", "breathsPerMinute", "Breaths per minute", "br/min"
        ),
        summary_label="Last night",
    ),
    Metric(
        slug="distance",
        label="Distance",
        group=DAILY,
        chart="bar",
        unit="km",
        source=SOURCE_ROLLUP,
        series=distance_series,
        intraday_slug="distance",
        summary_label="Today",
    ),
    Metric(
        slug="active-minutes",
        label="Active minutes",
        group=DAILY,
        chart="bar",
        unit="min",
        source=SOURCE_ROLLUP,
        series=active_minutes_series,
        summary_label="Today",
    ),
    Metric(
        slug="exercise",
        label="Exercise sessions",
        group=EVENT,
        chart="bar",
        unit="min",
        series=exercise_series,
        rows=exercise_rows,
        summary_label="Sessions",
    ),
    Metric(
        slug="daily-vo2-max",
        historical_metric_type="vo2-max",
        label="VO2 max (daily)",
        group=UNAVAILABLE,
        chart="none",
        empty_message="No VO2 max data yet - requires a GPS-tracked outdoor run of 10+ minutes.",
    ),
    Metric(
        slug="run-vo2-max",
        historical_metric_type="vo2-max",
        label="VO2 max (from runs)",
        group=UNAVAILABLE,
        chart="none",
        empty_message="No VO2 max data yet - requires a GPS-tracked outdoor run of 10+ minutes.",
    ),
    Metric(
        slug="vo2-max",
        historical_metric_type="vo2-max",
        label="VO2 max (measured)",
        group=UNAVAILABLE,
        chart="none",
        empty_message="No VO2 max data yet - requires a GPS-tracked outdoor run of 10+ minutes.",
    ),
]

BY_SLUG = {metric.slug: metric for metric in METRICS}


def get(slug: str) -> Metric | None:
    return BY_SLUG.get(slug)


def fetchable() -> list[Metric]:
    """Metrics that need an API call. VO2 max and sleep score need none."""
    return [metric for metric in METRICS if metric.is_api_backed]


# ---------------------------------------------------------------------------
# series helpers
# ---------------------------------------------------------------------------


def tail(series: dict | None, count: int) -> dict | None:
    """Last `count` points of a chart series, for the overview sparklines.

    The dashboard fetches one long window and slices it, rather than issuing a
    second short-window request for the same metrics.
    """
    if not series or not series.get("labels"):
        return series
    return chart_data(
        series["labels"][-count:],
        [
            {**dataset, "data": dataset["data"][-count:]}
            for dataset in series["datasets"]
        ],
    )


def as_daily_pairs(series: dict | None, dataset_index: int = 0) -> list:
    """Chart series -> [(date, value)], which is what health.insights consumes.

    Reuses the extractors that are already pinned to real API responses instead
    of parsing the payloads a second time.
    """
    if not series or not series.get("datasets"):
        return []
    data = series["datasets"][dataset_index]["data"]
    pairs = []
    for label, value in zip(series["labels"], data):
        if value is None:
            continue
        try:
            day = date.fromisoformat(label)
        except (TypeError, ValueError):
            continue
        pairs.append((day, float(value)))
    return pairs


def latest_value(series: dict):
    """Last non-null value of the first dataset, for the summary tile."""
    if not series or not series.get("datasets"):
        return None, None
    dataset = series["datasets"][0]
    for label, value in zip(reversed(series["labels"]), reversed(dataset["data"])):
        if value is not None:
            return value, label
    return None, None
