"""Registry of the Google Health API data types this app reads.

Every name here was taken from the live API discovery document
(https://health.googleapis.com/$discovery/rest?version=v4, revision 20260916)
and the data-type reference at https://developers.google.com/health/data-types.

Two spellings matter and they are NOT the same:
  * ``slug``         - kebab-case, used in the URL path:
                       /v4/users/me/dataTypes/daily-resting-heart-rate/dataPoints
  * ``filter_field`` - snake_case, used inside AIP-160 filter expressions:
                       daily_resting_heart_rate.date >= "2026-09-01"
"""

from dataclasses import dataclass
from datetime import date

SCOPE_ACTIVITY = "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly"
SCOPE_METRICS = "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly"
SCOPE_SLEEP = "https://www.googleapis.com/auth/googlehealth.sleep.readonly"

# Interval, sample, daily and session types each expose a different filterable
# time field. Sessions use the interval start time, except sleep, which the API
# only lets you filter on its END time.
_TIME_FIELD_BY_KIND = {
    "interval": "{field}.interval.civil_start_time",
    "sample": "{field}.sample_time.civil_time",
    "daily": "{field}.date",
    "session": "{field}.interval.civil_start_time",
}


@dataclass(frozen=True)
class DataType:
    slug: str
    filter_field: str
    label: str
    kind: str
    scope: str
    payload_key: str
    max_page_size: int = 10000
    supports_daily_rollup: bool = False
    rollup_max_days: int = 90
    notes: str = ""
    time_filter_override: str | None = None
    rollup_value_key: str | None = None

    @property
    def time_filter_field(self) -> str:
        if self.time_filter_override:
            return self.time_filter_override
        return _TIME_FIELD_BY_KIND[self.kind].format(field=self.filter_field)

    def time_filter(self, start: date, end: date) -> str:
        """Build a half-open [start, end) AIP-160 filter expression.

        Only >= and < are supported by the API; there is no BETWEEN.
        """
        field = self.time_filter_field
        return f'{field} >= "{start.isoformat()}" AND {field} < "{end.isoformat()}"'


# Ordered roughly the way the dashboard will present them.
REGISTRY: dict[str, DataType] = {}


def _register(*data_types: DataType):
    for dt in data_types:
        REGISTRY[dt.slug] = dt


_register(
    DataType(
        slug="exercise",
        filter_field="exercise",
        label="Exercise sessions",
        kind="session",
        scope=SCOPE_ACTIVITY,
        payload_key="exercise",
        max_page_size=25,
        notes="Session type. The API caps both the default and the maximum page size at 25.",
    ),
    DataType(
        slug="steps",
        rollup_value_key="steps",
        filter_field="steps",
        label="Steps",
        kind="interval",
        scope=SCOPE_ACTIVITY,
        payload_key="steps",
        supports_daily_rollup=True,
        notes="list returns intraday buckets; use dailyRollUp for daily totals.",
    ),
    DataType(
        slug="distance",
        rollup_value_key="distance",
        filter_field="distance",
        label="Distance",
        kind="interval",
        scope=SCOPE_ACTIVITY,
        payload_key="distance",
        supports_daily_rollup=True,
    ),
    DataType(
        slug="active-minutes",
        rollup_value_key="activeMinutes",
        filter_field="active_minutes",
        label="Active minutes",
        kind="interval",
        scope=SCOPE_ACTIVITY,
        payload_key="activeMinutes",
        supports_daily_rollup=True,
        rollup_max_days=14,
    ),
    DataType(
        slug="heart-rate",
        rollup_value_key="heartRate",
        filter_field="heart_rate",
        label="Heart rate",
        kind="sample",
        scope=SCOPE_METRICS,
        payload_key="heartRate",
        supports_daily_rollup=True,
        rollup_max_days=14,
        notes="High-frequency samples. Prefer dailyRollUp over list for charts.",
    ),
    DataType(
        slug="daily-resting-heart-rate",
        filter_field="daily_resting_heart_rate",
        label="Resting heart rate (daily)",
        kind="daily",
        scope=SCOPE_METRICS,
        payload_key="dailyRestingHeartRate",
        # The docs list this under dailyRollUp, but the live API returns HTTP 400:
        # "DailyRollup is not supported for data type daily-resting-heart-rate,
        # but the following actions are supported: list, reconcile".
        # It is already one point per day, so list is what we want anyway.
        supports_daily_rollup=False,
    ),
    DataType(
        slug="heart-rate-variability",
        filter_field="heart_rate_variability",
        label="HRV (samples)",
        kind="sample",
        scope=SCOPE_METRICS,
        payload_key="heartRateVariability",
        notes="RMSSD in milliseconds, plus a standard deviation.",
    ),
    DataType(
        slug="daily-heart-rate-variability",
        filter_field="daily_heart_rate_variability",
        label="HRV (daily)",
        kind="daily",
        scope=SCOPE_METRICS,
        payload_key="dailyHeartRateVariability",
        # Same as daily-resting-heart-rate: documented as rollup-capable, but the
        # live API rejects it with HTTP 400. Already daily granularity.
        supports_daily_rollup=False,
        notes="Daily RMSSD average, deep-sleep RMSSD and entropy.",
    ),
    DataType(
        slug="oxygen-saturation",
        filter_field="oxygen_saturation",
        label="SpO2 (samples)",
        kind="sample",
        scope=SCOPE_METRICS,
        payload_key="oxygenSaturation",
    ),
    DataType(
        slug="daily-oxygen-saturation",
        filter_field="daily_oxygen_saturation",
        label="SpO2 (daily)",
        kind="daily",
        scope=SCOPE_METRICS,
        payload_key="dailyOxygenSaturation",
        notes="Average plus the upper/lower bound of the overnight confidence interval.",
    ),
    DataType(
        slug="daily-respiratory-rate",
        filter_field="daily_respiratory_rate",
        label="Respiratory rate (daily)",
        kind="daily",
        scope=SCOPE_METRICS,
        payload_key="dailyRespiratoryRate",
    ),
    DataType(
        slug="respiratory-rate-sleep-summary",
        filter_field="respiratory_rate_sleep_summary",
        label="Respiratory rate by sleep stage",
        kind="sample",
        scope=SCOPE_METRICS,
        payload_key="respiratoryRateSleepSummary",
        notes="Breaks the night down into light / deep / REM statistics.",
    ),
    DataType(
        slug="sleep",
        filter_field="sleep",
        label="Sleep",
        kind="session",
        scope=SCOPE_SLEEP,
        payload_key="sleep",
        max_page_size=25,
        time_filter_override="sleep.interval.civil_end_time",
        notes=(
            "Filtered on interval.civil_end_time, not start time. The API exposes no "
            "sleep score; efficiency is computed from summary.minutesAsleep and "
            "summary.minutesInSleepPeriod."
        ),
    ),
    DataType(
        slug="daily-vo2-max",
        filter_field="daily_vo2_max",
        label="VO2 max (daily cardio fitness)",
        kind="daily",
        scope=SCOPE_ACTIVITY,
        payload_key="dailyVo2Max",
        notes=(
            "The passive daily estimate, with a cardioFitnessLevel label and an "
            "'estimated' flag that marks low-confidence values. list and reconcile only."
        ),
    ),
    DataType(
        slug="run-vo2-max",
        rollup_value_key="runVo2Max",
        filter_field="run_vo2_max",
        label="VO2 max (from runs)",
        kind="sample",
        scope=SCOPE_ACTIVITY,
        payload_key="runVo2Max",
        supports_daily_rollup=True,
        notes="Only populated by outdoor runs recorded with GPS.",
    ),
    DataType(
        slug="vo2-max",
        filter_field="vo2_max",
        label="VO2 max (measured)",
        kind="sample",
        scope=SCOPE_ACTIVITY,
        payload_key="vo2Max",
    ),
)

def get(slug: str) -> DataType | None:
    return REGISTRY.get(slug)


def all_types() -> list[DataType]:
    return list(REGISTRY.values())
