"""Parsers for metrics that only exist in a Google Takeout export.

A metric earns a place here only if the live API cannot supply it. Verified
against a real export and a live 90-day query on 2026-09-19:

  * **Sleep score** - no endpoint exists at all. The API returns sleep stages and
    minute counts but no score, so this is genuinely unobtainable live.
  * **VO2 max** - the passive `demographic_vo2_max` estimate was retired in the
    Fitbit to Google Health migration; live VO2 max only comes from GPS runs.

Deliberately *not* imported, because the live API already serves them:
`Active Zone Minutes` (live returns 913 points/90d), `Computed Temperature`
(`daily-sleep-temperature-derivations`, 49 points), HRV and SpO2. `Stress Score`
and `Daily Readiness` are header-only files in the export - no rows to import.

Everything here reads local files. Nothing contacts Google.
"""

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from django.utils import timezone

logger = logging.getLogger(__name__)


class TakeoutParseError(Exception):
    """The file was found but could not be read as the expected format."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Takeout writes US-style MM/DD/YY in the VO2 max export. Confirmed against a real
# export whose records run 07/15/26 -> 09/11/26 one day apart: read as DD/MM that
# sequence is impossible.
VO2MAX_DATE_FORMAT = "%m/%d/%y %H:%M:%S"


def parse_vo2max_date(raw: str):
    return datetime.strptime(raw, VO2MAX_DATE_FORMAT).date()


def parse_local_date(raw: str):
    """ISO timestamp -> the civil date it falls on in the configured timezone.

    Sleep score rows are stamped in UTC at wake time. A night that ends at
    07:56 local is 11:56Z, so taking the UTC date happens to work - but only
    by luck, and it breaks for any wake time that crosses midnight UTC. The
    live sleep data keys each night by its civil END date, so this matches.
    """
    text = raw.strip().replace("Z", "+00:00")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        # Naive stamps in this export are already local wall-clock time.
        return moment.date()
    return timezone.localtime(moment).date()


def _number(value):
    if value is None or value == "" or value == "NaN":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value):
    number = _number(value)
    return None if number is None else int(number)


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def parse_vo2max(path: Path) -> list[dict]:
    """Global Export Data/demographic_vo2_max-<date>.json

    Each record carries a raw estimate and a filtered one. The filtered series
    is what Fitbit displayed as the Cardio Fitness Score - its error term
    shrinks as readings accumulate - so that is the value stored.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise TakeoutParseError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise TakeoutParseError(
            f"{path.name}: expected a JSON array, got {type(payload).__name__}"
        )

    readings = []
    for index, record in enumerate(payload):
        value = (record or {}).get("value") or {}
        filtered = value.get("filteredDemographicVO2Max")
        raw = value.get("demographicVO2Max")
        best = filtered if filtered is not None else raw

        if best is None or "dateTime" not in record:
            continue
        try:
            day = parse_vo2max_date(record["dateTime"])
        except ValueError:
            raise TakeoutParseError(
                f"{path.name} record {index}: cannot parse dateTime "
                f"{record['dateTime']!r} as {VO2MAX_DATE_FORMAT!r}"
            )

        readings.append(
            {
                "date": day,
                "value": round(float(best), 2),
                "extra": {
                    "raw_estimate": raw,
                    "error": value.get("filteredDemographicVO2MaxError"),
                    "source_file": path.name,
                },
            }
        )
    return readings


def parse_sleep_score(path: Path) -> list[dict]:
    """Sleep Score/sleep_score.csv

    Columns: sleep_log_entry_id, timestamp, overall_score, composition_score,
    revitalization_score, duration_score, deep_sleep_in_minutes,
    resting_heart_rate, restlessness.

    `overall_score` is the headline 0-100 figure; the sub-scores are often blank
    in real exports, so they are kept in `extra` rather than charted.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if rows and "overall_score" not in rows[0]:
        raise TakeoutParseError(
            f"{path.name}: no 'overall_score' column - columns are {list(rows[0])}"
        )

    readings = []
    for row in rows:
        score = _number(row.get("overall_score"))
        stamp = (row.get("timestamp") or "").strip()
        if score is None or not stamp:
            continue
        try:
            day = parse_local_date(stamp)
        except ValueError:
            raise TakeoutParseError(
                f"{path.name}: cannot parse timestamp {stamp!r}"
            )

        readings.append(
            {
                "date": day,
                "value": round(score, 1),
                # Keeps the whole row usable later without a re-import.
                "extra": {
                    "composition_score": _int(row.get("composition_score")),
                    "revitalization_score": _int(row.get("revitalization_score")),
                    "duration_score": _int(row.get("duration_score")),
                    "deep_sleep_in_minutes": _int(row.get("deep_sleep_in_minutes")),
                    "resting_heart_rate": _int(row.get("resting_heart_rate")),
                    "restlessness": _number(row.get("restlessness")),
                    "sleep_log_entry_id": row.get("sleep_log_entry_id"),
                    "timestamp": stamp,
                    "source_file": path.name,
                },
            }
        )
    return readings


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyImport:
    metric_type: str
    label: str
    glob: str
    unit: str
    source: str
    parser: Callable[[Path], list[dict]]
    #: Where in the export it lives, for error messages.
    expected_in: str
    why: str


IMPORTS: dict[str, LegacyImport] = {
    "vo2-max": LegacyImport(
        metric_type="vo2-max",
        label="VO2 max",
        glob="*demographic_vo2_max*.json",
        unit="ml/kg/min",
        source="Fitbit demographic estimate",
        parser=parse_vo2max,
        expected_in="Takeout/Google Health/Global Export Data/",
        why="The passive estimate was retired in the migration; live VO2 max needs a GPS run.",
    ),
    "sleep-score": LegacyImport(
        metric_type="sleep-score",
        label="Sleep score",
        glob="sleep_score*.csv",
        unit="/100",
        source="Fitbit sleep score",
        parser=parse_sleep_score,
        expected_in="Takeout/Google Health/Sleep Score/",
        why="The Google Health API exposes no sleep score of any kind.",
    ),
}


def collect(root: Path, only: str | None = None) -> dict[str, list[dict]]:
    """Find and parse every supported legacy export under `root`.

    `root` may be a single file, the folder holding it, or the Takeout root.
    Returns {metric_type: readings}, newest file last so later exports win.
    """
    wanted = [IMPORTS[only]] if only else list(IMPORTS.values())
    results: dict[str, list[dict]] = {}

    for spec in wanted:
        if root.is_file():
            paths = [root] if root.match(spec.glob) else []
        else:
            paths = sorted(root.rglob(spec.glob))
        if not paths:
            continue

        by_date: dict = {}
        for path in paths:
            for reading in spec.parser(path):
                # One reading per day; a later file's version wins.
                by_date[reading["date"]] = reading
        if by_date:
            results[spec.metric_type] = [by_date[d] for d in sorted(by_date)]

    return results
