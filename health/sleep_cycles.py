"""Sleep cycle length, estimated from recorded stage transitions.

A sleep cycle is conventionally measured from one REM onset to the next, so the
stage timeline the API returns is in principle enough to measure it: `stages`
carries real timestamps per segment rather than only the nightly totals.

Four things stop that from being a clean measurement, and all four are reported
alongside the number rather than hidden inside it.

* The stages are themselves a model output. Wrist staging agrees with clinical
  polysomnography on roughly 60-70% of epochs for four-class staging, so every
  cycle length here inherits that error. This is an estimate derived from an
  estimate, exactly like the arousal signal.
* Wearable hypnograms fragment. Short REM bouts separated by a few minutes of
  light sleep are one REM period physiologically, but arrive as several
  segments. Without a merge rule the estimator would report spurious
  twenty-minute cycles, so bouts closer together than MERGE_GAP are joined.
* The first cycle of a night is genuinely shorter than later ones. A single
  median flattens that, which is why the spread is reported with it.
* A handful of nights is not a sample. The estimate is marked unreliable, and
  the interface falls back to the textbook figure, until the gates below pass.

The textbook fallback is the familiar ~90 minute cycle plus ~15 minutes to fall
asleep. It is a population average, not a measurement of anyone.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

from .metrics import _parse_instant

#: A REM segment shorter than this is treated as noise, not a REM period.
MIN_REM_BOUT_MINUTES = 5.0

#: REM periods separated by less than this much non-REM are one period.
MERGE_GAP_MINUTES = 15.0

#: Cycles outside this range are dropped as staging artefacts rather than
#: averaged in. Physiological NREM-REM cycles sit around 70-120 minutes.
MIN_PLAUSIBLE_CYCLE = 45.0
MAX_PLAUSIBLE_CYCLE = 150.0

#: Gates before a personal estimate is offered in place of the textbook one.
MIN_NIGHTS = 5
MIN_CYCLES = 10

#: The standard assumption, used when the gates fail.
TEXTBOOK_CYCLE_MINUTES = 90
FALL_ASLEEP_MINUTES = 15

#: Cycle counts offered as bedtime options. Five and six cycles are 7.5 and 9
#: hours at the textbook length, which brackets the usual adult recommendation.
CYCLE_OPTIONS = (4, 5, 6)


def _rem_periods(sleep: dict) -> list:
    """Merged REM periods for one night, as (start, end) datetimes."""
    bouts = sorted(
        (
            (_parse_instant(segment["startTime"]), _parse_instant(segment["endTime"]))
            for segment in sleep.get("stages") or []
            if segment.get("type") == "REM"
            and segment.get("startTime")
            and segment.get("endTime")
        ),
        key=lambda bout: bout[0],
    )

    merged = []
    for start, end in bouts:
        if merged and (start - merged[-1][1]).total_seconds() / 60 < MERGE_GAP_MINUTES:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            continue
        merged.append((start, end))

    return [
        (start, end)
        for start, end in merged
        if (end - start).total_seconds() / 60 >= MIN_REM_BOUT_MINUTES
    ]


def night_cycles(point: dict) -> dict:
    """Cycle lengths for one night, plus what had to be discarded to get them."""
    sleep = (point or {}).get("sleep") or {}
    periods = _rem_periods(sleep)

    lengths, dropped = [], 0
    for earlier, later in zip(periods, periods[1:]):
        minutes = (later[0] - earlier[0]).total_seconds() / 60
        if MIN_PLAUSIBLE_CYCLE <= minutes <= MAX_PLAUSIBLE_CYCLE:
            lengths.append(round(minutes, 1))
        else:
            dropped += 1

    return {
        "cycles": lengths,
        "rem_periods": len(periods),
        "dropped": dropped,
        "has_stages": bool(sleep.get("stages")),
    }


@dataclass
class Estimate:
    """A personal cycle length, and an honest account of what it rests on."""

    #: True only when the gates pass. False means use `TEXTBOOK_CYCLE_MINUTES`.
    reliable: bool = False
    reason: str = ""

    cycle_minutes: int | None = None
    spread_minutes: int | None = None
    shortest: float | None = None
    longest: float | None = None

    nights_with_stages: int = 0
    nights_contributing: int = 0
    cycles_measured: int = 0
    cycles_dropped: int = 0

    per_night: list = field(default_factory=list)


def estimate(points) -> Estimate:
    """Median cycle length across every night with usable stage transitions."""
    lengths, per_night = [], []
    nights_with_stages = nights_contributing = dropped = 0

    for point in points or []:
        night = night_cycles(point)
        if not night["has_stages"]:
            continue
        nights_with_stages += 1
        dropped += night["dropped"]
        if night["cycles"]:
            nights_contributing += 1
            lengths.extend(night["cycles"])
            per_night.append(night["cycles"])

    result = Estimate(
        nights_with_stages=nights_with_stages,
        nights_contributing=nights_contributing,
        cycles_measured=len(lengths),
        cycles_dropped=dropped,
        per_night=per_night,
    )

    if not lengths:
        result.reason = (
            "No night in this window has two separate REM periods recorded, so "
            "there is no interval to measure a cycle across."
        )
        return result

    result.cycle_minutes = round(median(lengths))
    result.shortest = min(lengths)
    result.longest = max(lengths)
    # Half the central range, which describes the night-to-night spread without
    # implying the distribution is normal.
    ordered = sorted(lengths)
    lower = ordered[len(ordered) // 4]
    upper = ordered[(3 * len(ordered)) // 4]
    result.spread_minutes = round((upper - lower) / 2)

    if nights_contributing < MIN_NIGHTS or len(lengths) < MIN_CYCLES:
        result.reason = (
            f"Measured {len(lengths)} cycle"
            f"{'' if len(lengths) == 1 else 's'} across {nights_contributing} "
            f"night{'' if nights_contributing == 1 else 's'}. That is below the "
            f"{MIN_CYCLES} cycles over {MIN_NIGHTS} nights this needs before it "
            "is worth preferring to the standard figure."
        )
        return result

    result.reliable = True
    return result


def bedtimes(wake_time, cycle_minutes: int, *, latency_minutes: int = FALL_ASLEEP_MINUTES,
             counts=CYCLE_OPTIONS) -> list:
    """Bedtime for each cycle count, working back from a target wake time.

    `wake_time` is a datetime.time. The date is irrelevant to the arithmetic
    and the result is a wall clock, so this deliberately does not care which
    day it is or what timezone the sleeper is in.
    """
    anchor = datetime(2000, 1, 2, wake_time.hour, wake_time.minute)

    options = []
    for count in counts:
        asleep = count * cycle_minutes
        bedtime = anchor - timedelta(minutes=asleep + latency_minutes)
        options.append(
            {
                "cycles": count,
                "bedtime": bedtime.strftime("%H:%M"),
                "asleep_minutes": asleep,
                "asleep_label": f"{asleep // 60}h {asleep % 60:02d}m",
                "total_minutes": asleep + latency_minutes,
            }
        )
    return options
