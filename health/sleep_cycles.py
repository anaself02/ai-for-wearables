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
* A missed REM period joins two cycles into one. When the result exceeds the
  ceiling it is discarded and counted, but a miss that lands under the ceiling
  is indistinguishable from one long cycle and inflates the median. Nothing in
  the data can correct for that, so `dropped_long` is surfaced instead: it is
  the only available evidence of how often REM is being missed at all.
* The first cycle of a night is normally the shortest. The estimator measures
  that rather than assuming it, because if the two come out equal the staging
  is probably not resolving cycle architecture, and the reader should be told.
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

#: Bedtime options are chosen by how much sleep they buy, not by a fixed set
#: of cycle counts. At the textbook 90 minutes, four to six cycles lands on
#: 6h/7h30/9h and a fixed (4, 5, 6) works. At a measured 110 it would offer
#: eleven hours in bed, which is not advice anyone wants, so the counts follow
#: the cycle length instead.
MIN_SENSIBLE_SLEEP = 330   # 5h30
MAX_SENSIBLE_SLEEP = 570   # 9h30
MAX_OPTIONS = 3

#: Used only if a cycle length is so long or short that nothing lands in the
#: sensible band at all.
CYCLE_OPTIONS = (4, 5, 6)


def cycle_counts(cycle_minutes: int) -> tuple:
    """Cycle counts whose total sleep lands in a range worth suggesting."""
    counts = [
        count
        for count in range(2, 10)
        if MIN_SENSIBLE_SLEEP <= count * cycle_minutes <= MAX_SENSIBLE_SLEEP
    ]
    return tuple(counts[:MAX_OPTIONS]) or CYCLE_OPTIONS


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

    lengths, short, overlong = [], 0, 0
    for earlier, later in zip(periods, periods[1:]):
        minutes = (later[0] - earlier[0]).total_seconds() / 60
        if minutes < MIN_PLAUSIBLE_CYCLE:
            short += 1
        elif minutes > MAX_PLAUSIBLE_CYCLE:
            # Almost certainly a missed REM period: the interval spans two
            # cycles or more. Counted separately because the count is evidence
            # about the estimate itself - see Estimate.dropped_long.
            overlong += 1
        else:
            lengths.append(round(minutes, 1))

    return {
        "cycles": lengths,
        "rem_periods": len(periods),
        "dropped": short + overlong,
        "dropped_short": short,
        "dropped_long": overlong,
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

    #: Intervals too short to be a cycle: stage fragmentation.
    dropped_short: int = 0
    #: Intervals too long to be one cycle, which means a REM period was missed.
    #: This one is not just bookkeeping. A miss long enough to exceed the
    #: ceiling gets discarded, but a miss that lands under it is counted as a
    #: single long cycle and silently inflates the median. The visible count is
    #: the only evidence available about how often that is happening.
    dropped_long: int = 0

    #: The first cycle of a night is normally the shortest. If these two come
    #: out equal, the staging is probably not resolving cycle architecture.
    first_cycle_minutes: int | None = None
    later_cycle_minutes: int | None = None

    per_night: list = field(default_factory=list)


def estimate(points) -> Estimate:
    """Median cycle length across every night with usable stage transitions."""
    lengths, per_night = [], []
    nights_with_stages = nights_contributing = 0
    dropped = dropped_short = dropped_long = 0

    for point in points or []:
        night = night_cycles(point)
        if not night["has_stages"]:
            continue
        nights_with_stages += 1
        dropped += night["dropped"]
        dropped_short += night["dropped_short"]
        dropped_long += night["dropped_long"]
        if night["cycles"]:
            nights_contributing += 1
            lengths.extend(night["cycles"])
            per_night.append(night["cycles"])

    result = Estimate(
        nights_with_stages=nights_with_stages,
        nights_contributing=nights_contributing,
        cycles_measured=len(lengths),
        cycles_dropped=dropped,
        dropped_short=dropped_short,
        dropped_long=dropped_long,
        per_night=per_night,
    )

    firsts = [night[0] for night in per_night]
    laters = [value for night in per_night for value in night[1:]]
    if firsts:
        result.first_cycle_minutes = round(median(firsts))
    if laters:
        result.later_cycle_minutes = round(median(laters))

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
             counts=None) -> list:
    """Bedtime for each cycle count, working back from a target wake time.

    `wake_time` is a datetime.time. The date is irrelevant to the arithmetic
    and the result is a wall clock, so this deliberately does not care which
    day it is or what timezone the sleeper is in.
    """
    anchor = datetime(2000, 1, 2, wake_time.hour, wake_time.minute)

    options = []
    for count in counts if counts is not None else cycle_counts(cycle_minutes):
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
