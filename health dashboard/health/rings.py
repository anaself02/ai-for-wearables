"""The row of dials, and the three different things a full arc can mean.

Most of these dials are a fraction whose denominator is the reader's own median
over the window, not a target somebody invented. That is a deliberate refusal.
Ten thousand steps is the name of a 1960s Japanese pedometer, not a finding,
and the WHO's 150 weekly minutes describes moderate-to-vigorous activity while
this account's `active-minutes` sums LIGHT as well, so the two are not the same
quantity and comparing them would be arithmetic dressed as guidance.

A full ring therefore means "a typical day for you". Half a ring means the day
on show ran at half your usual, which is information rather than a verdict.
Nothing here is scored, graded or coloured for approval.

The day being shown is the most recent one that has readings, not the calendar
date. Google stamps a day's rollup some hours after the fact, so at breakfast
"today" is usually empty and a row of blank dials would be the first thing on
the page every morning. Whichever day it lands on is named on screen, so the
figure is never passed off as more current than it is.

That day is excluded from its own denominator: a day still in progress is a
partial count, and letting it drag the median down would flatter the very
comparison it is being made against.

Two dials do not work that way, and both say so on their face.

The sleep score is out of 100 because Fitbit chose 100, and it can never change
again. VO2 max is measured against a published norm table rather than against
this reader, and the reason is arithmetic rather than taste: the stored series
moves 4% across two months, so the latest reading sits at 99.7% of its own
median and would pin the arc full for ever. A dial that is always full reads as
a target hit every day, which is the one verdict this row exists to refuse - and
it would arrive here by accident rather than by decision. A percentile is
already a fraction of a hundred, moves when the reading moves (the same series
runs from the 68th down to the 61st over those two months), and is the half of
health.fitness that survives a modelled input, because it compares rather than
inverts. So the arc carries the percentile and the fitness age is not a dial at
all.
"""

from dataclasses import dataclass
from datetime import timedelta
from statistics import median

#: Geometry of the dial. The arc is drawn with stroke-dasharray, so the
#: template needs the circumference to set the gap.
RADIUS = 19
CIRCUMFERENCE = round(2 * 3.141592653589793 * RADIUS, 2)

#: Below this many prior days there is no meaningful median to divide by.
MIN_DAYS_FOR_BASELINE = 5

#: Short enough to sit under a 92px dial. The full sentence, with the link to
#: settings, belongs in the panel below, where there is room for it.
NEEDS_PROFILE = "Needs your date of birth and sex."
NEVER_IMPORTED = "Never imported."
OFF_TABLE = "Norms cover ages 20 to 79."


@dataclass
class Ring:
    """One dial: what it measures, against what, and how sure it is."""

    key: str
    label: str

    available: bool = False
    #: Why not, when not. Shown in place of the comparison.
    note: str = ""

    display: str = "--"
    #: The real ratio, uncapped: 140% is worth reading even though the arc stops.
    percent: int | None = None
    #: Length of the drawn arc. Capped at the full circle.
    dash: float = 0.0

    #: What the arc is a fraction of, in words. Not every dial divides by the
    #: same thing, so each one says which denominator it used rather than
    #: leaving a single footnote to cover three different claims.
    comparison: str = ""
    baseline_display: str = ""
    #: Which day the figure is from, always named: "so far today", "yesterday",
    #: or the date. Google stamps a rollup hours late, so a dial often lands on
    #: yesterday even when the page is opened at noon.
    when: str = ""

    #: Imported once from Takeout and never updated. Drawn dashed, in warn,
    #: the same as every other legacy surface in the app.
    legacy: bool = False
    legacy_date: str = ""

    @property
    def circumference(self) -> float:
        return CIRCUMFERENCE


def _baseline(pairs, shown_day):
    """Median of every day but the one on show, or None if there are too few."""
    earlier = [value for day, value in pairs if day != shown_day]
    if len(earlier) < MIN_DAYS_FOR_BASELINE:
        return None
    return median(earlier)


def _latest(pairs):
    """The most recent (day, value) there is, or (None, None)."""
    if not pairs:
        return None, None
    return max(pairs, key=lambda pair: pair[0])


def _when(day, today) -> str:
    """How to name the day on screen.

    Today is always said out loud as "so far today". A day in progress against
    a median of whole days is an unfair comparison by construction - at two
    minutes past midnight it reads nought percent, which is true and useless -
    and the only thing that makes it honest is saying the day is not over.
    """
    if day == today:
        return "so far today"
    if day == today - timedelta(days=1):
        return "yesterday"
    return day.isoformat()


def _compare(ring: Ring, pairs, today, fmt) -> Ring:
    """Fill a ring in from [(date, value)] pairs, or say why it stayed empty."""
    if not pairs:
        ring.note = "No readings in this window."
        return ring

    day, value = _latest(pairs)
    baseline = _baseline(pairs, day)

    ring.when = _when(day, today)
    if not baseline:
        ring.note = f"Needs {MIN_DAYS_FOR_BASELINE} earlier days to compare against."
        ring.display = fmt(value)
        return ring

    ratio = value / baseline
    ring.available = True
    ring.display = fmt(value)
    ring.percent = round(ratio * 100)
    ring.dash = round(CIRCUMFERENCE * min(ratio, 1.0), 2)
    ring.baseline_display = fmt(baseline)
    ring.comparison = f"{ring.percent}% of your usual {ring.baseline_display}"
    return ring


def _count(value) -> str:
    return f"{int(round(value)):,}"


def _minutes(value) -> str:
    hours, remainder = divmod(int(round(value)), 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


def _ordinal(number: int) -> str:
    """61 -> '61st'. The teens are the exception every implementation forgets."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _cohort(assessment) -> str:
    """'men 20-29': who the published table compared this reading against."""
    from . import fitness

    bounds = fitness.bracket_for(assessment.age, assessment.sex)
    people = {"male": "men", "female": "women"}.get(assessment.sex, assessment.sex)
    if bounds is None:
        return people
    return f"{people} {bounds[0]}-{bounds[1]}"


def _vo2_max(assessment, reading) -> Ring:
    """VO2 max as a percentile of the published norms, drawn as legacy.

    Legacy in the same sense as the sleep score, and for a stronger reason: the
    live endpoint exists but only ever produces a figure from a GPS-tracked
    outdoor run, of which this account has none. Every stored reading is a
    Fitbit demographic estimate from a one-off Takeout import, and the series
    stopped the day the import did. Drawing it as a live dial would be the one
    thing this row is not allowed to do.

    The arc is the percentile rather than a ratio against the reader's own
    median; the module docstring has the arithmetic for why.
    """
    ring = Ring(key="vo2-max", label="VO2 max", legacy=True)

    if reading is None:
        ring.note = NEVER_IMPORTED
        return ring

    ring.legacy_date = str(reading.date)

    if assessment is None or not assessment.available:
        # The reading exists, so the date is real even when the comparison is
        # not. Showing the figure without a percentile is still honest; showing
        # a percentile without a date of birth would not be.
        ring.display = f"{round(float(reading.value), 1)}"
        reason = getattr(assessment, "reason", "") if assessment else ""
        from . import fitness

        if reason == fitness.NO_PROFILE:
            ring.note = NEEDS_PROFILE
        elif reason == fitness.OFF_TABLE:
            ring.note = OFF_TABLE
        else:
            ring.note = NEVER_IMPORTED
        return ring

    ring.available = True
    ring.display = f"{assessment.vo2max}"
    ring.percent = round(assessment.percentile)
    ring.dash = round(CIRCUMFERENCE * min(assessment.percentile / 100, 1.0), 2)

    # The table stops at the 5th and the 95th. Beyond either end the honest
    # answer is a bound, so the arc is drawn at the bound and the words say so.
    if assessment.position == "below":
        ring.comparison = f"below the 5th percentile, {_cohort(assessment)}"
    elif assessment.position == "above":
        ring.comparison = f"above the 95th percentile, {_cohort(assessment)}"
    else:
        ring.comparison = (
            f"{_ordinal(ring.percent)} percentile, {_cohort(assessment)}"
        )
    return ring


def _sleep_score(reading) -> Ring:
    """The one quantity here whose denominator somebody else chose.

    A score out of 100, picked by Fitbit, and also the one quantity that can
    never change again - so it is drawn dashed and dated rather than sitting in
    the row looking as current as its neighbours.
    """
    ring = Ring(key="sleep-score", label="Sleep score", legacy=True)
    if reading is None:
        ring.note = NEVER_IMPORTED
        return ring

    ring.available = True
    ring.display = f"{int(round(reading.value))}"
    ring.percent = int(round(reading.value))
    ring.dash = round(CIRCUMFERENCE * min(reading.value / 100, 1.0), 2)
    ring.legacy_date = str(reading.date)
    ring.comparison = "out of 100, Fitbit's own score"
    return ring


def build(points_by_slug: dict, historical: dict, today, assessment=None) -> list:
    """The row of dials for the overview.

    Reads the points the overview has already fetched, and the historical
    readings it has already loaded. No extra API call and no extra query: a
    decorative row at the top of a page must not cost a round trip.

    `assessment` is the health.fitness.Assessment the overview built for the
    panel under the dials. It is passed in rather than computed here so that
    the dial and the fitness age below it can never be derived from two
    different calls, which is the failure this project has already had once.
    """
    from . import metrics

    rings = [_vo2_max(assessment, historical.get("vo2-max"))]

    for key, label, fmt in (
        ("steps", "Steps", _count),
        ("active-minutes", "Active", _minutes),
    ):
        metric = metrics.get(key)
        series = metric.series(points_by_slug.get(key) or []) if metric.series else None
        rings.append(
            _compare(
                Ring(key=key, label=label),
                metrics.as_daily_pairs(series),
                today,
                fmt,
            )
        )

    rings.append(_sleep_score(historical.get("sleep-score")))

    return rings
