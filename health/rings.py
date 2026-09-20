"""Today, measured against the only reference this app is willing to use: you.

Every ring here is a fraction whose denominator is the reader's own median
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
    return ring


def _count(value) -> str:
    return f"{int(round(value)):,}"


def _minutes(value) -> str:
    hours, remainder = divmod(int(round(value)), 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


def build(points_by_slug: dict, historical: dict, today) -> list:
    """The row of dials for the overview.

    Reads the points the overview has already fetched. No extra API call: a
    decorative row at the top of a page must not cost a round trip.
    """
    from . import metrics

    rings = []

    # -- sleep ------------------------------------------------------------
    asleep, _deep = metrics.sleep_pairs(points_by_slug.get("sleep") or [])
    rings.append(
        _compare(
            Ring(key="sleep", label="Sleep"),
            asleep,
            today,
            _minutes,
        )
    )

    # -- steps and active minutes ----------------------------------------
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

    # -- the legacy one ---------------------------------------------------
    #
    # A score out of 100 is the one quantity here with a denominator that is
    # not the reader, because Fitbit chose it. It is also the one quantity
    # that can never change again, so it is drawn dashed and dated rather than
    # sitting in the row looking as current as its neighbours.
    reading = historical.get("sleep-score")
    score = Ring(key="sleep-score", label="Sleep score", legacy=True)
    if reading is None:
        score.note = "Never imported."
    else:
        score.available = True
        score.display = f"{int(round(reading.value))}"
        score.percent = int(round(reading.value))
        score.dash = round(CIRCUMFERENCE * min(reading.value / 100, 1.0), 2)
        score.legacy_date = str(reading.date)
    rings.append(score)

    return rings
