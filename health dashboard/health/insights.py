"""Derived observations computed from the daily series the dashboard already loads.

Design notes:

* Every function here is **pure**. It takes plain ``[(date, value)]`` series and
  returns an :class:`Insight`. No ORM access, no HTTP, no template logic - so
  each one is testable against a synthetic series whose answer is known in
  advance.
* Every function **degrades on its own**. Too little data returns an Insight
  with ``available=False`` and a note saying what is missing, so one thin metric
  never blanks the whole section.
* Wording is deliberately observational. These are n<=90 samples from a single
  person with no control, no randomisation and no blinding: they can support
  "X was higher on days when Y was lower", never "Y causes X". The one place a
  p-value is reported it is labelled as a description of *this* sample, and the
  circularity caveat below is stated in the insight itself.

A note on resting heart rate and sleep: the API reports
``calculationMethod: "WITH_SLEEP"`` for daily resting heart rate, meaning the
figure is *derived from* the same night's sleep recording. Correlating the two
is therefore partly circular and is labelled as such rather than presented as an
independent finding.
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import date

# A point is (date, value). Series are lists of points, oldest first.
Point = tuple[date, float]

MINUTES_PER_HOUR = 60

#: Default nightly sleep target used for the sleep-debt running total.
#: Comparing against the person's own average would make the debt ~0 by
#: construction, so a fixed reference is used and their average reported beside it.
DEFAULT_SLEEP_TARGET_MINUTES = 450  # 7h30


#: How much weight a reader should put on an insight. Rendered as a badge, and
#: the detail sentence is phrased to match - an insight that fails its test says
#: so in its first clause rather than burying it in the footnote.
SIG_DESCRIPTIVE = "descriptive"  # arithmetic on the sample; no test applies
SIG_NONE = "not distinguishable from chance"
SIG_SAMPLE = "significant in this sample"
SIG_CORRECTED = "survives multiple-comparison correction"

ALPHA = 0.05


@dataclass(frozen=True)
class Insight:
    key: str
    title: str
    #: One plain-language sentence. Observational phrasing only, and it must
    #: not read as significant when the statistics do not support it.
    detail: str
    available: bool = True
    value: float | None = None
    unit: str = ""
    #: Shown under the detail line: sample size, caveats, statistics.
    footnote: str = ""
    #: Set when the insight cannot be computed yet.
    note: str = ""
    sample_size: int = 0
    direction: str = ""  # "up" | "down" | "flat" | ""
    significance: str = SIG_DESCRIPTIVE
    p_value: float | None = None
    #: Numbers the methodology section quotes back, e.g. pairings tested.
    meta: dict = field(default_factory=dict)

    @classmethod
    def unavailable(cls, key, title, note):
        return cls(key=key, title=title, detail="", available=False, note=note)

    @property
    def is_significant(self) -> bool:
        return self.significance in (SIG_SAMPLE, SIG_CORRECTED)


# ---------------------------------------------------------------------------
# small statistics helpers (no numpy/scipy - this stays a pure-Python project)
# ---------------------------------------------------------------------------


def clean(series) -> list[Point]:
    """Drop missing values and sort oldest first."""
    return sorted(
        ((day, float(value)) for day, value in series if value is not None),
        key=lambda point: point[0],
    )


def align(left, right) -> tuple[list[float], list[float], list[date]]:
    """Pair two series on their shared dates."""
    left_by_date = dict(clean(left))
    right_by_date = dict(clean(right))
    shared = sorted(set(left_by_date) & set(right_by_date))
    return (
        [left_by_date[day] for day in shared],
        [right_by_date[day] for day in shared],
        shared,
    )


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Ordinary least squares slope of y on x."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_value_for_r(r: float, n: int) -> float | None:
    """Two-tailed p for a Pearson r under the usual t approximation.

    Reported to describe *this* sample. It assumes roughly linear, roughly
    normal, independent observations - consecutive days from one person are not
    independent, so treat it as descriptive rather than inferential.
    """
    if n < 3:
        return None
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = abs(r) * math.sqrt(df / (1.0 - r * r))
    return _incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def describe_strength(r: float) -> str:
    magnitude = abs(r)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.5:
        return "moderate"
    if magnitude >= 0.3:
        return "weak"
    return "very weak"


def format_p(p: float | None) -> str:
    if p is None:
        return "p unavailable"
    return f"p ≈ {p:.3f}" if p >= 0.001 else "p < 0.001"


def _stats_footnote(r: float, n: int) -> str:
    p = p_value_for_r(r, n)
    verdict = (
        "not distinguishable from chance at the 0.05 level"
        if p is None or p >= ALPHA
        else "unlikely under the null at the 0.05 level"
    )
    return (
        f"Pearson r = {r:+.2f} over {n} paired days, {format_p(p)} - {verdict}. "
        f"Association only; consecutive days from one person are not independent "
        f"observations, so this describes the sample rather than proving anything."
    )


# ---------------------------------------------------------------------------
# insights
# ---------------------------------------------------------------------------


def trend(
    series,
    *,
    key: str,
    title: str,
    label: str,
    unit: str,
    min_days: int = 14,
    decimals: int = 1,
    lower_is_better: bool | None = None,
) -> Insight:
    """Least-squares drift across the window, expressed as total change."""
    points = clean(series)
    if len(points) < min_days:
        return Insight.unavailable(
            key, title, f"Needs {min_days} days of data; have {len(points)}."
        )

    first_day = points[0][0]
    xs = [(day - first_day).days for day, _ in points]
    ys = [value for _, value in points]
    slope = linear_slope(xs, ys)
    if slope is None:
        return Insight.unavailable(key, title, "Values do not vary over the window.")

    span_days = xs[-1]
    total_change = slope * span_days
    direction = "flat" if abs(total_change) < 10 ** -decimals else ("up" if total_change > 0 else "down")

    # A fitted line always has *some* slope. Testing it against zero is what
    # separates a trend from day-to-day scatter, and the sentence is worded to
    # match the answer rather than asserting a trend either way.
    r = pearson([float(x) for x in xs], ys)
    p = p_value_for_r(r, len(points)) if r is not None else None
    significant = p is not None and p < ALPHA
    significance = SIG_SAMPLE if significant else SIG_NONE

    if direction == "flat":
        detail = f"{label} has been essentially flat over the last {span_days} days."
        significance = SIG_DESCRIPTIVE
    elif significant:
        detail = (
            f"{label} trended {direction} by "
            f"{abs(total_change):.{decimals}f} {unit} across the last {span_days} days."
        )
    else:
        detail = (
            f"{label} drifted {direction} by about "
            f"{abs(total_change):.{decimals}f} {unit} across the last {span_days} days, "
            f"but the day-to-day scatter is wide enough that this is not "
            f"distinguishable from no trend at all."
        )

    # A perfectly constant series has no correlation to report, which is not an
    # error - it is the flat case, already described above.
    fit = f"fit r = {r:+.2f}, {format_p(p)}. " if r is not None else ""
    footnote = (
        f"Least-squares slope over {len(points)} readings ({slope:+.3f} {unit}/day); "
        f"{fit}A line fitted to {len(points)} days describes this window only - "
        f"it is not a forecast."
    )
    return Insight(
        key=key,
        title=title,
        detail=detail,
        value=round(total_change, decimals),
        unit=unit,
        footnote=footnote,
        sample_size=len(points),
        direction=direction,
        significance=significance,
        p_value=p,
    )


def sleep_debt(
    nights,
    *,
    target_minutes: int = DEFAULT_SLEEP_TARGET_MINUTES,
    min_nights: int = 7,
) -> Insight:
    """Running total of minutes slept above or below a fixed nightly target.

    A personal trailing average is deliberately *not* used as the reference: the
    debt against one's own mean is zero by construction. The personal average is
    reported alongside instead.
    """
    points = clean(nights)
    if len(points) < min_nights:
        return Insight.unavailable(
            "sleep-debt",
            "Sleep debt",
            f"Needs {min_nights} nights; have {len(points)}.",
        )

    deficits = [value - target_minutes for _, value in points]
    balance = sum(deficits)
    average = statistics.mean(value for _, value in points)
    short_nights = sum(1 for d in deficits if d < 0)

    target_h = target_minutes / MINUTES_PER_HOUR
    balance_h = balance / MINUTES_PER_HOUR
    average_h = average / MINUTES_PER_HOUR

    if balance < 0:
        detail = (
            f"Running a sleep deficit of {abs(balance_h):.1f} hours against a "
            f"{target_h:.1f}h nightly target across {len(points)} nights."
        )
    else:
        detail = (
            f"Running a sleep surplus of {balance_h:.1f} hours against a "
            f"{target_h:.1f}h nightly target across {len(points)} nights."
        )

    footnote = (
        f"Average {average_h:.2f}h per night; {short_nights} of {len(points)} nights "
        f"fell below the {target_h:.1f}h reference. The target is an arbitrary "
        f"reference point, not a clinical recommendation."
    )
    return Insight(
        key="sleep-debt",
        title="Sleep debt",
        detail=detail,
        value=round(balance_h, 1),
        unit="h",
        footnote=footnote,
        sample_size=len(points),
        direction="down" if balance < 0 else "up",
    )


def resting_hr_after_sleep(
    nights,
    resting_hr,
    *,
    min_pairs: int = 10,
) -> Insight:
    """Compare resting HR on mornings following the shortest vs longest nights.

    Nights are keyed by the civil date they end on, which is the same date the
    daily resting heart rate is reported for - so the pairing is same-date.
    """
    durations, beats, days = align(nights, resting_hr)
    if len(days) < min_pairs:
        return Insight.unavailable(
            "rhr-after-sleep",
            "Resting HR and sleep length",
            f"Needs {min_pairs} nights with a matching resting HR; have {len(days)}.",
        )

    paired = sorted(zip(durations, beats))
    cut = max(1, len(paired) // 3)
    short_nights = paired[:cut]
    long_nights = paired[-cut:]

    short_hr = statistics.mean(hr for _, hr in short_nights)
    long_hr = statistics.mean(hr for _, hr in long_nights)
    short_sleep_h = statistics.mean(d for d, _ in short_nights) / MINUTES_PER_HOUR
    long_sleep_h = statistics.mean(d for d, _ in long_nights) / MINUTES_PER_HOUR
    difference = short_hr - long_hr

    r = pearson(durations, beats)
    p = p_value_for_r(r, len(days)) if r is not None else None
    significant = p is not None and p < ALPHA

    if abs(difference) < 0.5:
        detail = (
            f"Resting heart rate was about the same after the shortest nights "
            f"({short_sleep_h:.1f}h) and the longest ({long_sleep_h:.1f}h)."
        )
    else:
        higher_lower = "higher" if difference > 0 else "lower"
        observed = (
            f"Resting heart rate was {abs(difference):.1f} bpm {higher_lower} on mornings "
            f"after the shortest nights ({short_sleep_h:.1f}h avg, {short_hr:.1f} bpm) "
            f"than after the longest ({long_sleep_h:.1f}h avg, {long_hr:.1f} bpm)"
        )
        # The difference is a fact about the sample either way; whether it means
        # anything is a separate question, and the sentence has to answer both.
        detail = (
            f"{observed}."
            if significant
            else (
                f"{observed} - a gap this small is within normal day-to-day "
                f"variation over {len(days)} nights."
            )
        )

    footnote = (
        f"Bottom and top third of {len(days)} nights by duration. "
        + (_stats_footnote(r, len(days)) if r is not None else "")
        + " Note that the API derives daily resting heart rate from the same night's "
        "sleep recording (calculationMethod WITH_SLEEP), so the two are not "
        "independent measurements and part of any relationship is definitional."
    )
    return Insight(
        key="rhr-after-sleep",
        title="Resting HR and sleep length",
        detail=detail,
        value=round(difference, 1),
        unit="bpm",
        footnote=footnote,
        sample_size=len(days),
        direction="up" if difference > 0 else "down",
        significance=SIG_SAMPLE if significant else SIG_NONE,
        p_value=p,
        meta={"partly_definitional": True},
    )


def pair_correlation(
    left,
    right,
    *,
    key: str,
    title: str,
    left_label: str,
    right_label: str,
    min_pairs: int = 14,
) -> Insight:
    """A single named correlation between two daily series."""
    xs, ys, days = align(left, right)
    if len(days) < min_pairs:
        return Insight.unavailable(
            key, title, f"Needs {min_pairs} paired days; have {len(days)}."
        )

    r = pearson(xs, ys)
    if r is None:
        return Insight.unavailable(key, title, "One of the series does not vary.")

    p = p_value_for_r(r, len(days))
    significant = p is not None and p < ALPHA
    relation = "higher" if r > 0 else "lower"

    if significant:
        detail = (
            f"{describe_strength(r).capitalize()} association: days with higher "
            f"{left_label} tended to show {relation} {right_label}."
        )
    else:
        # Leading with the strength label reads as a finding. When the test
        # fails, the sentence has to say so before describing the direction.
        detail = (
            f"Tested, no clear relationship: {left_label} and {right_label} moved "
            f"{'together' if r > 0 else 'in opposite directions'} slightly "
            f"(r = {r:+.2f}), which is within what random variation would produce "
            f"over {len(days)} days."
        )

    return Insight(
        key=key,
        title=title,
        detail=detail,
        value=round(r, 2),
        unit="r",
        footnote=_stats_footnote(r, len(days)),
        sample_size=len(days),
        direction="up" if r > 0 else "down",
        significance=SIG_SAMPLE if significant else SIG_NONE,
        p_value=p,
    )


def consistency(
    series,
    *,
    key: str,
    title: str,
    label: str,
    min_days: int = 10,
) -> Insight:
    """How steady a nightly figure is, as a spread in hours."""
    points = clean(series)
    if len(points) < min_days:
        return Insight.unavailable(
            key, title, f"Needs {min_days} nights; have {len(points)}."
        )

    values = [value for _, value in points]
    spread = statistics.pstdev(values) / MINUTES_PER_HOUR
    average = statistics.mean(values) / MINUTES_PER_HOUR
    shortest = min(values) / MINUTES_PER_HOUR
    longest = max(values) / MINUTES_PER_HOUR

    if spread < 0.5:
        verdict = "very consistent"
    elif spread < 1.0:
        verdict = "fairly consistent"
    else:
        verdict = "variable"

    detail = (
        f"{label} was {verdict}: {average:.1f}h on average, "
        f"give or take {spread:.1f}h night to night."
    )
    footnote = (
        f"Population standard deviation across {len(points)} nights; "
        f"shortest {shortest:.1f}h, longest {longest:.1f}h."
    )
    return Insight(
        key=key,
        title=title,
        detail=detail,
        value=round(spread, 2),
        unit="h",
        footnote=footnote,
        sample_size=len(points),
    )


#: Pairs where one series is a component of the other, so a strong correlation is
#: arithmetic rather than a finding. Deep sleep is part of total time asleep.
DEFINITIONAL_PAIRS = {frozenset({"deep sleep", "time asleep"})}


def strongest_relationship(series_map: dict, *, min_pairs: int = 14) -> Insight:
    """Scan every pair of daily series and surface the strongest association.

    Exploratory by construction: testing many pairs and reporting the largest
    inflates the apparent strength of whatever comes top, which the footnote says.
    Pairs listed in DEFINITIONAL_PAIRS are skipped - a metric correlating with a
    quantity it is part of tells you nothing.
    """
    labels = sorted(series_map)
    best = None
    tested = 0

    for i, left_key in enumerate(labels):
        for right_key in labels[i + 1 :]:
            if frozenset({left_key, right_key}) in DEFINITIONAL_PAIRS:
                continue
            xs, ys, days = align(series_map[left_key], series_map[right_key])
            if len(days) < min_pairs:
                continue
            r = pearson(xs, ys)
            if r is None:
                continue
            tested += 1
            if best is None or abs(r) > abs(best[0]):
                best = (r, left_key, right_key, len(days))

    if best is None:
        return Insight.unavailable(
            "strongest-relationship",
            "Strongest relationship",
            f"No pair of metrics has {min_pairs} overlapping days yet.",
        )

    r, left_key, right_key, n = best
    relation = "together" if r > 0 else "in opposite directions"

    # This is the maximum of `tested` correlations, so the usual single-test
    # threshold does not apply. Bonferroni is crude but it is the honest floor:
    # claiming significance off the back of the biggest of twenty comparisons is
    # exactly the error this insight would otherwise invite.
    p = p_value_for_r(r, n)
    threshold = ALPHA / tested if tested else ALPHA
    survives = p is not None and p < threshold

    lead = (
        f"The strongest of {tested} relationships tested: {left_key} and "
        f"{right_key} moved {relation} ({describe_strength(r)}, r = {r:+.2f})"
    )
    if survives:
        detail = (
            f"{lead}, and it still holds after correcting for having tested "
            f"{tested} pairs."
        )
    else:
        detail = (
            f"{lead} - though not significant after correcting for multiple "
            f"comparisons, so this is a lead to look into rather than a real effect."
        )

    footnote = (
        f"Pearson r = {r:+.2f} over {n} paired days, {format_p(p)}. Picking the "
        f"largest of {tested} comparisons inflates it, so the single-test 0.05 "
        f"threshold does not apply; the Bonferroni-corrected threshold is "
        f"{threshold:.4f}, which this {'clears' if survives else 'does not clear'}."
    )
    return Insight(
        key="strongest-relationship",
        title="Strongest relationship",
        detail=detail,
        value=round(r, 2),
        unit="r",
        footnote=footnote,
        sample_size=n,
        direction="up" if r > 0 else "down",
        significance=SIG_CORRECTED if survives else SIG_NONE,
        p_value=p,
        meta={
            "pairs_tested": tested,
            "bonferroni_threshold": threshold,
            "survives_correction": survives,
        },
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build(series_map: dict) -> list[Insight]:
    """Compute every insight from a map of named daily series.

    Expected keys (all optional - a missing one simply yields an unavailable
    insight rather than an error):
    resting_hr, hrv, spo2, respiratory_rate, sleep_minutes, deep_sleep_minutes,
    steps, sleep_score.
    """
    get = lambda name: series_map.get(name) or []

    insights = [
        trend(
            get("resting_hr"),
            key="rhr-trend",
            title="Resting heart rate trend",
            label="Resting heart rate",
            unit="bpm",
        ),
        trend(
            get("hrv"),
            key="hrv-trend",
            title="HRV trend",
            label="Heart rate variability",
            unit="ms",
        ),
        sleep_debt(get("sleep_minutes")),
        resting_hr_after_sleep(get("sleep_minutes"), get("resting_hr")),
        consistency(
            get("sleep_minutes"),
            key="sleep-consistency",
            title="Sleep consistency",
            label="Time asleep",
        ),
        pair_correlation(
            get("hrv"),
            get("resting_hr"),
            key="hrv-rhr",
            title="HRV and resting heart rate",
            left_label="heart rate variability",
            right_label="resting heart rate",
        ),
        pair_correlation(
            get("deep_sleep_minutes"),
            get("hrv"),
            key="deep-hrv",
            title="Deep sleep and HRV",
            left_label="deep sleep",
            right_label="heart rate variability",
        ),
        strongest_relationship(
            {
                name: series
                for name, series in {
                    "resting HR": get("resting_hr"),
                    "HRV": get("hrv"),
                    "SpO2": get("spo2"),
                    "respiratory rate": get("respiratory_rate"),
                    "time asleep": get("sleep_minutes"),
                    "deep sleep": get("deep_sleep_minutes"),
                    "steps": get("steps"),
                }.items()
                if series
            }
        ),
    ]
    return insights


def methodology(computed) -> list[dict]:
    """Caveats shown on the page itself, with the real numbers filled in.

    Built from the computed insights rather than written as fixed prose, so the
    counts and thresholds quoted here cannot drift away from what was actually
    calculated.
    """
    by_key = {insight.key: insight for insight in computed if insight.available}
    notes = [
        {
            "key": "sample",
            "heading": "What this is",
            "body": (
                "Every figure here is computed from one person's own readings over "
                "the displayed window - typically 20 to 30 days. There is no control "
                "group, no randomisation and no blinding, so these describe patterns "
                "in the data. They are not clinical findings and cannot show that one "
                "thing causes another."
            ),
        }
    ]

    rhr_sleep = by_key.get("rhr-after-sleep")
    if rhr_sleep is not None and rhr_sleep.meta.get("partly_definitional"):
        notes.append(
            {
                "key": "circularity",
                "heading": "Resting HR and sleep overlap by construction",
                "body": (
                    "The API reports daily resting heart rate with "
                    "calculationMethod \"WITH_SLEEP\" - the figure is derived from "
                    "that same night's sleep recording. The two are therefore not "
                    "independent measurements, so the resting-HR-versus-sleep "
                    "comparison is partly definitional rather than purely "
                    "correlational, and part of any relationship is built into how "
                    "the number is produced."
                ),
            }
        )

    strongest = by_key.get("strongest-relationship")
    if strongest is not None:
        tested = strongest.meta.get("pairs_tested", 0)
        threshold = strongest.meta.get("bonferroni_threshold", ALPHA)
        survives = strongest.meta.get("survives_correction", False)
        notes.append(
            {
                "key": "multiple-comparisons",
                "heading": "The strongest relationship was cherry-picked",
                "body": (
                    f"\"Strongest relationship\" is the largest of {tested} "
                    f"correlations tested, and reporting the biggest of many "
                    f"comparisons inflates it. Its raw p-value "
                    f"({format_p(strongest.p_value)}) therefore cannot be read "
                    f"against the usual 0.05 threshold. Correcting for {tested} "
                    f"tests (Bonferroni) gives {threshold:.4f}, which the current "
                    f"result {'clears' if survives else 'does not clear'}. It is "
                    f"{'worth taking seriously' if survives else 'a lead to look into, not a real effect'}."
                ),
            }
        )

    excluded = sorted(" and ".join(sorted(pair)) for pair in DEFINITIONAL_PAIRS)
    if excluded:
        notes.append(
            {
                "key": "nested-metrics",
                "heading": "Overlapping metrics were excluded",
                "body": (
                    f"Pairs where one metric is a component of the other were left "
                    f"out of the scan ({'; '.join(excluded)}). Deep sleep is part of "
                    f"total time asleep, so those two are guaranteed to correlate by "
                    f"arithmetic. Including them would surface a strong number that "
                    f"says nothing about health."
                ),
            }
        )

    notes.append(
        {
            "key": "independence",
            "heading": "Consecutive days are not independent",
            "body": (
                "p-values come from the usual t approximation for a Pearson "
                "correlation, which assumes independent observations. Readings from "
                "consecutive days in one person are autocorrelated, so the true "
                "uncertainty is wider than the quoted p-values suggest. Treat them "
                "as a rough description of this sample."
            ),
        }
    )
    return notes


def available(insights) -> list[Insight]:
    return [insight for insight in insights if insight.available]


def pending(insights) -> list[Insight]:
    return [insight for insight in insights if not insight.available]
