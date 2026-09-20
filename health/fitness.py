"""Where one VO2 max figure sits against published population norms.

This module reports both a percentile and a fitness age, and the reason the two
are not equally trustworthy is worth stating in the code rather than only in
the interface.

The stored figure, 51.15 ml/kg/min, is Google's `demographicVO2Max`. Google
documents that value as being derived from resting heart rate, age, sex and
weight, with no exercise input at all. A fitness age is produced by asking "at
what age would this VO2 max be average?", so computing one from a demographic
estimate feeds age in and reads age back out. The answer would track the
calendar by construction, and the only independent signal inside it would be
resting heart rate. That is not a rounding error, it is the whole number.

The percentile does not have that problem: where this figure sits in the
normative distribution for a given age and sex is honest even when the input is
modelled, because it compares rather than inverts.

A fitness age is still reported, because the comparison between a calendar age
and a fitness age is the thing people actually want from this. What the page
does instead of withholding it is refuse to let it stand alone: it is shown
beside the percentile, labelled as an estimate of cardiorespiratory fitness
rather than of a body, and accompanied by the reason the absolute figure is
partly circular. The gap between the two ages is flagged as the more
trustworthy half, because age appears on both sides of that subtraction and
largely cancels out of it.

Source, and exactly how far it has been checked
-----------------------------------------------
Both the percentile and the fitness age are read off one table: The Cooper
Institute, Aerobics Center Longitudinal Study, as published in ACSM's
Guidelines for Exercise Testing and Prescription, 11th ed., Wolters Kluwer,
2021, Table 4.7.

Provenance, stated precisely rather than generously: the values in NORMS were
transcribed from a secondary web rendering of that table. Two of the 84
entries - men 20-29 at the 50th (48.0) and men 70-79 at the 50th (24.4) -
were spot-checked against a different secondary rendering and matched. Nothing
here has been compared against the printed edition. This is an unverified
secondary transcription, spot-checked in two places, and should be read as
that until someone opens the book.

The idea of a fitness age comes from the HUNT3 work at NTNU: Nes BM, Janszky
I, Wisloff U, Stoylen A, Karlsen T, "Estimating VO2peak from a nonexercise
prediction model: the HUNT Study, Norway", Med Sci Sports Exerc.
2011;43(11):2024-30. The concept is theirs. The numbers here are not.

An earlier version of this module inverted the HUNT3 Norwegian means while
taking percentiles from Cooper, and the two cohorts differ by about six
ml/kg/min in the same age band - Cooper puts the median male 20-29 at 48.0,
HUNT3 puts the mean at 54.0. That let one reading be called above average for
its age and older than its age in the same panel. One population now answers
both questions, so they cannot contradict each other again.
"""

from dataclasses import dataclass, field
from datetime import date

#: The percentile each column of NORMS corresponds to.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

#: VO2 max in ml/kg/min by (age_low, age_high) bracket, Cooper / ACSM.
#: Unverified secondary transcription; see the provenance note above.
NORMS = {
    "male": {
        (20, 29): (29.0, 32.1, 40.1, 48.0, 55.2, 61.8, 66.3),
        (30, 39): (27.2, 30.2, 35.9, 42.4, 49.2, 56.5, 59.8),
        (40, 49): (24.2, 26.8, 31.9, 37.8, 45.0, 52.1, 55.6),
        (50, 59): (20.9, 22.8, 27.1, 32.6, 39.7, 45.6, 50.7),
        (60, 69): (17.4, 19.8, 23.7, 28.2, 34.5, 40.3, 43.0),
        (70, 79): (16.3, 17.1, 20.4, 24.4, 30.4, 36.6, 39.7),
    },
    "female": {
        (20, 29): (21.7, 23.9, 30.5, 37.6, 44.7, 51.3, 56.0),
        (30, 39): (19.0, 20.9, 25.3, 30.2, 36.1, 41.4, 45.8),
        (40, 49): (17.0, 18.8, 22.1, 26.7, 32.4, 38.4, 41.7),
        (50, 59): (16.0, 17.3, 19.9, 23.4, 27.6, 32.0, 35.9),
        (60, 69): (13.4, 14.6, 17.2, 20.0, 23.8, 27.0, 29.4),
        (70, 79): (13.1, 13.6, 15.6, 18.3, 20.8, 23.1, 24.1),
    },
}

#: Plain-language bands. Deliberately coarse: the input is an estimate with a
#: stated error of about 3 ml/kg/min, which is wider than a single percentile.
BANDS = (
    (90, "well above average"),
    (75, "above average"),
    (25, "around average"),
    (10, "below average"),
    (0, "well below average"),
)


def age_on(born: date, today: date) -> int:
    """Whole years, counting a birthday that has not happened yet as not had."""
    had_birthday = (today.month, today.day) >= (born.month, born.day)
    return today.year - born.year - (0 if had_birthday else 1)


def bracket_for(age: int, sex: str):
    """The (low, high) norm bracket covering this age, or None if off the table."""
    for bounds in NORMS.get(sex, {}):
        if bounds[0] <= age <= bounds[1]:
            return bounds
    return None


def _interpolate(x, points):
    """Linear interpolation over (x, y) points sorted by x. Clamps at the ends."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def percentile_for(vo2max: float, age: int, sex: str):
    """Percentile of this VO2 max within its age and sex bracket.

    Returns (percentile, position) where position is "inside", "below" or
    "above": the published table stops at the 5th and 95th, and pretending to
    resolve beyond its ends would be inventing numbers.
    """
    bounds = bracket_for(age, sex)
    if bounds is None:
        return None, "off-table"

    values = NORMS[sex][bounds]
    if vo2max < values[0]:
        return 5.0, "below"
    if vo2max > values[-1]:
        return 95.0, "above"

    points = list(zip(values, PERCENTILES))
    return round(_interpolate(vo2max, points), 1), "inside"


def band_for(percentile: float) -> str:
    for floor, label in BANDS:
        if percentile >= floor:
            return label
    return BANDS[-1][1]


def median_curve(sex: str) -> list:
    """(age, median VO2 max) anchors for one sex, youngest first.

    The published table reports a decade band rather than a value per year, so
    each band's 50th percentile is anchored at the midpoint of its own age
    range. Everything between two anchors is straight-line interpolation done
    here. That interpolation is mine, not the publication's; only the anchors
    are published figures.
    """
    return [
        ((low + high) // 2, values[PERCENTILES.index(50)])
        for (low, high), values in NORMS.get(sex, {}).items()
    ]


def fitness_age(vo2max: float, sex: str):
    """The age at which this VO2 max would be the median for the population.

    Read off the same Cooper/ACSM table as the percentile, deliberately: the
    two answers describe one cohort, so they can no longer disagree about
    whether a reading is good for its age.

    Returns (age, position). "below" and "above" mean the reading falls off the
    young or the old end of the table, and the answer is a bound rather than a
    number - there is no published anchor beyond it to interpolate towards,
    and inventing one is exactly what this module exists not to do.
    """
    curve = median_curve(sex)
    if not curve:
        return None, "off-table"

    youngest_age, highest_median = curve[0]
    oldest_age, lowest_median = curve[-1]
    if vo2max > highest_median:
        return youngest_age, "below"
    if vo2max < lowest_median:
        return oldest_age, "above"

    # Ascending by VO2 max, which is descending by age.
    points = [(value, age) for age, value in reversed(curve)]
    return round(_interpolate(vo2max, points)), "inside"


@dataclass
class Assessment:
    """What can honestly be said about one VO2 max figure."""

    available: bool
    reason: str = ""

    vo2max: float | None = None
    source: str = ""
    #: True only for a VO2 max derived from exercise, not from a profile model.
    measured: bool = False

    age: int | None = None
    sex: str = ""

    percentile: float | None = None
    position: str = ""
    band: str = ""

    fitness_age: int | None = None
    fitness_age_position: str = ""
    #: age - fitness_age. Positive means the fitness age is the lower number.
    difference: int | None = None
    #: True when the VO2 max behind it was modelled from age rather than
    #: measured from exercise. The number is still shown; see WHY_CIRCULAR.
    circular: bool = False

    notes: list = field(default_factory=list)


NO_PROFILE = (
    "Population norms are specific to age and sex, so this needs a date of "
    "birth and a sex recorded in settings before it can say anything."
)
NO_VO2MAX = (
    "No VO2 max figure is available. The live API derives one only from "
    "GPS-tracked outdoor runs, and no Takeout estimate has been imported."
)
OFF_TABLE = (
    "The published norms cover ages 20 to 79. Outside that range there is no "
    "table to compare against."
)
WHY_CIRCULAR = (
    "The VO2 max behind this is a demographic estimate, which Google derives "
    "from resting heart rate, age, sex and weight. Because your age is one of "
    "its inputs, the absolute fitness age partly restates the birthday it was "
    "calculated from."
)

WHY_THE_GAP_IS_THE_PART_TO_READ = (
    "The difference is the more honest half of it. Age appears on both sides "
    "of the subtraction and largely cancels, so what is left is driven by the "
    "inputs that are not your age: resting heart rate and weight. A gap in "
    "either direction says your heart rate for your age sits above or below "
    "what the model expects, which is a real thing to know. The absolute "
    "number will only stand on its own once a VO2 max measured from a "
    "GPS-tracked run exists."
)


def assess(vo2max, *, source: str, measured: bool, born, sex: str, today: date):
    """Everything the fitness page needs, or the reason it has nothing."""
    if born is None or not sex:
        return Assessment(available=False, reason=NO_PROFILE)
    if vo2max is None:
        return Assessment(available=False, reason=NO_VO2MAX)

    age = age_on(born, today)
    if bracket_for(age, sex) is None:
        return Assessment(available=False, reason=OFF_TABLE, age=age, sex=sex)

    percentile, position = percentile_for(float(vo2max), age, sex)
    assessment = Assessment(
        available=True,
        vo2max=round(float(vo2max), 1),
        source=source,
        measured=measured,
        age=age,
        sex=sex,
        percentile=percentile,
        position=position,
        band=band_for(percentile),
    )

    value, where = fitness_age(float(vo2max), sex)
    assessment.fitness_age = value
    assessment.fitness_age_position = where
    assessment.circular = not measured
    # Only when the fitness age is a number rather than a bound: the distance
    # to "under 24" is not a number of years.
    if value is not None and where == "inside":
        assessment.difference = age - value

    return assessment
