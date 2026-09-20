"""Where one VO2 max figure sits against published population norms.

This module deliberately does not compute a "fitness age" from the VO2 max this
project actually has, and the reason is worth stating in the code rather than
only in the interface.

The stored figure, 51.15 ml/kg/min, is Google's `demographicVO2Max`. Google
documents that value as being derived from resting heart rate, age, sex and
weight, with no exercise input at all. A fitness age is produced by asking "at
what age would this VO2 max be average?", so computing one from a demographic
estimate feeds age in and reads age back out. The answer would track the
calendar by construction, and the only independent signal inside it would be
resting heart rate. That is not a rounding error, it is the whole number.

So the default output here is a percentile: where this figure sits in the
normative distribution for a given age and sex. That comparison is honest even
when the input is an estimate, because it makes no claim to invert anything.

A fitness age is computed only from a *measured* VO2 max, which on this account
means one derived from a GPS-tracked outdoor run. Until such a run exists, the
page says why the number is missing rather than printing a circular one.

Sources
-------
Percentiles:
    The Cooper Institute, Aerobics Center Longitudinal Study, as published in
    ACSM's Guidelines for Exercise Testing and Prescription, 11th ed. (2021),
    Table 4.7. Values below transcribed from a secondary rendering of that
    table and cross-checked against two independent renderings.

Population means used for the age inversion:
    The HUNT3 Fitness Study, reported by NTNU/CERG, which is the source of the
    "fitness age" idea itself. Nes BM, Janszky I, Wisloff U, Stoylen A,
    Karlsen T. "Estimating VO2peak from a nonexercise prediction model: the
    HUNT Study, Norway." Med Sci Sports Exerc. 2011;43(11):2024-30.
    https://www.ntnu.edu/cerg/fitness-numbers
"""

from dataclasses import dataclass, field
from datetime import date

#: The percentile each column of NORMS corresponds to.
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

#: VO2 max in ml/kg/min by (age_low, age_high) bracket. Cooper / ACSM.
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

#: Mean VO2 max at the midpoint of each HUNT3 decade band, ml/kg/min.
HUNT_MEANS = {
    "male": ((25, 54.0), (35, 49.0), (45, 47.0), (55, 42.0), (65, 39.0), (75, 34.0)),
    "female": ((25, 43.0), (35, 40.0), (45, 38.0), (55, 34.0), (65, 31.0), (75, 27.0)),
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


def fitness_age(vo2max: float, sex: str):
    """The age at which this VO2 max would be the population mean.

    Inverts the HUNT3 mean curve, which is what "fitness age" means in the
    literature that coined the term. Returns (age, position); position is
    "below"/"above" when the value falls off the ends of the published range,
    because the curve says nothing outside 25 to 75.
    """
    means = HUNT_MEANS.get(sex)
    if not means:
        return None, "off-table"

    # Sorted by increasing VO2 max, which means decreasing age.
    points = [(value, age) for age, value in reversed(means)]
    highest = points[-1][0]
    lowest = points[0][0]
    if vo2max > highest:
        return points[-1][1], "below"
    if vo2max < lowest:
        return points[0][1], "above"
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

    #: Populated only when `measured` is True. See the module docstring.
    fitness_age: int | None = None
    fitness_age_position: str = ""
    fitness_age_withheld_because: str = ""

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
WHY_NO_FITNESS_AGE = (
    "This figure is a demographic estimate, which Google derives from resting "
    "heart rate, age, sex and weight. A fitness age asks at what age a VO2 max "
    "would be average, so computing one here would put age in and take age "
    "back out. It is withheld until a VO2 max measured from a GPS-tracked run "
    "exists."
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

    if measured:
        value, where = fitness_age(float(vo2max), sex)
        assessment.fitness_age = value
        assessment.fitness_age_position = where
    else:
        assessment.fitness_age_withheld_because = WHY_NO_FITNESS_AGE

    return assessment
