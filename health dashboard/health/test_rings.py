"""The dials, and the comparisons they refuse to make.

The arithmetic matters less here than what the module declines to do: divide
by a target nobody can defend, let a partial day flatter its own median, or
draw a legacy score as though it were current.
"""

from datetime import date, timedelta

from django.test import TestCase

from health import rings
from health.models import HistoricalMetric

TODAY = date(2026, 9, 19)


def daily(values, end=TODAY):
    """[(date, value)] ending today, oldest first."""
    return [
        (end - timedelta(days=len(values) - 1 - index), float(value))
        for index, value in enumerate(values)
    ]


class BaselineTests(TestCase):
    def test_today_is_left_out_of_its_own_median(self):
        """A day still running is a partial count; including it flatters it."""
        # Ten earlier days at 1000, today at 0.
        pairs = daily([1000] * 10 + [0])
        self.assertEqual(rings._baseline(pairs, TODAY), 1000)

    def test_too_few_earlier_days_is_not_a_baseline(self):
        pairs = daily([1000, 1000, 0])
        self.assertIsNone(rings._baseline(pairs, TODAY))

    def test_the_day_on_show_is_the_one_left_out(self):
        """Not the calendar date: whichever day the dial is actually drawing."""
        pairs = daily([1000] * 10 + [0], end=TODAY - timedelta(days=1))
        self.assertEqual(rings._baseline(pairs, TODAY - timedelta(days=1)), 1000)

    def test_the_median_is_not_the_mean(self):
        """One 40,000-step day must not move everybody else's typical day."""
        pairs = daily([8000, 8000, 8000, 8000, 8000, 40000, 0])
        self.assertEqual(rings._baseline(pairs, TODAY), 8000)


class CompareTests(TestCase):
    def _ring(self, values):
        return rings._compare(
            rings.Ring(key="steps", label="Steps"), daily(values), TODAY, rings._count
        )

    def test_a_typical_day_fills_the_dial(self):
        ring = self._ring([8000] * 10 + [8000])
        self.assertTrue(ring.available)
        self.assertEqual(ring.percent, 100)
        self.assertEqual(ring.dash, rings.CIRCUMFERENCE)

    def test_half_a_usual_day_is_half_an_arc(self):
        ring = self._ring([8000] * 10 + [4000])
        self.assertEqual(ring.percent, 50)
        self.assertAlmostEqual(ring.dash, rings.CIRCUMFERENCE / 2, places=1)

    def test_the_arc_stops_at_full_but_the_number_does_not(self):
        """Beating your median by half is worth reading; the circle cannot show it."""
        ring = self._ring([8000] * 10 + [12000])
        self.assertEqual(ring.percent, 150)
        self.assertEqual(ring.dash, rings.CIRCUMFERENCE)

    def test_it_falls_back_to_the_last_day_with_data_and_names_it(self):
        """A rollup lands hours late, so at breakfast today is always empty."""
        pairs = daily([8000] * 10 + [4000], end=TODAY - timedelta(days=1))
        ring = rings._compare(
            rings.Ring(key="steps", label="Steps"), pairs, TODAY, rings._count
        )
        self.assertTrue(ring.available)
        self.assertEqual(ring.display, "4,000")
        self.assertEqual(ring.percent, 50)
        self.assertEqual(ring.when, "yesterday")

    def test_a_day_older_than_yesterday_is_named_by_its_date(self):
        pairs = daily([8000] * 10, end=TODAY - timedelta(days=4))
        ring = rings._compare(
            rings.Ring(key="steps", label="Steps"), pairs, TODAY, rings._count
        )
        self.assertEqual(ring.when, (TODAY - timedelta(days=4)).isoformat())

    def test_a_day_still_running_says_so(self):
        """Otherwise 17 steps at two minutes past midnight reads as a bad day."""
        ring = self._ring([8000] * 10 + [17])
        self.assertEqual(ring.when, "so far today")
        self.assertEqual(ring.percent, 0)

    def test_no_readings_at_all_says_that_instead(self):
        ring = rings._compare(
            rings.Ring(key="steps", label="Steps"), [], TODAY, rings._count
        )
        self.assertFalse(ring.available)
        self.assertIn("No readings", ring.note)

    def test_a_value_without_enough_history_is_shown_but_not_compared(self):
        ring = self._ring([8000, 9000, 5000])
        self.assertFalse(ring.available)
        self.assertEqual(ring.display, "5,000")
        self.assertIn("earlier days", ring.note)


class FormattingTests(TestCase):
    def test_counts_get_thousands_separators(self):
        self.assertEqual(rings._count(8397.4), "8,397")

    def test_minutes_read_as_hours_when_there_are_hours(self):
        self.assertEqual(rings._minutes(450), "7h 30m")
        self.assertEqual(rings._minutes(83), "1h 23m")

    def test_under_an_hour_stays_in_minutes(self):
        """A leading 0h is noise, so it is dropped."""
        self.assertEqual(rings._minutes(42), "42m")


class BuildTests(TestCase):
    def test_the_sleep_score_dial_is_marked_legacy_and_dated(self):
        """It can never update, so it must not sit in the row looking current."""
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.create(username="t@example.com")
        reading = HistoricalMetric.objects.create(
            user=user,
            metric_type="sleep-score",
            date=date(2026, 9, 11),
            value=78,
            unit="/100",
            source="Fitbit sleep score",
        )

        built = rings.build({}, {"sleep-score": reading}, TODAY)
        score = next(ring for ring in built if ring.key == "sleep-score")

        self.assertTrue(score.legacy)
        self.assertTrue(score.available)
        self.assertEqual(score.display, "78")
        self.assertEqual(score.legacy_date, "2026-09-11")
        # 78 of 100, not 78 of the reader's own median.
        self.assertAlmostEqual(score.dash, rings.CIRCUMFERENCE * 0.78, places=1)

    def test_a_missing_sleep_score_is_still_a_legacy_dial(self):
        built = rings.build({}, {}, TODAY)
        score = next(ring for ring in built if ring.key == "sleep-score")
        self.assertTrue(score.legacy)
        self.assertFalse(score.available)
        self.assertIn("imported", score.note)

    def test_every_dial_is_present_even_with_no_data_at_all(self):
        """An empty dial that explains itself beats a dial that vanishes."""
        built = rings.build({}, {}, TODAY)
        self.assertEqual(
            [ring.key for ring in built],
            ["vo2-max", "steps", "active-minutes", "sleep-score"],
        )
        for ring in built:
            self.assertTrue(ring.note, ring.key)


class OrdinalTests(TestCase):
    def test_the_common_cases(self):
        self.assertEqual(rings._ordinal(61), "61st")
        self.assertEqual(rings._ordinal(62), "62nd")
        self.assertEqual(rings._ordinal(63), "63rd")
        self.assertEqual(rings._ordinal(64), "64th")

    def test_the_teens_are_the_exception(self):
        self.assertEqual(rings._ordinal(11), "11th")
        self.assertEqual(rings._ordinal(12), "12th")
        self.assertEqual(rings._ordinal(13), "13th")
        self.assertEqual(rings._ordinal(111), "111th")


class Vo2MaxDialTests(TestCase):
    """The one dial whose denominator is a published table rather than the reader.

    Two things are being pinned here. That it is drawn as legacy, because every
    reading is a frozen Takeout import and the live endpoint has never produced
    one. And that the arc is the percentile rather than a ratio against the
    reader's own median - the stored series moves 4% in two months, so a median
    denominator would pin the arc full for ever and read as a target hit daily.
    """

    def setUp(self):
        from django.contrib.auth import get_user_model

        self.user = get_user_model().objects.create(username="v@example.com")

    def _reading(self, value=51.15, on=date(2026, 9, 11)):
        return HistoricalMetric.objects.create(
            user=self.user,
            metric_type="vo2-max",
            date=on,
            value=value,
            unit="ml/kg/min",
            source="Fitbit demographic estimate",
        )

    def _assess(self, value=51.15, born=date(2002, 3, 9), sex="male"):
        from health import fitness

        return fitness.assess(
            value,
            source="Fitbit demographic estimate",
            measured=False,
            born=born,
            sex=sex,
            today=TODAY,
        )

    def _dial(self, **kwargs):
        reading = kwargs.pop("reading", None)
        assessment = kwargs.pop("assessment", None)
        built = rings.build(
            {}, {"vo2-max": reading} if reading else {}, TODAY, assessment=assessment
        )
        return next(ring for ring in built if ring.key == "vo2-max")

    def test_it_is_always_legacy_even_when_it_has_a_number(self):
        """The live endpoint needs a GPS-tracked run. This never came from one."""
        dial = self._dial(reading=self._reading(), assessment=self._assess())
        self.assertTrue(dial.legacy)
        self.assertTrue(dial.available)
        self.assertEqual(dial.legacy_date, "2026-09-11")

    def test_the_arc_is_the_percentile_not_a_ratio_against_the_reader(self):
        dial = self._dial(reading=self._reading(), assessment=self._assess())
        self.assertEqual(dial.percent, 61)
        self.assertAlmostEqual(dial.dash, rings.CIRCUMFERENCE * 0.609, places=1)

    def test_the_centre_shows_the_measurement_and_the_note_the_comparison(self):
        """Same shape as every other dial: value inside, denominator beneath."""
        dial = self._dial(reading=self._reading(), assessment=self._assess())
        self.assertEqual(dial.display, "51.1")
        self.assertEqual(dial.comparison, "61st percentile, men 20-29")

    def test_the_cohort_follows_the_sex_on_the_profile(self):
        dial = self._dial(
            reading=self._reading(value=40.0),
            assessment=self._assess(value=40.0, sex="female"),
        )
        self.assertIn("women 20-29", dial.comparison)

    def test_a_reading_off_the_top_of_the_table_is_a_bound_not_a_number(self):
        """The table stops at the 95th. Resolving past it would be invention."""
        dial = self._dial(
            reading=self._reading(value=80.0), assessment=self._assess(value=80.0)
        )
        self.assertEqual(dial.comparison, "above the 95th percentile, men 20-29")
        self.assertEqual(dial.percent, 95)

    def test_a_reading_off_the_bottom_of_the_table_is_also_a_bound(self):
        dial = self._dial(
            reading=self._reading(value=12.0), assessment=self._assess(value=12.0)
        )
        self.assertEqual(dial.comparison, "below the 5th percentile, men 20-29")

    def test_no_reading_at_all_is_still_a_legacy_dial_that_says_so(self):
        dial = self._dial(assessment=None)
        self.assertTrue(dial.legacy)
        self.assertFalse(dial.available)
        self.assertEqual(dial.note, rings.NEVER_IMPORTED)
        self.assertEqual(dial.legacy_date, "")

    def test_a_reading_without_a_profile_shows_the_figure_but_no_percentile(self):
        """The date is real even when the comparison cannot be made."""
        dial = self._dial(
            reading=self._reading(), assessment=self._assess(born=None, sex="")
        )
        self.assertFalse(dial.available)
        # The same rounding as the assessed path, so the figure cannot change
        # depending on whether a date of birth happens to be on file.
        self.assertEqual(dial.display, "51.1")
        self.assertEqual(dial.legacy_date, "2026-09-11")
        self.assertEqual(dial.note, rings.NEEDS_PROFILE)
        self.assertEqual(dial.dash, 0.0)

    def test_an_age_outside_the_published_norms_says_which_ages_it_covers(self):
        dial = self._dial(
            reading=self._reading(),
            assessment=self._assess(born=date(2012, 3, 9)),
        )
        self.assertFalse(dial.available)
        self.assertEqual(dial.note, rings.OFF_TABLE)

    def test_the_dial_and_the_fitness_page_cannot_disagree(self):
        """One assessment feeds the arc and the panel under it, by construction.

        The failure this guards against has already happened once in this
        project: a percentile and a fitness age read off two different cohorts,
        so one panel called the same reading above average for its age and
        older than its age at the same time.
        """
        assessment = self._assess()
        dial = self._dial(reading=self._reading(), assessment=assessment)
        self.assertEqual(dial.percent, round(assessment.percentile))
