"""The fitness percentile, the fitness age, and the caveat they travel with."""

from datetime import date

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from accounts.models import Profile
from health import fitness
from health.models import HistoricalMetric

User = get_user_model()

TODAY = date(2026, 9, 19)


class AgeTests(TestCase):
    def test_birthday_not_yet_had_this_year_does_not_count(self):
        self.assertEqual(fitness.age_on(date(1990, 12, 25), TODAY), 35)

    def test_birthday_already_had_counts(self):
        self.assertEqual(fitness.age_on(date(1990, 1, 5), TODAY), 36)

    def test_birthday_today_counts(self):
        self.assertEqual(fitness.age_on(date(1990, 9, 19), TODAY), 36)


class PercentileTests(TestCase):
    def test_a_median_value_lands_on_the_fiftieth(self):
        # 42.4 is the published 50th percentile for men aged 30-39.
        percentile, position = fitness.percentile_for(42.4, 35, "male")
        self.assertEqual(position, "inside")
        self.assertEqual(percentile, 50.0)

    def test_interpolates_between_published_columns(self):
        # Between the 50th (42.4) and the 75th (49.2) for men 30-39.
        percentile, _ = fitness.percentile_for(45.8, 35, "male")
        self.assertGreater(percentile, 50)
        self.assertLess(percentile, 75)

    def test_below_the_table_is_reported_as_below_not_extrapolated(self):
        percentile, position = fitness.percentile_for(10.0, 35, "male")
        self.assertEqual(position, "below")
        self.assertEqual(percentile, 5.0)

    def test_above_the_table_is_reported_as_above(self):
        _, position = fitness.percentile_for(80.0, 35, "male")
        self.assertEqual(position, "above")

    def test_sex_changes_the_answer(self):
        male, _ = fitness.percentile_for(38.0, 35, "male")
        female, _ = fitness.percentile_for(38.0, 35, "female")
        self.assertLess(male, female)

    def test_an_age_outside_the_published_range_has_no_bracket(self):
        self.assertIsNone(fitness.bracket_for(12, "male"))
        percentile, position = fitness.percentile_for(45.0, 12, "male")
        self.assertIsNone(percentile)
        self.assertEqual(position, "off-table")


class FitnessAgeTests(TestCase):
    def test_a_median_value_returns_the_midpoint_of_its_own_band(self):
        # 42.4 is the 50th percentile for men 30-39, whose midpoint is 34.
        age, position = fitness.fitness_age(42.4, "male")
        self.assertEqual(position, "inside")
        self.assertEqual(age, 34)

    def test_fitter_than_the_youngest_median_is_a_bound_not_a_number(self):
        age, position = fitness.fitness_age(70.0, "male")
        self.assertEqual(position, "below")
        self.assertEqual(age, 24)

    def test_below_the_oldest_median_is_a_bound_at_the_other_end(self):
        age, position = fitness.fitness_age(15.0, "male")
        self.assertEqual(position, "above")
        self.assertEqual(age, 74)

    def test_a_higher_vo2max_gives_a_lower_fitness_age(self):
        younger, _ = fitness.fitness_age(40.0, "male")
        older, _ = fitness.fitness_age(30.0, "male")
        self.assertLess(younger, older)

    def test_the_curve_is_the_fiftieth_percentile_column_of_the_norms(self):
        """Not a second table: the anchors are read out of NORMS itself."""
        curve = dict(fitness.median_curve("male"))
        for (low, high), values in fitness.NORMS["male"].items():
            self.assertEqual(curve[(low + high) // 2], values[3])

    def test_percentile_and_fitness_age_cannot_contradict_each_other(self):
        """The invariant the old two-cohort version broke.

        Someone sitting exactly on the median for their own band must come out
        as the 50th percentile AND as a fitness age equal to their own band
        midpoint. Reading the two numbers off one table is what guarantees it.
        """
        for sex in ("male", "female"):
            for (low, high), values in fitness.NORMS[sex].items():
                midpoint = (low + high) // 2
                median = values[3]
                percentile, _ = fitness.percentile_for(median, midpoint, sex)
                age, position = fitness.fitness_age(median, sex)
                self.assertEqual(percentile, 50.0, f"{sex} {low}-{high}")
                self.assertEqual(position, "inside", f"{sex} {low}-{high}")
                self.assertEqual(age, midpoint, f"{sex} {low}-{high}")


class AssessTests(TestCase):
    def test_without_a_profile_it_asks_for_one_rather_than_guessing(self):
        result = fitness.assess(
            51.15, source="Fitbit demographic estimate", measured=False,
            born=None, sex="", today=TODAY,
        )
        self.assertFalse(result.available)
        self.assertIn("date of birth", result.reason)

    def test_without_a_vo2max_it_says_so(self):
        result = fitness.assess(
            None, source="", measured=False,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertFalse(result.available)
        self.assertIn("No VO2 max", result.reason)

    def test_a_demographic_estimate_is_reported_but_flagged_circular(self):
        """The number is shown; what it rests on is not hidden."""
        result = fitness.assess(
            51.15, source="Fitbit demographic estimate", measured=False,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.percentile)
        # Above the median of every band, so the answer is the young bound.
        self.assertEqual(result.fitness_age, 24)
        self.assertEqual(result.fitness_age_position, "below")
        self.assertTrue(result.circular)

    def test_the_difference_is_calendar_age_minus_fitness_age(self):
        result = fitness.assess(
            42.4, source="Fitbit demographic estimate", measured=False,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertEqual(result.age, 36)
        self.assertEqual(result.fitness_age, 34)
        self.assertEqual(result.difference, 2)

    def test_a_bounded_fitness_age_has_no_difference_in_years(self):
        """The distance to "under 24" is not a number of years, so it is None."""
        result = fitness.assess(
            51.15, source="Fitbit demographic estimate", measured=False,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertEqual(result.fitness_age_position, "below")
        self.assertIsNone(result.difference)

    def test_a_measured_value_is_not_flagged_circular(self):
        result = fitness.assess(
            51.15, source="GPS-tracked run", measured=True,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.fitness_age)
        self.assertFalse(result.circular)


class FitnessPageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.client.force_login(self.user)

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse("health:fitness"))
        self.assertEqual(response.status_code, 302)

    def test_without_a_profile_it_points_at_settings(self):
        response = self.client.get(reverse("health:fitness"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("accounts:settings"))

    def test_the_page_shows_both_ages_and_never_calls_it_biological(self):
        Profile.objects.create(
            user=self.user, date_of_birth="1990-05-01", sex="male"
        )
        HistoricalMetric.objects.create(
            user=self.user,
            metric_type="vo2-max",
            date=date(2026, 9, 11),
            value=51.15,
            unit="ml/kg/min",
            source="Fitbit demographic estimate",
        )
        response = self.client.get(reverse("health:fitness"))

        self.assertEqual(response.status_code, 200)
        assessment = response.context["assessment"]
        self.assertTrue(assessment.available)

        # Both numbers on the page, and the caveat that has to travel with them.
        self.assertEqual(assessment.fitness_age, 24)
        self.assertEqual(assessment.fitness_age_position, "below")
        self.assertContains(response, "Fitness age")
        self.assertContains(response, "Your age")
        # A bound, phrased as one, rather than a made-up single number - and
        # the caveat panel gives the advice that fits a bound, since there is
        # no gap to subtract.
        self.assertContains(response, "under 24")
        self.assertContains(response, "Read the percentile instead")
        self.assertContains(response, "partly restates the birthday")
        self.assertNotContains(response, "Read the gap, not the number")
        # Never claimed as biology.
        self.assertContains(response, "not a biological age")

    def test_a_fitness_age_inside_the_table_reads_as_a_gap_in_years(self):
        """The other branch: a real number, so the gap advice applies."""
        Profile.objects.create(
            user=self.user, date_of_birth="1990-05-01", sex="male"
        )
        HistoricalMetric.objects.create(
            user=self.user,
            metric_type="vo2-max",
            date=date(2026, 9, 11),
            value=42.4,
            unit="ml/kg/min",
            source="Fitbit demographic estimate",
        )
        response = self.client.get(reverse("health:fitness"))

        self.assertEqual(response.context["assessment"].difference, 2)
        self.assertContains(response, "2 years younger")
        self.assertContains(response, "Read the gap, not the number")

    def test_the_page_states_how_far_the_source_was_actually_checked(self):
        """The citation claims no more than was done."""
        Profile.objects.create(
            user=self.user, date_of_birth="1990-05-01", sex="male"
        )
        HistoricalMetric.objects.create(
            user=self.user,
            metric_type="vo2-max",
            date=date(2026, 9, 11),
            value=51.15,
            unit="ml/kg/min",
            source="Fitbit demographic estimate",
        )
        response = self.client.get(reverse("health:fitness"))

        self.assertContains(response, "11th ed., Wolters Kluwer, 2021, Table 4.7")
        self.assertContains(response, "spot-checked")
        self.assertContains(response, "not verified against the")
