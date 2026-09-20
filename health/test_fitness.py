"""The fitness percentile, and the fitness age it refuses to compute."""

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
    def test_a_mean_value_returns_the_age_it_is_the_mean_for(self):
        # 49.0 is the HUNT3 mean for men in their thirties, midpoint 35.
        age, position = fitness.fitness_age(49.0, "male")
        self.assertEqual(position, "inside")
        self.assertEqual(age, 35)

    def test_fitter_than_the_youngest_band_is_reported_as_below_it(self):
        age, position = fitness.fitness_age(70.0, "male")
        self.assertEqual(position, "below")
        self.assertEqual(age, 25)

    def test_a_higher_vo2max_gives_a_lower_fitness_age(self):
        younger, _ = fitness.fitness_age(50.0, "male")
        older, _ = fitness.fitness_age(40.0, "male")
        self.assertLess(younger, older)


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

    def test_a_demographic_estimate_gets_a_percentile_but_no_fitness_age(self):
        """The whole point: age goes into that figure, so it cannot come out."""
        result = fitness.assess(
            51.15, source="Fitbit demographic estimate", measured=False,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.percentile)
        self.assertIsNone(result.fitness_age)
        self.assertIn("age back out", result.fitness_age_withheld_because)

    def test_a_measured_value_does_get_a_fitness_age(self):
        result = fitness.assess(
            51.15, source="GPS-tracked run", measured=True,
            born=date(1990, 5, 1), sex="male", today=TODAY,
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.fitness_age)
        self.assertEqual(result.fitness_age_withheld_because, "")


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

    def test_the_page_never_prints_a_fitness_age_for_a_demographic_estimate(self):
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
        self.assertTrue(response.context["assessment"].available)
        self.assertIsNone(response.context["assessment"].fitness_age)
        self.assertContains(response, "Withheld, on purpose")
        # And it never claims to be a biological age anywhere on the page.
        self.assertNotContains(response, "biological age measurement")
