"""Fitness age on the overview: what it says, and what it must never say.

It is deliberately not a dial - a ring is a fraction of something and an age is
not - so what is pinned here is the wording rather than any geometry. Two of
these tests exist because the same mistake has already been made once in this
project: a None comparison inside a Django `{% if %}` evaluates false in
silence, which would print a confident verdict for a reading the published
table cannot place at all.
"""

from datetime import date
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from accounts.models import GoogleCredential, Profile
from health.models import HistoricalMetric

User = get_user_model()


class OverviewFitnessAgeTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )
        self.client.force_login(self.user)

    def _profile(self, born="2002-03-09", sex="male"):
        return Profile.objects.create(
            user=self.user, date_of_birth=born, sex=sex
        )

    def _vo2(self, value=51.15):
        return HistoricalMetric.objects.create(
            user=self.user,
            metric_type="vo2-max",
            date=date(2026, 9, 11),
            value=value,
            unit="ml/kg/min",
            source="Fitbit demographic estimate",
        )

    def _overview(self):
        """The overview with the API stubbed out: none of this needs a call."""
        fake = mock.MagicMock()
        fake.list_data_points.side_effect = lambda dt, **kw: []
        fake.daily_roll_up.side_effect = lambda dt, **kw: []
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            return self.client.get(reverse("health:index"))

    # -- the real account's case --------------------------------------------

    def test_it_shows_both_ages_on_the_overview(self):
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "health/_fitness_age.html")
        self.assertContains(response, "Fitness age")
        self.assertContains(response, "Your age")
        self.assertTrue(response.context["assessment"].available)

    def test_a_fitness_age_off_the_young_end_is_a_bound_not_a_number(self):
        """51.15 beats the median for every band the table covers."""
        self._profile()
        self._vo2()
        response = self._overview()

        assessment = response.context["assessment"]
        self.assertEqual(assessment.fitness_age_position, "below")
        self.assertEqual(assessment.fitness_age, 24)
        self.assertIsNone(assessment.difference)
        self.assertContains(response, "&lt;24")
        self.assertContains(response, "bound rather than a number")

    def test_a_bound_never_prints_a_verdict_it_cannot_support(self):
        """The None-comparison trap: `{% if difference > 0 %}` is false for None."""
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertNotContains(response, "Median for your age")
        self.assertNotContains(response, "years younger")
        self.assertNotContains(response, "years older")

    def test_a_bound_does_not_tell_the_reader_to_read_a_gap(self):
        """There is nothing to subtract, so the usual advice would be wrong."""
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertNotContains(response, "Read the gap")
        self.assertContains(response, "no gap to read")

    # -- the ordinary case, where there is a difference ----------------------

    def test_a_reading_inside_the_table_reports_the_gap_in_years(self):
        self._profile(born="1980-03-09")
        self._vo2(value=42.4)  # the published median for men 30-39
        response = self._overview()

        assessment = response.context["assessment"]
        self.assertEqual(assessment.fitness_age_position, "inside")
        self.assertEqual(assessment.age, 46)
        self.assertEqual(assessment.fitness_age, 34)
        self.assertContains(response, "12 years younger")
        self.assertContains(response, "Read the gap, not the number")

    def test_a_reading_below_the_curve_reports_the_gap_the_other_way(self):
        self._profile(born="1996-03-09")
        self._vo2(value=37.8)  # the published median for men 40-49
        response = self._overview()

        assessment = response.context["assessment"]
        self.assertEqual(assessment.difference, -14)
        self.assertContains(response, "14 years older")

    # -- the empty states, which must not vanish -----------------------------

    def test_without_a_date_of_birth_it_says_so_and_offers_the_fix(self):
        """An absent block reads as a page that never had the feature."""
        self._vo2()
        response = self._overview()

        self.assertFalse(response.context["assessment"].available)
        self.assertContains(response, "Fitness age")
        self.assertContains(response, reverse("accounts:settings"))

    def test_without_a_vo2_max_reading_it_says_that_instead(self):
        self._profile()
        response = self._overview()

        assessment = response.context["assessment"]
        self.assertFalse(assessment.available)
        self.assertContains(response, "GPS-tracked outdoor runs")
        self.assertContains(response, reverse("health:fitness"))

    # -- the invariant -------------------------------------------------------

    def test_the_overview_and_the_fitness_page_agree(self):
        """One function answers both, so they cannot drift apart.

        This project has already shipped a panel where a percentile and a
        fitness age came from two different cohorts and contradicted each
        other. Two surfaces deriving the same number twice is how that starts.
        """
        self._profile()
        self._vo2()

        overview = self._overview().context["assessment"]
        page = self.client.get(reverse("health:fitness")).context["assessment"]

        self.assertEqual(overview.age, page.age)
        self.assertEqual(overview.percentile, page.percentile)
        self.assertEqual(overview.fitness_age, page.fitness_age)
        self.assertEqual(overview.fitness_age_position, page.fitness_age_position)
        self.assertEqual(overview.circular, page.circular)

    def test_the_dial_arc_is_the_same_percentile_the_panel_was_built_from(self):
        self._profile()
        self._vo2()
        response = self._overview()

        assessment = response.context["assessment"]
        dial = next(r for r in response.context["rings"] if r.key == "vo2-max")
        self.assertEqual(dial.percent, round(assessment.percentile))

    def test_fitness_age_is_not_one_of_the_dials(self):
        """It is not a fraction of anything, so it does not get a ring."""
        self._profile()
        self._vo2()
        response = self._overview()

        keys = [r.key for r in response.context["rings"]]
        self.assertEqual(keys, ["vo2-max", "steps", "active-minutes", "sleep-score"])
