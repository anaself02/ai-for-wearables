"""Fitness age on the overview: what it says, and what it must never say.

It stands in the dial row, drawn in a circle, and it is not a dial. That is the
whole point of the component: an arc is a fraction of something, and a fitness
age is not a fraction of anything. So the load-bearing tests here pin absences
rather than values - no arc, no prose, no calendar age - because each of those
is a thing that would grow back under pressure to make the circle match its
four neighbours.

The bound case carries its own trap. A None comparison inside a Django
`{% if %}` evaluates false in silence, which is how a reading the published
table cannot place would quietly print a verdict anyway.
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

    def test_it_shows_the_fitness_age_in_the_dial_row(self):
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "health/_fitness_age.html")
        self.assertContains(response, "Fitness age")
        self.assertTrue(response.context["assessment"].available)

    def test_the_circle_has_no_arc_because_there_is_no_fraction(self):
        """The whole objection to a ring here was the denominator it would imply.

        A circular frame answers that only for as long as nothing proportional
        is drawn inside it, so this pins the absence: one plain circle, and no
        track or arc anywhere in the component. Standing in the row beside four
        real dials, this is the test that stops one being added to make it fit.
        """
        self._profile()
        self._vo2()
        html = self._overview().content.decode("utf-8")
        frame = html[html.index('class="ring is-frame') :]
        frame = frame[: frame.index('<div class="cards"')]

        self.assertIn('class="ring-frame"', frame)
        self.assertNotIn("stroke-dasharray", frame)
        self.assertNotIn("ring-arc", frame)
        self.assertNotIn("ring-track", frame)

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

    def test_a_reading_inside_the_table_shows_the_number_bare(self):
        self._profile(born="1980-03-09")
        self._vo2(value=42.4)  # the published median for men 30-39

        response = self._overview()
        assessment = response.context["assessment"]

        self.assertEqual(assessment.fitness_age_position, "inside")
        self.assertEqual(assessment.fitness_age, 34)

    # -- what the component must not say -------------------------------------

    def test_the_calendar_age_comparison_is_gone(self):
        """It belongs on /fitness/, where a gap can be explained when it exists."""
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertNotContains(response, "Your age")
        self.assertNotContains(response, "Read the gap")
        self.assertNotContains(response, "years younger")
        self.assertNotContains(response, "years older")

    def test_the_circle_carries_no_prose_beside_it(self):
        """A glance, not an explanation: the caveat moved to /fitness/.

        Pinned because the explanation is the thing that keeps growing back -
        every honest caveat wants to be next to the number it qualifies.
        """
        self._profile()
        self._vo2()
        html = self._overview().content.decode("utf-8")
        frame = html[html.index('class="ring is-frame') :]
        frame = frame[: frame.index('<div class="cards"')]

        self.assertNotIn("A bound rather than a number", frame)
        self.assertNotIn("Full method", frame)
        self.assertNotIn("rings-age-note", frame)
        self.assertNotIn("<a ", frame)

    def test_the_caveat_survives_for_a_reader_who_cannot_see_the_layout(self):
        """Moved for visual quiet, not deleted: the label still says all of it."""
        self._profile()
        self._vo2()
        response = self._overview()

        self.assertTrue(response.context["assessment"].circular)
        self.assertContains(response, "not a biological age")
        self.assertContains(response, "partly circular")

    def test_the_panel_explains_nothing_in_prose_at_all(self):
        """The dials carry their own denominators; the panel carries no essay.

        Each note under a dial already names what its arc is a fraction of -
        "94% of your usual 8,938", "79th percentile, men 30-39", "out of 100,
        Fitbit's own score" - so the row is self-describing without a footnote
        under it. The method lives on /fitness/, reachable from the top nav.
        """
        self._profile()
        self._vo2()
        html = self._overview().content.decode("utf-8")
        panel = html[html.index('<div class="rings">') : html.index('<div class="cards"')]

        self.assertNotIn("rings-footnote", panel)
        self.assertNotIn("Full method", panel)
        self.assertNotIn("<p", panel)

    # -- the empty states, which must not vanish -----------------------------

    def test_without_a_date_of_birth_the_circle_is_empty_rather_than_absent(self):
        """An absent circle reads as a row that never had the feature."""
        self._vo2()
        response = self._overview()

        self.assertFalse(response.context["assessment"].available)
        self.assertContains(response, "Fitness age")
        self.assertContains(response, "&mdash;")
        self.assertContains(response, "ring is-frame is-empty")

    def test_without_a_vo2_max_reading_it_is_also_empty(self):
        self._profile()
        response = self._overview()

        self.assertFalse(response.context["assessment"].available)
        self.assertContains(response, "ring is-frame is-empty")
        self.assertContains(response, "&mdash;")

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
        """It is drawn in a circle, but it is not in the row and not a Ring.

        Nothing about it is built by health.rings, so it cannot pick up an arc
        by being handled alongside things that have one.
        """
        self._profile()
        self._vo2()
        response = self._overview()

        keys = [r.key for r in response.context["rings"]]
        self.assertEqual(keys, ["vo2-max", "steps", "active-minutes", "sleep-score"])
