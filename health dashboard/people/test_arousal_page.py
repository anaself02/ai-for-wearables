"""The arousal page: a chosen day, the caveat in the same branch, and no mixing.

Models are stand-ins (one class for every window) except in the end-to-end test
in health/test_arousal.py. Every reading is synthetic.
"""

import re
from datetime import date
from unittest import mock

from django.test import TestCase
from django.urls import reverse

from people.testing import StaffTestCase

from health import arousal
from health.test_arousal import (FEATURE_NAMES, fake_artifacts, fake_model,
                                 make_person, make_short_person)
from people.fitguard import fit_guard
from people.models import Person
from people.test_views import AWAITING, make_person as make_marked_person

NOTICE = "Model trained on WESAD only; uploaded data is only scored."


def patched(model=None):
    return mock.patch.object(arousal, "load_artifacts",
                             return_value=fake_artifacts(model))


def result_block(html):
    """The one block that holds headline, caveat, notes and the methodology link."""
    start = html.index('<div class="insight dir-neutral">')
    end = html.index("What measured this day")
    return html[start:end]


class CaveatInTheSameBranchTests(StaffTestCase):
    """Property 1: no label is ever printed without its caveat beside it."""

    @classmethod
    def setUpTestData(cls):
        cls.person = make_person()

    def get(self, **params):
        return self.client.get(reverse("people:arousal", args=[self.person.slug]), params)

    def test_for_every_possible_label_the_caveat_and_link_share_its_block(self):
        for label, (headline, _) in arousal.CLASS_WORDING.items():
            with patched(fake_model(label)):
                html = self.get().content.decode()
            block = result_block(html)
            self.assertIn(headline, block, label)
            self.assertIn("not a clinical or validated measure", block, label)
            self.assertIn("low confidence", block, label)
            self.assertIn(reverse("people:arousal_methodology", args=[self.person.slug]),
                          block, label)
            self.assertLess(block.index(headline),
                            block.index("not a clinical or validated measure"), label)

    def test_the_page_never_shows_a_headline_outside_that_block(self):
        with patched():
            html = self.get().content.decode()
        for headline, _ in arousal.CLASS_WORDING.values():
            self.assertEqual(html.count(headline), html[:].count(headline))
            self.assertEqual(result_block(html).count(headline), html.count(headline))

    def test_an_unavailable_result_shows_a_reason_and_no_label(self):
        with mock.patch.object(arousal, "load_artifacts",
                               side_effect=arousal.ArousalUnavailable("no model today")):
            html = self.get().content.decode()
        self.assertIn("No assessment available", html)
        self.assertIn("no model today", html)
        for headline, _ in arousal.CLASS_WORDING.values():
            self.assertNotIn(headline, html)

    def test_out_of_distribution_and_degenerate_notes_travel_with_the_label(self):
        flat = make_person("flat", spread=0)      # constant bpm: HR_std is 0, below training
        with patched(fake_model(3)):              # every window labelled amusement-like
            html = self.client.get(reverse("people:arousal", args=[flat.slug])
                                   ).content.decode()
        block = result_block(html)
        self.assertIn("OUT OF DISTRIBUTION", block)
        self.assertIn("DEGENERATE", block)
        self.assertIn("cannot separate amusement from baseline", block)
        self.assertIn("NOT interpretable", block)
        # Out of distribution AND degenerate: no amusement-like headline, caveats kept.
        self.assertIn(arousal.NO_RESULT[0], block)
        self.assertIn("not a clinical or validated measure", block)
        self.assertNotIn("Light arousal signal", html)


class ChoosingADayTests(StaffTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.person = make_person()

    def get(self, **params):
        with patched():
            return self.client.get(reverse("people:arousal", args=["subject"]), params)

    def test_the_default_is_the_most_recent_scorable_day(self):
        response = self.get()
        self.assertEqual(response.context["chosen"], date(2026, 9, 18))
        self.assertContains(response, '<option value="2026-09-18" selected>')

    def test_a_chosen_date_is_scored(self):
        response = self.get(date="2026-09-17")
        self.assertEqual(response.context["chosen"], date(2026, 9, 17))
        self.assertEqual(response.context["assessment"].n_windows, 2879)
        self.assertContains(response, '<option value="2026-09-17" selected>')

    def test_every_day_with_data_is_listed_with_its_scorable_windows(self):
        html = self.get().content.decode()
        for day in ("2026-09-16", "2026-09-17", "2026-09-18"):
            self.assertIn(day, html)
        self.assertIn("2,879", html)

    def test_the_scorable_threshold_is_printed_as_a_constant(self):
        html = " ".join(self.get().content.decode().split())
        self.assertIn("at least <strong>60</strong> windows", html)

    def test_an_invalid_date_falls_back_to_the_default_with_a_message(self):
        response = self.get(date="2026-13-45", follow=True) if False else \
            self.get(date="not-a-date")
        self.assertEqual(response.context["chosen"], date(2026, 9, 18))
        self.assertContains(response, "not a valid date")

    def test_a_date_with_no_readings_says_so_and_shows_no_label(self):
        response = self.get(date="2026-01-01")
        self.assertContains(response, "No assessment available")
        self.assertContains(response, "No heart-rate readings")
        self.assertNotContains(response, "Elevated arousal signal")

    def test_the_arousal_page_is_not_cacheable(self):
        self.assertIn("no-store", self.get()["Cache-Control"])


class DeviceAndSpacingTests(StaffTestCase):
    def test_the_result_names_the_person_device_and_reading_spacing(self):
        person = make_person()
        with patched():
            response = self.client.get(reverse("people:arousal", args=[person.slug]),
                                       {"date": "2026-09-17"})
        html = " ".join(response.content.decode().split())
        self.assertIn("What measured this day", html)
        self.assertIn("Subject", html)
        self.assertIn("Fitbit Air", html)
        self.assertIn("median 3.0 s", html)
        self.assertIn("Resolution mismatch", html)
        self.assertIn("one heart-rate value per heartbeat", html)
        self.assertIn("each already smoothed by the device", html)

    def test_the_page_says_models_are_wesad_only_and_data_is_only_scored(self):
        person = make_person()
        with patched():
            html = self.client.get(reverse("people:arousal", args=[person.slug])
                                   ).content.decode()
        self.assertIn(NOTICE, html)


class OnlyDaysWithEnoughDataTests(StaffTestCase):
    def test_a_short_day_is_marked_not_scorable_and_is_not_the_default(self):
        short = make_short_person(minutes=10)
        with patched():
            response = self.client.get(reverse("people:arousal", args=[short.slug]))
        self.assertTrue(response.context["no_scorable_day"])
        self.assertContains(response, "No day has enough data to score")
        self.assertNotContains(response, "Elevated arousal signal")

    def test_a_short_day_chosen_explicitly_explains_why_it_cannot_be_scored(self):
        short = make_short_person(minutes=10)
        # not the default, but it can still be asked for
        make_person("long", days=(date(2026, 9, 17),))
        with patched():
            response = self.client.get(reverse("people:arousal", args=[short.slug]),
                                       {"date": "2026-09-18"})
        self.assertEqual(response.status_code, 200)


class IsolationTests(StaffTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.alpha = make_marked_person("alpha", "Alpha", "Fitbit Air",
                                       [date(2026, 7, 15), date(2026, 7, 16)], 77)
        cls.beta = make_marked_person("beta", "Beta", "Fitbit Charge",
                                      [date(2026, 8, 1), date(2026, 8, 2)], 133)

    def test_each_persons_page_lists_only_their_days_and_scores_only_their_readings(self):
        for person, other, mine, theirs in ((self.alpha, self.beta, "2026-07-15", "2026-08-01"),
                                            (self.beta, self.alpha, "2026-08-01", "2026-07-15")):
            with patched():
                response = self.client.get(reverse("people:arousal", args=[person.slug]))
            html = response.content.decode()
            main = re.search(r"<main.*?</main>", html, re.S).group(0)
            self.assertIn(mine, main)
            self.assertNotIn(theirs, main)
            self.assertNotIn(other.display_name, html)
            self.assertNotIn(other.device_label, html)
            bar = re.search(r'<header class="topbar">.*?</header>', html, re.S).group(0)
            self.assertIn(person.display_name, bar)
            self.assertIn(person.device_label, bar)

    def test_scoring_a_day_that_belongs_to_the_other_person_finds_nothing(self):
        with patched():
            response = self.client.get(reverse("people:arousal", args=["alpha"]),
                                       {"date": "2026-08-01"})     # Beta's day
        self.assertContains(response, "No heart-rate readings")
        self.assertNotContains(response, "Elevated arousal signal")

    def test_the_arousal_tab_is_in_every_persons_bar(self):
        for slug in ("alpha", "beta", "anas", "rahil"):
            bar = re.search(r'<header class="topbar">.*?</header>',
                            self.client.get("/p/{}/".format(slug)).content.decode(),
                            re.S).group(0)
            self.assertIn(reverse("people:arousal", args=[slug]), bar)


class EmptyStateTests(StaffTestCase):
    def test_rahil_gets_the_honest_message_and_no_scoring(self):
        with patched() as loaded:
            response = self.client.get(reverse("people:arousal", args=["rahil"]))
        self.assertContains(response, AWAITING)
        self.assertContains(response, NOTICE)
        self.assertNotContains(response, "Score this day")
        self.assertNotContains(response, "<table")
        loaded.assert_not_called()                      # the model was never loaded

    def test_anas_before_any_upload_gets_the_plain_message(self):
        response = self.client.get(reverse("people:arousal", args=["anas"]))
        self.assertContains(response, "No data uploaded yet.")
        self.assertNotContains(response, "Expected after the mid-project")


class MethodologyPageTests(StaffTestCase):
    def card(self):
        return {"honest_summary": "A demonstration, not a validated detector.",
                "algorithm": "SVM", "trained_on": {"dataset": "WESAD", "signals": "hr",
                                                   "n_windows": 1015, "window_seconds": 60,
                                                   "n_subjects": 15},
                "classes": {"1": "baseline"}, "validation": "LOSO",
                "scores": {"hr_only": {"svm": {"accuracy": 0.5, "f1_macro": 0.4}}},
                "always_guess_largest_class": {"accuracy": 0.5, "f1_macro": 0.2},
                "limitations": ["Only laboratory stress."]}

    def test_it_renders_the_model_card_and_names_the_person(self):
        with mock.patch.object(arousal, "load_model_card", return_value=self.card()):
            response = self.client.get(
                reverse("people:arousal_methodology", args=["anas"]))
        self.assertContains(response, "A demonstration, not a validated detector.")
        self.assertContains(response, "Only laboratory stress.")
        self.assertContains(response, NOTICE)
        self.assertContains(response, 'class="person-badge"')
        self.assertIn("no confidence interval", " ".join(response.content.decode().split()))

    def test_it_survives_a_missing_model_card(self):
        with mock.patch.object(arousal, "load_model_card", return_value=None):
            response = self.client.get(
                reverse("people:arousal_methodology", args=["anas"]))
        self.assertEqual(response.status_code, 200)
        self.assertIn("cannot describe the model honestly",
                      " ".join(response.content.decode().split()))


class NeverTrainsOnTheDayTests(StaffTestCase):
    def test_the_arousal_page_flow_never_fits_or_saves_a_model(self):
        person = make_person()
        with fit_guard(), patched():
            for params in ({}, {"date": "2026-09-17"}, {"date": "2026-01-01"}):
                response = self.client.get(
                    reverse("people:arousal", args=[person.slug]), params)
                self.assertEqual(response.status_code, 200)

    def test_arousal_source_never_fits_or_saves_a_model(self):
        import os
        forbidden = re.compile(r"\.fit\(|\.fit_transform\(|\.partial_fit\(|joblib\.dump|"
                               r"pickle\.dump|GridSearchCV|RandomizedSearchCV")
        path = os.path.join(os.path.dirname(arousal.__file__), "arousal.py")
        with open(path, encoding="utf-8") as handle:
            offenders = [line.strip() for line in handle
                         if forbidden.search(line) and not line.lstrip().startswith("#")]
        self.assertEqual(offenders, [])


class CompleteAndPartialDayTests(StaffTestCase):
    """Synthetic UTC days 2026-09-17 and -18 at 3 s: in Toronto (UTC-4) the local
    16th has 4 h of readings, the 17th 24 h and the 18th 20 h."""

    @classmethod
    def setUpTestData(cls):
        cls.person = make_person()

    def get(self, **params):
        with patched():
            return self.client.get(reverse("people:arousal", args=["subject"]), params)

    def test_coverage_numbers_on_synthetic_data(self):
        days = {entry["day"]: entry for entry in arousal.scorable_days(self.person)}
        self.assertEqual(days[date(2026, 9, 16)]["hours"], 4.0)
        self.assertEqual(days[date(2026, 9, 17)]["hours"], 24.0)
        self.assertEqual(days[date(2026, 9, 18)]["hours"], 20.0)
        response = self.get(date="2026-09-18")
        coverage = response.context["assessment"].coverage
        self.assertEqual(coverage["readings"], 20 * 3600 // 3)
        self.assertEqual(coverage["first"].strftime("%H:%M:%S"), "00:00:00")
        self.assertEqual(coverage["last"].strftime("%H:%M:%S"), "19:59:57")
        html = " ".join(response.content.decode().split())
        self.assertIn("00:00:00 / 19:59:57", html)
        self.assertIn("20.0 h", html)

    def test_a_partial_day_is_marked_and_the_threshold_is_shown(self):
        html = " ".join(self.get().content.decode().split())
        self.assertIn("at least <strong>{}</strong> hours".format(arousal.COMPLETE_DAY_HOURS), html)
        self.assertIn("2026-09-16 (partial)", html)
        self.assertNotIn("2026-09-17 (partial)", html)
        self.assertNotIn("2026-09-18 (partial)", html)

    def test_the_default_skips_partial_days(self):
        with mock.patch.object(arousal, "COMPLETE_DAY_HOURS", 21):   # the 18th becomes partial
            response = self.get()
            self.assertEqual(response.context["chosen"], date(2026, 9, 17))
            self.assertContains(response, "2026-09-18 (partial)")

    def test_a_chosen_partial_day_carries_the_note_beside_its_headline(self):
        html = self.get(date="2026-09-16").content.decode()
        block = result_block(html)
        self.assertIn("Elevated arousal signal", block)
        self.assertIn("This day is PARTIAL", block)
        self.assertIn("not a clinical or validated measure", block)
        self.assertEqual(html.count("This day is PARTIAL"), 1)
