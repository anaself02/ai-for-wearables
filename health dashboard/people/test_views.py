"""The home page, the person page, and the guarantee that people never mix.

Two people with data are built from synthetic files whose heart rate is a
constant marker (77 for Alpha, 133 for Beta), so any page can be searched for
the other person's number. Nothing here is real personal data.
"""

import re
from datetime import date

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from people import ingest, repository
from people.models import HeartRateDay, Person
from people.sources import SourceFile
from people.testing import utc_day_file, StaffTestCase

AWAITING = "No data uploaded yet. Expected after the mid-project presentation."


def make_person(slug, name, device, days, marker):
    person = Person.objects.create(slug=slug, display_name=name,
                                   device_label=device,
                                   adapter_key="fitbit-takeout")
    files = [SourceFile(*utc_day_file(d, marker=marker, source=device)) for d in days]
    ingest.commit_plan(ingest.build_plan(person, files))
    return person


class TwoPeopleCase(StaffTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.alpha = make_person("alpha", "Alpha", "Fitbit Air",
                                [date(2026, 7, 15), date(2026, 7, 16)], 77)
        cls.beta = make_person("beta", "Beta", "Fitbit Charge",
                               [date(2026, 8, 1), date(2026, 8, 2)], 133)


class HomePageTests(StaffTestCase):
    def test_the_seeded_people_are_listed_with_device_and_status(self):
        response = self.client.get(reverse("people:home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Anas")
        self.assertContains(response, "Fitbit Air")
        self.assertContains(response, "Rahil")
        self.assertContains(response, "EmotiBit")
        self.assertContains(response, "Awaiting data")

    def test_home_needs_no_sign_in_and_mentions_no_google(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Sign in with Google")
        self.assertNotContains(response, "oauth")

    def test_rahil_shows_the_honest_empty_state_with_no_numbers(self):
        response = self.client.get(reverse("people:home"))
        self.assertContains(response, AWAITING)
        self.assertNotContains(response, "bpm")

    def test_home_states_that_uploaded_data_is_only_scored(self):
        text = " ".join(self.client.get(reverse("people:home"))
                        .content.decode().split())
        self.assertIn("Models are trained on the public WESAD dataset only", text)
        self.assertIn("Uploaded data is only scored", text)

    def test_the_home_page_is_not_cacheable(self):
        response = self.client.get(reverse("people:home"))
        self.assertIn("no-store", response["Cache-Control"])


class HomeIsolationTests(TwoPeopleCase):
    def cards(self):
        html = self.client.get(reverse("people:home")).content.decode()
        sections = re.findall(r'<section class="person-card".*?</section>', html, re.S)
        return {re.search(r'<h2 id="p-([^"]+)"', s).group(1): s for s in sections}

    def test_each_card_holds_only_that_persons_range_and_device(self):
        cards = self.cards()
        # Local ranges, in Toronto: UTC day 07-15 starts on the evening of 07-14.
        self.assertIn("2026-07-14 to 2026-07-16", cards["alpha"])
        self.assertNotIn("2026-08", cards["alpha"])
        self.assertNotIn("2026-07-31", cards["alpha"])
        self.assertIn("2026-07-31 to 2026-08-02", cards["beta"])
        self.assertNotIn("2026-07-1", cards["beta"])
        self.assertIn("Fitbit Air", cards["alpha"])
        self.assertNotIn("Fitbit Charge", cards["alpha"])
        self.assertIn("Fitbit Charge", cards["beta"])
        self.assertNotIn("Fitbit Air", cards["beta"])

    def test_card_reading_counts_are_each_persons_own(self):
        cards = self.cards()
        self.assertIn("57,600", cards["alpha"])
        self.assertIn("57,600", cards["beta"])
        self.assertNotIn("115,200", cards["alpha"] + cards["beta"])   # never pooled


class PersonPageIsolationTests(TwoPeopleCase):
    def get(self, person):
        return self.client.get(reverse("people:person", args=[person.slug]))

    def test_alphas_page_has_alphas_numbers_and_none_of_betas(self):
        html = self.get(self.alpha).content.decode()
        self.assertIn("77.0", html)
        self.assertNotIn("133", html)
        self.assertNotIn("Beta", html)
        self.assertNotIn("Fitbit Charge", html)
        self.assertNotIn("2026-08", html)

    def test_betas_page_has_betas_numbers_and_none_of_alphas(self):
        html = self.get(self.beta).content.decode()
        self.assertIn("133.0", html)
        self.assertNotIn("77.0", html)
        self.assertNotIn("Alpha", html)
        self.assertNotIn("Fitbit Air", html)
        self.assertNotIn("2026-07-1", html)          # Alpha's dates

    def test_the_top_bar_names_the_person_and_device_and_offers_switch(self):
        for person in (self.alpha, self.beta):
            response = self.get(person)
            self.assertContains(response, 'class="person-badge"')
            bar = re.search(r'<header class="topbar">.*?</header>',
                            response.content.decode(), re.S).group(0)
            self.assertIn(person.display_name, bar)
            self.assertIn(person.device_label, bar)
            self.assertIn(reverse("people:home"), bar)
            self.assertIn(">Switch<", bar)
            other = self.beta if person is self.alpha else self.alpha
            self.assertNotIn(other.display_name, bar)

    def test_every_query_for_a_person_page_is_filtered_to_that_person(self):
        for person, other in ((self.alpha, self.beta), (self.beta, self.alpha)):
            with CaptureQueriesContext(connection) as queries:
                self.get(person)
            touching = [q["sql"] for q in queries
                        if "people_heartrateday" in q["sql"].lower()
                        or "people_recording" in q["sql"].lower()]
            self.assertTrue(touching)
            for sql in touching:
                self.assertRegex(sql, r'"person_id" = {}\b'.format(person.id), sql)
                self.assertNotRegex(sql, r'"person_id" = {}\b'.format(other.id), sql)

    def test_an_unknown_person_is_a_404_not_somebody_elses_page(self):
        self.assertEqual(self.client.get("/p/nobody/").status_code, 404)

    def test_person_pages_are_not_cacheable(self):
        self.assertIn("no-store", self.get(self.alpha)["Cache-Control"])


class RahilEmptyStateTests(TwoPeopleCase):
    def test_his_page_is_honest_and_contains_no_data(self):
        rahil = Person.objects.get(slug="rahil")
        response = self.client.get(reverse("people:person", args=[rahil.slug]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, AWAITING)
        self.assertContains(response, "Rahil")
        self.assertContains(response, "EmotiBit")
        html = response.content.decode()
        main = re.search(r"<main.*?</main>", html, re.S).group(0)
        self.assertNotIn("<table", html)
        self.assertNotIn("<canvas", html)
        self.assertNotRegex(main, r"\b\d{2,3}(\.\d)?\b")        # no bpm-like numbers
        self.assertNotRegex(main, r"\d{4}-\d{2}-\d{2}")         # no dates either
        for other in ("77.0", "133.0", "Alpha", "Beta", "Anas"):
            self.assertNotIn(other, html.replace("Anas's", ""))

    def test_a_person_with_a_working_reader_but_no_data_gets_the_plain_message(self):
        anas = Person.objects.get(slug="anas")
        response = self.client.get(reverse("people:person", args=[anas.slug]))
        self.assertContains(response, "No data uploaded yet.")
        self.assertNotContains(response, "Expected after the mid-project")


class RepositoryGuardTests(TwoPeopleCase):
    def test_every_function_refuses_anything_but_a_person(self):
        for bad in ("alpha", 1, None, {"slug": "alpha"}):
            for call in (repository.local_day_readings,):
                with self.assertRaises(TypeError):
                    call(bad, date(2026, 7, 16))
            with self.assertRaises(TypeError):
                repository.local_dates_with_data(bad)
            with self.assertRaises(TypeError):
                repository.recent_day_summaries(bad)
            with self.assertRaises(TypeError):
                repository.totals(bad)

    def test_readings_of_a_local_day_belong_to_that_person_only(self):
        epochs, bpm = repository.local_day_readings(self.alpha, date(2026, 7, 16))
        self.assertGreater(len(epochs), 0)
        self.assertEqual(set(bpm), {77.0})
        epochs, bpm = repository.local_day_readings(self.beta, date(2026, 7, 16))
        self.assertEqual(len(epochs), 0)       # Beta has nothing on Alpha's dates

    def test_a_toronto_day_is_assembled_from_two_utc_days(self):
        # Alpha has UTC days 07-15 and 07-16; local 07-15 spans both.
        epochs, _ = repository.local_day_readings(self.alpha, date(2026, 7, 15))
        start, end = repository.local_day_bounds(date(2026, 7, 15), "America/Toronto")
        self.assertEqual(int(epochs.min()), start)
        self.assertEqual(int(epochs.max()), end - 3)
        self.assertEqual(len(epochs), 28800)    # 24 h at 3 s

    def test_storage_rows_are_one_per_person_and_utc_day(self):
        self.assertEqual(HeartRateDay.objects.filter(person=self.alpha).count(), 2)
        self.assertEqual(HeartRateDay.objects.filter(person=self.beta).count(), 2)


class LoadedDataLocationTests(TwoPeopleCase):
    """The data summary lives on the Import data page, not on Overview."""

    MARKERS = ("Loaded data", "Local date range", "Device name(s) in the files",
               "the files themselves are in UTC")

    def test_the_summary_is_on_the_import_page_and_not_on_overview(self):
        main = lambda html: re.search(r"<main.*?</main>", html, re.S).group(0)
        imported = main(self.client.get(
            reverse("people:import_data", args=["alpha"])).content.decode())
        overview = main(self.client.get(
            reverse("people:person", args=["alpha"])).content.decode())
        for marker in self.MARKERS:
            self.assertIn(marker, imported, marker)
            self.assertNotIn(marker, overview, marker)
        self.assertIn("2026-07-14 to 2026-07-16", imported)
        self.assertIn("57,600", imported)
        self.assertLess(imported.index("Loaded data"), imported.index('enctype="multipart/form-data"'))
        self.assertNotIn("Import more data", overview)
        self.assertIn("Heart rate in detail", overview)
        self.assertIn("Raw uploaded rows", overview)

    def test_rahils_import_page_shows_the_empty_state_and_no_numbers(self):
        html = self.client.get(reverse("people:import_data", args=["rahil"])).content.decode()
        section = re.search(r'<section class="loaded-data".*?</section>', html, re.S).group(0)
        self.assertIn(AWAITING, section)
        self.assertNotIn("Local date range", section)
        self.assertNotRegex(section, r"\d{4}-\d{2}-\d{2}")
