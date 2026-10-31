"""Every person-scoped page: content, empty states, isolation and the bar.

Two people with data are built from synthetic files whose heart rate is a
constant marker (77 for Alpha, 133 for Beta). Nothing here is real data.
"""

import json
import re
from datetime import date, datetime, timezone

from django.test import TestCase
from django.urls import reverse

from people import ingest
from people.models import Person
from people.sources import SourceFile
from people.test_views import AWAITING, TwoPeopleCase, make_person
from people.testing import hr_csv_bytes, utc_day_file


def routes(person, day):
    """Every person-scoped URL, for one person and one local day."""
    slug = person.slug
    return {
        "overview": reverse("people:person", args=[slug]),
        "heart_rate": reverse("people:heart_rate", args=[slug]),
        "heart_rate_7": reverse("people:heart_rate", args=[slug]) + "?days=7",
        "heart_rate_day": reverse("people:heart_rate_day", args=[slug, day]),
        "raw_index": reverse("people:raw_index", args=[slug]),
        "raw_day": reverse("people:raw_day", args=[slug, day]),
    }


def bar_of(response):
    return re.search(r'<header class="topbar">.*?</header>',
                     response.content.decode(), re.S).group(0)


class EveryPageIsolationTests(TwoPeopleCase):
    def test_no_page_of_one_person_mentions_the_other(self):
        pairs = ((self.alpha, self.beta, date(2026, 7, 15), "77", "133"),
                 (self.beta, self.alpha, date(2026, 8, 1), "133", "77"))
        for person, other, day, own, foreign in pairs:
            for name, url in routes(person, day).items():
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200, (person.slug, name))
                html = response.content.decode()
                main = re.search(r"<main.*?</main>", html, re.S).group(0)
                self.assertNotIn(other.display_name, html, (person.slug, name))
                self.assertNotIn(other.device_label, html, (person.slug, name))
                self.assertNotRegex(main, r"\b{}(\.0)?\b".format(foreign),
                                    (person.slug, name))
                self.assertRegex(bar_of(response), person.display_name)

    def test_every_page_names_the_person_and_device_in_the_bar_with_switch(self):
        for person, day in ((self.alpha, date(2026, 7, 15)),
                            (self.beta, date(2026, 8, 1))):
            for name, url in routes(person, day).items():
                bar = bar_of(self.client.get(url))
                self.assertIn('class="person-badge"', bar, name)
                self.assertIn(person.display_name, bar, name)
                self.assertIn(person.device_label, bar, name)
                self.assertIn('href="/"', bar, name)
                self.assertIn(">Switch<", bar, name)

    def test_every_page_title_carries_name_and_device(self):
        for name, url in routes(self.alpha, date(2026, 7, 15)).items():
            html = self.client.get(url).content.decode()
            title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
            self.assertIn("Alpha", title, name)
            self.assertIn("Fitbit Air", title, name)

    def test_every_page_is_served_no_store(self):
        for name, url in routes(self.alpha, date(2026, 7, 15)).items():
            self.assertIn("no-store", self.client.get(url)["Cache-Control"], name)

    def test_pages_for_a_day_that_belongs_to_the_other_person_are_empty_not_borrowed(self):
        beta_day = date(2026, 8, 1)
        response = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", beta_day]))
        self.assertContains(response, "No readings on this local day.")
        self.assertNotContains(response, "133")
        response = self.client.get(reverse("people:raw_day", args=["alpha", beta_day]))
        self.assertContains(response, "No rows on this local day.")


class EmptyStateOnEveryPageTests(TestCase):
    def test_rahils_pages_are_all_honest_and_hold_no_numbers(self):
        day = date(2026, 7, 16)
        for name, url in routes(Person.objects.get(slug="rahil"), day).items():
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, name)
            self.assertContains(response, AWAITING, msg_prefix=name)
            html = response.content.decode()
            main = re.search(r"<main.*?</main>", html, re.S).group(0)
            main = re.sub(r'href="[^"]*"', "", main)   # link targets are not data
            main = main.replace(str(day), "")          # the date asked for in the URL
            self.assertNotIn("<table", html, name)
            self.assertNotIn("<canvas", html, name)
            self.assertNotRegex(main, r"\d{4}-\d{2}-\d{2}", name)
            self.assertNotRegex(main, r"\b\d{2,3}(\.\d)?\b", name)
            self.assertIn("Rahil", bar_of(response), name)
            self.assertIn("EmotiBit", bar_of(response), name)

    def test_anas_before_any_upload_gets_the_plain_message_on_every_page(self):
        day = date(2026, 7, 16)
        for name, url in routes(Person.objects.get(slug="anas"), day).items():
            response = self.client.get(url)
            self.assertContains(response, "No data uploaded yet.", msg_prefix=name)
            self.assertNotContains(response, "Expected after the mid-project",
                                   msg_prefix=name)


class HeartRatePageTests(TwoPeopleCase):
    def test_window_chips_and_fallback(self):
        url = reverse("people:heart_rate", args=["alpha"])
        for days in (7, 30, 90):
            html = self.client.get(url + "?days={}".format(days)).content.decode()
            self.assertRegex(html, r'chip active" href="\?days={}"'.format(days))
        html = self.client.get(url + "?days=999").content.decode()
        self.assertRegex(html, r'chip active" href="\?days=30"')
        html = self.client.get(url + "?days=abc").content.decode()
        self.assertRegex(html, r'chip active" href="\?days=30"')

    def test_the_chart_data_is_this_persons_daily_series(self):
        response = self.client.get(reverse("people:heart_rate", args=["alpha"]))
        html = response.content.decode()
        payload = json.loads(re.search(
            r'<script id="chart-heart-rate" type="application/json">(.*?)</script>',
            html, re.S).group(1))
        self.assertEqual(payload["labels"], ["2026-07-14", "2026-07-15", "2026-07-16"])
        names = [d["label"] for d in payload["datasets"]]
        self.assertEqual(names, ["Average", "Min", "Max"])
        for dataset in payload["datasets"]:
            self.assertEqual(set(dataset["data"]), {77.0})

    def test_the_day_picker_points_at_this_persons_day_pages(self):
        html = self.client.get(reverse("people:heart_rate", args=["beta"])).content.decode()
        self.assertIn('data-day-picker="/p/beta/heart-rate/day/"', html)
        self.assertNotIn("/p/alpha/", html)

    def test_computed_values_are_marked_derived_and_explained(self):
        html = self.client.get(reverse("people:heart_rate", args=["alpha"])).content.decode()
        self.assertIn('class="num derived"', html)
        text = " ".join(html.split())
        self.assertIn("computed by this app", text)
        self.assertIn("exactly as in the file", text)


class DayPageTests(TwoPeopleCase):
    def payload(self, html):
        return json.loads(re.search(
            r'<script id="day-heart-rate" type="application/json">(.*?)</script>',
            html, re.S).group(1))

    def test_a_day_shows_per_minute_means_device_and_spacing(self):
        response = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", date(2026, 7, 15)]))
        html = response.content.decode()
        series = self.payload(html)
        self.assertEqual(len(series["labels"]), 1440)
        self.assertEqual({v for v in series["datasets"][0]["data"]}, {77.0})
        self.assertIn("Synthetic Band", html)
        self.assertIn("median 3.0 s", html)
        self.assertIn("28,800", html)
        self.assertIn("00:00 / 23:59", html)       # first / last reading, local
        self.assertIn("computed by this app", " ".join(html.split()))

    def test_a_partial_day_leaves_gaps_open_instead_of_bridging_them(self):
        # Alpha's UTC day 07-16 ends at 00:00Z 07-17; local 07-16 ends 20:00 Toronto.
        response = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", date(2026, 7, 16)]))
        data = self.payload(response.content.decode())["datasets"][0]["data"]
        self.assertIn(None, data)
        self.assertEqual(data[0], 77.0)

    def test_previous_and_next_links_only_go_to_days_with_data(self):
        html = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", date(2026, 7, 15)])
        ).content.decode()
        self.assertIn("/p/alpha/heart-rate/day/2026-07-14/", html)
        self.assertIn("/p/alpha/heart-rate/day/2026-07-16/", html)
        html = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", date(2026, 7, 16)])
        ).content.decode()
        self.assertNotIn("2026-07-17", html)

    def test_a_day_with_no_data_says_so_and_invents_nothing(self):
        response = self.client.get(
            reverse("people:heart_rate_day", args=["alpha", date(2026, 1, 1)]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No readings on this local day.")
        self.assertNotContains(response, "<canvas")

    def test_an_impossible_date_is_a_404(self):
        self.assertEqual(self.client.get("/p/alpha/heart-rate/day/2026-13-45/").status_code, 404)
        self.assertEqual(self.client.get("/p/alpha/raw/2026-02-30/").status_code, 404)
        self.assertEqual(self.client.get("/p/alpha/heart-rate/day/not-a-date/").status_code, 404)


class RawPageTests(TwoPeopleCase):
    def test_the_index_lists_this_persons_files_and_days_only(self):
        html = self.client.get(reverse("people:raw_index", args=["alpha"])).content.decode()
        self.assertIn("heart_rate_2026-07-15.csv", html)
        self.assertIn("heart_rate_2026-07-16.csv", html)
        self.assertNotIn("heart_rate_2026-08-01.csv", html)
        for recording in self.alpha.recordings.all():
            self.assertIn(recording.sha256[:12], html)
        for recording in self.beta.recordings.all():
            self.assertNotIn(recording.sha256[:12], html)

    def test_a_day_shows_the_first_rows_in_the_files_own_utc_format(self):
        html = self.client.get(
            reverse("people:raw_day", args=["alpha", date(2026, 7, 15)])).content.decode()
        self.assertIn("Showing the first 100 of 28,800 rows", html)
        self.assertIn("2026-07-15T04:00:00Z", html)        # 00:00 Toronto = 04:00Z
        self.assertIn("2026-07-15T04:00:03Z", html)
        self.assertEqual(html.count("<tr>"), 101)           # header + 100 rows
        self.assertIn("77.0", html)


class DaylightSavingDayTests(TestCase):
    def test_a_25_hour_day_has_1500_minutes_and_a_repeated_hour(self):
        person = Person.objects.create(slug="dst", display_name="Dst",
                                       device_label="Fitbit Air",
                                       adapter_key="fitbit-takeout")
        files = [SourceFile(*utc_day_file(date(2026, 11, 1), marker=70)),
                 SourceFile(*utc_day_file(date(2026, 11, 2), marker=70))]
        ingest.commit_plan(ingest.build_plan(person, files))

        response = self.client.get(
            reverse("people:heart_rate_day", args=["dst", date(2026, 11, 1)]))
        html = response.content.decode()
        series = json.loads(re.search(
            r'<script id="day-heart-rate" type="application/json">(.*?)</script>',
            html, re.S).group(1))
        self.assertEqual(len(series["labels"]), 1500)
        self.assertEqual(series["labels"].count("01:00"), 2)    # the hour that repeats


class RemovedRoutesTests(TestCase):
    def test_the_old_signed_in_pages_are_gone(self):
        for url in ("/dashboard/", "/insights/", "/fitness/", "/sleep-calculator/",
                    "/raw/", "/metric/heart-rate/", "/arousal/compute/",
                    "/welcome/", "/welcome/warm/"):
            self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_no_page_mentions_signing_in_with_google(self):
        for url in ("/", "/p/anas/", "/p/rahil/"):
            html = self.client.get(url).content.decode()
            for phrase in ("Sign in", "Sign out", "oauth", "Google"):
                self.assertNotIn(phrase, html, (url, phrase))
