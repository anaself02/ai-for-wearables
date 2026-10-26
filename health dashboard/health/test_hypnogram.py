"""Tests for the per-night sleep-stage timeline.

The fixture mirrors the shape confirmed against a live response on 2026-09-19:
`stages` tiles the night with no gaps or overlaps, each segment carries its own
UTC offset, and `shortAwakenings` are nested inside the stage segments rather
than sitting alongside them.
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase

from accounts.models import GoogleCredential
from health import metrics

User = get_user_model()

OFFSET = "-14400s"  # UTC-4, as the live data reports


def segment(start, end, stage):
    return {
        "startTime": f"2026-09-19T{start}:00Z",
        "startUtcOffset": OFFSET,
        "endTime": f"2026-09-19T{end}:00Z",
        "endUtcOffset": OFFSET,
        "type": stage,
    }


# A four-hour night: 30m awake, 60m light, 45m deep, 60m REM, 45m light.
# Totals: AWAKE 30, LIGHT 105, DEEP 45, REM 60. Runs 04:00Z -> 08:00Z,
# which is 00:00 -> 04:00 local at UTC-4.
NIGHT = {
    "sleep": {
        "type": "STAGES",
        "metadata": {"mainSleep": True, "stagesStatus": "SUCCEEDED"},
        "interval": {
            "startTime": "2026-09-19T04:00:00Z",
            "startUtcOffset": OFFSET,
            "endTime": "2026-09-19T08:00:00Z",
            "endUtcOffset": OFFSET,
            "civilEndTime": {"date": {"year": 2026, "month": 9, "day": 19}},
        },
        "stages": [
            segment("04:00", "04:30", "AWAKE"),
            segment("04:30", "05:30", "LIGHT"),
            segment("05:30", "06:15", "DEEP"),
            segment("06:15", "07:15", "REM"),
            segment("07:15", "08:00", "LIGHT"),
        ],
        # Nested inside the stages above: 5 + 3 = 8 minutes.
        "shortAwakenings": [
            segment("05:00", "05:05", "AWAKE"),
            segment("06:40", "06:43", "AWAKE"),
        ],
        "summary": {
            "minutesAsleep": "210",
            "minutesInSleepPeriod": "240",
            "minutesAwake": "30",
            "minutesToFallAsleep": "0",
            "stagesSummary": [
                {"type": "AWAKE", "minutes": "30", "count": "1"},
                {"type": "LIGHT", "minutes": "105", "count": "2"},
                {"type": "DEEP", "minutes": "45", "count": "1"},
                {"type": "REM", "minutes": "60", "count": "1"},
            ],
        },
    }
}


class HypnogramTests(TestCase):
    def setUp(self):
        self.chart = metrics.hypnogram(NIGHT)

    def _row(self, label):
        return next(r for r in self.chart["rows"] if r["label"] == label)

    def test_five_bands_in_display_order(self):
        # Restlessness gets its own band: it is nested inside the stages, and a
        # separate row shows it without adding its minutes to the awake total.
        self.assertEqual(
            [row["label"] for row in self.chart["rows"]],
            ["Total awake time", "Restlessness", "REM", "Light", "Deep"],
        )

    def test_each_band_carries_its_total_duration(self):
        totals = {row["label"]: row["minutes"] for row in self.chart["rows"]}
        self.assertEqual(
            totals,
            {
                "Total awake time": 30,
                "Restlessness": 8,
                "REM": 60,
                "Light": 105,
                "Deep": 45,
            },
        )

    def test_durations_are_formatted_for_display(self):
        self.assertEqual(self._row("Light")["duration"], "1h 45m")
        self.assertEqual(self._row("Total awake time")["duration"], "30m")
        self.assertEqual(self._row("Restlessness")["duration"], "8m")

    def test_stage_totals_match_the_api_stages_summary(self):
        # Verified live: segment durations sum exactly to stagesSummary.
        reported = {
            entry["type"]: int(entry["minutes"])
            for entry in NIGHT["sleep"]["summary"]["stagesSummary"]
        }
        computed = {
            row["type"]: row["minutes"]
            for row in self.chart["rows"]
            if not row["nested"]
        }
        self.assertEqual(computed, reported)

    def test_restlessness_is_not_added_to_the_awake_band(self):
        # Its 8 minutes already sit inside other stages; counting them again
        # under Awake would inflate the night.
        self.assertEqual(self._row("Total awake time")["minutes"], 30)
        self.assertTrue(self._row("Restlessness")["nested"])
        self.assertFalse(self._row("Deep")["nested"])

    def test_segments_are_positioned_as_percentages_of_the_night(self):
        first = self._row("Total awake time")["segments"][0]
        # 04:00-04:30 of a 240-minute night: starts at 0%, spans 12.5%.
        self.assertEqual(first["left"], 0.0)
        self.assertEqual(first["width"], 12.5)

        last = self._row("Light")["segments"][-1]
        # 07:15-08:00 -> starts at 195/240, spans 45/240.
        self.assertEqual(last["left"], 81.25)
        self.assertEqual(last["width"], 18.75)

    def test_every_segment_carries_its_own_clock_range(self):
        deep = self._row("Deep")["segments"][0]
        self.assertEqual((deep["from"], deep["to"]), ("01:30", "02:15"))
        self.assertEqual(deep["minutes"], 45.0)

    def test_segments_are_ordered_across_the_night(self):
        for row in self.chart["rows"]:
            lefts = [seg["left"] for seg in row["segments"]]
            self.assertEqual(lefts, sorted(lefts), row["label"])

    def test_segment_counts_are_exposed_per_band(self):
        self.assertEqual(self._row("Light")["count"], 2)
        self.assertEqual(self._row("Restlessness")["count"], 2)
        self.assertEqual(self.chart["segment_count"], 7)

    def test_axis_labels_use_the_devices_local_offset(self):
        # 04:00Z at UTC-4 is midnight locally, not 04:00.
        self.assertEqual(self.chart["bedtime"], "00:00")
        self.assertEqual(self.chart["wake_time"], "04:00")
        self.assertEqual(self.chart["start_minute_of_day"], 0)

    def test_the_midpoint_label_sits_halfway_through_the_night(self):
        self.assertEqual(self.chart["midpoint"], "02:00")

    def test_a_midpoint_that_crosses_midnight_wraps(self):
        night = {
            "sleep": {
                "interval": {
                    "startTime": "2026-09-19T02:00:00Z",  # 22:00 local
                    "startUtcOffset": OFFSET,
                    "endTime": "2026-09-19T10:00:00Z",
                    "endUtcOffset": OFFSET,
                },
                "stages": [segment("02:00", "10:00", "LIGHT")],
            }
        }
        chart = metrics.hypnogram(night)
        self.assertEqual(chart["bedtime"], "22:00")
        self.assertEqual(chart["midpoint"], "02:00")
        self.assertEqual(chart["wake_time"], "06:00")

    def test_total_span_covers_the_whole_interval(self):
        self.assertEqual(self.chart["total_minutes"], 240)

    def test_a_night_without_stage_data_returns_none(self):
        classic = {
            "sleep": {
                "type": "classic",
                "interval": {"startTime": "2026-09-19T04:00:00Z"},
                "summary": {"minutesAsleep": "200"},
            }
        }
        self.assertIsNone(metrics.hypnogram(classic))

    def test_an_empty_point_returns_none(self):
        self.assertIsNone(metrics.hypnogram({}))
        self.assertIsNone(metrics.hypnogram(None))

    def test_a_zero_length_night_returns_none_rather_than_dividing_by_zero(self):
        night = {
            "sleep": {
                "interval": {
                    "startTime": "2026-09-19T04:00:00Z",
                    "endTime": "2026-09-19T04:00:00Z",
                },
                "stages": [segment("04:00", "04:00", "LIGHT")],
            }
        }
        self.assertIsNone(metrics.hypnogram(night))

    def test_zero_length_segments_are_dropped(self):
        night = {
            "sleep": {
                "interval": {
                    "startTime": "2026-09-19T04:00:00Z",
                    "startUtcOffset": OFFSET,
                    "endTime": "2026-09-19T05:00:00Z",
                    "endUtcOffset": OFFSET,
                },
                "stages": [
                    segment("04:00", "04:00", "LIGHT"),
                    segment("04:00", "05:00", "DEEP"),
                ],
            }
        }
        chart = metrics.hypnogram(night)
        self.assertEqual(chart["segment_count"], 1)
        deep = next(r for r in chart["rows"] if r["label"] == "Deep")
        self.assertEqual(deep["count"], 1)

    def test_a_night_with_no_short_awakenings_still_shows_the_band(self):
        sleep = dict(NIGHT["sleep"])
        sleep.pop("shortAwakenings")
        chart = metrics.hypnogram({"sleep": sleep})
        restless = next(r for r in chart["rows"] if r["label"] == "Restlessness")
        self.assertEqual(restless["minutes"], 0)
        self.assertEqual(restless["segments"], [])
        self.assertEqual(restless["duration"], "0m")


class SleepNightViewTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )
        self.client.force_login(self.user)

    def _get(self, day="2026-09-19", points=None):
        fake = mock.MagicMock()
        fake.list_data_points.return_value = points if points is not None else [NIGHT]
        fake.daily_roll_up.return_value = []
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            return self.client.get(f"/metric/sleep/night/{day}/")

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get("/metric/sleep/night/2026-09-19/")
        self.assertEqual(response.status_code, 302)

    def test_renders_all_five_bands_with_their_durations(self):
        response = self._get()
        self.assertEqual(response.status_code, 200)

        chart = response.context["hypnogram"]
        self.assertEqual(
            [row["label"] for row in chart["rows"]],
            ["Total awake time", "Restlessness", "REM", "Light", "Deep"],
        )
        for label, duration in [
            ("Total awake time", "30m"),
            ("Restlessness", "8m"),
            ("REM", "1h 00m"),
            ("Light", "1h 45m"),
            ("Deep", "45m"),
        ]:
            self.assertContains(response, label)
            self.assertContains(response, duration)

    def test_bands_are_positioned_elements_not_a_canvas(self):
        response = self._get()
        self.assertContains(response, 'class="hypno-track"')
        self.assertContains(response, "hypno-seg seg-deep")
        self.assertContains(response, "left:0.0%;width:12.5%")
        # No charting library is needed on this page at all.
        self.assertNotContains(response, "<canvas")
        self.assertNotContains(response, "chart.umd.min.js")

    def test_every_band_is_labelled_for_a_screen_reader(self):
        response = self._get()
        self.assertContains(response, 'aria-label="Deep: 45m across 1 segment"')
        self.assertContains(response, 'aria-label="Light: 1h 45m across 2 segments"')

    def test_bedtime_midpoint_and_wake_time_appear_on_the_axis(self):
        response = self._get()
        self.assertContains(response, "00:00")
        self.assertContains(response, "02:00")  # midpoint
        self.assertContains(response, "04:00")
        self.assertContains(response, "Bedtime")
        self.assertContains(response, "Wake time")

    def test_restlessness_is_marked_nested_rather_than_a_share_of_the_night(self):
        response = self._get()
        self.assertContains(response, "nested")
        self.assertContains(response, "not added to the awake")

    def test_a_night_with_no_session_says_so(self):
        response = self._get(points=[])
        self.assertIsNone(response.context["hypnogram"])
        self.assertContains(response, "No sleep session recorded")

    def test_a_classic_night_explains_why_there_is_nothing_to_plot(self):
        classic = {
            "sleep": {
                "type": "classic",
                "metadata": {"mainSleep": True},
                "interval": {
                    "startTime": "2026-09-19T04:00:00Z",
                    "endTime": "2026-09-19T08:00:00Z",
                    "civilEndTime": {"date": {"year": 2026, "month": 9, "day": 19}},
                },
                "summary": {"minutesAsleep": "200", "minutesInSleepPeriod": "240"},
            }
        }
        response = self._get(points=[classic])
        self.assertTrue(response.context["no_stages"])
        self.assertContains(response, "no stage breakdown")

    def test_naps_are_not_charted_as_the_night(self):
        nap = {
            "sleep": {
                "type": "STAGES",
                "metadata": {"nap": True},
                "interval": {
                    "startTime": "2026-09-19T18:00:00Z",
                    "startUtcOffset": OFFSET,
                    "endTime": "2026-09-19T19:00:00Z",
                    "endUtcOffset": OFFSET,
                    "civilEndTime": {"date": {"year": 2026, "month": 9, "day": 19}},
                },
                "stages": [segment("18:00", "19:00", "LIGHT")],
                "summary": {"minutesAsleep": "60", "stagesSummary": []},
            }
        }
        response = self._get(points=[nap, NIGHT])
        # The real night must win, not the nap.
        self.assertEqual(response.context["hypnogram"]["total_minutes"], 240)

    def test_a_malformed_date_is_404(self):
        self.assertEqual(
            self.client.get("/metric/sleep/night/19-09-2026/").status_code, 404
        )

    def test_the_sleep_table_links_into_the_night_view(self):
        fake = mock.MagicMock()
        fake.list_data_points.return_value = [NIGHT]
        fake.daily_roll_up.return_value = []
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/metric/sleep/?days=30")
        self.assertContains(response, "/metric/sleep/night/2026-09-19/")

    def test_the_route_is_not_shadowed_by_the_generic_day_view(self):
        response = self._get()
        self.assertTemplateUsed(response, "health/sleep_night.html")
