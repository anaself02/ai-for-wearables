"""Tests for the per-night sleep-stage timeline.

The fixture mirrors the shape confirmed against a live response on 2026-09-19:
`stages` tiles the night with no gaps or overlaps, each segment carries its own
UTC offset, and `shortAwakenings` are nested inside the stage segments rather
than sitting alongside them.
"""

from datetime import date
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
        "shortAwakenings": [
            segment("05:00", "05:05", "AWAKE"),
            segment("06:40", "06:43", "AWAKE"),
        ],
        "summary": {
            "minutesAsleep": "210",
            "minutesInSleepPeriod": "240",
            "minutesAwake": "30",
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

    def test_all_four_stage_bands_are_present_in_display_order(self):
        self.assertEqual(
            [row["label"] for row in self.chart["rows"]],
            ["Awake", "REM", "Light", "Deep"],
        )

    def test_each_band_carries_its_total_duration(self):
        totals = {row["label"]: row["minutes"] for row in self.chart["rows"]}
        self.assertEqual(totals, {"Awake": 30, "REM": 60, "Light": 105, "Deep": 45})

    def test_durations_are_formatted_for_display(self):
        durations = {row["label"]: row["duration"] for row in self.chart["rows"]}
        self.assertEqual(durations["Light"], "1h 45m")
        self.assertEqual(durations["Awake"], "30m")

    def test_computed_totals_match_the_api_stages_summary(self):
        # Verified live: segment durations sum exactly to stagesSummary.
        reported = {
            entry["type"]: int(entry["minutes"])
            for entry in NIGHT["sleep"]["summary"]["stagesSummary"]
        }
        computed = {row["type"]: row["minutes"] for row in self.chart["rows"]}
        self.assertEqual(computed, reported)

    def test_segments_are_positioned_in_minutes_from_bedtime(self):
        first = self.chart["segments"][0]
        self.assertEqual((first["start"], first["end"]), (0, 30))
        self.assertEqual(first["stage"], "Awake")
        last = self.chart["segments"][-1]
        self.assertEqual((last["start"], last["end"]), (195, 240))

    def test_each_segment_targets_its_own_stage_row(self):
        rows_by_stage = {
            row["label"]: index for index, row in enumerate(self.chart["rows"])
        }
        for seg in self.chart["segments"]:
            self.assertEqual(seg["row"], rows_by_stage[seg["stage"]])

    def test_segments_are_ordered_across_the_night(self):
        starts = [seg["start"] for seg in self.chart["segments"]]
        self.assertEqual(starts, sorted(starts))

    def test_bedtime_and_wake_labels_use_the_devices_local_offset(self):
        # 04:00Z at UTC-4 is midnight locally, not 04:00.
        self.assertEqual(self.chart["bedtime"], "00:00")
        self.assertEqual(self.chart["wake_time"], "04:00")
        self.assertEqual(self.chart["start_minute_of_day"], 0)

    def test_total_span_covers_the_whole_interval(self):
        self.assertEqual(self.chart["total_minutes"], 240)

    def test_brief_awakenings_are_reported_but_not_added_to_the_awake_band(self):
        # They are nested inside the stage segments; counting them in Awake
        # would double-count time that is already attributed to a stage.
        self.assertEqual(self.chart["brief_awakenings"], 2)
        self.assertEqual(self.chart["brief_awakenings_minutes"], 8)
        awake = next(r for r in self.chart["rows"] if r["type"] == "AWAKE")
        self.assertEqual(awake["minutes"], 30)

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
        self.assertEqual(len(chart["segments"]), 1)
        self.assertEqual(chart["segments"][0]["stage"], "Deep")


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

    def test_renders_all_four_bands_with_their_durations(self):
        response = self._get()
        self.assertEqual(response.status_code, 200)

        chart = response.context["hypnogram"]
        self.assertEqual(
            [row["label"] for row in chart["rows"]], ["Awake", "REM", "Light", "Deep"]
        )
        for label, duration in [
            ("Awake", "30m"),
            ("REM", "1h 00m"),
            ("Light", "1h 45m"),
            ("Deep", "45m"),
        ]:
            self.assertContains(response, label)
            self.assertContains(response, duration)

    def test_bedtime_and_wake_time_appear_on_the_page(self):
        response = self._get()
        self.assertContains(response, "00:00")
        self.assertContains(response, "04:00")

    def test_the_chart_payload_is_embedded_for_the_canvas(self):
        response = self._get()
        self.assertContains(response, 'id="hypnogram-data"')
        self.assertContains(response, 'data-hypnogram="hypnogram-data"')

    def test_brief_awakenings_are_explained_rather_than_charted(self):
        response = self._get()
        self.assertContains(response, "2 brief awakenings")
        self.assertContains(response, "not part of the Awake total")

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
        self.assertEqual(self.client.get("/metric/sleep/night/19-09-2026/").status_code, 404)

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
