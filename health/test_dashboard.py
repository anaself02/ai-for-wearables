"""Dashboard tests: metric routing, extractor correctness, empty states.

The API payloads below are verbatim slices of real responses captured on
2026-09-19, so they pin the exact shapes the extractors were written against.
"""

from datetime import date, datetime, timedelta, timezone as dt_timezone
from unittest import mock

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import RequestFactory, TestCase, override_settings
from django.utils import timezone as django_timezone

from accounts.models import GoogleCredential
from health import metrics, views

User = get_user_model()


def _civil(day):
    return {"date": {"year": 2026, "month": 9, "day": day}, "time": {}}


# -- real response fragments ------------------------------------------------

STEPS_ROLLUP = [
    {"civilStartTime": _civil(19), "civilEndTime": _civil(20), "steps": {"countSum": "1652"}},
    {"civilStartTime": _civil(18), "civilEndTime": _civil(19), "steps": {"countSum": "9004"}},
]

DISTANCE_ROLLUP = [
    {"civilStartTime": _civil(19), "distance": {"millimetersSum": "1233700"}},
]

ACTIVE_MINUTES_ROLLUP = [
    {
        "civilStartTime": _civil(19),
        "activeMinutes": {
            "activeMinutesRollupByActivityLevel": [
                {"activityLevel": "LIGHT", "activeMinutesSum": "52"},
                {"activityLevel": "MODERATE", "activeMinutesSum": "8"},
            ]
        },
    }
]

HEART_RATE_ROLLUP = [
    {
        "civilStartTime": _civil(19),
        "heartRate": {
            "beatsPerMinuteAvg": 57.93969891042169,
            "beatsPerMinuteMax": 103,
            "beatsPerMinuteMin": 43,
        },
    }
]

RESTING_HR_LIST = [
    {
        "dailyRestingHeartRate": {
            "date": {"year": 2026, "month": 9, "day": 19},
            "beatsPerMinute": "54",
        }
    }
]

SPO2_LIST = [
    {
        "dailyOxygenSaturation": {
            "date": {"year": 2026, "month": 9, "day": 19},
            "averagePercentage": 97.6,
            "lowerBoundPercentage": 94.9,
            "upperBoundPercentage": 99.7,
        }
    }
]

SLEEP_LIST = [
    {
        "sleep": {
            "interval": {
                "startTime": "2026-09-19T04:59:00Z",
                "endTime": "2026-09-19T14:29:00Z",
                "civilEndTime": {"date": {"year": 2026, "month": 9, "day": 19}},
            },
            "type": "STAGES",
            "summary": {
                "minutesInSleepPeriod": "570",
                "minutesAsleep": "526",
                "minutesAwake": "44",
                "minutesToFallAsleep": "0",
                "stagesSummary": [
                    {"type": "AWAKE", "minutes": "44", "count": "3"},
                    {"type": "LIGHT", "minutes": "309", "count": "16"},
                    {"type": "DEEP", "minutes": "104", "count": "8"},
                    {"type": "REM", "minutes": "113", "count": "7"},
                ],
            },
        }
    }
]

EXERCISE_LIST = [
    {
        "exercise": {
            "interval": {
                "startTime": "2026-09-19T02:37:41Z",
                "endTime": "2026-09-19T03:21:16Z",
            },
            "exerciseType": "WALKING",
            "displayName": "Marche",
            "activeDuration": "2611.095s",
            "metricsSummary": {
                "caloriesKcal": 335,
                "distanceMillimeters": 3143477,
                "steps": "3919",
                "averageHeartRateBeatsPerMinute": "95",
            },
            "exerciseMetadata": {"hasGps": True},
        }
    }
]

HEART_RATE_SAMPLES = [
    {
        "heartRate": {
            "sampleTime": {
                "physicalTime": "2026-09-19T20:13:00Z",
                "civilTime": {
                    "date": {"year": 2026, "month": 9, "day": 19},
                    "time": {"hours": 16, "minutes": 13},
                },
            },
            "beatsPerMinute": "76",
        }
    }
]


class ExtractorTests(TestCase):
    def test_steps_rollup_reads_count_sum_and_sorts_ascending(self):
        series = metrics.steps_series(STEPS_ROLLUP)
        self.assertEqual(series["labels"], ["2026-09-18", "2026-09-19"])
        self.assertEqual(series["datasets"][0]["data"], [9004, 1652])

    def test_distance_rollup_converts_millimetres_to_kilometres(self):
        series = metrics.distance_series(DISTANCE_ROLLUP)
        self.assertEqual(series["datasets"][0]["data"], [1.23])

    def test_active_minutes_sums_every_activity_level(self):
        # The rollup returns one entry per level rather than a scalar.
        series = metrics.active_minutes_series(ACTIVE_MINUTES_ROLLUP)
        self.assertEqual(series["datasets"][0]["data"], [60])

    def test_heart_rate_rollup_yields_average_min_and_max(self):
        series = metrics.heart_rate_series(HEART_RATE_ROLLUP)
        labels = [dataset["label"] for dataset in series["datasets"]]
        self.assertEqual(labels, ["Average", "Min", "Max"])
        self.assertEqual(series["datasets"][0]["data"], [57.9])
        self.assertEqual(series["datasets"][1]["data"], [43.0])
        self.assertEqual(series["datasets"][2]["data"], [103.0])

    def test_daily_list_metrics_read_their_own_date_field(self):
        # list points carry the date inside the payload, not at civilStartTime.
        series = metrics.get("daily-resting-heart-rate").series(RESTING_HR_LIST)
        self.assertEqual(series["labels"], ["2026-09-19"])
        self.assertEqual(series["datasets"][0]["data"], [54])

    def test_spo2_uses_the_average_percentage(self):
        series = metrics.get("daily-oxygen-saturation").series(SPO2_LIST)
        self.assertEqual(series["datasets"][0]["data"], [97.6])

    def test_string_and_numeric_fields_both_coerce(self):
        self.assertEqual(metrics.as_int("1652"), 1652)
        self.assertEqual(metrics.as_float(57.93), 57.93)
        self.assertIsNone(metrics.as_int(None))
        self.assertIsNone(metrics.as_float(""))

    def test_sleep_series_is_one_stacked_bar_per_stage(self):
        series = metrics.sleep_series(SLEEP_LIST)
        self.assertEqual(series["labels"], ["2026-09-19"])
        self.assertEqual(
            [dataset["label"] for dataset in series["datasets"]],
            ["Deep", "Rem", "Light", "Awake"],
        )
        self.assertEqual(series["datasets"][0]["data"], [104])

    def test_sleep_efficiency_is_computed_because_the_api_has_no_score(self):
        row = metrics.sleep_rows(SLEEP_LIST)[0]
        self.assertEqual(row["asleep"], 526)
        self.assertEqual(row["asleep_hm"], "8h 46m")
        self.assertEqual(row["efficiency"], round(526 / 570 * 100))

    def test_exercise_rows_expose_table_columns(self):
        row = metrics.exercise_rows(EXERCISE_LIST)[0]
        self.assertEqual(row["type"], "WALKING")
        self.assertEqual(row["calories"], 335)
        self.assertEqual(row["km"], 3.14)
        self.assertEqual(row["avg_hr"], 95)
        self.assertEqual(row["minutes"], 43)

    def test_intraday_extractor_labels_by_local_clock_time(self):
        series = metrics.INTRADAY_EXTRACTORS["heart-rate"](HEART_RATE_SAMPLES)
        self.assertEqual(series["labels"], ["16:13"])
        self.assertEqual(series["datasets"][0]["data"], [76])

    def test_latest_value_skips_trailing_gaps(self):
        series = metrics.chart_data(
            ["2026-09-17", "2026-09-18", "2026-09-19"],
            [{"label": "Steps", "data": [10, 20, None]}],
        )
        self.assertEqual(metrics.latest_value(series), (20, "2026-09-18"))


class MetricRoutingTests(TestCase):
    """Each metric must use the endpoint the live API actually accepts."""

    def test_rollup_metrics_are_exactly_the_types_the_api_accepts(self):
        rollup = {m.slug for m in metrics.METRICS if m.source == metrics.SOURCE_ROLLUP}
        self.assertEqual(rollup, {"steps", "distance", "active-minutes", "heart-rate"})

    def test_daily_types_that_reject_rollup_use_list(self):
        # Verified live: these four return HTTP 400 "DailyRollup is not supported".
        for slug in [
            "daily-resting-heart-rate",
            "daily-heart-rate-variability",
            "daily-oxygen-saturation",
            "daily-respiratory-rate",
        ]:
            metric = metrics.get(slug)
            self.assertEqual(metric.source, metrics.SOURCE_LIST, slug)
            self.assertFalse(metric.data_type.supports_daily_rollup, slug)

    def test_every_rollup_metric_is_flagged_in_the_datatype_registry(self):
        for metric in metrics.METRICS:
            if metric.source == metrics.SOURCE_ROLLUP:
                self.assertTrue(metric.data_type.supports_daily_rollup, metric.slug)

    def test_high_volume_types_have_an_intraday_drill_down(self):
        # These return thousands of raw points over a month.
        for slug in ["heart-rate", "daily-oxygen-saturation", "daily-heart-rate-variability"]:
            self.assertIsNotNone(metrics.get(slug).intraday_slug, slug)

    def test_every_intraday_slug_has_an_extractor_and_a_datatype(self):
        for metric in metrics.METRICS:
            if metric.intraday_slug:
                self.assertIn(metric.intraday_slug, metrics.INTRADAY_EXTRACTORS, metric.slug)
                self.assertIsNotNone(metric.intraday_type, metric.slug)

    def test_vo2max_metrics_are_unavailable_and_need_no_api_call(self):
        slugs = {m.slug for m in metrics.METRICS if m.group == metrics.UNAVAILABLE}
        self.assertEqual(slugs, {"daily-vo2-max", "run-vo2-max", "vo2-max"})
        self.assertNotIn("vo2-max", {m.slug for m in metrics.fetchable()})
        for slug in slugs:
            self.assertIn("GPS", metrics.get(slug).empty_message)

    def test_rollup_ranges_are_chunked_to_the_api_limit(self):
        # heart-rate caps dailyRollUp at 14 days, so 90 days needs several calls.
        chunks = list(metrics._chunk_ranges(date(2026, 6, 1), date(2026, 8, 30), 14))
        self.assertEqual(chunks[0], (date(2026, 6, 1), date(2026, 6, 15)))
        self.assertEqual(chunks[-1][1], date(2026, 8, 30))
        self.assertTrue(all((end - start).days <= 14 for start, end in chunks))


class TimezoneWindowTests(TestCase):
    """The day window must resolve in the API's civil timezone, not the host's.

    The API stamps civil times with the device's own offset (observed
    `startUtcOffset: "-14400s"`, i.e. UTC-4). For the hours between local
    midnight and UTC midnight the two disagree by a full day, which would
    silently shift every chart.
    """

    #: 02:00 UTC on the 19th is still 22:00 on the 18th in New York.
    LATE_EVENING_UTC = datetime(2026, 9, 19, 2, 0, tzinfo=dt_timezone.utc)

    def _window_at(self, instant, days=7):
        request = RequestFactory().get(f"/?days={days}")
        with mock.patch("django.utils.timezone.now", return_value=instant):
            return views._window(request)

    def test_settings_pin_the_timezone_to_the_devices_zone(self):
        self.assertEqual(settings.TIME_ZONE, "America/New_York")
        self.assertTrue(settings.USE_TZ)

    @override_settings(TIME_ZONE="America/New_York")
    def test_window_follows_local_date_not_utc_date(self):
        days, start, end = self._window_at(self.LATE_EVENING_UTC)
        # Local date is the 18th, so the exclusive end is the 19th.
        # Using date.today()/UTC here would wrongly give the 20th.
        self.assertEqual(end, date(2026, 9, 19))
        self.assertEqual(start, date(2026, 9, 12))
        self.assertEqual(days, 7)

    @override_settings(TIME_ZONE="UTC")
    def test_same_instant_under_utc_gives_the_next_day(self):
        # Proves the assertion above is actually sensitive to the setting.
        _, _, end = self._window_at(self.LATE_EVENING_UTC)
        self.assertEqual(end, date(2026, 9, 20))

    @override_settings(TIME_ZONE="America/New_York")
    def test_window_is_half_open_so_today_is_included(self):
        days, start, end = self._window_at(
            datetime(2026, 9, 19, 16, 0, tzinfo=dt_timezone.utc), days=1
        )
        self.assertEqual(start, date(2026, 9, 19))
        self.assertEqual(end, date(2026, 9, 20))

    @override_settings(TIME_ZONE="America/New_York")
    def test_window_clamps_out_of_range_values(self):
        self.assertEqual(self._window_at(self.LATE_EVENING_UTC, days=9999)[0], 90)
        self.assertEqual(self._window_at(self.LATE_EVENING_UTC, days=0)[0], 1)

    def test_token_expiry_still_compares_in_utc(self):
        # accounts.models uses timezone.now(), which stays absolute regardless
        # of TIME_ZONE - changing the display zone must not expire tokens early.
        user = User.objects.create(username="tz@example.com")
        credential = GoogleCredential.objects.create(
            user=user,
            google_sub="tz-sub",
            refresh_token="1//refresh",
            expires_at=django_timezone.now() + timedelta(hours=1),
        )
        with override_settings(TIME_ZONE="Pacific/Kiritimati"):
            self.assertFalse(credential.is_expired)
        with override_settings(TIME_ZONE="Pacific/Niue"):
            self.assertFalse(credential.is_expired)


class DashboardViewTests(TestCase):
    def setUp(self):
        # The dashboard and the steps detail page now share a 30-day cache key,
        # so without this a previous test's fetch satisfies the next one.
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )
        self.client.force_login(self.user)

    def test_dashboard_requires_login(self):
        self.client.logout()
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/?next="))

    def test_metric_detail_requires_login(self):
        self.client.logout()
        self.assertEqual(self.client.get("/metric/steps/").status_code, 302)

    def test_unknown_metric_is_404(self):
        self.assertEqual(self.client.get("/metric/not-a-metric/").status_code, 404)

    def test_vo2max_detail_renders_empty_state_without_calling_the_api(self):
        with mock.patch(
            "health.views.GoogleHealthClient", side_effect=AssertionError("no API call")
        ):
            response = self.client.get("/metric/daily-vo2-max/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GPS-tracked outdoor run")
        self.assertTemplateUsed(response, "health/metric_unavailable.html")

    def _fake_client(self, points_by_slug):
        client = mock.MagicMock()
        client.list_data_points.side_effect = lambda dt, **kw: points_by_slug.get(dt.slug, [])
        client.daily_roll_up.side_effect = lambda dt, **kw: points_by_slug.get(dt.slug, [])
        return client

    def test_dashboard_renders_every_card(self):
        fake = self._fake_client(
            {
                "steps": STEPS_ROLLUP,
                "sleep": SLEEP_LIST,
                "exercise": EXERCISE_LIST,
                "heart-rate": HEART_RATE_ROLLUP,
                "daily-resting-heart-rate": RESTING_HR_LIST,
                "daily-oxygen-saturation": SPO2_LIST,
            }
        )
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["cards"]), len(metrics.METRICS))
        self.assertContains(response, "Steps")
        self.assertContains(response, "1652")
        # VO2 max empty state, rendered rather than errored.
        self.assertContains(response, "GPS-tracked outdoor run")

    def test_the_overview_greets_by_name(self):
        """Same greeting as the welcome screen, from the same two functions."""
        from accounts.greeting import display_name, greeting_for
        from django.utils import timezone

        self.user.first_name = "ANAS"
        self.user.save(update_fields=["first_name"])

        fake = self._fake_client({"steps": STEPS_ROLLUP})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/dashboard/")

        expected = greeting_for(timezone.localtime())
        self.assertEqual(response.context["greeting"], expected)
        self.assertContains(response, f"{expected}, Anas")
        # Calmed down here exactly as it is on the welcome screen.
        self.assertNotContains(response, "ANAS")

    def test_the_overview_greets_without_a_name_when_there_is_none(self):
        fake = self._fake_client({"steps": STEPS_ROLLUP})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/dashboard/")

        self.assertEqual(response.context["name"], "")
        self.assertNotContains(response, f"{response.context['greeting']},")

    def test_detail_chooses_rollup_for_steps_and_list_for_resting_hr(self):
        fake = self._fake_client({"steps": STEPS_ROLLUP})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            self.client.get("/metric/steps/?days=30")
        self.assertTrue(fake.daily_roll_up.called)
        self.assertFalse(fake.list_data_points.called)

        fake = self._fake_client({"daily-resting-heart-rate": RESTING_HR_LIST})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            self.client.get("/metric/daily-resting-heart-rate/?days=30")
        self.assertTrue(fake.list_data_points.called)
        self.assertFalse(fake.daily_roll_up.called)

    def test_detail_clamps_the_window_and_marks_the_active_chip(self):
        fake = self._fake_client({"steps": STEPS_ROLLUP})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/metric/steps/?days=9999")
        self.assertEqual(response.context["days"], 90)

    def test_day_view_fetches_the_intraday_type_for_one_day(self):
        fake = self._fake_client({"heart-rate": HEART_RATE_SAMPLES})
        with mock.patch("health.views.GoogleHealthClient", return_value=fake):
            response = self.client.get("/metric/heart-rate/day/2026-09-19/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["count"], 1)
        _, kwargs = fake.list_data_points.call_args
        self.assertEqual((kwargs["end"] - kwargs["start"]).days, 1)

    def test_day_view_rejects_a_metric_without_a_drill_down(self):
        self.assertEqual(
            self.client.get("/metric/daily-respiratory-rate/day/2026-09-19/").status_code,
            404,
        )

    def test_day_view_rejects_a_malformed_date(self):
        self.assertEqual(
            self.client.get("/metric/heart-rate/day/19-09-2026/").status_code, 404
        )
