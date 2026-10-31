"""Tests for the API response cache.

The two properties that matter most are that cache keys are scoped per user -
these are personal health readings - and that a failed call is never cached.
"""

from datetime import date, timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings

from accounts.models import GoogleCredential
from health import cache as health_cache
from health import datatypes, metrics
from health.client import GoogleHealthError

User = get_user_model()

START = date(2026, 9, 12)
END = date(2026, 9, 19)

POINTS = [
    {"civilStartTime": {"date": {"year": 2026, "month": 9, "day": 18}}, "steps": {"countSum": "9004"}}
]


class CacheBehaviourTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        self.metric = metrics.get("steps")

    def _client(self, points=POINTS):
        client = mock.MagicMock()
        client.daily_roll_up.return_value = points
        client.list_data_points.return_value = points
        return client

    def test_second_call_is_served_from_cache(self):
        client = self._client()
        first = health_cache.fetch_points(
            client, self.metric, START, END, user=self.user
        )
        second = health_cache.fetch_points(
            client, self.metric, START, END, user=self.user
        )

        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(second.points, POINTS)
        # Only one trip to Google.
        self.assertEqual(client.daily_roll_up.call_count, 1)

    def test_refresh_forces_a_new_call(self):
        client = self._client()
        health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        again = health_cache.fetch_points(
            client, self.metric, START, END, user=self.user, force=True
        )

        self.assertFalse(again.from_cache)
        self.assertEqual(client.daily_roll_up.call_count, 2)

    def test_cache_is_scoped_per_user(self):
        other = User.objects.create(username="other@example.com")
        mine = self._client([{"steps": {"countSum": "1"}}])
        theirs = self._client([{"steps": {"countSum": "2"}}])

        health_cache.fetch_points(mine, self.metric, START, END, user=self.user)
        result = health_cache.fetch_points(theirs, self.metric, START, END, user=other)

        # A shared key here would leak one user's health data to another.
        self.assertFalse(result.from_cache)
        self.assertEqual(result.points, [{"steps": {"countSum": "2"}}])

    def test_different_windows_do_not_share_an_entry(self):
        client = self._client()
        health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        other_window = health_cache.fetch_points(
            client, self.metric, START - timedelta(days=30), END, user=self.user
        )
        self.assertFalse(other_window.from_cache)

    def test_different_metrics_do_not_share_an_entry(self):
        client = self._client()
        health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        distance = health_cache.fetch_points(
            client, metrics.get("distance"), START, END, user=self.user
        )
        self.assertFalse(distance.from_cache)

    def test_failures_are_not_cached(self):
        client = mock.MagicMock()
        client.daily_roll_up.side_effect = GoogleHealthError("boom", status_code=500)

        with self.assertRaises(GoogleHealthError):
            health_cache.fetch_points(client, self.metric, START, END, user=self.user)

        # The next request must retry rather than serve a remembered failure.
        client.daily_roll_up.side_effect = None
        client.daily_roll_up.return_value = POINTS
        result = health_cache.fetch_points(
            client, self.metric, START, END, user=self.user
        )
        self.assertFalse(result.from_cache)
        self.assertEqual(result.points, POINTS)

    @override_settings(HEALTH_CACHE_SECONDS=0)
    def test_caching_can_be_switched_off(self):
        client = self._client()
        health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        again = health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        self.assertFalse(again.from_cache)
        self.assertEqual(client.daily_roll_up.call_count, 2)

    def test_oldest_picks_the_least_fresh_fetch(self):
        client = self._client()
        first = health_cache.fetch_points(client, self.metric, START, END, user=self.user)
        second = health_cache.fetch_points(
            client, metrics.get("distance"), START, END, user=self.user
        )
        self.assertIs(health_cache.oldest([second, first, None]), first)
        self.assertIsNone(health_cache.oldest([None]))


class IntradayCacheTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        self.data_type = datatypes.get("heart-rate")

    def _client(self):
        client = mock.MagicMock()
        client.list_data_points.return_value = [{"heartRate": {"beatsPerMinute": "70"}}]
        return client

    def test_a_past_day_is_held_far_longer_than_today(self):
        client = self._client()
        yesterday = date.today() - timedelta(days=400)

        with mock.patch.object(health_cache, "_load", wraps=health_cache._load) as spy:
            health_cache.fetch_intraday(client, self.data_type, yesterday, user=self.user)
            past_ttl = spy.call_args[0][1]

            health_cache.fetch_intraday(
                client, self.data_type, date.today(), user=self.user
            )
            today_ttl = spy.call_args[0][1]

        # A finished day cannot gain new samples.
        self.assertGreater(past_ttl, today_ttl)

    def test_intraday_results_are_cached_per_day(self):
        client = self._client()
        day = date(2026, 9, 18)
        health_cache.fetch_intraday(client, self.data_type, day, user=self.user)
        same = health_cache.fetch_intraday(client, self.data_type, day, user=self.user)
        other = health_cache.fetch_intraday(
            client, self.data_type, day - timedelta(days=1), user=self.user
        )

        self.assertTrue(same.from_cache)
        self.assertFalse(other.from_cache)
        self.assertEqual(client.list_data_points.call_count, 2)
