from datetime import date

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase

from . import datatypes
from . import views as views_module
from .client import _civil

User = get_user_model()


class DataTypeRegistryTests(TestCase):
    """The registry is the single place data type names are spelled out.

    Every expectation here comes from the live API discovery document
    (revision 20260916), so a wrong name fails a test rather than a request.
    """

    def test_interval_types_filter_on_civil_start_time(self):
        steps = datatypes.get("steps")
        self.assertEqual(
            steps.time_filter(date(2026, 9, 1), date(2026, 9, 8)),
            'steps.interval.civil_start_time >= "2026-09-01" '
            'AND steps.interval.civil_start_time < "2026-09-08"',
        )

    def test_daily_types_filter_on_date(self):
        rhr = datatypes.get("daily-resting-heart-rate")
        self.assertEqual(rhr.time_filter_field, "daily_resting_heart_rate.date")

    def test_sample_types_filter_on_sample_time(self):
        hrv = datatypes.get("heart-rate-variability")
        self.assertEqual(
            hrv.time_filter_field, "heart_rate_variability.sample_time.civil_time"
        )

    def test_sleep_is_filtered_on_its_end_time(self):
        # The API does not support filtering sleep sessions by start time.
        self.assertEqual(
            datatypes.get("sleep").time_filter_field, "sleep.interval.civil_end_time"
        )

    def test_exercise_and_sleep_page_size_is_capped_at_25(self):
        self.assertEqual(datatypes.get("exercise").max_page_size, 25)
        self.assertEqual(datatypes.get("sleep").max_page_size, 25)

    def test_heart_rate_rollup_window_is_capped_at_14_days(self):
        self.assertEqual(datatypes.get("heart-rate").rollup_max_days, 14)
        self.assertEqual(datatypes.get("steps").rollup_max_days, 90)

    def test_slugs_are_kebab_case_and_filter_fields_snake_case(self):
        for dt in datatypes.all_types():
            self.assertNotIn("_", dt.slug, f"{dt.slug} should be kebab-case")
            self.assertNotIn("-", dt.filter_field, f"{dt.filter_field} should be snake_case")
            self.assertEqual(dt.filter_field, dt.slug.replace("-", "_"))

    def test_every_type_uses_a_scope_the_app_requests(self):
        for dt in datatypes.all_types():
            self.assertIn(dt.scope, settings.GOOGLE_OAUTH_SCOPES, dt.slug)

    def test_requested_metrics_are_all_registered(self):
        for slug in [
            "steps",
            "heart-rate",
            "daily-resting-heart-rate",
            "heart-rate-variability",
            "daily-heart-rate-variability",
            "oxygen-saturation",
            "daily-oxygen-saturation",
            "daily-respiratory-rate",
            "exercise",
            "sleep",
            "daily-vo2-max",
            "run-vo2-max",
        ]:
            self.assertIsNotNone(datatypes.get(slug), slug)


class RollUpRequestTests(TestCase):
    def test_civil_datetime_wraps_the_date_in_a_date_object(self):
        # range.start is a CivilDateTime, not a bare Date.
        self.assertEqual(
            _civil(date(2026, 9, 19)),
            {"date": {"year": 2026, "month": 9, "day": 19}},
        )


class OutboundNetworkTests(TestCase):
    """Cover for the ~169s-per-request stall caused by dead IPv6 routes."""

    def test_force_ipv4_restricts_urllib3_to_af_inet(self):
        import socket

        import urllib3.util.connection as u3

        from config.net import force_ipv4

        original = u3.HAS_IPV6
        try:
            force_ipv4()
            self.assertFalse(u3.HAS_IPV6)
            self.assertEqual(u3.allowed_gai_family(), socket.AF_INET)
        finally:
            u3.HAS_IPV6 = original

    def test_connect_and_read_timeouts_are_separate(self):
        from .client import REQUEST_TIMEOUT

        # A scalar timeout is spent per candidate address on connect, so the
        # connect budget must be kept small and distinct from the read budget.
        connect, read = REQUEST_TIMEOUT
        self.assertLess(connect, read)
        self.assertLessEqual(connect, 10)


class RawViewTruncationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")
        self.client.force_login(self.user)

    def _patch_fetch(self, point_count):
        from unittest import mock

        from . import views

        points = [{"steps": {"count": str(i)}} for i in range(point_count)]
        return mock.patch.object(
            views.GoogleHealthClient, "list_data_points", return_value=points
        ), points

    def test_large_responses_are_truncated_for_display(self):
        from unittest import mock

        from accounts.models import GoogleCredential

        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-1", refresh_token="1//refresh"
        )
        patcher, _ = self._patch_fetch(500)
        with patcher, mock.patch.object(views_module.GoogleHealthClient, "__init__", return_value=None):
            response = self.client.get("/raw/steps/?days=7")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["count"], 500)
        self.assertEqual(response.context["shown"], views_module.MAX_RENDERED_POINTS)
        self.assertTrue(response.context["truncated"])
        self.assertContains(response, "Showing the first")

    def test_small_responses_are_not_truncated(self):
        from unittest import mock

        from accounts.models import GoogleCredential

        GoogleCredential.objects.create(
            user=self.user, google_sub="sub-2", refresh_token="1//refresh"
        )
        patcher, _ = self._patch_fetch(3)
        with patcher, mock.patch.object(views_module.GoogleHealthClient, "__init__", return_value=None):
            response = self.client.get("/raw/steps/?days=7")

        self.assertEqual(response.context["count"], 3)
        self.assertFalse(response.context["truncated"])
        self.assertNotContains(response, "Showing the first")


class AccessControlTests(TestCase):
    def test_dashboard_requires_login(self):
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/?next="))

    def test_raw_view_requires_login(self):
        response = self.client.get("/raw/exercise/")
        self.assertEqual(response.status_code, 302)

    def test_unknown_data_type_is_404(self):
        user = User.objects.create(username="tester@example.com")
        self.client.force_login(user)
        self.assertEqual(self.client.get("/raw/not-a-real-type/").status_code, 404)

    def test_signed_in_user_without_credentials_is_sent_back_to_landing(self):
        user = User.objects.create(username="tester@example.com")
        self.client.force_login(user)
        response = self.client.get("/raw/exercise/")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/")
