from datetime import date

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase

from . import datatypes

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
