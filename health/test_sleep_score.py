"""Tests for the legacy sleep score import and its dashboard card.

Sleep score is the one metric in the Takeout export with no live equivalent at
all - the Google Health API returns sleep stages and minute counts but no score
of any kind - so it is the reason the LEGACY presentation group exists.

The fixture mirrors a real `sleep_score.csv`, including blank sub-score columns,
a row with no `overall_score`, and one naive (non-UTC) timestamp.
"""

from datetime import date
from io import StringIO
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from health import metrics, takeout
from health.models import HistoricalMetric

User = get_user_model()

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SLEEP_SCORE = FIXTURES / "sleep_score.csv"


class SleepScoreParsingTests(TestCase):
    def test_reads_every_scored_row(self):
        readings = takeout.parse_sleep_score(SLEEP_SCORE)
        # The fourth row has a blank overall_score and must be skipped.
        self.assertEqual(len(readings), 3)
        self.assertEqual([r["value"] for r in readings], [78.0, 83.0, 71.0])

    def test_keeps_the_sub_scores_and_sleep_metrics(self):
        extra = takeout.parse_sleep_score(SLEEP_SCORE)[0]["extra"]
        self.assertEqual(extra["revitalization_score"], 79)
        self.assertEqual(extra["deep_sleep_in_minutes"], 109)
        self.assertEqual(extra["resting_heart_rate"], 53)
        self.assertAlmostEqual(extra["restlessness"], 0.15758896151053015)
        self.assertEqual(extra["sleep_log_entry_id"], "53548224804")

    def test_blank_sub_scores_become_none_not_zero(self):
        # Real exports leave composition_score and duration_score empty.
        extra = takeout.parse_sleep_score(SLEEP_SCORE)[0]["extra"]
        self.assertIsNone(extra["composition_score"])
        self.assertIsNone(extra["duration_score"])

    def test_a_naive_timestamp_is_taken_as_local_wall_clock(self):
        readings = takeout.parse_sleep_score(SLEEP_SCORE)
        naive_row = [r for r in readings if r["value"] == 71.0][0]
        self.assertEqual(naive_row["date"], date(2026, 9, 9))

    def test_a_file_without_the_score_column_is_reported(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as handle:
            handle.write("timestamp,something_else\n2026-09-11T11:56:30Z,5\n")
            path = Path(handle.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        with self.assertRaises(takeout.TakeoutParseError) as ctx:
            takeout.parse_sleep_score(path)
        self.assertIn("overall_score", str(ctx.exception))


class LocalDateConversionTests(TestCase):
    """A UTC stamp must be resolved in the configured zone, not taken as-is."""

    @override_settings(TIME_ZONE="America/New_York")
    def test_utc_stamp_before_local_midnight_belongs_to_the_previous_day(self):
        # 02:30Z on the 9th is 22:30 on the 8th in New York.
        self.assertEqual(
            takeout.parse_local_date("2026-09-09T02:30:00Z"), date(2026, 9, 8)
        )

    @override_settings(TIME_ZONE="UTC")
    def test_the_same_stamp_under_utc_stays_on_the_ninth(self):
        # Proves the assertion above is actually sensitive to the setting.
        self.assertEqual(
            takeout.parse_local_date("2026-09-09T02:30:00Z"), date(2026, 9, 9)
        )

    @override_settings(TIME_ZONE="America/New_York")
    def test_a_normal_morning_wake_time_keeps_its_own_date(self):
        self.assertEqual(
            takeout.parse_local_date("2026-09-11T11:56:30Z"), date(2026, 9, 11)
        )


class CollectTests(TestCase):
    def test_collects_both_metrics_from_one_folder(self):
        collected = takeout.collect(FIXTURES)
        self.assertEqual(set(collected), {"vo2-max", "sleep-score"})

    def test_only_narrows_to_a_single_metric(self):
        collected = takeout.collect(FIXTURES, only="sleep-score")
        self.assertEqual(set(collected), {"sleep-score"})

    def test_readings_come_back_oldest_first(self):
        readings = takeout.collect(FIXTURES, only="sleep-score")["sleep-score"]
        dates = [r["date"] for r in readings]
        self.assertEqual(dates, sorted(dates))

    def test_a_single_file_path_only_matches_its_own_metric(self):
        collected = takeout.collect(SLEEP_SCORE)
        self.assertEqual(set(collected), {"sleep-score"})

    def test_registry_only_holds_metrics_with_no_live_equivalent(self):
        # Active zone minutes, temperature, HRV and SpO2 are all served live and
        # must not be imported from Takeout.
        self.assertEqual(set(takeout.IMPORTS), {"vo2-max", "sleep-score"})
        for spec in takeout.IMPORTS.values():
            self.assertTrue(spec.why)


class ImportTakeoutCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")

    def run_import(self, *args, **kwargs):
        out = StringIO()
        call_command("import_takeout", *args, stdout=out, **kwargs)
        return out.getvalue()

    def test_imports_every_supported_metric(self):
        output = self.run_import(str(FIXTURES))

        self.assertEqual(
            HistoricalMetric.objects.filter(metric_type="sleep-score").count(), 3
        )
        self.assertEqual(
            HistoricalMetric.objects.filter(metric_type="vo2-max").count(), 3
        )
        self.assertIn("Sleep score", output)
        self.assertIn("VO2 max", output)

    def test_sleep_score_rows_carry_unit_and_source(self):
        self.run_import(str(FIXTURES), only="sleep-score")
        row = HistoricalMetric.objects.filter(metric_type="sleep-score").first()
        self.assertEqual(row.unit, "/100")
        self.assertEqual(row.source, "Fitbit sleep score")
        self.assertEqual(row.date, date(2026, 9, 11))
        self.assertEqual(row.value, 78.0)

    def test_only_limits_what_is_written(self):
        self.run_import(str(FIXTURES), only="sleep-score")
        self.assertEqual(HistoricalMetric.objects.filter(metric_type="vo2-max").count(), 0)

    def test_unknown_only_value_is_rejected(self):
        with self.assertRaises(CommandError) as ctx:
            self.run_import(str(FIXTURES), only="stress-score")
        self.assertIn("Unknown metric", str(ctx.exception))

    def test_dry_run_writes_nothing(self):
        output = self.run_import(str(FIXTURES), dry_run=True)
        self.assertEqual(HistoricalMetric.objects.count(), 0)
        self.assertIn("Dry run", output)

    def test_re_running_updates_rather_than_duplicating(self):
        self.run_import(str(FIXTURES))
        before = HistoricalMetric.objects.count()
        output = self.run_import(str(FIXTURES))
        self.assertEqual(HistoricalMetric.objects.count(), before)
        self.assertIn("updated    6", output)

    def test_empty_folder_explains_where_to_look(self):
        import tempfile

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(CommandError) as ctx:
                self.run_import(empty)
        message = str(ctx.exception)
        self.assertIn("Sleep Score", message)
        self.assertIn("Global Export Data", message)

    def test_vo2max_alias_still_works(self):
        out = StringIO()
        call_command("import_takeout_vo2max", str(FIXTURES), stdout=out)
        self.assertEqual(HistoricalMetric.objects.filter(metric_type="vo2-max").count(), 3)
        self.assertEqual(
            HistoricalMetric.objects.filter(metric_type="sleep-score").count(), 0
        )
