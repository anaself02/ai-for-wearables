"""Tests for the one-time Takeout VO2 max import and the legacy card it feeds.

The fixture is a verbatim slice of a real export, including its unusual
`MM/DD/YY` dateTime format and the pair of raw/filtered values per record.
"""

import json
from datetime import date
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import GoogleCredential
from health import metrics
from health import takeout as importer
from health.models import HistoricalMetric

User = get_user_model()

# Named exactly as Takeout names it, so the directory-scan path is exercised too.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demographic_vo2_max-2026-07-12.json"


class DateParsingTests(TestCase):
    def test_takeout_dates_are_month_first(self):
        # 09/11/26 is 11 September, not 9 November. Read the other way round the
        # real export's daily sequence (07/15 -> 09/11) would be impossible.
        self.assertEqual(importer.parse_vo2max_date("09/11/26 00:00:00"), date(2026, 9, 11))
        self.assertEqual(importer.parse_vo2max_date("07/15/26 00:00:00"), date(2026, 7, 15))

    def test_a_day_that_cannot_be_a_month_still_parses(self):
        self.assertEqual(importer.parse_vo2max_date("12/31/26 00:00:00"), date(2026, 12, 31))


class ReadRecordsTests(TestCase):
    def test_reads_every_record_from_the_fixture(self):
        readings = importer.parse_vo2max(FIXTURE)
        self.assertEqual(len(readings), 3)
        self.assertEqual(readings[0]["date"], date(2026, 7, 15))
        self.assertEqual(readings[-1]["date"], date(2026, 9, 11))

    def test_stores_the_filtered_value_not_the_raw_estimate(self):
        # filteredDemographicVO2Max is the smoothed series Fitbit displayed.
        readings = importer.parse_vo2max(FIXTURE)
        self.assertEqual(readings[-1]["value"], 51.15)
        self.assertEqual(readings[-1]["extra"]["raw_estimate"], 51.113800000000005)

    def test_keeps_the_error_term_and_source_file(self):
        reading = importer.parse_vo2max(FIXTURE)[-1]
        self.assertEqual(reading["extra"]["error"], 0.36325)
        self.assertEqual(reading["extra"]["source_file"], FIXTURE.name)

    def test_records_without_a_value_are_skipped_not_fatal(self):
        path = Path(self.mktemp())
        path.write_text(
            json.dumps(
                [
                    {"dateTime": "07/15/26 00:00:00", "value": {}},
                    {"dateTime": "07/16/26 00:00:00", "value": {"demographicVO2Max": 50.0}},
                ]
            ),
            encoding="utf-8",
        )
        readings = importer.parse_vo2max(path)
        self.assertEqual(len(readings), 1)
        # Falls back to the raw estimate when no filtered value is present.
        self.assertEqual(readings[0]["value"], 50.0)

    def test_a_malformed_date_is_reported_rather_than_guessed(self):
        path = Path(self.mktemp())
        path.write_text(
            json.dumps([{"dateTime": "2026-07-15", "value": {"demographicVO2Max": 50.0}}]),
            encoding="utf-8",
        )
        with self.assertRaises(importer.TakeoutParseError) as ctx:
            importer.parse_vo2max(path)
        self.assertIn("cannot parse dateTime", str(ctx.exception))

    def mktemp(self):
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name


class ImportCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="tester@example.com")

    def run_import(self, *args, **kwargs):
        out = StringIO()
        call_command("import_takeout_vo2max", *args, stdout=out, **kwargs)
        return out.getvalue()

    def test_imports_the_fixture(self):
        output = self.run_import(str(FIXTURE))

        rows = HistoricalMetric.objects.order_by("date")
        self.assertEqual(rows.count(), 3)
        self.assertEqual(rows.last().value, 51.15)
        self.assertEqual(rows.last().date, date(2026, 9, 11))
        self.assertEqual(rows.last().unit, "ml/kg/min")
        self.assertEqual(rows.last().source, "Fitbit demographic estimate")
        self.assertEqual(rows.last().metric_type, HistoricalMetric.MetricType.VO2_MAX)
        self.assertIn("created    3", output)

    def test_running_twice_updates_rather_than_duplicating(self):
        self.run_import(str(FIXTURE))
        output = self.run_import(str(FIXTURE))
        self.assertEqual(HistoricalMetric.objects.count(), 3)
        self.assertIn("updated    3", output)

    def test_dry_run_writes_nothing(self):
        output = self.run_import(str(FIXTURE), dry_run=True)
        self.assertEqual(HistoricalMetric.objects.count(), 0)
        self.assertIn("Dry run", output)

    def test_accepts_a_directory_and_finds_the_file(self):
        self.run_import(str(FIXTURE.parent))
        self.assertEqual(HistoricalMetric.objects.count(), 3)

    def test_missing_path_is_a_clean_error(self):
        with self.assertRaises(CommandError):
            self.run_import(str(FIXTURE.parent / "nope"))

    def test_directory_without_the_export_explains_where_to_look(self):
        import tempfile

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(CommandError) as ctx:
                self.run_import(empty)
        self.assertIn("Global Export Data", str(ctx.exception))

    def test_ambiguous_user_requires_the_flag(self):
        User.objects.create(username="second@example.com")
        with self.assertRaises(CommandError) as ctx:
            self.run_import(str(FIXTURE))
        self.assertIn("--user", str(ctx.exception))

    def test_named_user_is_honoured(self):
        other = User.objects.create(username="second@example.com")
        self.run_import(str(FIXTURE), user="second@example.com")
        self.assertEqual(HistoricalMetric.objects.filter(user=other).count(), 3)
        self.assertEqual(HistoricalMetric.objects.filter(user=self.user).count(), 0)
