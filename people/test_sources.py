"""Adapter tests, on synthetic files only."""

from datetime import date, datetime, timezone

from django.test import SimpleTestCase

from people.sources import (SourceError, SourceFile, SourceNotImplemented,
                            get_adapter)
from people.sources.fitbit_takeout import FitbitTakeoutAdapter
from people.testing import utc_day_file

UTC = timezone.utc
ADAPTER = FitbitTakeoutAdapter()


def parse(name, data):
    return ADAPTER.parse_file(SourceFile(name, data))


class AcceptsTests(SimpleTestCase):
    def test_only_dated_heart_rate_csvs_are_accepted(self):
        for name in ("heart_rate_2026-07-16.csv", "heart_rate_1999-01-01.csv"):
            self.assertTrue(ADAPTER.accepts(name), name)
        for name in ("user_profile.csv", "heart_rate_readme.txt", "Devices.csv",
                     "Stress Score.csv", "heart_rate_2026-07-16.csv.bak",
                     "daily_resting_heart_rate.csv", "Heart Rate.csv",
                     "../heart_rate_2026-07-16.csv", "heart_rate_2026-7-6.csv"):
            self.assertFalse(ADAPTER.accepts(name), name)


class ParseTests(SimpleTestCase):
    def test_a_good_file_is_parsed_sorted_and_counted(self):
        name, data = utc_day_file(date(2026, 7, 16), seed=1)
        parsed = parse(name, data)
        self.assertEqual(parsed.report.status, "parsed")
        self.assertEqual(parsed.report.rows, 28800)
        self.assertTrue((parsed.epochs[1:] > parsed.epochs[:-1]).all())
        self.assertEqual(parsed.sources, {"Fitbit Air": 28800})
        self.assertEqual(parsed.report.first_utc, datetime(2026, 7, 16, tzinfo=UTC))

    def test_unsorted_and_duplicate_rows_are_fixed_and_reported(self):
        data = (b"timestamp,beats per minute,data source\n"
                b"2026-07-16T00:00:06Z,62.0,Band\n"
                b"2026-07-16T00:00:00Z,60.0,Band\n"
                b"2026-07-16T00:00:03Z,61.0,Band\n"
                b"2026-07-16T00:00:03Z,61.0,Band\n")
        parsed = parse("heart_rate_2026-07-16.csv", data)
        self.assertEqual(list(parsed.bpm), [60.0, 61.0, 62.0])
        self.assertEqual(parsed.report.dropped_duplicate, 1)

    def test_unreadable_rows_are_dropped_and_reported_not_hidden(self):
        data = (b"timestamp,beats per minute,data source\n"
                b"2026-07-16T00:00:00Z,60.0,Band\n"
                b"not-a-time,61.0,Band\n"
                b"2026-07-16T00:00:06Z,,Band\n"
                b"2026-07-16T00:00:09Z,63.0,Band\n")
        parsed = parse("heart_rate_2026-07-16.csv", data)
        self.assertEqual(parsed.report.rows, 2)
        self.assertEqual(parsed.report.dropped_unparseable, 2)

    def test_two_sources_in_one_file_are_both_named(self):
        data = (b"timestamp,beats per minute,data source\n"
                b"2026-07-16T00:00:00Z,60.0,Fitbit Air\n"
                b"2026-07-16T00:00:03Z,61.0,Fitbit\n")
        parsed = parse("heart_rate_2026-07-16.csv", data)
        self.assertEqual(parsed.sources, {"Fitbit Air": 1, "Fitbit": 1})


class BadFileTests(SimpleTestCase):
    def assertPlainError(self, data, fragment, name="heart_rate_2026-07-16.csv"):
        with self.assertRaises(SourceError) as ctx:
            parse(name, data)
        self.assertIn(fragment, str(ctx.exception))
        self.assertNotIn("Traceback", str(ctx.exception))

    def test_wrong_columns(self):
        self.assertPlainError(b"time,hr\n1,2\n", "must have the columns")

    def test_header_only(self):
        self.assertPlainError(b"timestamp,beats per minute,data source\n",
                              "no readings")

    def test_timestamps_without_a_timezone_are_refused_not_guessed(self):
        self.assertPlainError(
            b"timestamp,beats per minute,data source\n2026-07-16 00:00:00,60,Band\n",
            "without a timezone")

    def test_not_a_csv_at_all(self):
        self.assertPlainError(b"\x00\x01\x02\x03binary", "must have the columns")

    def test_empty_file(self):
        self.assertPlainError(b"", "must have the columns")

    def test_values_that_are_not_bpm(self):
        self.assertPlainError(
            b"timestamp,beats per minute,data source\n2026-07-16T00:00:00Z,99999,Band\n",
            "does not look like bpm")

    def test_nothing_readable(self):
        self.assertPlainError(
            b"timestamp,beats per minute,data source\n2026-07-16T00:00:00Z,y,Band\n",
            "no readable heart-rate rows")


class EmotiBitStubTests(SimpleTestCase):
    def test_it_accepts_no_files(self):
        self.assertFalse(get_adapter("emotibit").accepts("anything.csv"))

    def test_parsing_raises_a_clear_not_implemented_message(self):
        with self.assertRaises(SourceNotImplemented) as ctx:
            get_adapter("emotibit").parse_file(SourceFile("x.csv", b"1,2,3"))
        message = str(ctx.exception)
        self.assertIn("not implemented", message)
        self.assertIn("awaiting a sample export", message)
        self.assertIn("mid-project presentation", message)

    def test_the_stub_documents_what_is_needed(self):
        import people.sources.emotibit as module
        for need in ("heart rate", "sampling rate", "timestamp format",
                     "raw ppg", "units"):
            self.assertIn(need, module.__doc__.lower())

    def test_an_unknown_adapter_is_a_plain_error(self):
        with self.assertRaises(SourceError):
            get_adapter("nonsense")
