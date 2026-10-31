"""Ingest tests: parse, report, hash dedupe, merge, command. Synthetic data only."""

import builtins
import io
import os
import tempfile
import zipfile
from datetime import date, datetime, timezone
from unittest import mock

import numpy as np
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from people import ingest
from people.models import HeartRateDay, Person, Recording
from people.sources import SourceError, SourceFile, get_adapter
from people.testing import hr_csv_bytes, utc_day_file

UTC = timezone.utc


def anas():
    return Person.objects.get(slug="anas")


def source_files(days, **kwargs):
    return [SourceFile(*utc_day_file(d, **kwargs)) for d in days]


def load(person, files, skipped=0):
    plan = ingest.build_plan(person, files, skipped_other=skipped)
    return plan, ingest.commit_plan(plan)


class LoadTests(TestCase):
    def test_three_days_are_stored_and_read_back_exactly(self):
        person = anas()
        days = [date(2026, 7, 15), date(2026, 7, 16), date(2026, 7, 17)]
        files = source_files(days, seed=5)
        plan, result = load(person, files)

        self.assertEqual((result.files_loaded, result.days_new), (3, 3))
        self.assertEqual(person.heart_rate_days.count(), 3)
        self.assertEqual(person.recordings.count(), 3)
        self.assertEqual(plan.report.rows_to_load, 3 * 28800)

        stored = HeartRateDay.objects.get(person=person, utc_date=days[1])
        epochs, bpm = stored.unpack()
        parsed = get_adapter("fitbit-takeout").parse_file(files[1])
        self.assertEqual(list(epochs), list(parsed.epochs))
        self.assertEqual(list(bpm), list(parsed.bpm))
        self.assertEqual(stored.sources, {"Synthetic Band": 28800})
        self.assertEqual(stored.gap_median, 3.0)

    def test_the_report_names_range_timezone_and_device(self):
        plan = ingest.build_plan(anas(), source_files([date(2026, 7, 16)]))
        r = plan.report
        self.assertEqual(r.timezone_assumed, "America/Toronto")
        self.assertEqual(r.first_utc, datetime(2026, 7, 16, tzinfo=UTC))
        # 00:00Z on the 16th is the evening of the 15th in Toronto
        self.assertEqual(r.first_local, date(2026, 7, 15))
        self.assertEqual(r.devices, {"Synthetic Band": 28800})
        text = ingest.render_report(r)
        self.assertIn("America/Toronto", text)
        self.assertIn("Ready to load", text)

    def test_building_a_plan_writes_nothing(self):
        ingest.build_plan(anas(), source_files([date(2026, 7, 16)]))
        self.assertEqual(HeartRateDay.objects.count(), 0)
        self.assertEqual(Recording.objects.count(), 0)


class DedupeTests(TestCase):
    def test_reloading_the_same_files_changes_nothing(self):
        person = anas()
        files = source_files([date(2026, 7, 15), date(2026, 7, 16)])
        load(person, files)
        before = list(HeartRateDay.objects.values_list("id", "n", "utc_date"))

        plan = ingest.build_plan(person, files)
        self.assertEqual(plan.report.n_already, 2)
        self.assertEqual(plan.report.n_will_load, 0)
        self.assertIn("already loaded", plan.report.refusal())
        with self.assertRaises(SourceError):
            ingest.commit_plan(plan)
        self.assertEqual(list(HeartRateDay.objects.values_list("id", "n", "utc_date")),
                         before)
        self.assertEqual(Recording.objects.count(), 2)

    def test_the_same_file_twice_in_one_upload_is_loaded_once(self):
        files = source_files([date(2026, 7, 16)]) * 2
        plan, result = load(anas(), files)
        self.assertEqual(result.files_loaded, 1)
        self.assertEqual(plan.report.n_already, 1)
        self.assertEqual(HeartRateDay.objects.get().n, 28800)

    def test_the_same_content_under_another_filename_is_still_a_duplicate(self):
        name, data = utc_day_file(date(2026, 7, 16))
        load(anas(), [SourceFile(name, data)])
        plan = ingest.build_plan(anas(), [SourceFile("heart_rate_2026-07-17.csv", data)])
        self.assertEqual(plan.report.n_already, 1)

    def test_overlapping_exports_merge_by_timestamp_without_duplicates(self):
        person = anas()
        day = datetime(2026, 7, 16, tzinfo=UTC)
        first = hr_csv_bytes(day, day.replace(hour=12), seed=1)
        # a different export covering 06:00-18:00: 06:00-12:00 overlaps
        second = hr_csv_bytes(day.replace(hour=6), day.replace(hour=18), seed=2,
                              source="Other Band")
        load(person, [SourceFile("heart_rate_2026-07-16.csv", first)])
        plan, result = load(person, [SourceFile("heart_rate_2026-07-16.csv", second)])

        stored = HeartRateDay.objects.get(person=person)
        epochs, _ = stored.unpack()
        self.assertEqual(len(epochs), len(set(epochs)))
        self.assertEqual(len(epochs), 18 * 1200)          # 00:00-18:00 at 3 s
        self.assertTrue((np.diff(epochs) > 0).all())
        self.assertEqual(result.days_merged, 1)
        self.assertEqual(result.rows_already_present, 6 * 1200)
        # exact per-source counts: only genuinely new rows were attributed
        self.assertEqual(stored.sources, {"Synthetic Band": 12 * 1200,
                                          "Other Band": 6 * 1200})
        self.assertEqual(person.recordings.count(), 2)


class PersonIsolationInIngestTests(TestCase):
    def test_loading_for_one_person_never_touches_another(self):
        load(anas(), source_files([date(2026, 7, 16)]))
        rahil = Person.objects.get(slug="rahil")
        self.assertFalse(rahil.has_data)
        self.assertEqual(rahil.recordings.count(), 0)

    def test_the_same_file_may_be_loaded_for_two_people_independently(self):
        other = Person.objects.create(slug="demo", display_name="Demo",
                                      device_label="Fitbit Air",
                                      adapter_key="fitbit-takeout")
        files = source_files([date(2026, 7, 16)])
        load(anas(), files)
        plan, _ = load(other, files)
        self.assertEqual(plan.report.n_will_load, 1)
        self.assertEqual(HeartRateDay.objects.count(), 2)


class BadFileTests(TestCase):
    def test_one_bad_file_refuses_the_whole_upload_with_a_plain_message(self):
        good = source_files([date(2026, 7, 16)])
        bad = [SourceFile("heart_rate_2026-07-17.csv", b"time,hr\n1,2\n")]
        plan = ingest.build_plan(anas(), good + bad)
        self.assertEqual(plan.report.n_error, 1)
        self.assertFalse(plan.report.can_commit)
        self.assertIn("could not be read", plan.report.refusal())
        with self.assertRaises(SourceError):
            ingest.commit_plan(plan)
        self.assertEqual(HeartRateDay.objects.count(), 0)
        self.assertIn("must have the columns", ingest.render_report(plan.report))

    def test_an_upload_with_no_accepted_file_says_so(self):
        plan = ingest.build_plan(anas(), [], skipped_other=5)
        self.assertIn("No file in this upload", plan.report.refusal())

    def test_a_file_naming_a_different_device_raises_a_warning(self):
        files = source_files([date(2026, 7, 16)], source="EmotiBit")
        plan = ingest.build_plan(anas(), files)
        self.assertTrue(plan.report.warnings)
        self.assertIn("right person", plan.report.warnings[0])


class EmotiBitTests(TestCase):
    def test_rahils_upload_is_refused_up_front_and_nothing_is_stored(self):
        rahil = Person.objects.get(slug="rahil")
        plan = ingest.build_plan(rahil, source_files([date(2026, 7, 16)]))
        self.assertIn("not implemented", plan.report.fatal)
        self.assertIn("awaiting a sample export", plan.report.refusal())
        with self.assertRaises(SourceError):
            ingest.commit_plan(plan)
        self.assertEqual(HeartRateDay.objects.count(), 0)


class ZipTests(SimpleTestCase):
    def make_zip(self, members):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return buffer.getvalue()

    def test_only_accepted_members_are_read_decoys_are_not(self):
        name, data = utc_day_file(date(2026, 7, 16))
        blob = self.make_zip({
            "Takeout/Google Health/Physical Activity_GoogleData/Heart Rate/" + name: data,
            "Takeout/Google Health/User Profile_GoogleData/user_profile.csv": b"DECOY",
            "Takeout/Google Health/Paired Devices/Devices.csv": b"DECOY",
            "Takeout/Google Health/Stress Score/Stress Score.csv": b"DECOY",
        })
        read = []
        original = zipfile.ZipFile.read

        def spy(self_, name_or_info, *args, **kwargs):
            read.append(getattr(name_or_info, "filename", name_or_info))
            return original(self_, name_or_info, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "read", spy):
            files, skipped = ingest.files_from_zip(get_adapter("fitbit-takeout"), blob)

        self.assertEqual([f.name for f in files], [name])
        self.assertEqual(skipped, 3)
        self.assertEqual([os.path.basename(r) for r in read], [name])

    def test_a_bad_zip_is_a_plain_error(self):
        with self.assertRaises(SourceError) as ctx:
            ingest.files_from_zip(get_adapter("fitbit-takeout"), b"not a zip")
        self.assertIn("not a valid zip", str(ctx.exception))

    def test_an_oversized_zip_asks_for_only_the_heart_rate_folder(self):
        name, data = utc_day_file(date(2026, 7, 16))
        blob = self.make_zip({name: data})
        with mock.patch.object(ingest, "MAX_UNCOMPRESSED_BYTES", 10):
            with self.assertRaises(SourceError) as ctx:
                ingest.files_from_zip(get_adapter("fitbit-takeout"), blob)
        self.assertIn("only the Heart Rate folder", str(ctx.exception))


class FolderTests(SimpleTestCase):
    def test_only_accepted_names_are_opened(self):
        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(os.fspath(file) if not isinstance(file, int) else "")
            return real_open(file, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            name, data = utc_day_file(date(2026, 7, 16))
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
            for decoy in ("user_profile.csv", "Devices.csv", "heart_rate_readme.txt"):
                with open(os.path.join(tmp, decoy), "w") as f:
                    f.write("DECOY")
            opened.clear()
            with mock.patch.object(builtins, "open", spy):
                files, skipped = ingest.files_from_folder(
                    get_adapter("fitbit-takeout"), tmp)
            inside = [os.path.basename(p) for p in opened if tmp in os.path.abspath(p)]
        self.assertEqual([f.name for f in files], [name])
        self.assertEqual(skipped, 3)
        self.assertEqual(inside, [name])

    def test_a_missing_folder_is_a_plain_error(self):
        with self.assertRaises(SourceError):
            ingest.files_from_folder(get_adapter("fitbit-takeout"), "Z:/nope")


class CommandTests(TestCase):
    def folder(self, tmp, days):
        for d in days:
            name, data = utc_day_file(d)
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
        with open(os.path.join(tmp, "user_profile.csv"), "w") as f:
            f.write("DECOY")

    def run_command(self, *args):
        out = io.StringIO()
        call_command("import_heart_rate", *args, stdout=out)
        return out.getvalue()

    def test_load_then_reload_then_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.folder(tmp, [date(2026, 7, 15), date(2026, 7, 16)])

            dry = self.run_command("anas", tmp, "--dry-run")
            self.assertIn("nothing was stored", dry.lower())
            self.assertEqual(HeartRateDay.objects.count(), 0)

            text = self.run_command("anas", tmp)
            self.assertIn("Stored 57,600 readings from 2 file(s)", text)
            self.assertIn("1 other file name(s) skipped without being opened", text)
            self.assertEqual(HeartRateDay.objects.count(), 2)

            again = self.run_command("anas", tmp)
            self.assertIn("already loaded", again)
            self.assertIn("Nothing stored", again)
            self.assertEqual(HeartRateDay.objects.count(), 2)
            self.assertEqual(Recording.objects.count(), 2)

    def test_unknown_person_and_missing_folder_are_plain_errors(self):
        with self.assertRaises(CommandError) as ctx:
            self.run_command("nobody", ".")
        self.assertIn("Known: anas, rahil", str(ctx.exception))
        with self.assertRaises(CommandError):
            self.run_command("anas", "Z:/nope")

    def test_rahil_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.folder(tmp, [date(2026, 7, 16)])
            text = self.run_command("rahil", tmp)
        self.assertIn("CANNOT LOAD", text)
        self.assertEqual(HeartRateDay.objects.count(), 0)
