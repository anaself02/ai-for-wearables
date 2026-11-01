"""The Import data tab: what file types and layouts are accepted, and the limits.

Files are synthetic and use the real Fitbit column names. Sentinel strings stand in
for personal columns (an email, an address) so tests can prove they are never read,
shown or stored.
"""

import io
import os
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from openpyxl import Workbook

from people import ingest
from people.fitguard import fit_guard
from people.models import HeartRateDay, Person, Recording
from people.test_import import SpoolCase, upload_file
from people.testing import hr_csv_bytes, utc_day_file

HEADER = ["timestamp", "beats per minute", "data source"]
SENTINEL = "SENTINEL-PERSONAL-VALUE"


def csv_upload(name, text):
    return SimpleUploadedFile(name, text.encode(), content_type="text/csv")


def workbook_bytes(rows, header=HEADER, extra_sheet=None, sheet_title=None):
    workbook = Workbook()
    sheet = workbook.active
    if sheet_title:
        sheet.title = sheet_title
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    if extra_sheet:
        other = workbook.create_sheet("Second")
        for row in extra_sheet:
            other.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def xlsx_upload(name, rows, **kwargs):
    return SimpleUploadedFile(name, workbook_bytes(rows, **kwargs))


def good_rows(day=date(2026, 7, 16), n=300, source="Fitbit Air", start_bpm=60):
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return [[(start + timedelta(seconds=3 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             start_bpm + i % 7, source] for i in range(n)]


def text_of(response):
    return " ".join(response.content.decode().split())


class LooseCsvTests(SpoolCase):
    def test_a_csv_with_the_three_columns_is_accepted_whatever_it_is_called(self):
        start = datetime(2026, 7, 16, tzinfo=timezone.utc)
        data = hr_csv_bytes(start, start + timedelta(hours=2), source="Fitbit Air")
        response = self.validate("anas", [SimpleUploadedFile("my export (final).csv", data)])
        self.assertEqual(response.context["report"].n_will_load, 1)
        self.confirm("anas", self.token_of(response))
        self.assertEqual(Recording.objects.get().original_filename, "my export (final).csv")

    def test_a_csv_with_other_columns_is_refused_and_none_of_them_is_read_or_shown(self):
        text = ("timestamp,beats per minute,data source,email,home address\n"
                "2026-07-16T00:00:00Z,60,Fitbit Air,{0},{0}\n").format(SENTINEL)
        response = self.validate("anas", [csv_upload("export.csv", text)])
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("2 column(s) beyond the three heart-rate columns", html)
        self.assertNotIn(SENTINEL, html)
        self.assertNotIn("home address", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn('name="token"', html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_a_profile_style_file_is_refused_on_its_header_alone(self):
        text = "fitbit_id,email,first_name,birthday\n1,{0},{0},2000-01-01\n".format(SENTINEL)
        html = self.validate("anas", [csv_upload("user_profile.csv", text)]).content.decode()
        self.assertIn("must have the columns", html)
        self.assertNotIn(SENTINEL, html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_a_missing_column_is_named_but_no_layout_is_guessed(self):
        text = "timestamp,heart_rate,source\n2026-07-16T00:00:00Z,60,Fitbit Air\n"
        html = self.validate("anas", [csv_upload("x.csv", text)]).content.decode()
        self.assertIn("must have the columns timestamp, beats per minute, data source", html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_the_header_is_read_before_the_rest_of_the_file(self):
        # A header-only mismatch is refused even if the rest is not CSV at all.
        junk = b"patient,notes\n" + os.urandom(2000)
        html = self.validate("anas", [SimpleUploadedFile("scan.csv", junk)]).content.decode()
        self.assertIn("must have the columns", html)
        self.assertNotIn("Traceback", html)

    def test_a_utf8_bom_and_windows_line_endings_are_fine(self):
        text = ("﻿timestamp,beats per minute,data source\r\n"
                "2026-07-16T00:00:00Z,60,Fitbit Air\r\n"
                "2026-07-16T00:00:03Z,61,Fitbit Air\r\n")
        response = self.validate("anas", [csv_upload("bom.csv", text)])
        self.assertEqual(response.context["report"].rows_to_load, 2)


class XlsxTests(SpoolCase):
    def test_an_xlsx_first_sheet_is_read_and_loads(self):
        upload = xlsx_upload("heart.xlsx", good_rows(n=300))
        response = self.validate("anas", [upload])
        report = response.context["report"]
        self.assertEqual((report.n_will_load, report.rows_to_load), (1, 300))
        self.assertEqual(report.devices, {"Fitbit Air": 300})
        self.confirm("anas", self.token_of(response))
        stored = HeartRateDay.objects.get(person__slug="anas")
        epochs, bpm = stored.unpack()
        self.assertEqual(len(epochs), 300)
        self.assertEqual(list(bpm[:3]), [60.0, 61.0, 62.0])

    def test_only_the_first_sheet_is_read(self):
        upload = xlsx_upload("two.xlsx", good_rows(n=50),
                             extra_sheet=[["email"], [SENTINEL], [SENTINEL]])
        response = self.validate("anas", [upload])
        self.assertEqual(response.context["report"].rows_to_load, 50)
        self.assertNotIn(SENTINEL, response.content.decode())

    def test_an_xlsx_with_other_columns_is_refused_without_reading_them(self):
        rows = [[ts, bpm, src, SENTINEL] for ts, bpm, src in good_rows(n=5)]
        upload = xlsx_upload("extra.xlsx", rows, header=HEADER + ["email"])
        html = self.validate("anas", [upload]).content.decode()
        self.assertIn("1 column(s) beyond the three heart-rate columns", html)
        self.assertNotIn(SENTINEL, html)
        self.assertNotIn(">email<", html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_an_xlsx_with_a_different_layout_is_refused_plainly(self):
        upload = xlsx_upload("other.xlsx", [["a", 1]], header=["name", "value"])
        html = self.validate("anas", [upload]).content.decode()
        self.assertIn("is not a Fitbit heart-rate sheet", html)
        self.assertNotIn("Traceback", html)

    def test_excel_dates_without_a_timezone_are_refused_not_guessed(self):
        rows = [[datetime(2026, 7, 16, 0, 0, i), 60, "Fitbit Air"] for i in range(5)]
        html = self.validate("anas", [xlsx_upload("naive.xlsx", rows)]).content.decode()
        self.assertIn("without a timezone", html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_a_file_that_is_not_a_workbook_is_a_plain_message(self):
        html = self.validate("anas", [SimpleUploadedFile("bad.xlsx", b"not a workbook")]
                             ).content.decode()
        self.assertIn("is not a valid Excel (.xlsx) file", html)
        self.assertNotIn("Traceback", html)

    def test_a_zip_that_is_not_a_workbook_is_a_plain_message(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "hi")
        html = self.validate("anas", [SimpleUploadedFile("fake.xlsx", buffer.getvalue())]
                             ).content.decode()
        self.assertIn("could not be read as an Excel (.xlsx) file", html)

    def test_an_oversized_workbook_is_refused(self):
        with mock.patch("people.xlsx.MAX_XLSX_UNCOMPRESSED_BYTES", 10):
            html = self.validate("anas", [xlsx_upload("big.xlsx", good_rows(n=50))]
                                 ).content.decode()
        self.assertIn("too large to read as a workbook", html)

    def test_the_same_workbook_twice_is_loaded_once(self):
        def upload():
            return xlsx_upload("heart.xlsx", good_rows(n=100))

        self.confirm("anas", self.token_of(self.validate("anas", [upload()])))
        response = self.validate("anas", [upload()])
        self.assertEqual(response.context["report"].n_already, 1)
        self.assertNotIn('name="token"', response.content.decode())
        self.assertEqual(Recording.objects.count(), 1)

    def test_a_csv_and_an_xlsx_with_overlapping_readings_merge_without_duplicates(self):
        rows = good_rows(n=150)
        csv_text = "\n".join(["timestamp,beats per minute,data source"]
                             + ["{},{},{}".format(*r) for r in rows[:100]]) + "\n"
        self.confirm("anas", self.token_of(self.validate("anas", [csv_upload("a.csv", csv_text)])))
        response = self.validate("anas", [xlsx_upload("b.xlsx", rows[50:])])   # 50 overlap
        self.confirm("anas", self.token_of(response))
        stored = HeartRateDay.objects.get(person__slug="anas")
        epochs, _ = stored.unpack()
        self.assertEqual(len(epochs), 150)
        self.assertEqual(len(set(epochs)), 150)
        self.assertEqual(Recording.objects.count(), 2)

    def test_identical_readings_in_a_csv_and_an_xlsx_are_the_same_file(self):
        rows = good_rows(n=100)
        csv_text = "\n".join(["timestamp,beats per minute,data source"]
                             + ["{},{},{}".format(*r) for r in rows]) + "\n"
        self.confirm("anas", self.token_of(self.validate("anas", [csv_upload("a.csv", csv_text)])))
        response = self.validate("anas", [xlsx_upload("a.xlsx", rows)])
        self.assertEqual(response.context["report"].n_already, 1)        # same SHA-256


class ZipRulesTests(SpoolCase):
    def zip_upload(self, members, name="folder.zip"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for member, data in members.items():
                archive.writestr(member, data)
        return SimpleUploadedFile(name, buffer.getvalue())

    def good(self, day=date(2026, 7, 16)):
        return utc_day_file(day)

    def test_only_members_named_like_heart_rate_files_are_accepted(self):
        name, data = self.good()
        archive = self.zip_upload({
            "Heart Rate/" + name: data,
            "Heart Rate/Heart_Rate_2026-07-17.csv": data,        # wrong case
            "Heart Rate/heart_rate_readme.txt": b"DECOY",
            "Heart Rate/heart_rate_2026-07-18.csv.bak": b"DECOY",
            "User Profile/user_profile.csv": b"DECOY",
            "Account/Account_Management_Events_1.csv": b"DECOY",
            "Location/Gps Location 2026-07.csv": b"DECOY",
            "User Security Data/Security.csv": b"DECOY",
        })
        report = self.validate("anas", [archive]).context["report"]
        self.assertEqual(report.n_will_load, 1)
        self.assertEqual(report.skipped_other, 7)

    def test_skipped_members_are_never_opened(self):
        name, data = self.good()
        archive = self.zip_upload({name: data, "user_profile.csv": b"DECOY",
                                   "Devices.csv": b"DECOY"})
        opened = []
        real_open = zipfile.ZipFile.open

        def spy(self_, member, *args, **kwargs):
            opened.append(getattr(member, "filename", member))
            return real_open(self_, member, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "open", spy):
            self.validate("anas", [archive])
        # (numpy writes its own .npy members into the parked readings, so look only at
        # the members of the uploaded archive.)
        mine = [os.path.basename(m) for m in opened
                if os.path.basename(m) in {name, "user_profile.csv", "Devices.csv"}]
        self.assertEqual(mine, [name])

    def test_a_zip_is_never_extracted_to_disk(self):
        name, data = self.good()
        archive = self.zip_upload({name: data})
        with mock.patch.object(zipfile.ZipFile, "extract", side_effect=AssertionError("extract")), \
                mock.patch.object(zipfile.ZipFile, "extractall",
                                  side_effect=AssertionError("extractall")):
            response = self.validate("anas", [archive])
        self.assertEqual(response.context["report"].n_will_load, 1)

    def test_the_total_uncompressed_size_is_capped(self):
        name, data = self.good()
        archive = self.zip_upload({name: data})
        with mock.patch.object(ingest, "MAX_UNCOMPRESSED_BYTES", 100):
            html = self.validate("anas", [archive]).content.decode()
        self.assertIn("only the Heart Rate folder", html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_the_file_count_is_capped(self):
        members = {"heart_rate_2026-01-%02d.csv" % d: b"x" for d in range(1, 6)}
        with mock.patch.object(ingest, "MAX_ZIP_MEMBERS_READ", 3):
            html = self.validate("anas", [self.zip_upload(members)]).content.decode()
        self.assertIn("only the Heart Rate folder", html)

    def test_the_number_of_entries_in_the_archive_is_capped(self):
        members = {"junk/%d.txt" % i: b"x" for i in range(10)}
        with mock.patch.object(ingest, "MAX_ZIP_ENTRIES", 5):
            html = self.validate("anas", [self.zip_upload(members)]).content.decode()
        self.assertIn("too many files", html)

    def test_a_single_oversized_member_is_refused(self):
        name, data = self.good()
        with mock.patch.object(ingest, "MAX_MEMBER_BYTES", 100):
            html = self.validate("anas", [self.zip_upload({name: data})]).content.decode()
        self.assertIn("only the Heart Rate folder", html)

    def test_a_loose_file_above_the_size_limit_is_refused(self):
        with mock.patch("people.views_import.MAX_FILE_BYTES", 10):
            html = self.validate("anas", [upload_file(date(2026, 7, 16))]).content.decode()
        self.assertIn("too large to load here", html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_several_files_of_different_kinds_load_together(self):
        name, data = self.good(date(2026, 7, 15))
        response = self.validate("anas", [
            self.zip_upload({name: data}),
            xlsx_upload("two.xlsx", good_rows(day=date(2026, 7, 20), n=100)),
            upload_file(date(2026, 7, 17))])
        self.assertEqual(response.context["report"].n_will_load, 3)
        self.confirm("anas", self.token_of(response))
        self.assertEqual(Recording.objects.count(), 3)


class NoRetentionTests(SpoolCase):
    def spool_bytes(self):
        found = b""
        for base, _dirs, names in os.walk(self._tmp.name):
            for name in names:
                with open(os.path.join(base, name), "rb") as handle:
                    found += handle.read()
        return found

    def test_the_uploaded_file_is_not_kept_only_the_parsed_readings_wait(self):
        start = datetime(2026, 7, 16, tzinfo=timezone.utc)
        data = hr_csv_bytes(start, start + timedelta(hours=1), marker=71, source="Fitbit Air")
        response = self.validate("anas", [SimpleUploadedFile("heart.csv", data)])
        self.assertTrue(self.spool_dirs())
        parked = self.spool_bytes()
        self.assertNotIn(b"timestamp,beats per minute,data source", parked)
        self.assertNotIn(b"2026-07-16T00:00:03Z", parked)
        self.assertNotIn(data[:200], parked)
        self.assertTrue(response.context["token"])

    def test_nothing_is_left_after_confirm_or_cancel(self):
        token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 15))]))
        self.confirm("anas", token)
        self.assertEqual(self.spool_dirs(), [])
        token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 16))]))
        self.client.post(reverse("people:import_cancel", args=["anas"]), {"token": token})
        self.assertEqual(self.spool_dirs(), [])

    def test_confirming_uses_the_parked_readings_and_stores_exactly_them(self):
        files = [upload_file(date(2026, 7, 16), seed=3)]
        token = self.token_of(self.validate("anas", files))
        self.confirm("anas", token)
        stored = HeartRateDay.objects.get(person__slug="anas")
        reference = ingest.build_plan(
            Person.objects.create(slug="ref", display_name="Ref", device_label="Fitbit Air",
                                  adapter_key="fitbit-takeout"),
            [ingest.SourceFile(*utc_day_file(date(2026, 7, 16), seed=3))])
        expected = reference.loads[0][2]
        epochs, bpm = stored.unpack()
        self.assertEqual(list(epochs), list(expected.epochs))
        self.assertEqual(list(bpm), list(expected.bpm))

    def test_a_stale_import_is_cleaned_up(self):
        from people import spool
        self.validate("anas", [upload_file(date(2026, 7, 16))])
        self.assertEqual(len(self.spool_dirs()), 1)
        spool.cleanup(max_age=-1)
        self.assertEqual(self.spool_dirs(), [])


class NeverTrainsOnImportsTests(SpoolCase):
    def test_every_import_path_makes_no_fit_or_save_calls(self):
        name, data = utc_day_file(date(2026, 7, 14))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(name, data)
        with fit_guard():
            for upload in (upload_file(date(2026, 7, 15)),
                           xlsx_upload("a.xlsx", good_rows(n=100)),
                           SimpleUploadedFile("z.zip", buffer.getvalue())):
                response = self.validate("anas", [upload])
                self.confirm("anas", self.token_of(response))
            self.assertEqual(self.client.get("/p/anas/arousal/").status_code, 200)
        self.assertEqual(Recording.objects.count(), 3)
