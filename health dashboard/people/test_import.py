"""The upload flow: validate, then confirm. Synthetic files only."""

import io
import os
import re
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from people.fitguard import fit_guard
from people.models import HeartRateDay, Person, Recording
from people.test_views import make_person
from people.testing import hr_csv_bytes, utc_day_file, StaffTestCase

NOTICE = "Model trained on WESAD only; uploaded data is only scored."
EMOTIBIT = ("EmotiBit import is not available yet: it needs a sample export from "
            "Rahil.")


def upload_file(day, **kwargs):
    name, data = utc_day_file(day, **kwargs)
    return SimpleUploadedFile(name, data, content_type="text/csv")


def tiny_file(day):
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    data = hr_csv_bytes(start, start.replace(minute=1), spacing=3)
    return SimpleUploadedFile("heart_rate_{}.csv".format(day), data)


class SpoolCase(StaffTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        override = override_settings(UPLOAD_SPOOL_DIR=self._tmp.name)
        override.enable()
        self.addCleanup(override.disable)

    def spool_dirs(self):
        return [d for d in os.listdir(self._tmp.name)
                if os.path.isdir(os.path.join(self._tmp.name, d))]

    def validate(self, slug, files):
        return self.client.post(reverse("people:import_data", args=[slug]), {"files": files})

    def token_of(self, response):
        return re.search(r'name="token" value="([^"]+)"',
                         response.content.decode()).group(1)

    def confirm(self, slug, token, **extra):
        return self.client.post(reverse("people:import_confirm", args=[slug]),
                                {"token": token, **extra})


class FormTests(SpoolCase):
    def test_the_form_says_models_are_wesad_only_and_how_files_are_accepted(self):
        response = self.client.get(reverse("people:import_data", args=["anas"]))
        html = " ".join(response.content.decode().split())
        self.assertIn(NOTICE, html)
        self.assertIn('type="file"', html)
        self.assertIn("accepted by its header", html)
        self.assertIn("without being opened", html)
        self.assertIn("beats per minute", html)
        self.assertIn("Fitbit Air", html)
        self.assertIn(">Import data<", html)
    def test_rahils_page_has_the_form_the_message_and_what_is_needed(self):
        response = self.client.get(reverse("people:import_data", args=["rahil"]))
        html = " ".join(response.content.decode().split())
        self.assertIn('type="file"', html)
        self.assertIn(EMOTIBIT, html)
        for need in ("file layout", "whether heart rate is a signal", "raw PPG",
                     "sampling rate", "timestamp format", "timezone", "units"):
            self.assertIn(need.lower(), html.lower(), need)
    def test_posting_nothing_is_a_plain_message(self):
        response = self.client.post(reverse("people:import_data", args=["anas"]), {})
        self.assertContains(response, "Choose at least one file.")

    def test_the_upload_tab_is_in_the_bar(self):
        bar = re.search(r'<header class="topbar">.*?</header>',
                        self.client.get("/p/anas/").content.decode(), re.S).group(0)
        self.assertIn(reverse("people:import_data", args=["anas"]), bar)


class ValidateThenConfirmTests(SpoolCase):
    def test_validating_shows_a_full_report_and_stores_nothing(self):
        response = self.validate("anas", [upload_file(date(2026, 7, 15)),
                                          upload_file(date(2026, 7, 16))])
        self.assertEqual(response.status_code, 200)
        html = " ".join(response.content.decode().split())
        self.assertIn("Nothing has been saved yet", html)
        self.assertIn("2 file(s)", html)
        self.assertIn("57,600", html)
        self.assertIn("2026-07-15 00:00Z to 2026-07-16 23:59Z", html)   # UTC, not re-zoned
        self.assertIn("2026-07-14 to 2026-07-16", html)                  # local range
        self.assertIn("America/Toronto", html)
        self.assertIn("Fitbit Air", html)
        self.assertIn(NOTICE, html)
        self.assertEqual(HeartRateDay.objects.count(), 0)
        self.assertEqual(Recording.objects.count(), 0)
        self.assertEqual(len(self.spool_dirs()), 1)

    def test_confirming_stores_the_rows_and_clears_the_spool(self):
        files = [upload_file(date(2026, 7, 15)), upload_file(date(2026, 7, 16))]
        token = self.token_of(self.validate("anas", files))
        response = self.client.post(reverse("people:import_confirm", args=["anas"]),
                                    {"token": token}, follow=True)
        self.assertEqual(response.redirect_chain, [("/p/anas/", 302)])
        anas = Person.objects.get(slug="anas")
        self.assertEqual(anas.heart_rate_days.count(), 2)
        self.assertEqual(anas.recordings.count(), 2)
        self.assertEqual(self.spool_dirs(), [])
        self.assertContains(response, "Loaded 57,600 readings from 2 file(s) for Anas")

    def test_uploading_the_same_files_again_changes_nothing_and_offers_no_load(self):
        def files():
            return [upload_file(date(2026, 7, 15))]

        self.confirm("anas", self.token_of(self.validate("anas", files())))
        before = list(HeartRateDay.objects.values_list("id", "n"))

        response = self.validate("anas", files())
        html = response.content.decode()
        self.assertIn("already loaded", html)
        self.assertIn("Everything in this upload is already loaded", html)
        self.assertNotIn('name="token"', html)
        self.assertEqual(self.spool_dirs(), [])
        self.assertEqual(list(HeartRateDay.objects.values_list("id", "n")), before)
        self.assertEqual(Recording.objects.count(), 1)

    def test_cancel_discards_the_parked_files(self):
        token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 15))]))
        self.client.post(reverse("people:import_cancel", args=["anas"]), {"token": token})
        self.assertEqual(self.spool_dirs(), [])
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_a_confirmed_token_cannot_be_used_twice(self):
        token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 15))]))
        self.confirm("anas", token)
        response = self.confirm("anas", token)
        self.assertContains(response, "expired or was already used")
        self.assertEqual(Recording.objects.count(), 1)


class BadInputTests(SpoolCase):
    def test_a_bad_file_gives_a_plain_message_not_a_stack_trace(self):
        bad = SimpleUploadedFile("heart_rate_2026-07-16.csv", b"time,hr\n1,2\n")
        response = self.validate("anas", [bad])
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("must have the columns", html)
        self.assertIn("could not be read", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn('name="token"', html)
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_one_bad_file_blocks_the_whole_upload(self):
        bad = SimpleUploadedFile("heart_rate_2026-07-17.csv", b"time,hr\n1,2\n")
        response = self.validate("anas", [upload_file(date(2026, 7, 16)), bad])
        self.assertNotIn('name="token"', response.content.decode())
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_unsupported_file_types_are_skipped_without_being_read(self):
        decoys = [SimpleUploadedFile("notes.txt", b"DECOY"),
                  SimpleUploadedFile("profile.json", b"DECOY"),
                  SimpleUploadedFile("scan.pdf", b"DECOY")]
        response = self.validate("anas", [upload_file(date(2026, 7, 16))] + decoys)
        html = response.content.decode()
        self.assertIn("notes.txt", html)
        self.assertIn("Skipped without being read", html)
        self.assertEqual(response.context["report"].skipped_other, 3)
        self.confirm("anas", self.token_of(response))
        self.assertEqual(Recording.objects.count(), 1)
    def test_only_unsupported_files_gives_a_plain_refusal(self):
        response = self.validate("anas", [SimpleUploadedFile("notes.txt", b"x")])
        self.assertContains(response, "reader accepts")
    def test_a_broken_zip_is_a_plain_message(self):
        response = self.validate("anas", [SimpleUploadedFile("export.zip", b"not a zip")])
        self.assertContains(response, "not a valid zip")

    def test_an_oversized_upload_asks_for_only_the_heart_rate_folder(self):
        with override_settings(MAX_UPLOAD_BYTES=10):
            response = self.validate("anas", [upload_file(date(2026, 7, 16))])
        self.assertContains(response, "only the Heart Rate folder")
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_rahil_cannot_upload_and_nothing_is_saved(self):
        response = self.validate("rahil", [upload_file(date(2026, 7, 16))])
        html = " ".join(response.content.decode().split())
        self.assertIn(EMOTIBIT, html)
        self.assertIn("Nothing was read and nothing was saved", html)
        self.assertIn("raw PPG", html)
        self.assertNotIn('name="token"', html)
        self.assertEqual(HeartRateDay.objects.count(), 0)
        self.assertEqual(Recording.objects.count(), 0)
        self.assertEqual(self.spool_dirs(), [])
class ZipTests(SpoolCase):
    def zip_upload(self, members):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return SimpleUploadedFile("heart_rate_folder.zip", buffer.getvalue())

    def test_a_zip_of_the_folder_loads_and_decoys_inside_it_are_skipped(self):
        name, data = utc_day_file(date(2026, 7, 16))
        archive = self.zip_upload({
            "Takeout/Google Health/Physical Activity_GoogleData/Heart Rate/" + name: data,
            "Takeout/Google Health/User Profile_GoogleData/user_profile.csv": b"DECOY",
        })
        response = self.validate("anas", [archive])
        report = response.context["report"]
        self.assertEqual(report.n_will_load, 1)
        self.assertEqual(report.skipped_other, 1)
        self.confirm("anas", self.token_of(response))
        self.assertEqual(Recording.objects.get().original_filename, name)


class ManyFilesTests(SpoolCase):
    def test_more_than_a_hundred_files_in_one_upload_are_accepted(self):
        # Django refuses more than 100 files per request by default, and a real
        # Heart Rate folder holds about a hundred daily files.
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(120)]
        response = self.validate("anas", [tiny_file(d) for d in days])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["report"].n_will_load, 120)
        self.confirm("anas", self.token_of(response))
        self.assertEqual(Recording.objects.count(), 120)


class PersonGuardTests(SpoolCase):
    def test_a_report_for_one_person_cannot_be_loaded_into_another(self):
        token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 16))]))
        other = Person.objects.create(slug="other", display_name="Other",
                                      device_label="Fitbit Air",
                                      adapter_key="fitbit-takeout")
        response = self.confirm(other.slug, token)
        self.assertContains(response, "prepared for a different person")
        self.assertEqual(HeartRateDay.objects.count(), 0)
        self.assertEqual(len(self.spool_dirs()), 1)       # still Anas's to confirm

    def test_a_malformed_token_is_refused_and_cannot_traverse_paths(self):
        for token in ("../../etc", "a/b", "short", "", "x" * 200, "..\\..\\x"):
            response = self.confirm("anas", token)
            self.assertContains(response, "not valid")
        self.assertEqual(HeartRateDay.objects.count(), 0)

    def test_a_file_naming_another_device_needs_an_explicit_tick(self):
        response = self.validate(
            "anas", [upload_file(date(2026, 7, 16), source="EmotiBit")])
        self.assertContains(response, "Check this:")
        self.assertContains(response, 'name="confirm_person"')
        token = self.token_of(response)
        refused = self.confirm("anas", token)
        self.assertContains(refused, "Tick the box")
        self.assertEqual(HeartRateDay.objects.count(), 0)
        accepted = self.confirm("anas", token, confirm_person="1")
        self.assertRedirects(accepted, "/p/anas/")
        self.assertEqual(HeartRateDay.objects.count(), 1)

    def test_uploads_never_mix_into_the_other_person(self):
        beta = make_person("beta", "Beta", "Fitbit Charge", [date(2026, 8, 1)], 133)
        before = list(HeartRateDay.objects.filter(person=beta).values_list("id", "n"))
        self.confirm("anas", self.token_of(
            self.validate("anas", [upload_file(date(2026, 7, 16), marker=77)])))
        self.assertEqual(list(HeartRateDay.objects.filter(person=beta)
                              .values_list("id", "n")), before)
        self.assertEqual(Person.objects.get(slug="anas").recordings.count(), 1)


class NeverTrainsOnUploadsTests(SpoolCase):
    def test_the_whole_upload_flow_never_fits_or_saves_a_model(self):
        with fit_guard():
            token = self.token_of(self.validate("anas", [upload_file(date(2026, 7, 16))]))
            self.confirm("anas", token)
            for url in ("/", "/p/anas/", "/p/anas/heart-rate/", "/p/anas/raw/"):
                self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(HeartRateDay.objects.count(), 1)

    def test_the_guard_itself_fails_loudly(self):
        from sklearn.svm import SVC
        with fit_guard():
            with self.assertRaises(AssertionError):
                SVC().fit([[0.0], [1.0]], [0, 1])

    def test_no_application_source_fits_or_saves_a_model(self):
        root = os.path.dirname(os.path.abspath(__file__))
        forbidden = re.compile(r"\.fit\(|\.fit_transform\(|\.partial_fit\(|joblib\.dump|"
                               r"pickle\.dump|GridSearchCV|RandomizedSearchCV")
        offenders = []
        for base, _dirs, names in os.walk(root):
            for name in names:
                if (not name.endswith(".py") or name.startswith("test")
                        or name in ("testing.py", "fitguard.py")
                        or "migrations" in base):
                    continue
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as handle:
                    for number, line in enumerate(handle, 1):
                        if forbidden.search(line) and not line.lstrip().startswith("#"):
                            offenders.append("{}:{}: {}".format(name, number, line.strip()))
        self.assertEqual(offenders, [])
