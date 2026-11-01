"""The Sessions tab: entering labels, the labelling rule, grading, access, export.

Every reading is synthetic (people.testing). Most tests use a stand-in for the
WESAD-trained models; the guard and pinning tests use the real WESAD feature
table when the sibling wearable-ml project is present, and skip otherwise.
"""

import importlib.util
import logging
import os
import re
import sys
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from unittest import mock, skipUnless
from zoneinfo import ZoneInfo

from django.conf import settings
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from health import arousal, wesad_binary
from health.test_arousal import make_person
from people import sessions
from people.fitguard import fit_guard
from people.models import Session
from people.session_forms import SessionForm
from people.testing import StaffTestCase, make_user

TZ = "America/Toronto"
DAY = date(2026, 9, 17)          # synthetic readings cover 2026-09-17 and -18 UTC
NOTE = "zebra-note-should-never-leave"
WESAD_CSV = os.path.join(settings.WEARABLE_ML_DIR, "results", "features.csv")
HAS_WESAD = os.path.exists(WESAD_CSV)


class FakeSVM:
    """Stress when HR_mean is above 70; predicts only, never fits."""

    def predict(self, frame):
        return [int(v > 70) for v in frame["HR_mean"]]


def fake_models():
    stats = {"min": 40.0, "max": 150.0, "mean": 75.0, "std": 10.0}
    return wesad_binary.BinaryModels(
        svm=FakeSVM(), threshold=60.0, majority=0,
        distribution={name: dict(stats) for name in wesad_binary.FEATURE_NAMES},
        n_windows=1015, n_subjects=15)


def patched_models():
    return mock.patch.object(wesad_binary, "train", side_effect=fake_models)


def add(person, kind, start, minutes, note="", rating=None, day=DAY):
    """A session from local clock time on `day`."""
    begin = sessions.local_to_utc(day, start, person.timezone)
    return Session.objects.create(person=person, kind=kind, start_utc=begin,
                                  end_utc=begin + timedelta(minutes=minutes),
                                  note=note, rating=rating)


def form(person, instance=None, **data):
    values = {"date": "2026-09-17", "start_time": "10:00", "end_time": "10:10",
              "kind": "rest", "rating": "", "note": ""}
    values.update(data)
    return SessionForm(values, person=person, instance=instance)


def score_block(html):
    start = html.index('<div class="session-score">')
    end = html.index("</section>", start)
    return html[start:end]


class ValidationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.person = make_person("alpha", days=(DAY,))

    def test_end_must_be_after_start(self):
        for end in ("10:00", "09:59"):
            f = form(self.person, end_time=end)
            self.assertFalse(f.is_valid(), end)
            self.assertIn("The end must be after the start.", f.non_field_errors())

    def test_overlapping_sessions_are_rejected_touching_ones_are_not(self):
        add(self.person, "rest", time(10, 0), 10)
        overlap = form(self.person, start_time="10:05", end_time="10:20")
        self.assertFalse(overlap.is_valid())
        self.assertIn("overlaps", " ".join(overlap.non_field_errors()))
        inside = form(self.person, start_time="10:02", end_time="10:03")
        self.assertFalse(inside.is_valid())
        touching = form(self.person, start_time="10:10", end_time="10:20")
        self.assertTrue(touching.is_valid(), touching.errors)

    def test_editing_a_session_does_not_clash_with_itself(self):
        session = add(self.person, "rest", time(10, 0), 10)
        f = form(self.person, instance=session, end_time="10:15")
        self.assertTrue(f.is_valid(), f.errors)
        f.save()
        session.refresh_from_db()
        self.assertEqual(session.end_utc - session.start_utc, timedelta(minutes=15))

    def test_another_persons_session_is_not_an_overlap(self):
        other = make_person("beta", days=(DAY,))
        add(other, "rest", time(10, 0), 10)
        self.assertTrue(form(self.person).is_valid())

    def test_rating_must_be_1_to_5_and_note_is_short(self):
        self.assertFalse(form(self.person, rating="6").is_valid())
        self.assertFalse(form(self.person, note="x" * 201).is_valid())
        self.assertTrue(form(self.person, rating="3", note="ok").is_valid())


class DaylightSavingTests(TestCase):
    def test_local_time_is_stored_in_utc_with_the_right_offset(self):
        summer = sessions.local_to_utc(date(2026, 7, 1), time(10, 0), TZ)
        winter = sessions.local_to_utc(date(2026, 12, 1), time(10, 0), TZ)
        self.assertEqual(summer, datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))
        self.assertEqual(winter, datetime(2026, 12, 1, 15, 0, tzinfo=timezone.utc))

    def test_a_time_that_does_not_exist_is_refused(self):
        with self.assertRaisesRegex(sessions.SessionError, "does not exist"):
            sessions.local_to_utc(date(2026, 3, 8), time(2, 30), TZ)

    def test_a_time_that_happens_twice_is_refused(self):
        with self.assertRaisesRegex(sessions.SessionError, "happens twice"):
            sessions.local_to_utc(date(2026, 11, 1), time(1, 30), TZ)

    def test_a_session_across_the_change_has_its_real_length(self):
        start = sessions.local_to_utc(date(2026, 11, 1), time(0, 30), TZ)
        end = sessions.local_to_utc(date(2026, 11, 1), time(3, 0), TZ)
        self.assertEqual(end - start, timedelta(hours=3, minutes=30))
        start = sessions.local_to_utc(date(2026, 3, 8), time(1, 0), TZ)
        end = sessions.local_to_utc(date(2026, 3, 8), time(4, 0), TZ)
        self.assertEqual(end - start, timedelta(hours=2))

    def test_the_form_uses_the_same_conversion(self):
        person = make_person("alpha", days=(DAY,))
        f = form(person, date="2026-11-01", start_time="00:30", end_time="03:00")
        self.assertTrue(f.is_valid(), f.errors)
        session = f.save()
        self.assertEqual(session.end_utc - session.start_utc, timedelta(hours=3, minutes=30))


class WindowRuleTests(TestCase):
    def epochs(self, day, clock, minutes):
        start = int(sessions.local_to_utc(day, clock, TZ).timestamp())
        return start, start + minutes * 60

    def test_windows_lie_fully_inside_the_session_after_the_buffer(self):
        start, end = self.epochs(DAY, time(10, 0, 10), 10)   # off-grid start
        starts = sessions.label_window_starts(start, end, TZ)
        self.assertTrue(len(starts) > 0)
        self.assertTrue(all(s >= start + sessions.LABEL_LAG_SECONDS for s in starts))
        self.assertTrue(all(s + arousal.WINDOW_SECONDS <= end for s in starts))

    def test_windows_crossing_a_boundary_are_dropped(self):
        start, end = self.epochs(DAY, time(10, 0), 10)
        starts = list(sessions.label_window_starts(start, end, TZ))
        midnight = int(datetime(2026, 9, 17, tzinfo=ZoneInfo(TZ)).timestamp())
        grid = range(midnight, midnight + 86400, arousal.STEP_SECONDS)
        crossing = [s for s in grid if s < end < s + arousal.WINDOW_SECONDS
                    or s < start < s + arousal.WINDOW_SECONDS]
        self.assertTrue(crossing)
        self.assertFalse(set(crossing) & set(starts))
        # 10 min, minus the 60 s buffer, minus the last 60 s window length: 17 windows.
        self.assertEqual(len(starts), 17)

    def test_the_buffer_excludes_the_first_minute(self):
        start, end = self.epochs(DAY, time(10, 0), 2)
        self.assertEqual(list(sessions.label_window_starts(start, end, TZ)),
                         [start + 60])
        start, end = self.epochs(DAY, time(10, 0), 1)
        self.assertEqual(len(sessions.label_window_starts(start, end, TZ)), 0)

    def test_windows_are_on_the_same_grid_as_the_arousal_page(self):
        start, end = self.epochs(DAY, time(10, 0, 10), 30)
        day_grid = set(arousal.window_starts(DAY, TZ).tolist())
        self.assertTrue(set(sessions.label_window_starts(start, end, TZ).tolist()) <= day_grid)

    def test_label_mapping(self):
        self.assertEqual(sessions.PRIMARY_LABELS,
                         {Session.Kind.REST: 0, Session.Kind.STRESS_TASK: 1})
        self.assertNotIn(Session.Kind.RECOVERY, sessions.PRIMARY_LABELS)
        self.assertNotIn(Session.Kind.NATURAL_STRESS, sessions.PRIMARY_LABELS)


class GradingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.person = make_person("alpha", days=(DAY, DAY + timedelta(days=1)))

    def test_too_few_windows_shows_counts_only(self):
        add(self.person, "rest", time(10, 0), 10)
        add(self.person, "stress_task", time(10, 10), 10)
        with patched_models():
            result = sessions.score(self.person)
        self.assertTrue(result.available)
        self.assertFalse(result.enough)
        self.assertEqual(result.counts, {"non_stress": 17, "stress": 17, "other": 0})
        self.assertEqual(result.results, [])

    def test_recovery_and_natural_stress_are_not_in_the_primary_score(self):
        add(self.person, "rest", time(10, 0), 20)
        add(self.person, "stress_task", time(10, 20), 20)
        add(self.person, "recovery", time(10, 40), 20)
        add(self.person, "natural_stress", time(12, 0), 10)
        with patched_models():
            result = sessions.score(self.person)
        self.assertEqual(result.counts, {"non_stress": 37, "stress": 37, "other": 37 + 17})
        primary = [row["primary"] for row in result.per_session]
        self.assertEqual(primary, [True, True, False, False])

    def test_every_model_is_scored_on_the_same_windows(self):
        add(self.person, "rest", time(10, 0), 20)
        add(self.person, "stress_task", time(10, 20), 20)
        with patched_models():
            result = sessions.score(self.person)
        self.assertTrue(result.enough)
        self.assertEqual([title for title, _ in result.results],
                         [title for _, title in sessions.MODEL_NAMES])
        for title, m in result.results:
            rows = [sum(r) for r in m["confusion"]]
            self.assertEqual(rows, [37, 37], title)
            self.assertEqual([c["support"] for c in m["per_class"]], [37, 37], title)
        majority = dict(result.results)["Majority class (WESAD)"]
        self.assertEqual(majority["confusion"], [[37, 0], [37, 0]])

    def test_no_sessions_and_no_wesad_table_are_said_plainly(self):
        self.assertEqual(sessions.score(self.person).reason, "No sessions recorded yet.")
        add(self.person, "rest", time(10, 0), 10)
        with tempfile.TemporaryDirectory() as empty, override_settings(WEARABLE_ML_DIR=empty):
            result = sessions.score(self.person)
        self.assertFalse(result.available)
        self.assertIn("WESAD feature table was not found", result.reason)

    def test_coverage_counts_readings_and_warns_when_low(self):
        session = add(self.person, "rest", time(10, 0), 10)
        start = int(session.start_utc.timestamp())
        import numpy as np
        full = sessions.coverage(np.arange(start, start + 600, 3), start, start + 600, TZ)
        self.assertEqual((full["readings"], full["share"], full["low"]), (200, 1.0, False))
        half = sessions.coverage(np.arange(start, start + 300, 3), start, start + 600, TZ)
        self.assertEqual((half["share"], half["low"]), (0.5, True))
        none = sessions.coverage(np.array([], dtype="int64"), start, start + 600, TZ)
        self.assertEqual((none["readings"], none["first"], none["low"]), (0, None, True))


class PageAndCaveatTests(StaffTestCase):
    @classmethod
    def setUpTestData(cls):
        cls.person = make_person("alpha", days=(DAY, DAY + timedelta(days=1)))

    def get(self):
        with patched_models():
            return self.client.get(reverse("people:sessions", args=["alpha"])).content.decode()

    def test_the_score_never_renders_without_its_caveat(self):
        cases = [((10, 10)), ((20, 20))]          # not enough, then enough
        for rest, stress in cases:
            Session.objects.all().delete()
            add(self.person, "rest", time(10, 0), rest)
            add(self.person, "stress_task", time(11, 0), stress)
            html = self.get()
            block = score_block(html)
            for needed in ("not a clinical or validated measure", "outside the WESAD",
                           "Fitbit Air", "median reading spacing",
                           "No confidence interval is possible",
                           "do not generalise"):
                self.assertIn(needed, block, (rest, needed))
            outside = html.replace(block, "")
            for score_text in ("Macro-F1", "share predicted stress", "Primary score ("):
                self.assertNotIn(score_text, outside, score_text)
            if rest == 20:
                self.assertIn("Macro-F1", block)
                self.assertLess(block.index("not a clinical"), block.index("Macro-F1"))
            else:
                self.assertIn("Not enough labelled data to score", block)
                self.assertNotIn("Macro-F1", block)

    def test_a_score_cannot_be_built_without_its_evidence(self):
        with self.assertRaises(ValueError):
            sessions.SessionScore.graded(ood_rows=[])

    def test_the_template_prints_scores_only_inside_the_caveat_block(self):
        path = os.path.join(settings.BASE_DIR, "templates", "people", "sessions.html")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        block_start = source.index('<div class="session-score">')
        caveat = source.index("{{ score.caveat }}")
        self.assertLess(block_start, caveat)
        for name in ("score.results", "score.per_session", "score.counts", "score.enough"):
            for match in re.finditer(re.escape(name), source):
                self.assertGreater(match.start(), caveat, name)

    def test_the_page_shows_the_help_constants_and_missing_wesad(self):
        add(self.person, "rest", time(10, 0), 10)
        html = " ".join(self.get().split())
        for text in ("10 minutes of seated rest", "backwards aloud by 13",
                     "Only on yourself", "Stop at any time", "subjective",
                     "60 s</strong> after the session start", "30</strong> labelled windows per class"):
            self.assertIn(text, html, text)
        with tempfile.TemporaryDirectory() as empty, override_settings(WEARABLE_ML_DIR=empty):
            html = self.client.get(reverse("people:sessions", args=["alpha"])).content.decode()
        self.assertIn("WESAD feature table was not found", html)

    def test_add_edit_and_delete_through_the_page(self):
        url = reverse("people:sessions", args=["alpha"])
        with patched_models():
            response = self.client.post(url, {"date": "2026-09-17", "start_time": "10:00",
                                              "end_time": "10:10", "kind": "rest"})
            self.assertRedirects(response, url)
            session = Session.objects.get()
            edit = reverse("people:session_edit", args=["alpha", session.pk])
            self.assertEqual(self.client.get(edit).status_code, 200)
            self.client.post(edit, {"date": "2026-09-17", "start_time": "10:00",
                                    "end_time": "10:20", "kind": "stress_task"})
            session.refresh_from_db()
            self.assertEqual(session.kind, "stress_task")
            self.client.post(reverse("people:session_delete", args=["alpha", session.pk]))
        self.assertFalse(Session.objects.exists())


class AccessTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.alpha = make_person("alpha", days=(DAY,))
        cls.beta = make_person("beta", days=(DAY,))
        cls.owner_a, _ = make_user("a@example.com", linked_to=cls.alpha)
        cls.beta_session = add(cls.beta, "rest", time(10, 0), 10, note=NOTE)

    def as_user(self, user):
        client = Client()
        client.force_login(user)
        return client

    def test_an_owner_cannot_reach_the_other_persons_sessions(self):
        client = self.as_user(self.owner_a)
        pk = self.beta_session.pk
        for url in (reverse("people:sessions", args=["beta"]),
                    reverse("people:sessions_export", args=["beta"]),
                    reverse("people:session_edit", args=["beta", pk]),
                    reverse("people:session_edit", args=["alpha", pk])):
            self.assertEqual(client.get(url).status_code, 404, url)
        for url in (reverse("people:session_delete", args=["beta", pk]),
                    reverse("people:session_delete", args=["alpha", pk])):
            self.assertEqual(client.post(url).status_code, 404, url)
        self.assertTrue(Session.objects.filter(pk=pk).exists())

    def test_a_session_is_never_shown_under_another_name_even_for_staff(self):
        staff, _ = make_user("staff@example.com", staff=True)
        client = self.as_user(staff)
        self.assertEqual(client.get(reverse("people:session_edit",
                                            args=["alpha", self.beta_session.pk])).status_code, 404)
        with patched_models():
            html = client.get(reverse("people:sessions", args=["alpha"])).content.decode()
        self.assertNotIn(NOTE, html)
        self.assertIn("No sessions recorded yet.", html)

    def test_anonymous_is_sent_to_login(self):
        response = Client().get(reverse("people:sessions", args=["alpha"]))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])


class ExportTests(StaffTestCase):
    def test_export_has_no_notes_and_no_real_dates(self):
        person = make_person("alpha", days=(DAY, DAY + timedelta(days=1)))
        add(person, "rest", time(10, 0), 10, note=NOTE, rating=2)
        add(person, "stress_task", time(10, 30), 10, note=NOTE, rating=4)
        add(person, "recovery", time(9, 0), 10, note=NOTE, day=DAY + timedelta(days=1))
        with patched_models():
            response = self.client.get(reverse("people:sessions_export", args=["alpha"]))
        text = response.content.decode()
        self.assertEqual(response["Content-Type"], "text/csv")
        self.assertNotIn(NOTE, text)
        self.assertNotRegex(text, r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")
        self.assertNotRegex(text, r"(?i)\b(sep|september|2026)\b")
        lines = text.strip().splitlines()
        self.assertEqual(lines[0].split(","), sessions.EXPORT_COLUMNS)
        self.assertEqual([line.split(",")[0] for line in lines[1:]],
                         ["Session 1", "Session 2", "Session 3"])
        self.assertEqual([line.split(",")[3] for line in lines[1:]], ["0", "30", "1380"])
        self.assertNotIn("note", lines[0])

    def test_the_note_is_never_logged(self):
        person = make_person("alpha", days=(DAY,))
        url = reverse("people:sessions", args=["alpha"])
        with patched_models(), self.assertLogs(level=logging.DEBUG) as logs:
            logging.getLogger("probe").debug("probe")
            self.client.post(url, {"date": "2026-09-17", "start_time": "10:00",
                                   "end_time": "10:10", "kind": "rest", "note": NOTE})
            self.client.post(url, {"date": "2026-09-17", "start_time": "10:05",
                                   "end_time": "10:20", "kind": "rest", "note": NOTE})
            self.client.get(url)
            self.client.get(reverse("people:sessions_export", args=["alpha"]))
        self.assertNotIn(NOTE, "\n".join(logs.output))
        self.assertEqual(Session.objects.filter(person=person).count(), 1)


class NeverFitsOnSessionsTests(StaffTestCase):
    SOURCES = ("sessions.py", "views_sessions.py", "session_forms.py")

    def test_sessions_source_makes_no_fit_or_save_call(self):
        forbidden = re.compile(r"\bfit\(|\.fit_transform\(|partial_fit|joblib\.dump|"
                               r"pickle\.dump|GridSearchCV|RandomizedSearchCV|\bclone\(")
        root = os.path.dirname(os.path.abspath(sessions.__file__))
        offenders = []
        for name in self.SOURCES:
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                offenders += ["{}: {}".format(name, line.strip()) for line in handle
                              if forbidden.search(line)]
        self.assertEqual(offenders, [])

    def test_wesad_training_takes_no_data_argument(self):
        import inspect
        self.assertEqual(list(inspect.signature(wesad_binary.train).parameters), [])

    @skipUnless(HAS_WESAD, "wearable-ml/results/features.csv is not present")
    def test_the_whole_sessions_flow_runs_with_every_fit_patched_to_raise(self):
        wesad_binary.train()       # trained on WESAD before the guard goes up
        person = make_person("alpha", days=(DAY, DAY + timedelta(days=1)))
        add(person, "rest", time(10, 0), 20)
        add(person, "stress_task", time(10, 20), 20)
        url = reverse("people:sessions", args=["alpha"])
        with fit_guard():
            self.client.post(url, {"date": "2026-09-17", "start_time": "11:00",
                                   "end_time": "11:20", "kind": "recovery"})
            html = self.client.get(url).content.decode()
            self.assertIn("Macro-F1", html)
            self.assertEqual(self.client.get(
                reverse("people:sessions_export", args=["alpha"])).status_code, 200)
            result = sessions.score(person)
        self.assertTrue(result.enough)
        self.assertEqual(result.n_wesad_subjects, 15)


@skipUnless(HAS_WESAD, "wearable-ml/results/features.csv is not present")
class MatchesTheDemoTests(TestCase):
    """The in-memory binary model is the demo's: same rule, same SVM settings."""

    def test_threshold_and_svm_settings_match_wearable_ml(self):
        import pandas as pd

        ml = settings.WEARABLE_ML_DIR
        spec = importlib.util.spec_from_file_location(
            "wearable_ml_demo_wesad_pin", os.path.join(ml, "demo", "demo_wesad.py"))
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)

        table = pd.read_csv(WESAD_CSV, usecols=["label", "HR_mean"]).dropna()
        self.assertEqual(wesad_binary.best_threshold(table["HR_mean"], table["label"] == 2),
                         demo.best_threshold(table["HR_mean"], table["label"] == 2))

        scripts = os.path.join(ml, "scripts")
        before = set(sys.modules)
        sys.path.insert(0, scripts)
        try:
            train_module = importlib.import_module("03_train_model")
            theirs = train_module.build_models()["svm"].get_params()
        finally:
            sys.path.remove(scripts)
            # Drop only that project's own top-level modules (features,
            # project_config, 03_train_model), never a library's.
            for name in set(sys.modules) - before:
                path = getattr(sys.modules[name], "__file__", None) or ""
                if os.path.abspath(path).startswith(os.path.abspath(scripts)):
                    del sys.modules[name]
        ours = wesad_binary.svm_template().get_params()
        for key in ("classifier__kernel", "classifier__C", "classifier__gamma",
                    "classifier__class_weight", "classifier__random_state"):
            self.assertEqual(ours[key], theirs[key], key)
        self.assertEqual([name for name, _ in ours["steps"]],
                         [name for name, _ in theirs["steps"]])
