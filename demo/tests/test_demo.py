"""
Tests for the demo code only (demo/*.py). Standard library unittest, no new
dependencies:

    .venv\\Scripts\\python.exe -m unittest discover -s demo/tests -v

All Takeout-shaped data here is synthetic: seeded random numbers with the real
export's column names. No personal file is read or copied.
"""

import builtins
import contextlib
import hashlib
import io
import os
import re
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

sys.dont_write_bytecode = True

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(TESTS_DIR)
PROJECT_DIR = os.path.dirname(DEMO_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
for _path in (SCRIPTS_DIR, DEMO_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import demo_personal as personal  # noqa: E402
import demo_wesad as wesad  # noqa: E402
import run_demo  # noqa: E402
from demo_guard import FitGuard  # noqa: E402

TZ = "America/Toronto"
HR_NAME = re.compile(r"^heart_rate_\d{4}-\d{2}-\d{2}\.csv$")


def write_hr_csv(path, start_utc, end_utc, spacing=3, seed=0):
    """A synthetic Takeout heart-rate file: the real header, fake values."""
    rng = np.random.default_rng(seed)
    stamps = []
    t = start_utc
    while t < end_utc:
        stamps.append(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
        t += timedelta(seconds=spacing)
    bpm = rng.integers(55, 80, size=len(stamps)).astype(float)
    pd.DataFrame({"timestamp": stamps, "beats per minute": bpm,
                  "data source": "Synthetic Band"}).to_csv(path, index=False)


def make_takeout(folder, first_utc_day, n_days, decoys=True):
    """Write n full UTC days of synthetic data, plus decoy files that must
    never be opened."""
    os.makedirs(folder, exist_ok=True)
    for i in range(n_days):
        day = first_utc_day + timedelta(days=i)
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        write_hr_csv(os.path.join(folder, "heart_rate_{}.csv".format(day)),
                     start, start + timedelta(days=1), seed=i)
    if decoys:
        for name in ("user_profile.csv", "Devices.csv", "heart_rate_readme.txt",
                     "Stress Score.csv", "heart_rate_2026-07-16.csv.bak"):
            with open(os.path.join(folder, name), "w") as f:
                f.write("DECOY - must never be opened\n")


def snapshot(roots, files):
    """{relative path: (size, mtime_ns, sha256)} for directories and files."""
    result = {}
    paths = list(files)
    for root in roots:
        for base, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                paths.append(os.path.join(base, name))
    for path in paths:
        stat = os.stat(path)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        result[os.path.relpath(path, PROJECT_DIR)] = (
            stat.st_size, stat.st_mtime_ns, digest)
    return result


def project_listing():
    """Every file under the project except data, venv, git, caches and demo."""
    found = set()
    for base, dirs, names in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs
                   if d not in ("data", ".venv", ".git", "__pycache__", "demo")]
        for name in names:
            found.add(os.path.relpath(os.path.join(base, name), PROJECT_DIR))
    return found


# ---------------------------------------------------------------------------
# 1. no-fit guard
# ---------------------------------------------------------------------------

class FitGuardTests(unittest.TestCase):
    def test_guard_blocks_fit_and_dump_then_releases(self):
        X = np.array([[0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1])
        with FitGuard():
            with self.assertRaises(AssertionError):
                SVC().fit(X, y)
            with self.assertRaises(AssertionError):
                joblib.dump({}, os.devnull)
        SVC().fit(X, y)  # released: fitting works again outside the guard

    def test_guard_blocks_pipeline_and_scaler(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.array([[0.0], [1.0]])
        with FitGuard():
            with self.assertRaises(AssertionError):
                StandardScaler().fit(X)
            with self.assertRaises(AssertionError):
                Pipeline([("s", StandardScaler())]).fit(X, [0, 1])

    def test_scoring_a_synthetic_day_makes_no_fit_calls(self):
        model = joblib.load(os.path.join(PROJECT_DIR, "models", "svm_hr_only.joblib"))
        names = ["HR_mean", "HR_std", "HR_min", "HR_max", "HR_slope"]
        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 16), 2)
            samples, _ = personal.load_local_day(tmp, date(2026, 7, 16), TZ)
            windows = personal.build_windows(samples, date(2026, 7, 16), TZ)
            with FitGuard():
                predictions = personal.predict_only(model, windows, names)
        self.assertEqual(len(predictions), len(windows))
        self.assertGreater(len(windows), 2000)

    def test_personal_module_contains_no_fit_or_dump_calls(self):
        with open(os.path.join(DEMO_DIR, "demo_personal.py"),
                  encoding="utf-8") as f:
            source = f.read()
        for token in (r"\.fit\(", r"\.fit_transform\(", r"\.partial_fit\(",
                      r"joblib\.dump", r"pickle\.dump", r"import sklearn",
                      r"from sklearn"):
            self.assertIsNone(re.search(token, source),
                              "demo_personal.py must not contain " + token)


# ---------------------------------------------------------------------------
# 2. only heart-rate files are opened
# ---------------------------------------------------------------------------

class OnlyHeartRateFilesTests(unittest.TestCase):
    def test_decoy_files_are_never_opened(self):
        opened = []
        real_open = builtins.open
        real_read_csv = pd.read_csv

        def spy_open(file, *args, **kwargs):
            opened.append(os.fspath(file) if not isinstance(file, int) else "")
            return real_open(file, *args, **kwargs)

        def spy_read_csv(path, *args, **kwargs):
            opened.append(os.fspath(path))
            return real_read_csv(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 14), 5)
            with mock.patch.object(builtins, "open", spy_open), \
                    mock.patch.object(pd, "read_csv", spy_read_csv):
                personal.choose_day(tmp, TZ, min_windows=100)
                personal.load_local_day(tmp, date(2026, 7, 16), TZ)

            inside = [os.path.basename(p) for p in opened
                      if os.path.abspath(p).startswith(os.path.abspath(tmp))]
        self.assertTrue(inside, "expected some heart-rate files to be opened")
        for name in inside:
            self.assertRegex(name, HR_NAME, name + " must not be opened")


# ---------------------------------------------------------------------------
# 3. UTC -> America/Toronto
# ---------------------------------------------------------------------------

class TimezoneTests(unittest.TestCase):
    def test_summer_day_starts_at_04_00_utc(self):
        start, end = personal.local_day_bounds(date(2026, 7, 16), TZ)
        self.assertEqual(datetime.fromtimestamp(start, timezone.utc),
                         datetime(2026, 7, 16, 4, 0, tzinfo=timezone.utc))
        self.assertEqual(end - start, 24 * 3600)

    def test_winter_day_starts_at_05_00_utc(self):
        start, _ = personal.local_day_bounds(date(2026, 12, 10), TZ)
        self.assertEqual(datetime.fromtimestamp(start, timezone.utc),
                         datetime(2026, 12, 10, 5, 0, tzinfo=timezone.utc))

    def test_readings_land_on_the_right_local_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 15), 3, decoys=False)
            day15, _ = personal.load_local_day(tmp, date(2026, 7, 15), TZ)
            day16, opened = personal.load_local_day(tmp, date(2026, 7, 16), TZ)

        boundary = int(datetime(2026, 7, 16, 4, 0, tzinfo=timezone.utc).timestamp())
        self.assertEqual(opened, 2)  # a Toronto day spans two UTC files
        self.assertTrue((day16["epoch"] >= boundary).all())
        self.assertTrue((day15["epoch"] < boundary).all())
        # 03:59:57Z is still the 15th locally; the first reading on/after
        # 04:00:00Z is the 16th.
        self.assertEqual(int(day15["epoch"].max()), boundary - 3)
        self.assertEqual(int(day16["epoch"].min()), boundary)

    def test_a_missing_second_file_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 16), 1, decoys=False)
            with self.assertRaises(personal.TakeoutError):
                personal.load_local_day(tmp, date(2026, 7, 16), TZ)

    def test_daylight_saving_days_are_not_24_hours(self):
        expected = {date(2026, 11, 1): (25, 2999),    # clocks go back
                    date(2026, 3, 8): (23, 2759),     # clocks go forward
                    date(2026, 7, 16): (24, 2879)}    # ordinary day
        for day, (hours, n_windows) in expected.items():
            start, end = personal.local_day_bounds(day, TZ)
            self.assertEqual((end - start) // 3600, hours, str(day))
            self.assertEqual(len(personal.window_grid(day, TZ)), n_windows, str(day))

    def test_waking_flag_uses_local_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 16), 2, decoys=False)
            samples, _ = personal.load_local_day(tmp, date(2026, 7, 16), TZ)
            windows = personal.build_windows(samples, date(2026, 7, 16), TZ)
        # 08:00-23:00 local = 15 hours = 1800 half-minute starts
        self.assertEqual(int(windows["waking"].sum()), 1800)


# ---------------------------------------------------------------------------
# 4. rule baseline and notes
# ---------------------------------------------------------------------------

class RuleAndNotesTests(unittest.TestCase):
    def test_threshold_is_fitted_on_training_data_only(self):
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"HR_mean": np.r_[rng.normal(60, 3, 100),
                                               rng.normal(90, 3, 100)]})
        y = np.r_[np.zeros(100, int), np.ones(100, int)]
        rule = wesad.HRMeanThreshold().fit(train, y)
        self.assertTrue(70 < rule.threshold_ < 80)
        held_out = pd.DataFrame({"HR_mean": [55.0, 95.0]})
        self.assertEqual(list(rule.predict(held_out)), [0, 1])

    def test_three_class_rule_never_predicts_amusement(self):
        train = pd.DataFrame({"HR_mean": [60.0, 62, 61, 90, 92, 91, 70, 71]})
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3])
        rule = wesad.HRMeanThreshold(three_class=True).fit(train, y)
        self.assertNotIn(3, set(rule.predict(train)))

    def test_degenerate_amusement_day_gets_the_plain_note(self):
        shares = {"baseline": 0.2, "stress": 0.05, "amusement": 0.75}
        text = " ".join(personal.notes_for(shares, worst_ood_share=0.4))
        self.assertIn("cannot separate amusement from baseline", text)
        self.assertIn("OUT OF DISTRIBUTION", text)
        self.assertIn("NOT interpretable", text)

    def test_ordinary_in_range_day_gets_no_warning_notes(self):
        shares = {"baseline": 0.5, "stress": 0.3, "amusement": 0.2}
        self.assertEqual(personal.notes_for(shares, worst_ood_share=0.05), [])


# ---------------------------------------------------------------------------
# 5. the demo overwrites nothing
# ---------------------------------------------------------------------------

class NoOverwriteTests(unittest.TestCase):
    def test_full_demo_run_changes_no_file_in_the_project(self):
        roots = [os.path.join(PROJECT_DIR, "results"),
                 os.path.join(PROJECT_DIR, "models"),
                 os.path.join(PROJECT_DIR, "scripts")]
        singles = [os.path.join(PROJECT_DIR, name)
                   for name in ("RESULTS.md", "PRE_DECLARED.md", "README.md",
                                "requirements.txt")]
        before = snapshot(roots, singles)
        listing_before = project_listing()

        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 14), 5)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_demo.main(["--takeout", tmp, "--day", "2026-07-16"])
            text = out.getvalue()

        self.assertEqual(snapshot(roots, singles), before,
                         "the demo modified a file in results/, models/ or scripts/")
        self.assertEqual(project_listing(), listing_before,
                         "the demo created or removed a file")
        self.assertEqual(code, 0, text[-600:])
        self.assertIn(run_demo.BANNER, text)
        self.assertIn("matches saved scores_hr_only.json", text)
        self.assertIn("matches saved scores_full.json", text)
        self.assertIn("Files written: none", text)
        # day-level output only: no per-reading timestamps on screen
        self.assertNotRegex(text, r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")

    def test_masked_date_does_not_print_the_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_takeout(tmp, date(2026, 7, 14), 5)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_demo.main(["--takeout", tmp, "--day", "2026-07-16",
                               "--mask-date"])
        self.assertIn("Day #1 (date hidden)", out.getvalue())
        self.assertNotIn("2026-07-16", out.getvalue())


# ---------------------------------------------------------------------------
# 6. Amendment 1, labelling, wording and --save
# ---------------------------------------------------------------------------

def run_on_synthetic(extra_args=(), out_dir=None):
    with tempfile.TemporaryDirectory() as tmp:
        make_takeout(tmp, date(2026, 7, 14), 5)
        out = io.StringIO()
        patches = [mock.patch.object(run_demo, "OUT_DIR", out_dir)] if out_dir else []
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(out))
            code = run_demo.main(["--takeout", tmp, "--day", "2026-07-16",
                                  *extra_args])
    return code, out.getvalue()


class AmendmentAndWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code, cls.text = run_on_synthetic()  # one demo run for the class

    def test_binary_has_no_dominance_flag_but_keeps_the_ood_flag(self):
        shares = {"non-stress": 0.99, "stress": 0.01}
        self.assertEqual(personal.notes_for(shares, 0.0, dominance=False), [])
        text = " ".join(personal.notes_for(shares, 0.4, dominance=False))
        self.assertIn("OUT OF DISTRIBUTION", text)
        self.assertNotIn("DEGENERATE", text)

    def test_three_class_keeps_the_dominance_flag(self):
        shares = {"baseline": 0.1, "stress": 0.1, "amusement": 0.8}
        self.assertIn("DEGENERATE", " ".join(personal.notes_for(shares, 0.0)))

    def test_demo_prints_amendment_labels_and_comparison_note(self):
        code, text = self.code, self.text
        self.assertEqual(code, 0, text[-500:])
        self.assertIn("Amendment 1", text)
        self.assertIn("AFTER seeing", text)
        self.assertIn("2026-10-30 demo output", text)
        self.assertIn("single-run point estimates", text)
        self.assertIn("NO confidence interval", text)
        self.assertIn("does not claim either one is better", text)
        self.assertIn("[Model: BINARY SVM, trained in memory on WESAD", text)
        self.assertIn("[Model: SAVED 3-CLASS SVM, models/svm_hr_only.joblib", text)
        self.assertIn("default hyperparameters", text)

    def test_demo_never_says_one_model_beats_another_or_chance_level(self):
        text = self.text
        for phrase in ("beats", "outperform", "wins", "at chance", "chance level"):
            self.assertNotIn(phrase, text.lower())

    def test_binary_block_never_prints_a_degenerate_note(self):
        text = self.text
        start = text.index("[Model: BINARY SVM")
        end = text.index("[Model: SAVED 3-CLASS SVM")
        self.assertNotIn("DEGENERATE", text[start:end])

    def test_spacing_is_printed_next_to_the_wesad_resolution(self):
        text = self.text
        self.assertIn("Resolution mismatch", text)
        self.assertIn("WESAD training : beat-to-beat", text)
        self.assertIn("This Fitbit day: device-smoothed", text)
        self.assertIn("fewer values per window", text)


class SaveCopyTests(unittest.TestCase):
    def test_save_writes_one_masked_copy_only_under_the_out_dir(self):
        listing_before = project_listing()
        with tempfile.TemporaryDirectory() as out_dir:
            code, text = run_on_synthetic(["--save"], out_dir=out_dir)
            written = os.listdir(out_dir)
            self.assertEqual(len(written), 1)
            self.assertRegex(written[0], r"^demo_output_\d{8}-\d{6}\.txt$")
            with open(os.path.join(out_dir, written[0]), encoding="utf-8") as f:
                copy = f.read()
        self.assertEqual(code, 0)
        self.assertEqual(copy, text, "the saved copy must equal what was printed")
        self.assertIn("Day #1 (date hidden)", copy)
        self.assertNotIn("2026-07-16", copy)
        self.assertNotIn("2026-10-30", copy)   # the amendment must not leak a day
        self.assertIn("AFTER seeing", copy)
        self.assertIn("this --save copy only", copy)
        self.assertEqual(project_listing(), listing_before)

    def test_without_save_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as out_dir:
            run_on_synthetic([], out_dir=out_dir)
            self.assertEqual(os.listdir(out_dir), [])


if __name__ == "__main__":
    unittest.main()
