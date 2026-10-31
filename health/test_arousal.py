"""Tests for the arousal assessment: the WESAD model applied to one person's day.

Four properties matter here, and each is worth stating plainly because each
protects something that would otherwise fail quietly:

  1. A prediction is never shown without its caveat. The model is trained on
     laboratory stress and is measurably out of distribution on a consumer
     device's data, so a bare state label would overclaim. This is enforced
     twice: the `Assessment.predicted` constructor refuses to build without the
     out-of-distribution evidence, and the template renders the caveat in the
     same branch as the headline (people/test_arousal_page.py).

  2. Everything degrades to an explanation, never to a crash. A missing model
     file, an uninstalled ML stack, a scikit-learn that did not write the
     pickle, or a day with too little heart-rate data must all produce a result
     that says so.

  3. The feature computation matches the training project. `basic_features`
     here is a reimplementation of the one in wearable-ml/scripts/features.py;
     a golden-value test pins the two together so a drift fails here rather
     than silently changing predictions.

  4. Nothing learns from the data being scored. The models were fitted on WESAD
     only; the end-to-end test below runs the real model under a guard that
     makes every fit and save raise.

Except for the one end-to-end test, nothing here loads the real model. Every
reading is synthetic.
"""

import hashlib
import os
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from django.conf import settings
from django.test import TestCase, override_settings

from health import arousal
from people import ingest, repository
from people.fitguard import fit_guard
from people.models import Person
from people.sources import SourceFile
from people.testing import hr_csv_bytes, utc_day_file

DAY = date(2026, 9, 18)

# The five features the heart-rate-only model consumes, in fitted order.
FEATURE_NAMES = ["HR_mean", "HR_std", "HR_min", "HR_max", "HR_slope"]

# A stand-in for models/hr_training_distribution.json.
DISTRIBUTION = {
    "n_windows": 1015,
    "features": {
        "HR_mean": {"mean": 76.7, "std": 11.6, "min": 52.4, "median": 75.0, "max": 131.4},
        "HR_std": {"mean": 11.2, "std": 5.3, "min": 1.5, "median": 10.4, "max": 31.9},
        "HR_min": {"mean": 55.8, "std": 9.0, "min": 40.0, "median": 55.7, "max": 112.9},
        "HR_max": {"mean": 106.2, "std": 19.5, "min": 57.3, "median": 103.8, "max": 147.7},
        "HR_slope": {"mean": 0.0, "std": 0.25, "min": -2.0, "median": 0.0, "max": 2.6},
    },
}


def fake_model(predicted_class=2):
    """A stand-in for the scikit-learn pipeline: one class for every window."""
    model = mock.MagicMock()
    model.predict.side_effect = lambda frame: [predicted_class] * len(frame)
    return model


def fake_artifacts(model=None):
    return (model or fake_model(), FEATURE_NAMES, DISTRIBUTION)


def make_person(slug="subject", days=(date(2026, 9, 17), date(2026, 9, 18)),
                spacing=3, seed=1, base=62, spread=10, tz="America/Toronto"):
    """A synthetic person with full UTC days of readings at the given spacing."""
    person = Person.objects.create(slug=slug, display_name=slug.title(),
                                   device_label="Fitbit Air",
                                   adapter_key="fitbit-takeout", timezone=tz)
    files = [SourceFile(*utc_day_file(d, spacing=spacing, seed=seed, base=base,
                                      spread=spread)) for d in days]
    ingest.commit_plan(ingest.build_plan(person, files))
    return person


def make_short_person(minutes, spacing=3, slug="short"):
    person = Person.objects.create(slug=slug, display_name="Short",
                                   device_label="Fitbit Air",
                                   adapter_key="fitbit-takeout")
    start = datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc)
    data = hr_csv_bytes(start, start + timedelta(minutes=minutes), spacing=spacing)
    ingest.commit_plan(ingest.build_plan(
        person, [SourceFile("heart_rate_2026-09-18.csv", data)]))
    return person


class BasicFeaturesTests(TestCase):
    """Property 3: the feature code has not drifted from the training code."""

    def test_matches_training_implementation_on_known_values(self):
        # Golden values, produced by the training implementation itself:
        #
        #   wearable-ml/.venv/Scripts/python.exe -c \
        #     "import features; print(features.basic_features(
        #        [60.,62.,61.,65.,64.,63.,70.,68.], fs=1.0))"
        #
        # Regenerate them that way if the training code legitimately changes.
        # If they stop matching without that, the two implementations have
        # drifted and predictions on uploaded data no longer mean what the
        # training scores say they mean.
        #
        # Note std is the population standard deviation (numpy's default,
        # ddof=0), not the sample one - which is the sort of difference this
        # test exists to catch.
        values = [60.0, 62.0, 61.0, 65.0, 64.0, 63.0, 70.0, 68.0]
        result = arousal.basic_features(values)

        self.assertAlmostEqual(result["mean"], 64.125, places=10)
        self.assertAlmostEqual(result["std"], 3.2185982973959333, places=10)
        self.assertAlmostEqual(result["min"], 60.0, places=10)
        self.assertAlmostEqual(result["max"], 70.0, places=10)
        self.assertAlmostEqual(result["slope"], 1.20238095238095, places=10)

    def test_slope_is_zero_for_a_single_reading(self):
        self.assertEqual(arousal.basic_features([70.0])["slope"], 0.0)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            arousal.basic_features([])


class AssessmentConstructionTests(TestCase):
    """Property 1, enforced in the data model rather than by convention."""

    def test_a_prediction_cannot_be_built_without_its_caveat_evidence(self):
        with self.assertRaises(ValueError):
            arousal.Assessment.predicted(
                day=DAY, dominant=2, shares={2: 1.0}, waking_shares={2: 1.0},
                n_windows=10, n_waking_windows=10, mean_bpm=62.0,
                ood_rows=[],
            )

    def test_a_prediction_always_carries_a_caveat_and_low_confidence(self):
        assessment = arousal.Assessment.predicted(
            day=DAY, dominant=2, shares={2: 1.0}, waking_shares={2: 1.0},
            n_windows=10, n_waking_windows=10, mean_bpm=62.0,
            ood_rows=[{"feature": "HR_std", "share_outside": 0.52,
                       "median_shift": -1.85}],
        )

        self.assertTrue(assessment.caveat)
        self.assertIn("not a clinical or validated measure", assessment.caveat)
        self.assertEqual(assessment.confidence, "low confidence")
        # The plain-language label, never the raw class name.
        self.assertEqual(assessment.headline, "Elevated arousal signal")
        self.assertNotIn("stress", assessment.headline.lower())

    def test_unavailable_carries_a_reason_and_no_headline(self):
        assessment = arousal.Assessment.unavailable(DAY, "no model")
        self.assertFalse(assessment.available)
        self.assertEqual(assessment.reason, "no model")
        self.assertEqual(assessment.headline, "")
        self.assertEqual(assessment.caveat, "")


class NotesTests(TestCase):
    """The pre-declared 3-class display rules, printed with every label."""

    def test_a_dominant_amusement_day_gets_the_plain_notes(self):
        notes = " ".join(arousal.notes_for({1: 0.2, 2: 0.05, 3: 0.75}, 0.4))
        self.assertIn("OUT OF DISTRIBUTION", notes)
        self.assertIn("DEGENERATE", notes)
        self.assertIn("cannot separate amusement from baseline", notes)
        self.assertIn("NOT interpretable", notes)

    def test_an_ordinary_in_range_day_gets_no_warning_notes(self):
        self.assertEqual(arousal.notes_for({1: 0.5, 2: 0.3, 3: 0.2}, 0.05), [])

    def test_the_thresholds_are_the_predeclared_ones(self):
        self.assertEqual(arousal.OOD_FLAG_SHARE, 0.25)
        self.assertEqual(arousal.DOMINANT_CLASS_SHARE, 0.60)
        self.assertEqual(arousal.MIN_WINDOWS_TO_SCORE, 60)


class AssessTests(TestCase):
    """Property 2: every failure becomes an explanation, not an exception."""

    @classmethod
    def setUpTestData(cls):
        cls.person = make_person()

    def assess(self, person=None, day=DAY, artifacts=None):
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=artifacts or fake_artifacts()):
            return arousal.assess(person or self.person, day)

    def test_predicts_from_a_full_local_day_of_readings(self):
        assessment = self.assess(day=date(2026, 9, 17))

        self.assertTrue(assessment.available)
        self.assertEqual(assessment.headline, "Elevated arousal signal")
        self.assertEqual(assessment.n_windows, 2879)       # 24 h, 60 s / 30 s step
        self.assertEqual(assessment.n_waking_windows, 1800)  # 08:00-23:00 local
        self.assertEqual(len(assessment.ood_rows), len(FEATURE_NAMES))
        self.assertTrue(assessment.caveat)
        self.assertTrue(assessment.notes)

    def test_the_result_carries_device_spacing_and_timezone(self):
        assessment = self.assess(day=date(2026, 9, 17))
        self.assertEqual(assessment.device_names, ["Fitbit Air"])
        self.assertEqual(assessment.spacing["median"], 3.0)
        self.assertEqual(assessment.n_readings, 28800)
        self.assertEqual(assessment.timezone, "America/Toronto")
        self.assertEqual(assessment.median_values_per_window, 20.0)
        self.assertAlmostEqual(assessment.wesad_values_per_window, 76.7)

    def test_features_are_passed_in_the_fitted_column_order(self):
        # Feeding the right five numbers in the wrong order raises nothing -
        # it just returns a different answer - so the order is asserted.
        model = fake_model()
        self.assess(day=date(2026, 9, 17), artifacts=fake_artifacts(model))
        frame = model.predict.call_args[0][0]
        self.assertEqual(list(frame.columns), FEATURE_NAMES)

    def test_a_person_is_required(self):
        for bad in ("subject", None, 1):
            with self.assertRaises(TypeError):
                arousal.assess(bad, DAY)

    def test_missing_model_file_degrades_with_a_readable_reason(self):
        with override_settings(WEARABLE_ML_DIR="/nonexistent/wearable-ml"):
            assessment = arousal.assess(self.person, date(2026, 9, 17))
        self.assertFalse(assessment.available)
        self.assertIn("No trained model", assessment.reason)

    def test_uninstalled_ml_stack_degrades_with_a_readable_reason(self):
        with mock.patch.object(
                arousal, "load_artifacts",
                side_effect=arousal.ArousalUnavailable(
                    "scikit-learn is not installed in this environment")):
            assessment = arousal.assess(self.person, date(2026, 9, 17))
        self.assertFalse(assessment.available)
        self.assertIn("scikit-learn is not installed", assessment.reason)

    def test_wrong_sklearn_version_refuses_rather_than_migrating(self):
        with mock.patch.object(arousal, "_read_json", return_value={
                "feature_names_in_order": FEATURE_NAMES}), \
             mock.patch("os.path.exists", return_value=True), \
             mock.patch("sklearn.__version__", "1.9.9"):
            assessment = arousal.assess(self.person, date(2026, 9, 17))
        self.assertFalse(assessment.available)
        self.assertIn("Refusing to load", assessment.reason)

    def test_a_day_with_no_readings_degrades_with_a_readable_reason(self):
        assessment = self.assess(day=date(2026, 1, 1))
        self.assertFalse(assessment.available)
        self.assertIn("No heart-rate readings", assessment.reason)

    def test_too_few_windows_degrades_and_says_how_many_are_needed(self):
        short = make_short_person(minutes=10)           # ~19 windows
        assessment = self.assess(short, date(2026, 9, 18))
        self.assertFalse(assessment.available)
        self.assertIn("Too few scorable windows", assessment.reason)
        self.assertIn("at least 60", assessment.reason)

    def test_readings_too_sparse_for_any_window_degrades(self):
        sparse = make_short_person(minutes=30, spacing=10, slug="sparse")
        assessment = self.assess(sparse, date(2026, 9, 18))   # 6 readings / minute
        self.assertFalse(assessment.available)
        self.assertIn("Not enough heart-rate data", assessment.reason)

    def test_an_unforeseen_error_degrades_instead_of_raising(self):
        exploding = mock.MagicMock()
        exploding.predict.side_effect = RuntimeError("kaboom")
        assessment = self.assess(day=date(2026, 9, 17),
                                 artifacts=fake_artifacts(exploding))
        self.assertFalse(assessment.available)
        self.assertIn("kaboom", assessment.reason)

    def test_a_daylight_saving_day_is_scored_over_its_real_25_hours(self):
        person = make_person("dst", days=(date(2026, 11, 1), date(2026, 11, 2)))
        assessment = self.assess(person, date(2026, 11, 1))
        self.assertTrue(assessment.available)
        self.assertEqual(assessment.n_windows, 2999)       # 25 h, not 24


class ScorableDaysTests(TestCase):
    def test_days_are_listed_newest_first_with_window_counts(self):
        person = make_person()
        days = arousal.scorable_days(person)
        self.assertEqual([d["day"] for d in days],
                         [date(2026, 9, 18), date(2026, 9, 17), date(2026, 9, 16)])
        by_day = {d["day"]: d for d in days}
        self.assertEqual(by_day[date(2026, 9, 17)]["windows"], 2879)
        self.assertEqual(by_day[date(2026, 9, 17)]["readings"], 28800)
        self.assertTrue(all(d["scorable"] for d in days))

    def test_the_default_is_the_most_recent_scorable_day(self):
        person = make_person()
        self.assertEqual(arousal.default_day(arousal.scorable_days(person)),
                         date(2026, 9, 18))

    def test_a_day_with_too_few_windows_is_not_scorable_or_default(self):
        short = make_short_person(minutes=10)
        days = arousal.scorable_days(short)
        self.assertEqual(len(days), 1)
        self.assertFalse(days[0]["scorable"])
        self.assertIsNone(arousal.default_day(days))

    def test_a_person_is_required(self):
        with self.assertRaises(TypeError):
            arousal.scorable_days("subject")


class OutOfDistributionTests(TestCase):
    def test_flags_values_outside_the_training_range(self):
        # HR_std of 0.2 is below the training minimum of 1.5 in every window.
        rows = [{"HR_mean": 62.0, "HR_std": 0.2, "HR_min": 60.0,
                 "HR_max": 64.0, "HR_slope": 0.0} for _ in range(10)]

        report = arousal.out_of_distribution(rows, FEATURE_NAMES, DISTRIBUTION)
        by_feature = {row["feature"]: row for row in report}

        self.assertEqual(by_feature["HR_std"]["share_outside"], 1.0)
        self.assertEqual(by_feature["HR_mean"]["share_outside"], 0.0)
        self.assertLess(by_feature["HR_std"]["median_shift"], -2.0)


MODELS_DIR = os.path.join(settings.WEARABLE_ML_DIR, "models")


def fingerprint(directory):
    result = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                result[name] = (os.path.getsize(path), os.path.getmtime(path),
                                hashlib.sha256(handle.read()).hexdigest())
    return result


@unittest.skipUnless(os.path.exists(os.path.join(MODELS_DIR, "svm_hr_only.joblib")),
                     "the wearable-ml model files are not present")
class RealModelEndToEndTests(TestCase):
    """Property 4: the real WESAD-trained model scores a day and learns nothing."""

    def test_the_real_model_scores_a_day_under_the_no_fit_guard_and_is_untouched(self):
        person = make_person("real")
        before = fingerprint(MODELS_DIR)
        with fit_guard():
            assessment = arousal.assess(person, date(2026, 9, 17))
        self.assertTrue(assessment.available, assessment.reason)
        self.assertEqual(assessment.n_windows, 2879)
        self.assertEqual(sum(assessment.shares.values()), 1.0)
        self.assertTrue(assessment.caveat)
        self.assertEqual(fingerprint(MODELS_DIR), before,
                         "scoring must not modify any file in the model folder")

