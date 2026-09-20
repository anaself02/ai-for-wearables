"""Tests for the arousal card - the WESAD model applied to live data.

Three properties matter here, and each is worth stating plainly because each
protects something that would otherwise fail quietly:

  1. A prediction is never shown without its caveat. The model is trained on
     laboratory stress and is measurably out of distribution on this device's
     data, so a bare state label would overclaim. This is enforced twice: the
     `Assessment.predicted` constructor refuses to build without the
     out-of-distribution evidence, and the template renders the caveat in the
     same branch as the headline.

  2. Everything degrades to an explanation, never to a crash. A missing model
     file, an uninstalled ML stack, a scikit-learn that did not write the
     pickle, or a day with too little heart-rate data must all produce a card
     that says so.

  3. The feature computation matches the training project. `basic_features`
     here is a reimplementation of the one in wearable-ml/scripts/features.py;
     a golden-value test pins the two together so a drift fails here rather
     than silently changing predictions.

Nothing in this file loads the real model or calls the real API.
"""

from datetime import date
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from health import arousal

User = get_user_model()

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


def hr_point(hour, minute, second, bpm):
    """One data point in the shape the Google Health API really returns.

    beatsPerMinute is a string, and zero-valued time fields are omitted
    rather than sent as 0 - both confirmed against live responses.
    """
    time_part = {}
    if hour:
        time_part["hours"] = hour
    if minute:
        time_part["minutes"] = minute
    if second:
        time_part["seconds"] = second

    return {
        "heartRate": {
            "beatsPerMinute": str(bpm),
            "sampleTime": {
                "civilTime": {
                    "date": {"year": 2026, "month": 9, "day": 18},
                    "time": time_part,
                }
            },
        }
    }


def a_day_of_points(bpm=62, seconds_apart=3, count=1200):
    """A run of readings from midnight, spaced like the real device."""
    points = []
    for index in range(count):
        total = index * seconds_apart
        points.append(hr_point(total // 3600, (total // 60) % 60, total % 60,
                               bpm + (index % 5)))
    return points


def fake_model(predicted_class=2):
    """A stand-in for the scikit-learn pipeline: one class for every window."""
    model = mock.MagicMock()
    model.predict.side_effect = lambda frame: [predicted_class] * len(frame)
    return model


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
        # drifted and predictions on live data no longer mean what the
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


class ParseSamplesTests(TestCase):
    def test_parses_string_bpm_and_omitted_time_fields(self):
        samples = arousal.parse_samples([hr_point(0, 0, 0, 58)])

        self.assertEqual(len(samples), 1)
        moment, bpm = samples[0]
        self.assertEqual((moment.hour, moment.minute, moment.second), (0, 0, 0))
        self.assertEqual(bpm, 58.0)

    def test_skips_points_with_no_value(self):
        broken = {"heartRate": {"sampleTime": {"civilTime": {
            "date": {"year": 2026, "month": 9, "day": 18}, "time": {}}}}}
        self.assertEqual(arousal.parse_samples([broken]), [])


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


class AssessTests(TestCase):
    """Property 2: every failure becomes an explanation, not an exception."""

    def _artifacts(self, model=None):
        return (model or fake_model(), FEATURE_NAMES, DISTRIBUTION)

    def test_predicts_from_a_full_day_of_readings(self):
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=self._artifacts()):
            assessment = arousal.assess(a_day_of_points(), DAY)

        self.assertTrue(assessment.available)
        self.assertEqual(assessment.headline, "Elevated arousal signal")
        self.assertGreater(assessment.n_windows, 0)
        self.assertEqual(len(assessment.ood_rows), len(FEATURE_NAMES))
        self.assertTrue(assessment.caveat)

    def test_features_are_passed_in_the_fitted_column_order(self):
        # Feeding the right five numbers in the wrong order raises nothing -
        # it just returns a different answer - so the order is asserted.
        model = fake_model()
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=self._artifacts(model)):
            arousal.assess(a_day_of_points(), DAY)

        frame = model.predict.call_args[0][0]
        self.assertEqual(list(frame.columns), FEATURE_NAMES)

    def test_missing_model_file_degrades_with_a_readable_reason(self):
        with override_settings(WEARABLE_ML_DIR="/nonexistent/wearable-ml"):
            assessment = arousal.assess(a_day_of_points(), DAY)

        self.assertFalse(assessment.available)
        self.assertIn("No trained model", assessment.reason)

    def test_uninstalled_ml_stack_degrades_with_a_readable_reason(self):
        with mock.patch.object(
                arousal, "load_artifacts",
                side_effect=arousal.ArousalUnavailable(
                    "scikit-learn is not installed in this environment")):
            assessment = arousal.assess(a_day_of_points(), DAY)

        self.assertFalse(assessment.available)
        self.assertIn("scikit-learn is not installed", assessment.reason)

    def test_wrong_sklearn_version_refuses_rather_than_migrating(self):
        with mock.patch.object(arousal, "_read_json", return_value={
                "feature_names_in_order": FEATURE_NAMES}), \
             mock.patch("os.path.exists", return_value=True), \
             mock.patch("sklearn.__version__", "1.9.9"):
            assessment = arousal.assess(a_day_of_points(), DAY)

        self.assertFalse(assessment.available)
        self.assertIn("Refusing to load", assessment.reason)

    def test_no_readings_degrades_with_a_readable_reason(self):
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=self._artifacts()):
            assessment = arousal.assess([], DAY)

        self.assertFalse(assessment.available)
        self.assertIn("No heart-rate readings", assessment.reason)

    def test_too_few_readings_for_any_window_degrades(self):
        # Nine readings three seconds apart: no 60 s window reaches the
        # ten-reading minimum.
        sparse = a_day_of_points(count=9)
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=self._artifacts()):
            assessment = arousal.assess(sparse, DAY)

        self.assertFalse(assessment.available)
        self.assertIn("Not enough heart-rate data", assessment.reason)

    def test_an_unforeseen_error_degrades_instead_of_raising(self):
        exploding = mock.MagicMock()
        exploding.predict.side_effect = RuntimeError("kaboom")
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=self._artifacts(exploding)):
            assessment = arousal.assess(a_day_of_points(), DAY)

        self.assertFalse(assessment.available)
        self.assertIn("kaboom", assessment.reason)


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


class CardRenderingTests(TestCase):
    """The card and page as a visitor actually sees them."""

    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create_user(
            username="tester@example.com", password="pw")
        self.client.force_login(self.user)

    def _dashboard_with(self, assessment):
        """Render the dashboard with a given assessment already cached."""
        return self.client.get(
            reverse("health:index"),
            # The overview needs a credential; without one it redirects to
            # the reconnect page, so the card is rendered on its own instead.
        )

    def test_card_shows_label_and_caveat_together(self):
        assessment = arousal.Assessment.predicted(
            day=DAY, dominant=2, shares={2: 1.0}, waking_shares={2: 1.0},
            n_windows=120, n_waking_windows=100, mean_bpm=64.0,
            ood_rows=[{"feature": "HR_std", "share_outside": 0.52,
                       "median_shift": -1.85, "training_min": 1.5,
                       "training_max": 31.9, "today_median": 2.1}],
        )

        from django.template.loader import render_to_string
        html = render_to_string(
            "health/_arousal_card.html",
            {"arousal": assessment, "arousal_day": DAY},
        )

        # The label is present...
        self.assertIn("Elevated arousal signal", html)
        # ...and so is every piece of context that must accompany it.
        self.assertIn("low confidence", html)
        self.assertIn("not a clinical or validated measure", html)
        self.assertIn("How this works", html)
        self.assertIn("HR_std", html)

    def test_card_without_an_assessment_offers_to_compute(self):
        from django.template.loader import render_to_string
        html = render_to_string(
            "health/_arousal_card.html",
            {"arousal": None, "arousal_day": DAY},
        )

        self.assertIn("Not computed yet", html)
        self.assertIn("How this works", html)
        # No state label may appear when there is nothing to label.
        self.assertNotIn("Elevated arousal signal", html)

    def test_card_shows_the_reason_when_unavailable(self):
        from django.template.loader import render_to_string
        html = render_to_string(
            "health/_arousal_card.html",
            {"arousal": arousal.Assessment.unavailable(
                DAY, "No trained model at C:/models/svm_hr_only.joblib."),
             "arousal_day": DAY},
        )

        self.assertIn("No assessment available", html)
        self.assertIn("No trained model", html)
        self.assertNotIn("Elevated arousal signal", html)

    def test_methodology_page_renders_the_model_card(self):
        card = {
            "algorithm": "SVM (RBF kernel) inside a StandardScaler pipeline",
            "honest_summary": "Not a validated real-world stress detector.",
            "trained_on": {"dataset": "WESAD", "signals": "wrist heart rate",
                           "n_subjects": 15, "n_windows": 1015,
                           "window_seconds": 60},
            "classes": {"1": "baseline", "2": "stress", "3": "amusement"},
            "validation": "leave-one-subject-out cross-validation (15 folds)",
            "scores": {"hr_only": {"svm": {"accuracy": 0.549, "f1_macro": 0.499}}},
            "always_guess_largest_class": {"accuracy": 0.549, "f1_macro": 0.236},
            "limitations": ["Trained on laboratory-induced stress."],
        }

        with mock.patch.object(arousal, "load_model_card", return_value=card):
            response = self.client.get(reverse("health:arousal_methodology"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Not a validated real-world stress detector")
        self.assertContains(response, "Trained on laboratory-induced stress")
        self.assertContains(response, "0.499")
        self.assertContains(response, "no ground truth")

    def test_methodology_page_survives_a_missing_model_card(self):
        with mock.patch.object(arousal, "load_model_card", return_value=None):
            response = self.client.get(reverse("health:arousal_methodology"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "could not be read")

    def test_compute_endpoint_rejects_get(self):
        # A GET must not spend the work; it just returns to the dashboard.
        response = self.client.get(reverse("health:arousal_compute"))
        self.assertEqual(response.status_code, 302)


class CachingTests(TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.user = User.objects.create(username="tester@example.com")
        self.other = User.objects.create(username="someone-else@example.com")

    def test_nothing_is_computed_when_only_reading_the_cache(self):
        self.assertIsNone(arousal.cached_assessment(self.user, DAY))

    def test_a_computed_assessment_is_reused(self):
        client = mock.MagicMock()
        with mock.patch.object(arousal, "load_artifacts",
                               return_value=(fake_model(), FEATURE_NAMES,
                                             DISTRIBUTION)), \
             mock.patch("health.cache.fetch_intraday") as fetch:
            fetch.return_value = mock.MagicMock(points=a_day_of_points())

            first = arousal.compute_assessment(client, self.user, DAY)
            second = arousal.compute_assessment(client, self.user, DAY)

        self.assertTrue(first.available)
        self.assertEqual(first.headline, second.headline)
        # One fetch, not two: the second call came from the cache.
        self.assertEqual(fetch.call_count, 1)

    def test_assessments_are_scoped_per_user(self):
        stored = arousal.store_unavailable(self.user, DAY, "mine only")

        self.assertEqual(arousal.cached_assessment(self.user, DAY).reason,
                         stored.reason)
        self.assertIsNone(arousal.cached_assessment(self.other, DAY))
