"""eval/ tests: leakage, grouping, determinism, bootstrap, data access, reproduction.

    python -m unittest discover -s eval/tests -t .      (from wearable-ml)

Fast: small subject subsets and small grids, except the reproduction check
(15 SVM fits on the real feature table, a few seconds).
"""

import builtins
import importlib.util
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from eval import config, data, folds, metrics, models, nested, run_eval
from eval.reproduce import EXPECTED_FULL_SVM_MACRO_F1, full_svm_loso

SMALL_GRIDS = {"svm": {"clf__C": [1, 10]}, "rf": {"clf__max_depth": [None, 5]},
               "mlp": {"clf__alpha": [1e-4, 1e-2]}}


def synthetic_table(n_subjects=6, per_condition=8, seed=0):
    """Synthetic windows. HR_min encodes the subject (subject number x 1000) so a
    fitted model's training rows can be traced back to subjects."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2, 2 + n_subjects):
        t = 0.0
        for label, name, shift in ((1, "baseline", 0), (2, "stress", 15), (3, "amusement", 5)):
            for _ in range(per_condition):
                rows.append({"subject_id": "S{}".format(s), "label": label, "label_name": name,
                             "window_start_s": t,
                             "HR_mean": 70 + shift + rng.normal(0, 4),
                             "HR_std": 5 + rng.normal(0, 1), "HR_min": s * 1000.0,
                             "HR_max": 100 + shift + rng.normal(0, 4),
                             "HR_slope": rng.normal(0, 0.2)})
                t += 30.0
    return pd.DataFrame(rows)


class FoldTests(unittest.TestCase):
    SUBJECTS = ["S{}".format(i) for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]

    def test_no_subject_is_in_both_train_and_test(self):
        outer = folds.outer_folds(self.SUBJECTS)
        self.assertEqual(len(outer), 15)
        self.assertEqual(sorted(t for t, _ in outer), sorted(self.SUBJECTS))
        for test, train in outer:
            self.assertNotIn(test, train)
            self.assertEqual(len(train), 14)

    def test_inner_folds_are_grouped_by_subject_and_cover_the_training_set(self):
        for seed in config.SEEDS:
            for test, train in folds.outer_folds(self.SUBJECTS):
                inner = folds.inner_folds(train, seed)
                self.assertEqual(len(inner), config.INNER_FOLDS)
                flat = [s for group in inner for s in group]
                self.assertEqual(sorted(flat), sorted(train))        # each exactly once
                self.assertNotIn(test, flat)
                self.assertTrue(all(len(g) == 2 for g in inner))

    def test_inner_row_splits_never_mix_a_subject(self):
        table = synthetic_table()
        groups = table["subject_id"].to_numpy()
        subjects = sorted(set(groups), key=folds.subject_key)
        test, train = folds.outer_folds(subjects)[0]
        train_index = np.flatnonzero(np.isin(groups, train))
        for fit, val in folds.inner_splits(groups, train_index, folds.inner_folds(train, 0, 3)):
            fit_s, val_s = set(groups[train_index][fit]), set(groups[train_index][val])
            self.assertFalse(fit_s & val_s)
            self.assertNotIn(test, fit_s | val_s)
            self.assertEqual(len(fit) + len(val), len(train_index))

    def test_inner_folds_depend_on_the_seed_only(self):
        train = self.SUBJECTS[1:]
        self.assertEqual(folds.inner_folds(train, 3), folds.inner_folds(train, 3))
        self.assertNotEqual(folds.inner_folds(train, 0), folds.inner_folds(train, 1))

    def test_folds_json_records_outer_and_inner_subject_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = folds.write_folds(self.SUBJECTS, path=os.path.join(tmp, "folds.json"))
            import json
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertEqual(len(saved["outer"]), 15)
        self.assertEqual(sorted(saved["inner"]), ["0", "1", "2", "3", "4"])


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.table = synthetic_table()

    def test_the_test_subject_never_reaches_any_fit(self):
        seen = []

        def check(test_subject, fitted, train_index, test_index):
            seen.append(test_subject)
            scaler = fitted.named_steps["scaler"]
            # HR_min column = subject number x 1000: the scaler's training range
            # must not contain the test subject's code unless another subject has it.
            train_codes = set(self.table["HR_min"].iloc[train_index])
            self.assertNotIn(int(test_subject[1:]) * 1000.0, train_codes)
            self.assertAlmostEqual(scaler.mean_[2], np.mean(sorted(train_codes)), places=6)

        for model in config.MODELS:
            nested.run(self.table, "binary", model, 0, grid=SMALL_GRIDS[model],
                       inner_k=3, on_fitted=check)
        self.assertEqual(len(seen), 3 * 6)

    def test_the_scaler_is_fitted_on_training_rows_only(self):
        def check(test_subject, fitted, train_index, test_index):
            X = self.table[config.HR_FEATURES].to_numpy(dtype=float)
            np.testing.assert_allclose(fitted.named_steps["scaler"].mean_,
                                       X[train_index].mean(axis=0))
            self.assertFalse(np.allclose(fitted.named_steps["scaler"].mean_, X.mean(axis=0)))

        nested.run(self.table, "three", "svm", 0, grid=SMALL_GRIDS["svm"], inner_k=3,
                   on_fitted=check)

    def test_inner_selection_scores_scalers_fitted_on_inner_training_rows(self):
        from sklearn.preprocessing import StandardScaler
        fitted_sizes = []
        real_fit = StandardScaler.fit

        def spy(self_, X, y=None, sample_weight=None):
            fitted_sizes.append(len(X))
            return real_fit(self_, X, y)

        with mock.patch.object(StandardScaler, "fit", spy):
            nested.run(self.table, "binary", "svm", 0, grid={"clf__C": [1]}, inner_k=5,
                       subjects=["S2", "S3", "S4", "S5", "S6", "S7"])
        rows_per_subject = 24
        # per outer fold: 5 inner fits on 4 subjects, then the refit on 5 subjects
        self.assertEqual(sorted(set(fitted_sizes)), [4 * rows_per_subject, 5 * rows_per_subject])

    def test_oversampling_happens_only_inside_fit(self):
        X = np.random.default_rng(0).normal(size=(60, 5))
        y = np.array([0] * 50 + [1] * 10)
        mlp = models.OversampledMLP(random_state=0).fit(X, y)
        self.assertEqual(mlp.n_oversampled_, 40)
        self.assertEqual(len(mlp.predict(X[:7])), 7)
        self.assertEqual(mlp.predict_proba(X[:7]).shape, (7, 2))

    def test_threshold_baseline_is_fitted_on_training_rows_and_matches_the_demo(self):
        demo_path = os.path.join(config.ML_DIR, "demo", "demo_wesad.py")
        if not os.path.exists(demo_path):
            self.skipTest("demo/demo_wesad.py not present")
        spec = importlib.util.spec_from_file_location("demo_wesad_pin", demo_path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        rng = np.random.default_rng(1)
        x, positive = rng.normal(80, 10, 300), rng.random(300) > 0.7
        self.assertEqual(models.best_threshold(x, positive), demo.best_threshold(x, positive))
        self.assertEqual(config.HR_FEATURES[0], "HR_mean")


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_predictions(self):
        table = synthetic_table(n_subjects=5)
        for model in config.MODELS + config.BASELINES:
            a = nested.run(table, "binary", model, 2, grid=SMALL_GRIDS.get(model, {}), inner_k=2)
            b = nested.run(table, "binary", model, 2, grid=SMALL_GRIDS.get(model, {}), inner_k=2)
            self.assertEqual([(p["y_pred"], p["score"]) for p in a["predictions"]],
                             [(p["y_pred"], p["score"]) for p in b["predictions"]], model)
            self.assertEqual([f["chosen"] for f in a["folds"]], [f["chosen"] for f in b["folds"]])

    def test_predictions_carry_subject_condition_fold_seed_model_score(self):
        table = synthetic_table(n_subjects=4)
        result = nested.run(table, "binary", "svm", 1, grid=SMALL_GRIDS["svm"], inner_k=3)
        row = result["predictions"][0]
        for field in ("subject", "condition", "fold", "seed", "model", "score", "y_true", "y_pred"):
            self.assertIn(field, row)
        self.assertTrue(all(p["fold"] == p["subject"] for p in result["predictions"]))
        self.assertEqual(len(result["predictions"]), len(table))
        self.assertTrue(all(f["auroc"] is not None for f in result["folds"]))

    def test_a_failing_fold_is_recorded_and_the_others_continue(self):
        table = synthetic_table(n_subjects=4)
        calls = {"n": 0}
        real = models.make

        def flaky(name, task, seed):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return real(name, task, seed)

        with mock.patch.object(models, "make", flaky):
            result = nested.run(table, "binary", "svm", 0, grid=SMALL_GRIDS["svm"], inner_k=3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual([f["kind"] for f in result["failures"]], ["fold failed"])


class BootstrapTests(unittest.TestCase):
    def test_ci_contains_the_mean_and_has_the_expected_width(self):
        values = np.random.default_rng(3).normal(0.6, 0.1, 15)
        mean, lo, hi = metrics.bootstrap_ci(values)
        self.assertAlmostEqual(mean, values.mean())
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        expected = 2 * 1.96 * values.std(ddof=1) / np.sqrt(15)
        self.assertLess(abs((hi - lo) - expected) / expected, 0.25)

    def test_ci_is_reproducible_and_degenerate_for_constant_data(self):
        values = np.random.default_rng(4).normal(0, 1, 15)
        self.assertEqual(metrics.bootstrap_ci(values), metrics.bootstrap_ci(values))
        self.assertEqual(metrics.bootstrap_ci([0.5] * 15), (0.5, 0.5, 0.5))

    def test_paired_differences_average_seeds_first_and_flag_zero(self):
        rows = []
        for seed in range(3):
            for s in range(15):
                rows.append({"model": "mlp", "subject": "S{}".format(s), "seed": seed,
                             "macro_f1": 0.6 + 0.01 * seed})
                rows.append({"model": "svm", "subject": "S{}".format(s), "seed": seed,
                             "macro_f1": 0.5})
                rows.append({"model": "rf", "subject": "S{}".format(s), "seed": seed,
                             "macro_f1": 0.5 + (0.05 if s % 2 else -0.05)})
        diffs = metrics.paired_differences(rows, "macro_f1")
        self.assertAlmostEqual(diffs["mlp - svm"]["mean"], 0.11)
        self.assertTrue(diffs["mlp - svm"]["excludes_zero"])
        self.assertFalse(diffs["rf - svm"]["excludes_zero"])
        summary = metrics.summarise(rows, "macro_f1")
        self.assertEqual(summary["mlp"]["n_subjects"], 15)
        self.assertAlmostEqual(summary["mlp"]["sd"], 0.0)


class DataAccessTests(unittest.TestCase):
    def test_personal_paths_are_refused(self):
        for path in ("C:/x/health dashboard/data-samples/a.csv", "takeout/Heart Rate/x.csv",
                     "x/heart_rate_2026-10-30.csv", "../health dashboard/db.sqlite3"):
            with self.assertRaises(data.ForbiddenPath, msg=path):
                data.check_path(path)
        data.check_path(config.FEATURES)

    def test_only_the_feature_table_is_opened(self):
        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", spy), mock.patch("io.open", spy):
            table = data.load_table()
        self.assertEqual(len(table), 1015)
        for path in opened:
            self.assertFalse(re.search(r"data-samples|takeout|heart_rate_", path, re.I), path)

    def test_no_eval_source_mentions_personal_data_paths(self):
        offenders = []
        for name in sorted(os.listdir(config.EVAL_DIR)):
            if not name.endswith(".py") or name == "config.py":
                continue
            with open(os.path.join(config.EVAL_DIR, name), encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if re.search(r"data-samples|takeout|heart_rate_\d|db\.sqlite3", line, re.I):
                        offenders.append("{}:{}".format(name, number))
        self.assertEqual(offenders, [])


class ReproductionTests(unittest.TestCase):
    def test_full_feature_svm_reproduces_table_I(self):
        value, n_features = full_svm_loso()
        self.assertEqual(n_features, 40)
        self.assertAlmostEqual(value, EXPECTED_FULL_SVM_MACRO_F1, delta=0.005)

    def test_the_feature_table_is_the_one_the_protocol_froze(self):
        self.assertEqual(data.sha256(config.FEATURES), config.FEATURES_SHA256)

    def test_plan_counts(self):
        counts = run_eval.plan_counts()
        self.assertEqual(counts["svm"], (12 * 7 + 1) * 15 * 5 * 2)
        self.assertEqual(counts["rf"], (6 * 7 + 1) * 15 * 5 * 2)
        self.assertEqual(counts["mlp"], (4 * 7 + 1) * 15 * 5 * 2)

    def test_run_refuses_without_go(self):
        with self.assertRaises(SystemExit):
            run_eval.main(["run"])


if __name__ == "__main__":
    unittest.main()
