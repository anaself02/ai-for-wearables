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


def run1(*args, **kwargs):
    """nested.run with the grid search in THIS process (n_jobs=1), so spies and
    mocks see every fit. Parallel runs are tested separately."""
    kwargs.setdefault("n_jobs", 1)
    return nested.run(*args, **kwargs)


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
            run1(self.table, "binary", model, 0, grid=SMALL_GRIDS[model],
                       inner_k=3, on_fitted=check)
        self.assertEqual(len(seen), 3 * 6)

    def test_the_scaler_is_fitted_on_training_rows_only(self):
        def check(test_subject, fitted, train_index, test_index):
            X = self.table[config.HR_FEATURES].to_numpy(dtype=float)
            np.testing.assert_allclose(fitted.named_steps["scaler"].mean_,
                                       X[train_index].mean(axis=0))
            self.assertFalse(np.allclose(fitted.named_steps["scaler"].mean_, X.mean(axis=0)))

        run1(self.table, "three", "svm", 0, grid=SMALL_GRIDS["svm"], inner_k=3,
                   on_fitted=check)

    def test_inner_selection_scores_scalers_fitted_on_inner_training_rows(self):
        from sklearn.preprocessing import StandardScaler
        fitted_sizes = []
        real_fit = StandardScaler.fit

        def spy(self_, X, y=None, sample_weight=None):
            fitted_sizes.append(len(X))
            return real_fit(self_, X, y)

        with mock.patch.object(StandardScaler, "fit", spy):
            run1(self.table, "binary", "svm", 0, grid={"clf__C": [1]}, inner_k=5,
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

    def test_threshold_baseline_maximises_macro_f1_on_the_training_rows(self):
        """REPLACES the earlier test that pinned the cut to the demo's
        balanced-accuracy rule: Amendment 4 chooses the cut by macro-F1 instead
        (the demo and the dashboard are untouched and keep their own rule)."""
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(1)
        cases = [
            ("binary", 0, 1, [0, 1], rng.normal(80, 10, 300), rng.random(300) > 0.7),
            ("binary", 0, 1, [0, 1], np.round(rng.normal(80, 10, 200)), rng.random(200) > 0.5),
            ("three", 1, 2, [1, 2, 3], rng.normal(80, 10, 300), rng.integers(1, 4, 300)),
            ("three", 1, 2, [1, 2, 3], np.round(rng.normal(80, 10, 200)), rng.integers(1, 4, 200)),
        ]
        for name, low, high, task_labels, x, y in cases:
            y = np.asarray(y).astype(int)
            mine = models.best_threshold(x, y, low, high, task_labels)

            def macro(cut):
                return f1_score(y, np.where(x > cut, high, low), labels=task_labels,
                                average="macro", zero_division=0)

            distinct = np.unique(x)
            best = max(macro(0.5 * (a + b)) for a, b in zip(distinct[:-1], distinct[1:]))
            self.assertAlmostEqual(macro(mine), best, places=12, msg=name)
            self.assertTrue(distinct[0] < mine < distinct[-1])

    def test_threshold_edge_cases_and_the_three_class_rule(self):
        self.assertEqual(models.best_threshold([5.0, 5.0, 5.0], [0, 1, 0], 0, 1, [0, 1]), -np.inf)
        rng = np.random.default_rng(2)
        X = rng.normal(80, 10, (120, 5))
        y = rng.integers(1, 4, 120)
        rule = models.HRMeanThreshold(three_class=True).fit(X, y)
        self.assertEqual(set(np.unique(rule.predict(X))) - {1, 2}, set())     # never amusement
        binary = models.HRMeanThreshold().fit(X, (y == 2).astype(int))
        self.assertEqual(set(np.unique(binary.predict(X))) - {0, 1}, set())

    def test_the_threshold_is_computed_from_training_subjects_only(self):
        table = self.table
        seen = {}

        def check(test_subject, fitted, train_index, test_index):
            seen[test_subject] = fitted.threshold_
            y = (table["label"].to_numpy() == 2).astype(int)
            expected = models.best_threshold(table["HR_mean"].to_numpy()[train_index],
                                             y[train_index], 0, 1, [0, 1])
            self.assertEqual(fitted.threshold_, expected)

        run1(table, "binary", "threshold", 0, grid={}, on_fitted=check)
        self.assertEqual(len(seen), 6)
        self.assertGreater(len(set(seen.values())), 1)          # differs by held-out subject

    def test_the_mlp_has_no_early_stopping_and_a_fixed_max_iter(self):
        self.assertEqual(config.MLP_MAX_ITER, 300)
        pipeline = models.make("mlp", "binary", 0)
        X = np.random.default_rng(0).normal(size=(80, 5))
        y = np.array([0] * 60 + [1] * 20)
        pipeline.fit(X, y)
        inner = pipeline.named_steps["clf"].mlp_
        self.assertFalse(inner.early_stopping)
        self.assertEqual(inner.max_iter, 300)
        self.assertIsNone(getattr(inner, "validation_scores_", None))   # no validation split
        with open(os.path.join(config.EVAL_DIR, "models.py"), encoding="utf-8") as handle:
            code = [line for line in handle if not line.lstrip().startswith(("#", '"'))]
        self.assertFalse(any("early_stopping=True" in line for line in code))

    def test_oversampling_in_the_inner_folds_uses_only_inner_training_rows(self):
        """Every fit the grid search makes sees only the inner-TRAINING subjects
        (inner-validation rows go to predict only); oversampling then draws only
        from the rows it was given."""
        table = synthetic_table(n_subjects=6)
        seen = []
        real_fit = models.OversampledMLP.fit

        def spy(self_, X, y):
            seen.append(np.unique(np.round(np.asarray(X)[:, 2], 9)))   # HR_min = subject code
            return real_fit(self_, X, y)

        with mock.patch.object(models.OversampledMLP, "fit", spy):
            run1(table, "binary", "mlp", 0, grid={"clf__alpha": [1e-4]}, inner_k=3)

        subjects = sorted(set(table["subject_id"]), key=folds.subject_key)
        expected = []
        for test, train in folds.outer_folds(subjects):
            for held in folds.inner_folds(train, 0, 3):
                fit_subjects = [t for t in train if t not in held]
                codes = np.array([int(t[1:]) * 1000.0 for t in fit_subjects])
                expected.append(np.sort((codes - codes.mean()) / codes.std()))
            codes = np.array([int(t[1:]) * 1000.0 for t in train])               # the refit
            expected.append(np.sort((codes - codes.mean()) / codes.std()))
        self.assertEqual(len(seen), len(expected))
        for got, want in zip(seen, expected):
            np.testing.assert_allclose(np.sort(got), want, atol=1e-6)

    def test_oversampled_rows_are_copies_of_rows_given_to_fit(self):
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = np.array([0] * 50 + [1] * 10)
        captured = {}
        real_fit = MLPClassifier.fit

        def spy(self_, Xr, yr):
            captured["X"], captured["y"] = np.asarray(Xr), np.asarray(yr)
            return real_fit(self_, Xr, yr)

        with mock.patch.object(MLPClassifier, "fit", spy):
            models.OversampledMLP(random_state=0).fit(X, y)
        given = {tuple(row) for row in X}
        self.assertTrue(all(tuple(row) in given for row in captured["X"]))
        self.assertEqual(np.bincount(captured["y"]).tolist(), [50, 50])
        minority_rows = {tuple(row) for row in X[y == 1]}
        self.assertTrue(all(tuple(row) in minority_rows
                            for row in captured["X"][captured["y"] == 1]))


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_predictions(self):
        table = synthetic_table(n_subjects=5)
        for model in config.MODELS + config.BASELINES:
            a = run1(table, "binary", model, 2, grid=SMALL_GRIDS.get(model, {}), inner_k=2)
            b = run1(table, "binary", model, 2, grid=SMALL_GRIDS.get(model, {}), inner_k=2)
            self.assertEqual([(p["y_pred"], p["score"]) for p in a["predictions"]],
                             [(p["y_pred"], p["score"]) for p in b["predictions"]], model)
            self.assertEqual([f["chosen"] for f in a["folds"]], [f["chosen"] for f in b["folds"]])

    def test_predictions_carry_subject_condition_fold_seed_model_score(self):
        table = synthetic_table(n_subjects=4)
        result = run1(table, "binary", "svm", 1, grid=SMALL_GRIDS["svm"], inner_k=3)
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
            result = run1(table, "binary", "svm", 0, grid=SMALL_GRIDS["svm"], inner_k=3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual([f["kind"] for f in result["failures"]], ["fold failed"])


class ParallelAndConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.table = synthetic_table(n_subjects=5)

    def test_n_jobs_2_gives_the_same_predictions_and_choices_as_n_jobs_1(self):
        grid = SMALL_GRIDS["mlp"]
        one = nested.run(self.table, "binary", "mlp", 3, grid=grid, inner_k=2, n_jobs=1)
        two = nested.run(self.table, "binary", "mlp", 3, grid=grid, inner_k=2, n_jobs=2)
        self.assertEqual([(p["y_pred"], p["score"]) for p in one["predictions"]],
                         [(p["y_pred"], p["score"]) for p in two["predictions"]])
        self.assertEqual([f["chosen"] for f in one["folds"]], [f["chosen"] for f in two["folds"]])
        self.assertEqual([f["inner_macro_f1"] for f in one["folds"]],
                         [f["inner_macro_f1"] for f in two["folds"]])

    def test_convergence_is_recorded_per_fit_also_across_processes(self):
        grid = {"clf__max_iter": [2], "clf__alpha": [1e-4, 1e-2]}      # 2 epochs: cannot converge
        for n_jobs in (1, 2):
            result = nested.run(self.table, "binary", "mlp", 0, grid=grid, inner_k=3,
                                n_jobs=n_jobs)
            notes = [f for f in result["failures"] if f["kind"] == "convergence warnings"]
            self.assertEqual(len(notes), 5, n_jobs)                    # every outer fold
            for note in notes:
                self.assertEqual((note["inner_fits_not_converged"], note["inner_fits"]), (6, 6))
                self.assertFalse(note["outer_refit_converged"])

    def test_models_that_cannot_fail_to_converge_report_nothing(self):
        result = nested.run(self.table, "binary", "svm", 0, grid=SMALL_GRIDS["svm"],
                            inner_k=2, n_jobs=2)
        self.assertEqual([f for f in result["failures"] if f["kind"] == "convergence warnings"],
                         [])

    def test_run_all_saves_partials_and_resumes_and_refuses_other_code(self):
        table = synthetic_table(n_subjects=4)
        notes = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_eval, "log_experiment", notes.append), \
                mock.patch.object(folds, "write_folds", lambda *a, **k: None), \
                mock.patch.dict(config.GRIDS, {"svm": SMALL_GRIDS["svm"]}), \
                mock.patch.object(config, "INNER_FOLDS", 2):
            first = run_eval.run_all(["binary"], [0, 1], ["majority", "svm"], out_dir=tmp,
                                     table=table, say=lambda m: None)
            self.assertEqual(len(os.listdir(os.path.join(tmp, "partial"))), 4)
            self.assertEqual(len(notes), 4)
            second = run_eval.run_all(["binary"], [0, 1], ["majority", "svm"], out_dir=tmp,
                                      table=table, say=lambda m: None)
            self.assertEqual(len(notes), 4)                            # nothing re-run
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[2], second[2])
            with mock.patch.object(run_eval, "protocol_hash", lambda: "other code"):
                with self.assertRaises(SystemExit):
                    run_eval.run_all(["binary"], [0], ["majority"], out_dir=tmp, table=table,
                                     say=lambda m: None)


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
