"""Stage 2D tests: R1s parity, R2 resampling, no train/test subject overlap, determinism.

    python -m unittest discover -s eval/tests -t .      (from wearable-ml)

Only the parity tests read raw WESAD (one subject, ~1 GB pickle, skipped if absent).
Everything else uses synthetic series and tables.
"""

import json
import os
import unittest

import numpy as np
import pandas as pd

from eval import config, cross, folds, interaction, nested, resolution
from eval.tests.test_eval import synthetic_table

S2_PKL = os.path.join(resolution.WESAD_DIR, "S2", "S2.pkl")


def pulse_train(bpm=72.0, seconds=60, fs=64, seed=0):
    """A clean synthetic BVP: one smooth pulse per beat plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    signal = np.sin(2 * np.pi * (bpm / 60.0) * t) + 0.05 * rng.normal(size=len(t))
    return signal


class ResamplingTests(unittest.TestCase):
    def test_bin_mean_on_a_known_series(self):
        # beats at 0.5, 1.5, 2.5 (bin 0: 60, 70, 80), 3.0 (bin 1: 100), 59.9 (bin 19: 90)
        t = [0.5, 1.5, 2.5, 3.0, 59.9]
        bpm = [60.0, 70.0, 80.0, 100.0, 90.0]
        centres, means = resolution.resample_bin_mean(t, bpm)
        self.assertEqual(list(centres), [1.5, 4.5, 58.5])
        self.assertEqual(list(means), [70.0, 100.0, 90.0])      # t=3.0 belongs to bin 1

    def test_empty_bins_are_skipped_not_interpolated(self):
        centres, means = resolution.resample_bin_mean([1.0, 20.0], [60.0, 80.0])
        self.assertEqual(len(means), 2)
        self.assertEqual(list(means), [60.0, 80.0])

    def test_constant_rate_stays_constant_and_spacing_is_three_seconds(self):
        t = np.arange(0.4, 60.0, 0.8)                     # a beat every 0.8 s
        centres, means = resolution.resample_bin_mean(t, np.full(len(t), 75.0))
        self.assertEqual(len(means), 20)
        self.assertTrue(np.allclose(means, 75.0))
        self.assertTrue(np.allclose(np.diff(centres), resolution.BIN_SECONDS))

    def test_a_linear_ramp_has_the_right_slope_and_mean(self):
        t = np.arange(0.25, 60.0, 0.5)
        bpm = 60.0 + 0.5 * t                                # +0.5 bpm per second
        centres, means = resolution.resample_bin_mean(t, bpm)
        feats = resolution.r2_features(centres, means)
        self.assertAlmostEqual(feats["HR_slope"], 0.5, places=6)
        self.assertAlmostEqual(feats["HR_mean"], 60.0 + 0.5 * 30.0, places=1)
        self.assertLessEqual(feats["HR_min"], feats["HR_mean"])
        self.assertGreaterEqual(feats["HR_max"], feats["HR_mean"])

    def test_slope_is_bpm_per_second_and_identical_at_r1s_and_r2(self):
        t = np.arange(0.25, 60.0, 0.5)
        bpm = 60.0 + 0.5 * t
        r1s = resolution.per_second_features(t, bpm)               # beat level
        centres, means = resolution.resample_bin_mean(t, bpm)
        r2 = resolution.r2_features(centres, means)                # 3 s bins
        self.assertAlmostEqual(r1s["HR_slope"], 0.5, places=9)
        self.assertAlmostEqual(r2["HR_slope"], 0.5, places=6)
        # a pure change of unit would show here: per beat at 2 beats/s would be 0.25
        features, _ = resolution.legacy()
        per_beat = features.basic_features(bpm, fs=1.0)["slope"]
        self.assertAlmostEqual(per_beat, 0.5 * 0.5, places=9)       # the old definition
        self.assertNotAlmostEqual(per_beat, r1s["HR_slope"], places=3)

    def test_slope_uses_the_actual_timestamps_not_the_index(self):
        t = np.array([0.0, 1.0, 2.0, 10.0, 11.0])                  # a gap, then continuing
        x = np.array([60.0, 61.0, 62.0, 70.0, 71.0])               # exactly 1 bpm per second
        self.assertAlmostEqual(resolution.slope_per_second(t, x), 1.0, places=9)
        self.assertNotAlmostEqual(np.polyfit(np.arange(5), x, 1)[0], 1.0, places=2)

    def test_r1s_keeps_every_other_feature_of_r1_unchanged(self):
        features, _ = resolution.legacy()
        bvp = pulse_train(75.0, seed=9)
        t, bpm = resolution.beat_series(bvp)
        r1 = features.basic_features(features.heart_rate_series(bvp), fs=1.0)
        r1s = resolution.per_second_features(t, bpm)
        for stat in ("mean", "std", "min", "max"):
            self.assertEqual(r1[stat], r1s["HR_" + stat])
        self.assertNotEqual(r1["slope"], r1s["HR_slope"])
        self.assertEqual(resolution.slope_per_second([1.0], [70.0]), 0.0)

    def test_fewer_than_ten_readings_gives_no_window(self):
        t = np.arange(1.0, 28.0, 3.0)                       # 9 occupied bins
        centres, means = resolution.resample_bin_mean(t, np.full(len(t), 70.0))
        self.assertEqual(len(means), 9)
        self.assertIsNone(resolution.r2_features(centres, means))
        t = np.arange(1.0, 31.0, 3.0)                       # 10 occupied bins
        centres, means = resolution.resample_bin_mean(t, np.full(len(t), 70.0))
        self.assertIsNotNone(resolution.r2_features(centres, means))

    def test_empty_beat_series(self):
        centres, means = resolution.resample_bin_mean([], [])
        self.assertEqual(len(means), 0)
        self.assertIsNone(resolution.r2_features(centres, means))

    def test_beat_series_bpm_equals_the_original_heart_rate_series(self):
        features, _ = resolution.legacy()
        for bpm, seed in ((60.0, 1), (72.0, 2), (95.0, 3)):
            bvp = pulse_train(bpm, seed=seed)
            t, mine = resolution.beat_series(bvp)
            original = features.heart_rate_series(bvp)
            self.assertGreater(len(original), 20)
            self.assertTrue(np.array_equal(mine, original))      # same values, same order
            self.assertEqual(len(t), len(mine))
            self.assertTrue(np.all(np.diff(t) > 0))
            self.assertTrue(np.all((t >= 0) & (t < 60)))
            self.assertLess(abs(float(np.median(mine)) - bpm), 3.0)

    def test_r2_is_the_bin_mean_of_the_same_beats_that_r1_uses(self):
        bvp = pulse_train(70.0, seed=5)
        t, bpm = resolution.beat_series(bvp)
        centres, means = resolution.resample_bin_mean(t, bpm)
        r2 = resolution.r2_features(centres, means)
        # R2's mean is the mean of bin means: close to, not equal to, R1s's beat mean
        self.assertLess(abs(r2["HR_mean"] - float(np.mean(bpm))), 2.0)
        # binning smooths: R2's spread cannot exceed R1s's
        self.assertLessEqual(r2["HR_std"], float(np.std(bpm)) + 1e-9)


@unittest.skipUnless(os.path.exists(S2_PKL), "raw WESAD S2 not available")
class ParityTests(unittest.TestCase):
    def test_r1_for_subject_s2_matches_features_csv(self):
        rows = resolution.build_subject("S2")
        result = resolution.parity(rows, features_csv=config.FEATURES)
        ref = pd.read_csv(config.FEATURES)
        ref = ref[ref["subject_id"] == "S2"]
        mine = pd.DataFrame([{"window_start_s": r["window_start_s"], "label": r["label"],
                              **r["r1"]} for r in rows])
        self.assertEqual(len(rows), len(ref))
        self.assertEqual(set(mine["window_start_s"].round(6)), set(ref["window_start_s"].round(6)))
        merged = ref.merge(mine, on="window_start_s", suffixes=("_ref", "_new"))
        for f in config.HR_FEATURES:
            self.assertLessEqual(float(np.max(np.abs(merged[f + "_ref"] - merged[f + "_new"]))),
                                 resolution.PARITY_ATOL)
        self.assertTrue((merged["label_ref"] == merged["label_new"]).all())
        self.assertIsInstance(result, dict)               # whole-table parity is run by `parity`

    def test_the_full_parity_result_on_disk_is_ok(self):
        if not os.path.exists(resolution.PARITY_JSON):
            self.skipTest("run `python -m eval.interaction parity` first")
        with open(resolution.PARITY_JSON, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertTrue(result["ok"])
        self.assertTrue(result["same_window_set"])
        self.assertEqual(result["n_rebuilt"], 1015)
        self.assertLessEqual(max(result["max_abs_difference"].values()), resolution.PARITY_ATOL)

    def test_the_slope_comparison_on_disk_is_recorded(self):
        if not os.path.exists(resolution.PARITY_JSON):
            self.skipTest("run `python -m eval.interaction parity` first")
        with open(resolution.PARITY_JSON, encoding="utf-8") as handle:
            result = json.load(handle)
        comp = result["slope_comparison_R1s_vs_R1"]
        self.assertEqual(comp["n_windows"], result["n_windows_valid_at_R2"])
        self.assertGreater(comp["pearson_r"], 0.5)
        self.assertGreater(comp["ratio_of_sd"], 0.0)

    def test_parity_check_reports_missing_windows(self):
        features, _ = resolution.legacy()
        ref = pd.read_csv(config.FEATURES)
        ref = ref[ref["subject_id"] == "S2"]
        rows = [{"subject_id": "S2", "label": int(r.label), "window_start_s": float(r.window_start_s),
                 "r1": {f: float(getattr(r, f)) for f in config.HR_FEATURES}}
                for r in ref.itertuples()]
        good = resolution.parity(rows, features_csv=config.FEATURES)
        self.assertFalse(good["ok"])                        # S2 only: the other 14 subjects are missing
        self.assertFalse(good["same_window_set"])


def two_resolution_tables():
    """Row-aligned R1s/R2 synthetic tables; R2 is R1s with HR_mean shifted far away so a
    model trained on R1s and scored on R2 is visibly scored on different numbers."""
    r1 = synthetic_table(n_subjects=6, per_condition=8, seed=3)
    r2 = r1.copy()
    r2["HR_mean"] = r2["HR_mean"] + 1000.0
    return r1, r2


class CrossRunTests(unittest.TestCase):
    GRIDS = {"svm": {"clf__C": [1, 10]}, "rf": {"clf__max_depth": [None, 5]},
             "mlp": {"clf__alpha": [1e-4, 1e-2]}}

    def test_same_table_gives_the_same_result_as_nested_run(self):
        table = synthetic_table(n_subjects=6, per_condition=8, seed=1)
        for model in ("threshold", "svm"):
            a = nested.run(table, "binary", model, 0, grid=self.GRIDS.get(model), n_jobs=1, inner_k=2)
            b = cross.run(table, {"R1s": table}, model, 0, grid=self.GRIDS.get(model), n_jobs=1,
                          inner_k=2)
            self.assertEqual(len(a["folds"]), 6)                 # not vacuous: no failed folds
            self.assertEqual(a["failures"] + b["failures"], [])
            self.assertEqual([f["macro_f1"] for f in a["folds"]],
                             [f["macro_f1"] for f in b["results"]["R1s"]["folds"]])
            self.assertEqual([p["y_pred"] for p in a["predictions"]],
                             [p["y_pred"] for p in b["results"]["R1s"]["predictions"]])

    def test_no_subject_is_in_both_train_and_test_for_every_table(self):
        r1, r2 = two_resolution_tables()
        seen = []

        def spy(test_subject, fitted, train_index, test_index):
            seen.append((test_subject, set(r1["subject_id"].iloc[train_index]),
                         set(r1["subject_id"].iloc[test_index])))

        out = cross.run(r1, {"R1s": r1, "R1stoR2": r2}, "svm", 0, grid=self.GRIDS["svm"],
                        n_jobs=1, inner_k=2, on_fitted=spy)
        self.assertEqual(len(seen), 6)
        for test_subject, train_subjects, test_subjects in seen:
            self.assertEqual(test_subjects, {test_subject})
            self.assertNotIn(test_subject, train_subjects)
            self.assertEqual(len(train_subjects), 5)
        for name in ("R1s", "R1stoR2"):
            self.assertEqual({p["subject"] for p in out["results"][name]["predictions"]},
                             set(r1["subject_id"]))
            self.assertEqual(len(out["results"][name]["predictions"]), len(r1))   # each window once

    def test_cross_resolution_trains_on_the_training_table_only(self):
        r1, r2 = two_resolution_tables()
        cuts = []

        def spy(test_subject, fitted, train_index, test_index):
            cuts.append(fitted.threshold_)

        out = cross.run(r1, {"R1s": r1, "R1stoR2": r2}, "threshold", 0, n_jobs=1, on_fitted=spy)
        self.assertTrue(all(c is not None and c < 200 for c in cuts))     # cut learnt on R1s values
        # scored on R2 (every HR_mean >= 1000) the R1s-trained cut calls everything stress
        self.assertTrue(all(p["y_pred"] == 1 for p in out["results"]["R1stoR2"]["predictions"]))
        # the same fits scored on R1s do not
        self.assertTrue(any(p["y_pred"] == 0 for p in out["results"]["R1s"]["predictions"]))

    def test_misaligned_tables_are_refused(self):
        r1, r2 = two_resolution_tables()
        with self.assertRaises(ValueError):
            cross.run(r1, {"R2": r2.iloc[::-1].reset_index(drop=True)}, "svm", 0, n_jobs=1)
        with self.assertRaises(ValueError):
            cross.run(r1, {"R2": r2.iloc[:-1]}, "svm", 0, n_jobs=1)

    def test_deterministic_and_independent_of_the_number_of_workers(self):
        r1, r2 = two_resolution_tables()
        r2 = r1.copy()
        r2["HR_mean"] = r2["HR_mean"] * 0.99
        for model in ("svm", "mlp"):
            runs = [cross.run(r1, {"R1s": r1, "R1stoR2": r2}, model, 3, grid=self.GRIDS[model],
                              n_jobs=jobs, inner_k=2) for jobs in (1, 1, 2)]
            for name in ("R1s", "R1stoR2"):
                first = runs[0]["results"][name]
                self.assertEqual(len(first["folds"]), 6)
                self.assertEqual([f for f in runs[0]["failures"] if f["kind"] != "convergence warnings"], [])
                for other in runs[1:]:
                    self.assertEqual([p["y_pred"] for p in first["predictions"]],
                                     [p["y_pred"] for p in other["results"][name]["predictions"]])
                    self.assertEqual([p["score"] for p in first["predictions"]],
                                     [p["score"] for p in other["results"][name]["predictions"]])
                    self.assertEqual([f["chosen"] for f in first["folds"]],
                                     [f["chosen"] for f in other["results"][name]["folds"]])


def fold_row(model, condition, subject, seed, macro_f1):
    return {"model": model, "condition": condition, "subject": subject, "seed": seed,
            "macro_f1": macro_f1, "balanced_accuracy": macro_f1, "auroc": None}


class InteractionStatsTests(unittest.TestCase):
    def rows(self):
        out = []
        subjects = ["S{}".format(i) for i in range(2, 12)]
        for k, s in enumerate(subjects):
            for seed in (0, 1):
                for model, base in (("mlp", 0.8), ("svm", 0.7), ("rf", 0.75)):
                    out.append(fold_row(model, "R1s", s, seed, base + 0.01 * (k % 3) + 0.02 * seed))
                    # at R2 the MLP-SVM gap shrinks by 0.10; seed 1 adds +0.02 everywhere
                    shrink = -0.10 if model == "mlp" else 0.0
                    out.append(fold_row(model, "R2", s, seed, base + shrink + 0.01 * (k % 3)
                                        + 0.02 * seed))
        return out

    def test_interaction_contrast_is_the_difference_of_differences(self):
        rows = [r for r in interaction.interaction_contrasts(self.rows()) if r["kind"] == "interaction"]
        by = {r["contrast"]: r for r in rows}
        mlp_svm = by["(mlp-svm at R1s) - (mlp-svm at R2)"]
        self.assertAlmostEqual(mlp_svm["mean"], 0.10, places=9)
        self.assertAlmostEqual(mlp_svm["diff_at_R1"] - mlp_svm["diff_at_R2"], 0.10, places=9)
        self.assertAlmostEqual(by["(rf-svm at R1s) - (rf-svm at R2)"]["mean"], 0.0, places=9)
        self.assertEqual(len(rows), 3)
        self.assertTrue(mlp_svm["excludes_zero"])          # every subject has the same 0.10
        self.assertFalse(by["(rf-svm at R1s) - (rf-svm at R2)"]["excludes_zero"])

    def test_resolution_effect_per_model(self):
        rows = {r["contrast"]: r for r in interaction.interaction_contrasts(self.rows())
                if r["kind"] == "resolution"}
        self.assertAlmostEqual(rows["mlp at R2 - mlp at R1s"]["mean"], -0.10, places=9)
        self.assertAlmostEqual(rows["svm at R2 - svm at R1s"]["mean"], 0.0, places=9)

    def test_ranking_follows_the_means_and_can_change(self):
        ranks = interaction.rankings(self.rows())
        self.assertEqual(ranks["R1s"], ["mlp", "rf", "svm"])
        self.assertEqual(ranks["R2"], ["rf", "mlp", "svm"])

    def test_the_contrast_ci_uses_amendment_4_s_bootstrap(self):
        from eval import metrics
        row = [r for r in interaction.interaction_contrasts(self.rows())
               if r["kind"] == "interaction"][0]
        expected = metrics.bootstrap_ci([0.10] * 10)           # same function, n=10000, seed 6322
        for got, want in zip((row["mean"], row["ci_low"], row["ci_high"]), expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_plan_counts_two_training_tables(self):
        fits, seconds = interaction.plan()
        self.assertEqual(fits["majority"], 15 * 5 * 2)
        self.assertEqual(fits["svm"], (12 * 7 + 1) * 15 * 5 * 2)
        self.assertGreater(seconds, 3000)
        self.assertLess(seconds, 6000)


class NoLocalPathsTests(unittest.TestCase):
    def test_no_file_under_eval_has_a_hardcoded_local_path_or_username(self):
        import getpass
        banned = ["c:\\users", "c:/users", "/users/", "/home/", getpass.getuser().lower()]
        checked = 0
        for root, dirs, files in os.walk(config.EVAL_DIR):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "out")]
            for name in files:
                if not name.endswith((".py", ".md", ".json", ".csv")):
                    continue
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as handle:
                    text = handle.read().lower()
                if name == "test_resolution.py":
                    continue                                    # this file lists the patterns
                checked += 1
                for pattern in banned:
                    self.assertNotIn(pattern, text, "{} contains {!r}".format(name, pattern))
        self.assertGreater(checked, 10)

    def test_the_wesad_location_comes_from_an_argument_or_the_environment(self):
        import subprocess
        import sys
        self.assertIn("WESAD_DIR", open(resolution.__file__, encoding="utf-8").read())
        out = subprocess.run([sys.executable, "-m", "eval.resolution", "--help"], cwd=config.ML_DIR,
                             capture_output=True, text=True)
        self.assertIn("--wesad-dir", out.stdout)


class DesignIsFrozenTests(unittest.TestCase):
    def test_amendment_4_choices_are_unchanged(self):
        self.assertEqual(config.SEEDS, [0, 1, 2, 3, 4])
        self.assertEqual(config.GRIDS["svm"], {"clf__C": [0.1, 1, 10, 100],
                                              "clf__gamma": ["scale", 0.01, 0.1]})
        self.assertEqual(config.GRIDS["rf"], {"clf__max_depth": [None, 10, 20],
                                             "clf__min_samples_leaf": [1, 5]})
        self.assertEqual(config.MLP_MAX_ITER, 300)
        self.assertEqual(config.INNER_FOLDS, 7)
        self.assertEqual(resolution.BIN_SECONDS, 3.0)
        self.assertEqual(resolution.MIN_READINGS, 10)
        self.assertEqual(interaction.CONDITIONS, ["R1s", "R2", "R1stoR2"])

    def test_resolution_code_never_reads_personal_data(self):
        for name in ("resolution.py", "cross.py", "interaction.py"):
            with open(os.path.join(config.EVAL_DIR, name), encoding="utf-8") as handle:
                text = handle.read().lower()
            for forbidden in ("data-samples", "takeout", "heart_rate_2", "health dashboard"):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
