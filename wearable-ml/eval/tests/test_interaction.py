"""Tests for eval/resolution.py and eval/interaction.py (synthetic data only, no WESAD)."""

import os
import pickle
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from eval import config, interaction, resolution

sys.path.insert(0, resolution.SCRIPTS_DIR)
import features as script_features  # noqa: E402

FS_BVP, FS_LABEL = 64, 700


def synthetic_bvp(seconds, bpm_fn, rng):
    """A pulse train: one Gaussian pulse per beat plus a small dicrotic bump and noise."""
    t = np.arange(int(seconds * FS_BVP)) / FS_BVP
    signal = np.zeros_like(t)
    beat = 0.0
    while beat < seconds:
        signal += np.exp(-0.5 * ((t - beat) / 0.06) ** 2)
        signal += 0.3 * np.exp(-0.5 * ((t - beat - 0.3) / 0.05) ** 2)
        beat += 60.0 / max(40.0, bpm_fn(beat) + rng.normal(0, 2))
    return signal + rng.normal(0, 0.03, len(t))


def synthetic_subject(subject_index, rng):
    """baseline 300 s, transient 30 s, stress 240 s, transient 30 s, amusement 180 s."""
    plan = [(1, 300), (0, 30), (2, 240), (0, 30), (3, 180)]
    rest = 62 + 3 * subject_index
    level = {1: rest, 0: rest + 5, 2: rest + 18, 3: rest + 4}
    edges, labels, t0 = [], [], 0.0
    for code, dur in plan:
        edges.append((t0, t0 + dur, code))
        labels.append(np.full(dur * FS_LABEL, code, dtype=int))
        t0 += dur

    def bpm(t):
        for a, b, code in edges:
            if a <= t < b:
                return level[code] + 3 * np.sin(t / 7.0)
        return rest

    return {"label": np.concatenate(labels),
            "signal": {"wrist": {"BVP": synthetic_bvp(t0, bpm, rng).reshape(-1, 1)}}}


def write_fake_wesad(root, subjects):
    rng = np.random.default_rng(1)
    for i, subject in enumerate(subjects):
        os.makedirs(os.path.join(root, subject))
        with open(os.path.join(root, subject, subject + ".pkl"), "wb") as handle:
            pickle.dump(synthetic_subject(i, rng), handle)


class BeatSeriesTests(unittest.TestCase):
    def test_bpm_matches_the_frozen_heart_rate_series_exactly(self):
        rng = np.random.default_rng(0)
        for seed in range(5):
            bvp = synthetic_bvp(60, lambda t: 70 + 20 * np.sin(t / 9 + seed), rng)
            _, bpm = resolution.beat_series(bvp)
            np.testing.assert_array_equal(bpm, script_features.heart_rate_series(bvp))

    def test_native_features_match_the_frozen_hr_features(self):
        rng = np.random.default_rng(3)
        bvp = synthetic_bvp(60, lambda t: 80.0, rng)
        frozen = script_features.basic_features(script_features.heart_rate_series(bvp), fs=1.0)
        _, bpm = resolution.beat_series(bvp)
        mine = resolution.hr_features(bpm)
        for stat in ("mean", "std", "min", "max", "slope"):
            self.assertAlmostEqual(mine["HR_" + stat], frozen[stat], places=12)

    def test_times_are_increasing_and_inside_the_segment(self):
        rng = np.random.default_rng(4)
        times, bpm = resolution.beat_series(synthetic_bvp(60, lambda t: 75.0, rng))
        self.assertEqual(len(times), len(bpm))
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertTrue(times.min() >= 0 and times.max() <= 60)

    def test_flat_signal_gives_nothing(self):
        times, bpm = resolution.beat_series(np.zeros(64 * 60))
        self.assertEqual(len(times), 0)
        self.assertEqual(len(bpm), 0)


class BinningTests(unittest.TestCase):
    def test_bins_anchor_at_zero_and_average(self):
        times = np.array([0.5, 1.0, 3.2, 7.0, 7.5, 59.9])
        bpm = np.array([60, 80, 90, 100, 110, 70.0])
        np.testing.assert_allclose(resolution.bin_series(times, bpm, 60), [70, 90, 105, 70])

    def test_empty_bins_are_dropped_and_count_is_at_most_20(self):
        times = np.arange(0, 60, 0.8)
        out = resolution.bin_series(times, np.full(len(times), 72.0), 60)
        self.assertEqual(len(out), 20)
        self.assertEqual(len(resolution.bin_series(np.array([1.0, 50.0]), np.array([1, 2.]), 60)), 2)

    def test_binning_smooths(self):
        rng = np.random.default_rng(5)
        times = np.sort(rng.uniform(0, 60, 75))
        bpm = 75 + rng.normal(0, 8, 75)
        self.assertLess(np.std(resolution.bin_series(times, bpm, 60)), np.std(bpm))


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.subjects = ["S2", "S3", "S4", "S5"]
        write_fake_wesad(os.path.join(cls.tmp.name, "WESAD"), cls.subjects)
        cls.out = os.path.join(cls.tmp.name, "out")
        cls.tables, cls.report = resolution.build(
            os.path.join(cls.tmp.name, "WESAD"), cls.subjects, out_dir=cls.out,
            frozen_path=None, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_levels_hold_identical_windows(self):
        ids = resolution.ID_COLUMNS
        self.assertTrue(self.tables["native"][ids].equals(self.tables["3s"][ids]))
        on_disk = {lv: pd.read_csv(resolution.table_path(lv, self.out)) for lv in resolution.LEVELS}
        interaction.check_paired(on_disk)

    def test_windows_are_pure_and_never_transient(self):
        native = self.tables["native"]
        self.assertTrue(set(native["label"]) <= {1, 2, 3})
        # baseline 0-300 s with 60 s windows, 30 s step: starts 0..240 -> 9 windows
        s2 = native[(native.subject_id == "S2") & (native.label == 1)]
        self.assertEqual(list(s2["window_start_s"]), [30.0 * i for i in range(9)])

    def test_stress_has_higher_hr_at_both_levels(self):
        for level, table in self.tables.items():
            means = table.groupby("label")["HR_mean"].mean()
            self.assertGreater(means[2], means[1] + 10, level)

    def test_alignment_report(self):
        self.assertTrue(self.report["all_aligned"])
        spans = self.report["subjects"]["S2"]["alignment"]["spans_s"]
        self.assertEqual(spans["stress"], [330.0, 570.0])

    def test_misalignment_is_detected(self):
        check = resolution.alignment(np.ones(700 * 100), np.ones(64 * 90))
        self.assertFalse(check["aligned"])

    def test_compare_with_frozen_is_zero_on_itself(self):
        cmp = resolution.compare_with_frozen(self.tables["native"], self.tables["native"])
        self.assertEqual(cmp["label_mismatches"], 0)
        self.assertEqual(cmp["max_abs_diff_HR_mean"], 0.0)


def fake_rows(values):
    """values: {(level, model): {subject: macro_f1}} -> fold rows like nested.run's."""
    rows = []
    for (level, model), per_subject in values.items():
        for subject, v in per_subject.items():
            for seed in (0, 1):
                rows.append({"task": "binary", "level": level, "model": model, "seed": seed,
                             "subject": subject, "macro_f1": v + (0.01 if seed else -0.01)})
    return rows


class AnalysisTests(unittest.TestCase):
    subjects = ["S{}".format(i) for i in range(2, 12)]

    def test_known_interaction_is_recovered(self):
        # MLP beats SVM by 0.10 at native, loses by 0.05 at 3s: contrast +0.15 for everyone
        values = {}
        for i, s in enumerate(self.subjects):
            base = 0.5 + 0.02 * i
            values.setdefault(("native", "svm"), {})[s] = base
            values.setdefault(("native", "mlp"), {})[s] = base + 0.10
            values.setdefault(("3s", "svm"), {})[s] = base - 0.02
            values.setdefault(("3s", "mlp"), {})[s] = base - 0.07
        result = interaction.analyse(fake_rows(values), "binary", "macro_f1",
                                     pairs=[("mlp", "svm")])
        c = result["contrasts"][0]
        self.assertAlmostEqual(c["mean"], 0.15, places=10)
        self.assertTrue(c["excludes_zero"])
        self.assertTrue(c["sign_flips"])
        self.assertEqual(result["ranking"], {"native": ["mlp", "svm"], "3s": ["svm", "mlp"]})
        self.assertIn("Interaction supported", " ".join(interaction.verdict(result)))

    def test_no_interaction_when_gaps_are_equal(self):
        rng = np.random.default_rng(2)
        values = {}
        for s in self.subjects:
            base = rng.uniform(0.4, 0.7)
            shift = rng.uniform(-0.1, 0.1)
            for level, delta in (("native", 0.0), ("3s", shift)):
                values.setdefault((level, "svm"), {})[s] = base + delta
                values.setdefault((level, "mlp"), {})[s] = base + delta + 0.03
        c = interaction.analyse(fake_rows(values), "binary", "macro_f1",
                                pairs=[("mlp", "svm")])["contrasts"][0]
        self.assertAlmostEqual(c["mean"], 0.0, places=10)
        self.assertFalse(c["excludes_zero"])

    def test_resolution_effect_is_paired(self):
        values = {("native", "svm"): {s: 0.6 + 0.01 * i for i, s in enumerate(self.subjects)},
                  ("3s", "svm"): {s: 0.55 + 0.01 * i for i, s in enumerate(self.subjects)}}
        effect = interaction.analyse(fake_rows(values), "binary", "macro_f1",
                                     pairs=[])["effects"][0]
        self.assertAlmostEqual(effect["mean"], 0.05, places=10)
        self.assertAlmostEqual(effect["ci_low"], 0.05, places=10)   # identical per subject


class EndToEndTests(unittest.TestCase):
    """Small grids and one seed on the synthetic tables: wiring only, not a result."""

    def test_run_and_report(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        subjects = ["S2", "S3", "S4", "S5", "S6"]
        write_fake_wesad(os.path.join(tmp.name, "WESAD"), subjects)
        out = os.path.join(tmp.name, "res")
        tables, report = resolution.build(os.path.join(tmp.name, "WESAD"), subjects,
                                          out_dir=out, frozen_path=None, log=lambda *_: None)
        tables = {lv: t.sort_values(["subject_id", "window_start_s"]).reset_index(drop=True)
                  for lv, t in tables.items()}
        grids = {"svm": {"clf__C": [1, 10]}, "rf": {"clf__max_depth": [5]},
                 "mlp": {"clf__alpha": [1e-2]}}
        rows, failures, _ = interaction.run_all(
            tables, ["binary"], [0], interaction.ALL, grids=grids,
            log=lambda *_: None, pred_dir=None, inner_k=2)
        self.assertEqual(len(rows), 2 * len(interaction.ALL) * len(subjects))
        self.assertFalse([f for f in failures if f["kind"] in ("run failed", "fold failed")])
        for row in rows:
            self.assertIn(row["level"], resolution.LEVELS)
        path = interaction.write_report(rows, failures, ["binary"],
                                        out_dir=os.path.join(tmp.name, "report"),
                                        build_report=report)
        text = open(path, encoding="utf-8").read()
        self.assertIn("mlp - svm", text)
        self.assertTrue(os.path.exists(os.path.join(tmp.name, "report",
                                                    "heatmap_binary_macro_f1.png")))


if __name__ == "__main__":
    unittest.main()
