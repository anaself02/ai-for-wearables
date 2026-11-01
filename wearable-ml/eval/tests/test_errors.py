"""Tests for eval/errors.py on small hand-built tables (no WESAD)."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from eval import errors


def features_table(n_subjects=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        subject, rest, resp = "S{}".format(i + 2), 65 + 3 * i, 2 + 6 * i
        t = 0.0
        for label, n, hr in ((1, 8, rest), (2, 6, rest + resp), (3, 4, rest + 2)):
            for _ in range(n):
                rows.append({"subject_id": subject, "label": label,
                             "label_name": errors.CONDITIONS[label], "window_start_s": t,
                             "HR_mean": hr + rng.normal(0, 1), "HR_std": 5.0, "HR_min": hr - 8,
                             "HR_max": hr + 8, "HR_slope": 0.0,
                             "ACC_mag_mean": 60.0,
                             "ACC_mag_std": (2.5 if label == 2 else 1.0) + rng.uniform(0, 1)})
                t += 30.0
            t += 120.0
    return pd.DataFrame(rows)


class ErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.feats = features_table()
        self.feat_path = os.path.join(self.tmp.name, "features.csv")
        self.feats.to_csv(self.feat_path, index=False)

    def write(self, frame, name):
        path = os.path.join(self.tmp.name, name)
        frame.to_csv(path, index=False)
        return path

    def old_format(self):
        f = self.feats
        rest = f[f.label == 1].groupby("subject_id").HR_mean.mean()
        rel = f.HR_mean - f.subject_id.map(rest)
        pred = np.where(rel > 8, 2, 1)       # a "model" that only sees big responses
        return f[["subject_id", "label", "label_name", "window_start_s"]].assign(
            predicted_svm=pred, predicted_random_forest=1)

    def test_old_format_three_class_and_collapsed_binary(self):
        path = self.write(self.old_format(), "cv.csv")
        preds, task, desc = errors.load_predictions(path, "svm")
        self.assertEqual(task, "three")
        self.assertTrue((preds.error == (preds.y_true != preds.y_pred)).all())
        preds_b, task_b, desc_b = errors.load_predictions(path, "svm", "binary")
        self.assertEqual(task_b, "binary")
        self.assertIn("not a binary model", desc_b)
        self.assertTrue(set(preds_b.y_true) <= {0, 1})

    def test_eval_format_aggregates_seeds(self):
        f = self.feats
        base = pd.DataFrame({"task": "binary", "model": "svm", "subject": f.subject_id,
                             "fold": f.subject_id, "condition": f.label_name,
                             "window_start_s": f.window_start_s,
                             "y_true": (f.label == 2).astype(int)})
        seeds = []
        for seed, flip in ((0, False), (1, True), (2, False)):
            pred = base.y_true if not flip else 1 - base.y_true
            seeds.append(base.assign(seed=seed, y_pred=pred, score=np.where(pred, 1.0, -1.0)))
        path = self.write(pd.concat(seeds), "binary_svm_seed_all.csv")
        preds, task, _ = errors.load_predictions(path)
        self.assertEqual(task, "binary")
        self.assertTrue(np.allclose(preds.error, 1 / 3))      # wrong in 1 of 3 seeds
        self.assertFalse(preds.wrong.any())
        self.assertEqual(int(preds.n_seeds.max()), 3)

    def test_examples_only_list_errors_with_a_wrong_prediction(self):
        # seeds split: the shown prediction must differ from the truth and be the most common
        # WRONG one; a window wrong in only 2 of 5 seeds is not an error and must not appear
        f = self.feats
        per_seed = {0: [1, 1, 2, 3, 3],      # 3 of 5 wrong; mode of all five is 1 (the truth)
                    1: [1, 1, 1, 3, 3],      # 2 of 5 wrong: NOT an error
                    2: [2, 2, 3, 3, 1],      # 4 of 5 wrong; wrong tie 2 vs 3 -> lowest, 2
                    3: [3, 3, 3, 3, 3]}      # 5 of 5 wrong
        seeds = []
        for seed in range(5):
            pred = f.label.to_numpy().copy()
            for row, votes in per_seed.items():
                pred[row] = votes[seed]
            seeds.append(pd.DataFrame({
                "task": "three", "model": "svm", "seed": seed, "fold": f.subject_id,
                "subject": f.subject_id, "condition": f.label_name,
                "window_start_s": f.window_start_s, "y_true": f.label, "y_pred": pred,
                "score": np.nan}))
        path = self.write(pd.concat(seeds), "three_svm_seed_all.csv")
        preds, task, _ = errors.load_predictions(path)
        table = errors.attach_context(preds, self.feat_path)
        picked, _ = errors.examples(table, task, per_type=100)
        self.assertEqual(len(picked), 3)
        key = lambda r: (r.subject_id, r.window_start_s)
        shown = {key(r): r.predicted for r in picked.itertuples()}
        row = lambda i: (f.subject_id.iat[i], f.window_start_s.iat[i])
        self.assertEqual(shown[row(0)], "amusement")
        self.assertEqual(shown[row(2)], "stress")
        self.assertEqual(shown[row(3)], "amusement")
        self.assertNotIn(row(1), shown)
        truth = table.set_index(["subject_id", "window_start_s"])
        for (subject, start), name in shown.items():
            self.assertTrue(truth.loc[(subject, start), "wrong"])
            self.assertNotEqual(name, truth.loc[(subject, start), "condition"])

    def test_several_models_need_model_flag(self):
        f = self.feats.head(4)
        frame = pd.DataFrame({"task": "binary", "model": ["svm", "rf"] * 2, "subject": f.subject_id,
                              "window_start_s": f.window_start_s, "y_true": 0, "y_pred": 0,
                              "score": 0.0, "seed": 0})
        path = self.write(frame, "mixed.csv")
        with self.assertRaises(ValueError):
            errors.load_predictions(path)

    def test_full_run_finds_the_planted_patterns(self):
        path = self.write(self.old_format(), "cv.csv")
        out, res = errors.run(path, "svm", "binary", out_dir=os.path.join(self.tmp.name, "o"),
                              features_path=self.feat_path)
        # larger stress response -> better macro-F1, by construction
        self.assertGreater(res["rho"], 0.8)
        subj = res["subjects"].set_index("subject_id")
        self.assertAlmostEqual(subj.loc["S7", "stress_response"], 32, delta=2)
        for name in ("ERRORS.md", "per_subject.csv", "timelines.png", "error_by_motion.png",
                     "subject_response_vs_f1.png"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)

    def test_position_counts_from_one_in_each_block(self):
        preds, task, _ = errors.load_predictions(self.write(self.old_format(), "cv.csv"), "svm")
        table = errors.attach_context(preds, self.feat_path)
        first = table.groupby(["subject_id", "label"]).position.min()
        self.assertTrue((first == 1).all())
        self.assertEqual(int(table[table.label == 1].position.max()), 8)

    def test_unmatched_windows_are_refused(self):
        bad = self.old_format()
        bad.loc[0, "window_start_s"] = 99999.0
        preds, _, _ = errors.load_predictions(self.write(bad, "bad.csv"), "svm")
        with self.assertRaises(ValueError):
            errors.attach_context(preds, self.feat_path)


if __name__ == "__main__":
    unittest.main()
