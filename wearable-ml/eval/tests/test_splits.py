"""Tests for eval/splits.py: the saved splits are complete, disjoint and match the code."""

import csv
import json
import os
import tempfile
import unittest

from eval import config, folds, splits


def rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class SplitFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = splits.write(cls.tmp.name)
        cls.dir = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_five_files_are_written(self):
        names = sorted(os.path.basename(p) for p in self.paths)
        self.assertEqual(names, ["inner_folds.csv", "outer_folds.csv", "seeds.json",
                                 "subjects.csv", "windows.csv"])

    def test_window_table_has_subject_and_label_for_every_window(self):
        table = rows(os.path.join(self.dir, "windows.csv"))
        self.assertEqual(len(table), 1015)
        self.assertEqual({r["label"] for r in table}, {"1", "2", "3"})
        self.assertEqual(len({r["subject_id"] for r in table}), 15)
        stress = sum(1 for r in table if r["binary_label"] == "1")
        self.assertEqual(stress, sum(1 for r in table if r["label"] == "2"))

    def test_outer_folds_are_leave_one_subject_out(self):
        table = rows(os.path.join(self.dir, "outer_folds.csv"))
        self.assertEqual(len(table), 15)
        everyone = {r["test_subject"] for r in table}
        for r in table:
            train = r["train_subjects"].split()
            self.assertEqual(len(train), 14)
            self.assertNotIn(r["test_subject"], train)
            self.assertEqual(set(train) | {r["test_subject"]}, everyone)

    def test_inner_folds_are_seven_disjoint_groups_of_two_per_seed_and_fold(self):
        table = rows(os.path.join(self.dir, "inner_folds.csv"))
        self.assertEqual(len(table), 15 * len(config.SEEDS) * config.INNER_FOLDS)
        outer = {r["test_subject"]: set(r["train_subjects"].split())
                 for r in rows(os.path.join(self.dir, "outer_folds.csv"))}
        for seed in config.SEEDS:
            for test, train in outer.items():
                groups = [r["validation_subjects"].split() for r in table
                          if int(r["seed"]) == seed and r["outer_test_subject"] == test]
                self.assertEqual(len(groups), 7)
                self.assertTrue(all(len(g) == 2 for g in groups))
                flat = [s for g in groups for s in g]
                self.assertEqual(sorted(flat), sorted(train))      # each training subject once
                self.assertNotIn(test, flat)

    def test_seeds_file_records_the_frozen_values(self):
        with open(os.path.join(self.dir, "seeds.json"), encoding="utf-8") as handle:
            info = json.load(handle)
        self.assertEqual(info["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(info["bootstrap_seed"], 6322)
        self.assertEqual(info["inner_folds"], 7)
        self.assertEqual(info["features_sha256_crlf_to_lf"], config.FEATURES_SHA256)

    def test_writing_twice_gives_identical_files(self):
        with tempfile.TemporaryDirectory() as other:
            splits.write(other)
            for name in ("windows.csv", "outer_folds.csv", "inner_folds.csv", "seeds.json"):
                with open(os.path.join(self.dir, name), "rb") as a, open(os.path.join(other, name), "rb") as b:
                    self.assertEqual(a.read(), b.read(), name)

    def test_inner_assignment_equals_the_folds_module(self):
        table = rows(os.path.join(self.dir, "inner_folds.csv"))
        outer = {r["test_subject"]: r["train_subjects"].split()
                 for r in rows(os.path.join(self.dir, "outer_folds.csv"))}
        for seed, test in ((0, "S2"), (3, "S17")):
            expected = folds.inner_folds(outer[test], seed)
            got = [r["validation_subjects"].split() for r in table
                   if int(r["seed"]) == seed and r["outer_test_subject"] == test]
            self.assertEqual(got, expected)


@unittest.skipUnless(os.path.exists(config.FOLDS_FILE), "eval/folds.json not present")
class SavedFoldsFileTests(unittest.TestCase):
    def test_saved_folds_json_equals_the_folds_the_code_makes(self):
        self.assertEqual(splits.differences_with_folds_json(), [])


if __name__ == "__main__":
    unittest.main()
