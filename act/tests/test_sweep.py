"""The EXPLORATORY OOD boundary sweep (I35-I40): kept out of every primary result."""

import json
import unittest

from act import human, make_inputs, report, run, schema, synthetic
from act.tests.isolation import setUpModule, tearDownModule  # noqa: F401
from act.tests.test_act import FakeClient, Workspace
from act.tests.test_extension import REAL


class SweepInputTests(unittest.TestCase):
    def test_I35_to_I40_are_I34_with_only_the_ood_share_changed(self):
        records = make_inputs.build(REAL)
        i34, sweep = records[33]["payload"], records[34:]
        self.assertEqual(i34["input_id"], "I34")
        self.assertEqual([r["payload"]["input_id"] for r in sweep],
                         ["I35", "I36", "I37", "I38", "I39", "I40"])
        self.assertEqual([r["payload"]["share_out_of_distribution"] for r in sweep],
                         [0.15, 0.20, 0.22, 0.23, 0.25, 0.26])
        for r in sweep:
            self.assertTrue(r["meta"]["exploratory"])
            self.assertEqual(r["meta"]["source"], "synthetic")
            differ = {k for k in schema.FIELDS if r["payload"][k] != i34[k]}
            self.assertEqual(differ, {"input_id", "day_index", "share_out_of_distribution"})
        self.assertEqual([schema.caution_required(r["payload"]) for r in sweep],
                         [False, False, False, False, True, True])

    def test_adding_the_sweep_left_I01_to_I34_unchanged(self):
        records = make_inputs.build(REAL)
        self.assertEqual(records[:30], make_inputs.build(REAL, extension=False))
        self.assertEqual([r["meta"]["case"] for r in records[30:34]],
                         [case for case, _ in synthetic.EXTENSION_CASES])


class SweepRunTests(Workspace):
    def setUp(self):
        super().setUp()
        records = make_inputs.build([])
        self.sweep = [r for r in records if r["meta"].get("exploratory")]
        self.primary = [r for r in records if r["meta"].get("extension")][:1]
        with open(self.inputs, "w", encoding="utf-8") as handle:
            for record in self.primary + self.sweep:
                handle.write(json.dumps(record) + "\n")
        self.sweep_ids = [r["payload"]["input_id"] for r in self.sweep]

    def test_default_and_synthetic_only_plans_leave_the_sweep_out(self):
        for extra in ((), ("--synthetic-only",)):
            summary, out = self.go(FakeClient("m"), "--repeats", "5", *extra)
            self.assertEqual(summary["planned"], 1 * 2 * 5, extra)
            for input_id in self.sweep_ids:
                self.assertNotIn("input {} ".format(input_id), out)

    def test_naming_the_sweep_in_only_plans_it_and_marks_rows_exploratory(self):
        summary, _ = self.go(FakeClient("m"), "--repeats", "5", "--only", ",".join(self.sweep_ids),
                             "--synthetic-only")
        self.assertEqual(summary["planned"], 6 * 2 * 5)
        self.go(FakeClient("m"), "--approve", "--repeats", "5", "--arm", "llm",
                "--only", self.sweep_ids[0])
        self.assertTrue(all(r["exploratory"] for r in self.logged()))

    def test_sweep_rows_are_excluded_from_primary_metrics_and_reported_apart(self):
        refusing = FakeClient("m", text="This result cannot be interpreted.")
        self.go(refusing, "--approve", "--repeats", "5", "--arm", "llm")              # primary
        self.go(refusing, "--approve", "--repeats", "5", "--arm", "llm",
                "--only", ",".join(self.sweep_ids))                                   # sweep
        result = report.build(self.logged(), labels={})
        self.assertEqual(sum(c["usage"]["calls"] for c in result["cells"].values()), 1 * 2 * 5)
        self.assertEqual(result["false_caution_human"]["totals"]["outputs"], 1 * 2 * 5)
        cells = result["exploratory"]["cells"]
        self.assertEqual(len(cells), 6 * 2)
        low = cells["{}/table".format(self.sweep_ids[0])]
        self.assertEqual((low["rule3"], low["screen"], low["unlabelled"]), (False, 5, 5))
        high = cells["{}/table".format(self.sweep_ids[-1])]
        self.assertEqual((high["rule3"], high["caution_pass"]), (True, 5))
        self.assertEqual(result["exploratory"]["human"]["totals"]["outputs"], 4 * 2 * 5)
        markdown = report.markdown(result)
        self.assertIn("## EXPLORATORY: out-of-distribution boundary sweep (I35-I40)", markdown)
        primary_part = markdown.split("## EXPLORATORY")[0]
        for input_id in self.sweep_ids:
            self.assertNotIn("| {} |".format(input_id), primary_part)

    def test_sweep_labels_are_not_orphans_of_the_primary_comparison(self):
        self.go(FakeClient("m"), "--approve", "--repeats", "5", "--arm", "llm",
                "--only", self.sweep_ids[0])
        labels = {(self.sweep_ids[0], "table", 1): (False, "")}
        self.assertEqual(human.compare(self.logged(), labels)["orphans"], [])
        self.assertEqual(human.compare(self.logged(), labels, exploratory=True)
                         ["totals"]["unlabelled"], 9)


if __name__ == "__main__":
    unittest.main()
