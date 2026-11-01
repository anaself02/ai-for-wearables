"""Human false-caution labels: validation, screen vs human, the regex audit."""

import csv
import os
import tempfile
import unittest

from act import human, make_inputs, report, schema
from act.tests.isolation import setUpModule, tearDownModule  # noqa: F401
from act.tests.test_act import FakeClient, Workspace


def write_labels(path, rows, columns=human.COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class LabelFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "labels.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_labels_load(self):
        write_labels(self.path, [["I34", "sentence", "5", "yes", "missed"],
                                 ["I34", "table", "1", "No", ""]])
        self.assertEqual(human.load_labels(self.path),
                         {("I34", "sentence", 5): (True, "missed"),
                          ("I34", "table", 1): (False, "")})

    def test_bad_values_duplicates_and_columns_are_refused(self):
        write_labels(self.path, [["I34", "sentence", "5", "maybe", ""]])
        with self.assertRaises(human.LabelError):
            human.load_labels(self.path)
        write_labels(self.path, [["I34", "sentence", "5", "yes", ""],
                                 ["I34", "sentence", "5", "no", ""]])
        with self.assertRaises(human.LabelError):
            human.load_labels(self.path)
        write_labels(self.path, [["I34", "sentence", "5", "yes"]],
                     columns=["input_id", "presentation", "run_index", "label"])
        with self.assertRaises(human.LabelError):
            human.load_labels(self.path)

    def test_the_shipped_label_file_matches_the_owners_reading(self):
        labels = human.load_labels()
        self.assertEqual(len(labels), 60)
        self.assertEqual({k[0] for k in labels}, {"I03", "I18", "I31", "I32", "I33", "I34"})
        yes = sorted(k for k, (value, _) in labels.items() if value)
        self.assertEqual(yes, [("I34", "sentence", n) for n in range(1, 6)])


class CompareAndAuditTests(Workspace):
    TEXTS = {1: "This result cannot be interpreted.",            # screen yes
             2: "The result cannot be reliably interpreted.",    # screen no, human yes
             3: "18.3% of windows looked stress-like.",          # both no
             4: "Treat this with caution.",                      # regex hit, labelled no
             5: "All fine."}

    def setUp(self):
        super().setUp()
        built = make_inputs.build([])
        record = next(r for r in built if r["meta"].get("extension"))
        self.assertFalse(schema.caution_required(record["payload"]))
        self.input_id = record["payload"]["input_id"]
        import json
        with open(self.inputs, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.labels = os.path.join(self.tmp.name, "labels.csv")

    def run_texts(self):
        class ByRun(FakeClient):
            def __init__(self, texts):
                super().__init__("m")
                self.queue = [texts[n] for n in sorted(texts)] * 2   # table, then sentence

            def generate(self, system, user, settings):
                result = super().generate(system, user, settings)
                with self.lock:
                    result["text"] = self.queue.pop(0)
                return result
        self.go(ByRun(self.TEXTS), "--approve", "--repeats", "5", "--arm", "llm")

    def test_screen_and_human_counts_side_by_side_with_every_mismatch(self):
        self.run_texts()
        rows = [[self.input_id, pres, n, "yes" if n in (1, 2) else "no", ""]
                for pres in ("table", "sentence") for n in range(1, 6)]
        write_labels(self.labels, rows)
        result = human.compare(self.logged(), human.load_labels(self.labels))
        for pres in ("table", "sentence"):
            cell = result["cells"]["{}/{}".format(self.input_id, pres)]
            self.assertEqual((cell["outputs"], cell["screen"], cell["human"]), (5, 1, 2))
        self.assertEqual([m["key"][2] for m in result["mismatches"]], [2, 2])
        self.assertEqual(result["totals"], {"outputs": 10, "screen": 2, "human": 4,
                                            "unlabelled": 0})
        text = human.compare_markdown(result)
        self.assertIn("Mismatches (screen != human): 2", text)
        self.assertIn("cannot be reliably interpreted", text)

    def test_missing_labels_are_reported_not_guessed(self):
        self.run_texts()
        write_labels(self.labels, [[self.input_id, "table", 1, "yes", ""],
                                   ["I99", "table", 1, "no", ""]])
        result = human.compare(self.logged(), human.load_labels(self.labels))
        self.assertEqual(result["totals"]["unlabelled"], 9)
        self.assertEqual(result["orphans"], [("I99", "table", 1)])

    def test_the_audit_lists_regex_hits_not_labelled_yes(self):
        self.run_texts()
        rows = [[self.input_id, pres, n, "yes" if n in (1, 2) else "no", ""]
                for pres in ("table", "sentence") for n in range(1, 6)]
        write_labels(self.labels, rows)
        hits = human.audit(self.logged(), human.load_labels(self.labels))
        self.assertEqual(sorted((h["key"][1], h["key"][2]) for h in hits),
                         [("sentence", 4), ("table", 4)])
        self.assertEqual(hits[0]["matches"], ["caution"])

    def test_the_report_shows_the_side_by_side_section(self):
        self.run_texts()
        write_labels(self.labels, [[self.input_id, pres, n, "yes" if n in (1, 2) else "no", ""]
                                   for pres in ("table", "sentence") for n in range(1, 6)])
        result = report.build(self.logged(), labels=human.load_labels(self.labels))
        self.assertEqual(result["false_caution_human"]["totals"]["human"], 4)
        markdown = report.markdown(result)
        self.assertIn("### Screen vs human-adjudicated false cautions (LLM arm)", markdown)
        self.assertIn("| screen (phrase list) | human-adjudicated |", markdown)
