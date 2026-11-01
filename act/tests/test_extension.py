"""I31-I34 (appended inputs), the false-caution metric and the hand-reading view."""

import io
import json
import os
import unittest

from act import config, make_inputs, presentation, report, rubric, schema, show, synthetic
from act import template_arm
from act.run import settings
from act.tests.isolation import setUpModule, tearDownModule  # noqa: F401
from act.tests.test_act import FakeClient, Workspace, payload

REAL = [{"order": n, "summary": {**{k: v for k, v in payload().items()
                                    if k not in ("input_id", "day_index")},
                                 "mean_bpm": 60.0 + n, "share_out_of_distribution": 0.5}}
        for n in range(1, 23)]


class AppendedInputsTests(unittest.TestCase):
    def test_appending_keeps_every_earlier_id_payload_and_prompt_hash(self):
        before = make_inputs.build(REAL, extension=False)
        after = make_inputs.build(REAL)
        self.assertEqual(len(before), 30)
        self.assertEqual(len(after), 30 + 4 + 6)        # + I31-I34, + sweep I35-I40
        self.assertEqual(after[:30], before)
        for old, new in zip(before, after):
            for pres in presentation.PRESENTATIONS:
                self.assertEqual(
                    presentation.prompt_hash(presentation.build_prompt(old["payload"], pres),
                                             config.MODEL, settings()),
                    presentation.prompt_hash(presentation.build_prompt(new["payload"], pres),
                                             config.MODEL, settings()))

    def test_the_new_inputs_are_I31_to_I34_synthetic_and_unflagged(self):
        new = make_inputs.build(REAL)[30:34]
        self.assertEqual([r["payload"]["input_id"] for r in new], ["I31", "I32", "I33", "I34"])
        self.assertEqual([r["payload"]["day_index"] for r in new], [31, 32, 33, 34])
        for record in new:
            p = record["payload"]
            self.assertEqual(record["meta"]["source"], "synthetic")
            self.assertFalse(schema.caution_required(p), p["input_id"])
            self.assertLess(p["share_out_of_distribution"], config.OOD_FLAG_SHARE)
        self.assertEqual(max(r["payload"]["share_predicted_stress"] for r in new), 0.55)
        self.assertEqual(len({r["payload"]["mean_bpm"] for r in new}), 4)

    def test_rule_3_does_not_apply_to_exactly_six_inputs_with_the_real_layout(self):
        records = [r for r in make_inputs.build(REAL) if not r["meta"].get("exploratory")]
        unflagged = [r["payload"]["input_id"] for r in records
                     if not schema.caution_required(r["payload"])]
        self.assertEqual(len(unflagged), 2 + len(synthetic.EXTENSION_CASES))   # primary set


class FalseCautionTests(unittest.TestCase):
    def test_the_frozen_phrase_list(self):
        self.assertEqual(rubric.REFUSAL_PHRASES, [
            "cannot be interpreted", "not interpretable", "unreliable", "not reliable",
            "cannot be relied", "should not be relied", "should not be used",
            "not meaningful", "inconclusive"])

    def test_a_refusal_on_an_unflagged_input_is_a_false_caution(self):
        calm = payload()
        self.assertFalse(schema.caution_required(calm))
        for text in ("This result cannot be interpreted.", "The day is INCONCLUSIVE.",
                     "These numbers are unreliable.", "It should not be used."):
            self.assertTrue(rubric.false_caution(text, calm)[0], text)
        self.assertFalse(rubric.false_caution("18.3% of 2879 windows looked stress-like.", calm)[0])
        self.assertFalse(rubric.false_caution("5% of windows fall outside the training data.",
                                              calm)[0])

    def test_it_is_not_defined_where_rule_3_applies(self):
        flagged = payload(partial=True)
        self.assertIsNone(rubric.false_caution("This cannot be interpreted.", flagged)[0])

    def test_it_is_not_one_of_the_pre_declared_checks(self):
        self.assertEqual(rubric.CHECKS, ("numbers", "medical", "caution", "sentences"))
        self.assertNotIn("false_caution", rubric.score("x.", payload()))

    def test_the_template_arm_never_false_cautions(self):
        for record in make_inputs.build(REAL):
            if not schema.caution_required(record["payload"]):
                text = template_arm.feedback(record["payload"])
                self.assertFalse(rubric.false_caution(text, record["payload"])[0], text)


class ReportAndShowTests(Workspace):
    def setUp(self):
        super().setUp()
        built = make_inputs.build([])          # no real rows: the extension is I09-I12 here
        records = [r for r in built if r["meta"].get("extension")][:2]
        flagged = next(r for r in built if schema.caution_required(r["payload"]))
        self.unflagged_ids = {r["payload"]["input_id"] for r in records}
        with open(self.inputs, "w", encoding="utf-8") as handle:
            for record in records + [flagged]:
                handle.write(json.dumps(record) + "\n")
        self.flagged_id = flagged["payload"]["input_id"]

    def test_the_report_has_a_separate_false_caution_section(self):
        refusing = FakeClient("m", text="This day cannot be interpreted.")
        self.go(refusing, "--approve", "--repeats", "5")
        result = report.build(self.logged())
        self.assertEqual(result["false_caution"]["llm/table"][0], 1.0)
        self.assertEqual(result["false_caution"]["template/table"][0], 0.0)
        self.assertEqual(result["false_caution"]["B-A/table"][0], 1.0)
        self.assertEqual(len(result["false_caution_outputs"]), 2 * 2 * 5)
        markdown = report.markdown(result)
        self.assertIn("## False caution (separate metric; unflagged inputs only)", markdown)
        self.assertNotIn("false caution", result["cells"]["llm/table"]["pass"])

    def test_unflagged_view_prints_both_arms_only_for_unflagged_inputs(self):
        self.go(FakeClient("m"), "--approve", "--repeats", "5")
        out = io.StringIO()
        rows = show.main(["--unflagged", "--no-pilot"], log_path=self.log, out=out)
        self.assertEqual({r["arm"] for r in rows}, {"llm", "template"})
        self.assertEqual({r["input_id"] for r in rows}, self.unflagged_ids)
        self.assertNotIn(self.flagged_id, {r["input_id"] for r in rows})
        self.assertEqual(len(rows), 2 * 2 * 5 + 2 * 2)
        text = out.getvalue()
        self.assertIn("payload: ", text)
        self.assertIn("false caution: no", text)
        self.assertIn("raw_response:", text)
        self.assertFalse(os.path.abspath(self.log).startswith(
            os.path.abspath(os.path.join(config.ACT_DIR, "runs"))))
