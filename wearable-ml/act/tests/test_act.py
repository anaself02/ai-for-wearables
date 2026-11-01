"""ACT tests. A fake client only: no network, no API key, no real data.

    python -m unittest discover -s act/tests -t .      (from wearable-ml)
"""

import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from act import config, llm, make_inputs, presentation, rubric, run, schema, spotcheck, synthetic
from act import template_arm
from act import report
from act.tests.isolation import setUpModule, tearDownModule  # noqa: F401  (module fixtures)


def payload(**overrides):
    base = {"input_id": "I01", "day_index": 1, "n_windows": 2879, "mean_bpm": 72.4,
            "share_predicted_stress": 0.183, "share_out_of_distribution": 0.05,
            "degenerate": False, "partial": False, "coverage_hours": 23.5,
            "median_spacing_s": 2.9}
    base.update(overrides)
    return base


class FakeClient:
    """Stands in for GeminiClient. Can be told to fail on given call numbers."""

    def __init__(self, model, fail_on=None, error=None, text=None, delay=0.0):
        self.model = model
        self.sdk = "fake-sdk 0.0"
        self.calls = 0
        self.fail_on = set(fail_on or ())
        self.error = error
        self.text = text
        self.delay = delay
        self.starts = []
        self.lock = threading.Lock()

    def check_model(self):
        return "models/" + self.model

    def generate(self, system, user, settings):
        with self.lock:
            self.calls += 1
            number = self.calls
            self.starts.append(time.perf_counter())
        if self.delay:
            time.sleep(self.delay)
        if number in self.fail_on:
            raise self.error
        return {"text": self.text or "Day 1: 18.3% of 2879 windows looked stress-like.",
                "finish_reason": "STOP", "latency_s": 0.01,
                "tokens": {"prompt": 100, "output": 20, "thoughts": 0, "total": 120}}


class Workspace(unittest.TestCase):
    """Inputs (synthetic only) and a log in a temporary folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inputs = os.path.join(self.tmp.name, "inputs.jsonl")
        self.log = os.path.join(self.tmp.name, "runs", "calls.jsonl")
        records = make_inputs.build([])[:2]
        with open(self.inputs, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return run.parse_args(["--inputs", self.inputs, "--log", self.log, "--rpm", "6000",
                               "--repeats", "2", *extra])

    def go(self, client, *extra, sleep=lambda s: None):
        out = io.StringIO()
        summary = run.run(self.args(*extra), client_factory=lambda model: client,
                          out=out, sleep=sleep)
        return summary, out.getvalue()

    def logged(self):
        return run.read_log(self.log)


class AllowlistTests(unittest.TestCase):
    def test_a_valid_payload_passes(self):
        self.assertEqual(schema.validate_payload(payload()), payload())

    def test_extra_and_missing_fields_are_rejected(self):
        for extra in ({"note": "felt tense"}, {"date": "2026-10-30"}, {"name": "x"},
                      {"readings": [70, 71, 72]}, {"timestamp": 1730000000}):
            with self.assertRaises(schema.PayloadRejected, msg=extra):
                schema.validate_payload(payload(**extra))
        missing = payload()
        del missing["mean_bpm"]
        with self.assertRaises(schema.PayloadRejected):
            schema.validate_payload(missing)

    def test_ids_dates_names_and_wrong_types_are_rejected(self):
        bad = [{"input_id": "2026-10-30"}, {"input_id": "anas-1"}, {"input_id": "Rahil"},
               {"day_index": "3"}, {"mean_bpm": "72"}, {"degenerate": 1},
               {"n_windows": 12.5}, {"share_predicted_stress": 1.5},
               {"mean_bpm": float("nan")}, {"day_index": True}, {"coverage_hours": [1, 2]}]
        for change in bad:
            with self.assertRaises(schema.PayloadRejected, msg=change):
                schema.validate_payload(payload(**change))

    def test_the_finished_prompt_is_checked_for_dates_times_and_names(self):
        for text in ("on 2026-10-30", "at 14:05", "Monday was busy", "Oct 30 data",
                     "Anas's day", "for rahil"):
            with self.assertRaises(schema.PayloadRejected, msg=text):
                schema.assert_safe_text(text)
        schema.assert_safe_text(presentation.SYSTEM_PROMPT)

    def test_only_anas_may_be_used(self):
        self.assertEqual(schema.require_allowed_person("anas"), "anas")
        for slug in ("rahil", "Anas", "", None):
            with self.assertRaises(schema.PayloadRejected):
                schema.require_allowed_person(slug)

    def test_prompts_carry_the_numbers_but_never_the_input_id(self):
        for pres in presentation.PRESENTATIONS:
            prompt = presentation.build_prompt(payload(input_id="I07"), pres)
            self.assertNotIn("I07", prompt["user"])
            self.assertIn("0.183", prompt["user"])
            self.assertIn("2879", prompt["user"])
        with self.assertRaises(schema.PayloadRejected):
            presentation.build_prompt(payload(note="secret"), "table")

    def test_inputs_mark_real_and_synthetic_and_meta_is_never_in_a_prompt(self):
        real = [{"order": 1, "summary": {k: v for k, v in payload().items()
                                         if k not in ("input_id", "day_index")}}]
        records = make_inputs.build(real)
        self.assertEqual(len(records), 1 + len(synthetic.CASES) + len(synthetic.EXTENSION_CASES)
                         + len(synthetic.EXPLORATORY_CASES))
        self.assertEqual(sum(r["meta"]["source"] == "real" for r in records), 1)
        for record in records:
            schema.validate_payload(record["payload"])
            prompt = presentation.build_prompt(record["payload"], "sentence")["user"]
            self.assertNotIn("synthetic", prompt)
            self.assertNotIn("real", prompt.lower().replace("readings", ""))


class DryRunTests(Workspace):
    def test_dry_run_is_the_default_and_sends_and_writes_nothing(self):
        self.assertFalse(run.parse_args([]).approve)
        client = FakeClient("m")
        with mock.patch.object(llm, "load_api_key", side_effect=AssertionError("key read")):
            summary, out = self.go(client)
        self.assertEqual(client.calls, 0)
        self.assertFalse(os.path.exists(self.log))
        self.assertEqual(summary["planned"], 2 * 2 * 2)
        self.assertEqual(out.count("=== DRY RUN call"), 8)
        self.assertIn('"contents"', out)
        self.assertIn("Use only the numbers given", out)

    def test_approve_sends_and_logs_every_field(self):
        client = FakeClient("m")
        summary, _ = self.go(client, "--approve")
        self.assertEqual(client.calls, 8)
        records = self.logged()
        self.assertEqual(sum(r["arm"] == "template" for r in records), 4)
        llm_record = next(r for r in records if r["arm"] == "llm")
        for field in ("arm", "presentation", "input_id", "run_index", "model", "sdk_version",
                      "date", "prompt_hash", "prompt", "raw_response", "finish_reason",
                      "latency_s", "latency_total_s", "retries", "pilot",
                      "sdk_retry_attempts", "tokens", "error"):
            self.assertIn(field, llm_record, field)
        self.assertEqual(llm_record["sdk_version"], "fake-sdk 0.0")
        self.assertEqual(llm_record["model"], config.MODEL)


class ResumeAndQuotaTests(Workspace):
    def test_a_finished_run_is_not_sent_again(self):
        self.go(FakeClient("m"), "--approve")
        again = FakeClient("m")
        summary, out = self.go(again, "--approve")
        self.assertEqual(again.calls, 0)
        self.assertIn("Nothing left to send", out)
        self.assertEqual(len(self.logged()), 12)

    def test_a_quota_error_stops_the_run_and_a_rerun_finishes_only_the_rest(self):
        first = FakeClient("m", fail_on={3}, error=llm.QuotaExhausted("per day"))
        summary, _ = self.go(first, "--approve")
        self.assertEqual(first.calls, 3)
        self.assertIn("quota", summary["stopped"])
        self.assertEqual(sum(1 for r in self.logged() if r["arm"] == "llm" and not r["error"]), 2)
        second = FakeClient("m")
        self.go(second, "--approve")
        self.assertEqual(second.calls, 6)
        ok = [r for r in self.logged() if r["arm"] == "llm" and r["error"] is None]
        self.assertEqual(len({(r["prompt_hash"], r["run_index"]) for r in ok}), 8)

    def test_a_per_minute_limit_is_retried(self):
        client = FakeClient("m", fail_on={1}, error=llm.RateLimited("slow down", retry_after=1))
        summary, out = self.go(client, "--approve", "--limit", "1")
        self.assertIn("retry 1", out)
        self.assertEqual(summary["sent"], 1)

    def test_the_daily_budget_is_respected(self):
        client = FakeClient("m")
        summary, _ = self.go(client, "--approve", "--rpd", "3")
        self.assertEqual(client.calls, 3)
        self.assertIn("daily request budget", summary["stopped"])

    def test_errors_are_classified_and_the_key_is_redacted(self):
        class Err(Exception):
            def __init__(self, code, message):
                super().__init__(message)
                self.code = code
        key = "AIzaFAKEFAKEFAKEFAKEFAKEFAKE123"
        per_day = llm.classify(Err(429, "quotaId GenerateRequestsPerDayPerProject " + key), key)
        self.assertIsInstance(per_day, llm.QuotaExhausted)
        self.assertNotIn(key, str(per_day))
        per_minute = llm.classify(Err(429, "RESOURCE_EXHAUSTED 'retryDelay': '17s'"))
        self.assertIsInstance(per_minute, llm.RateLimited)
        self.assertEqual(per_minute.retry_after, 17.0)
        self.assertTrue(llm.classify(Err(503, "unavailable")).retryable)
        self.assertFalse(llm.classify(Err(400, "bad request")).retryable)


class NumberCheckTests(unittest.TestCase):
    def check(self, text, **overrides):
        return rubric.check_numbers(text, payload(**overrides))

    def test_numbers_at_the_precision_shown_match(self):
        for text in ("2,879 windows", "2879 windows", "mean 72.4 bpm", "about 72 bpm",
                     "18.3% stress-like", "18% of windows", "share 0.183", "0.18",
                     "23.5 hours", "2.9 s apart", "Day 1:", "under 25%", "below 20 hours"):
            self.assertTrue(self.check(text)[0], text)

    def test_invented_or_misrounded_numbers_fail(self):
        for text in ("mean 73 bpm", "19% of windows", "3,000 windows", "about 72.5 bpm",
                     "0.19 share", "45% outside", "a 10 minute walk"):
            ok, unmatched = self.check(text)
            self.assertFalse(ok, text)
            self.assertTrue(unmatched, text)

    def test_percent_is_only_compared_with_shares(self):
        self.assertFalse(self.check("72.4%")[0])
        self.assertTrue(self.check("5% outside", share_out_of_distribution=0.05)[0])


class OtherCheckTests(unittest.TestCase):
    def test_caution_is_required_only_when_flagged(self):
        flagged = payload(share_out_of_distribution=0.6)
        self.assertEqual(rubric.check_caution("All fine today.", payload())[0], None)
        self.assertFalse(rubric.check_caution("A calm day.", flagged)[0])
        self.assertTrue(rubric.check_caution("A calm day. This cannot be interpreted.", flagged)[0])
        self.assertTrue(rubric.check_caution("Most windows are outside the training data.",
                                             flagged)[0])
        self.assertFalse(rubric.check_caution("A calm day.", payload(degenerate=True))[0])
        self.assertFalse(rubric.check_caution("A calm day.", payload(partial=True))[0])

    def test_medical_wording_is_caught_unless_negated(self):
        self.assertFalse(rubric.check_medical("This may indicate anxiety.")[0])
        self.assertFalse(rubric.check_medical("Consider seeing a doctor.")[0])
        self.assertFalse(rubric.check_medical("A diagnostic signal.")[0])
        self.assertTrue(rubric.check_medical("This is not a medical assessment.")[0])
        self.assertTrue(rubric.check_medical("A stress-like pattern, no more.")[0])

    def test_sentence_count_ignores_decimal_points(self):
        self.assertEqual(rubric.check_sentences("Mean 72.4 bpm. Share 0.183. Fine.")[1], 3)
        self.assertFalse(rubric.check_sentences("One. Two. Three. Four.")[0])
        self.assertFalse(rubric.check_sentences("")[0])

    def test_the_template_arm_passes_its_own_rubric_on_every_synthetic_case(self):
        for record in make_inputs.build([]):
            text = template_arm.feedback(record["payload"])
            for check, (passed, detail) in rubric.score(text, record["payload"]).items():
                self.assertNotEqual(passed, False, (record["meta"]["case"], check, detail, text))
            if schema.caution_required(record["payload"]):
                self.assertIn("cannot be interpreted", text)


class ReportAndSpotcheckTests(Workspace):
    def test_report_and_blinded_spotcheck_round_trip(self):
        self.go(FakeClient("m"), "--approve", "--repeats", "5")       # not a pilot
        self.go(FakeClient("m"), "--approve")                         # a 2-repeat pilot
        self.assertEqual(sum(1 for r in self.logged() if r.get("pilot")), 8)
        result = report.build(self.logged())
        self.assertEqual(result["cells"]["llm/table"]["usage"]["calls"], 2 * 5)  # pilots left out
        self.assertEqual(report.build(self.logged(), include_pilot=True)
                         ["cells"]["llm/table"]["usage"]["calls"], 2 * 5 + 2 * 2)
        self.assertIn("llm/table", result["cells"])
        self.assertIn("numbers/table", result["differences"])
        markdown = report.markdown(result)
        self.assertIn("Threats to validity", markdown)
        self.assertIn("training data", markdown)

        fill = os.path.join(self.tmp.name, "fill.csv")
        key = os.path.join(self.tmp.name, "key.csv")
        n = spotcheck.make(self.logged(), n=8, to_fill=fill, key=key)
        with open(fill, encoding="utf-8") as handle:
            text = handle.read()
        for hidden in ("template", "llm", "table", "I0"):
            self.assertNotIn(hidden, text.split("\n", 1)[1], hidden)
        rows = text.strip().splitlines()
        self.assertEqual(len(rows), n + 1)
        filled = [rows[0]] + [line.rstrip(",") + ",2," if i % 2 else line
                              for i, line in enumerate(rows[1:], 1)]
        with open(fill, "w", encoding="utf-8") as handle:
            handle.write("\n".join(filled) + "\n")
        result, missing, invalid = spotcheck.aggregate(to_fill=fill, key=key)
        self.assertEqual(invalid, [])
        self.assertEqual(sum(v["n"] for v in result.values()) + missing, n)


if __name__ == "__main__":
    unittest.main()
