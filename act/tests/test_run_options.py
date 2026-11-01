"""--only, --repeats (pilots), --workers, retries and SDK settings. Fake client only."""

import json
import os
import random
import time
import unittest

from act import config, llm, make_inputs, report, run, spotcheck
from act.tests.isolation import REAL_RUNS, setUpModule, tearDownModule  # noqa: F401
from act.tests.test_act import FakeClient, Workspace, payload


class OnlyAndRepeatsTests(Workspace):
    def setUp(self):
        super().setUp()
        real_summary = {k: v for k, v in payload().items() if k not in ("input_id", "day_index")}
        records = make_inputs.build([{"order": 1, "summary": real_summary}])
        records = ([r for r in records if r["meta"]["source"] == "real"]
                   + [r for r in records if r["meta"]["source"] == "synthetic"][:3])
        with open(self.inputs, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        self.real = [r["payload"]["input_id"] for r in records if r["meta"]["source"] == "real"]
        self.synthetic = [r["payload"]["input_id"] for r in records
                          if r["meta"]["source"] == "synthetic"]

    def test_only_restricts_the_plan_and_the_dry_run_shows_it(self):
        pick = self.synthetic[:2]
        summary, out = self.go(FakeClient("m"), "--only", ",".join(pick), "--synthetic-only")
        self.assertEqual(summary["planned"], 2 * 2 * 2)
        listed = out.split("\n", 2)[1]                          # line after the pilot warning
        self.assertTrue(listed.startswith("PLAN: 2 input(s):"), listed)
        for input_id in pick:
            self.assertIn("{} (synthetic)".format(input_id), listed)
        shown = {line.split()[5] for line in out.splitlines() if line.startswith("=== DRY RUN call")}
        self.assertEqual(shown, set(pick))
        self.assertFalse(os.path.exists(self.log))

    def test_only_combines_with_limit(self):
        client = FakeClient("m")
        summary, _ = self.go(client, "--approve", "--synthetic-only", "--only",
                             self.synthetic[0], "--limit", "3")
        self.assertEqual(client.calls, 3)
        self.assertIn("limit", summary["stopped"])
        self.assertEqual({r["input_id"] for r in self.logged()}, {self.synthetic[0]})

    def test_synthetic_only_refuses_a_real_id_and_sends_nothing(self):
        client = FakeClient("m")
        with self.assertRaisesRegex(run.UsageError, "refuses real inputs: {}".format(self.real[0])):
            self.go(client, "--approve", "--synthetic-only", "--only",
                    "{},{}".format(self.synthetic[0], self.real[0]))
        self.assertEqual(client.calls, 0)
        self.assertFalse(os.path.exists(self.log))

    def test_unknown_or_malformed_ids_are_refused(self):
        for value in ("I99", "I2", "2026-10-30", "I01;x"):
            with self.assertRaises(run.UsageError, msg=value):
                self.go(FakeClient("m"), "--only", value)

    def test_repeats_other_than_5_warn_and_mark_rows_pilot(self):
        self.assertEqual(run.parse_args([]).repeats, 5)
        _, out = self.go(FakeClient("m"), "--approve", "--only", self.synthetic[0])
        self.assertIn("is not the pre-declared 5. This is a PILOT", out)
        self.assertTrue(all(r["pilot"] for r in self.logged() if r["arm"] == "llm"))
        _, out = self.go(FakeClient("m"), "--approve", "--only", self.synthetic[1],
                         "--repeats", "5")
        self.assertNotIn("PILOT", out)
        self.assertFalse(any(r["pilot"] for r in self.logged()
                             if r["arm"] == "llm" and r["input_id"] == self.synthetic[1]))

    def test_pilot_rows_never_stand_in_for_a_reported_run(self):
        self.go(FakeClient("m"), "--approve", "--only", self.synthetic[0])      # 2-repeat pilot
        client = FakeClient("m")
        self.go(client, "--approve", "--only", self.synthetic[0], "--repeats", "5")
        self.assertEqual(client.calls, 2 * 5)          # runs 1-2 re-sent, not taken from the pilot

    def test_rows_matching_the_pilot_registry_are_pilots(self):
        entry = {"prompt_hash": "a" * 64, "until": "2026-11-01T17:15:25+00:00"}
        before = {"prompt_hash": "a" * 64, "date": "2026-11-01T17:12:37+00:00"}
        after = {"prompt_hash": "a" * 64, "date": "2026-11-01T18:00:00+00:00"}
        self.assertTrue(run.is_pilot(before, [entry]))
        self.assertFalse(run.is_pilot(after, [entry]))
        self.assertTrue(run.is_pilot({**after, "pilot": True}, [entry]))
        registry = run.load_pilots()
        self.assertTrue(registry and registry[0]["prompt_hash"].startswith("f5b21d30"))


class ConcurrencyTests(Workspace):
    def test_workers_do_not_duplicate_or_interleave_calls(self):
        client = FakeClient("m", delay=0.02)
        self.go(client, "--approve", "--workers", "4")
        self.assertEqual(client.calls, 8)
        with open(self.log, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle.read().splitlines()]   # whole lines
        keys = [(r["prompt_hash"], r["run_index"]) for r in rows if r["arm"] == "llm"]
        self.assertEqual(len(keys), 8)
        self.assertEqual(len(set(keys)), 8)
        again = FakeClient("m")
        self.go(again, "--approve", "--workers", "4")
        self.assertEqual(again.calls, 0)                        # the cache still works

    def test_the_rate_limit_holds_across_workers(self):
        client = FakeClient("m", delay=0.05)
        began = time.monotonic()
        self.go(client, "--approve", "--workers", "4", "--rpm", "600", sleep=time.sleep)
        starts = sorted(client.starts)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertEqual(len(starts), 8)
        self.assertGreaterEqual(min(gaps), 0.1 - 0.01, gaps)   # 600/min = one start per 0.1 s
        self.assertGreaterEqual(time.monotonic() - began, 7 * 0.1 - 0.01)

    def test_a_failed_call_is_recorded_and_the_rest_continue(self):
        client = FakeClient("m", fail_on={2}, error=llm.CallFailed("400 bad request"))
        summary, _ = self.go(client, "--approve", "--workers", "3")
        self.assertEqual(client.calls, 8)
        self.assertEqual((summary["sent"], summary["errors"], summary["stopped"]), (7, 1, None))
        errors = [r for r in self.logged() if r["error"]]
        self.assertEqual(len(errors), 1)
        self.assertIn("400 bad request", errors[0]["error"])
        retry = FakeClient("m")
        self.go(retry, "--approve")
        self.assertEqual(retry.calls, 1)                        # only the failed one is resent

    def test_a_quota_error_stops_new_calls_across_workers(self):
        client = FakeClient("m", fail_on={1}, error=llm.QuotaExhausted("per day"), delay=0.01)
        summary, _ = self.go(client, "--approve", "--workers", "2", "--rpm", "600",
                             sleep=time.sleep)
        self.assertIn("quota", summary["stopped"])
        self.assertLess(client.calls, 8)

    def test_retries_are_counted_in_the_row(self):
        client = FakeClient("m", fail_on={1}, error=llm.RateLimited("slow", retry_after=0.001))
        self.go(client, "--approve", "--limit", "1")
        row = next(r for r in self.logged() if r["arm"] == "llm")
        self.assertEqual(row["retries"], 1)
        self.assertGreaterEqual(row["latency_total_s"], 0.0)     # measured around the retries
        self.assertEqual(row["sdk_retry_attempts"], 1)

    def test_the_report_does_not_depend_on_completion_order(self):
        self.go(FakeClient("m"), "--approve", "--workers", "4", "--repeats", "5")
        rows = self.logged()
        shuffled = rows[:]
        random.Random(1).shuffle(shuffled)
        self.assertEqual(report.markdown(report.build(rows)),
                         report.markdown(report.build(shuffled)))


class ClassifyAndSdkTests(unittest.TestCase):
    def test_timeouts_and_auth_errors_are_classified(self):
        class ReadTimeout(Exception):
            pass

        class Err(Exception):
            def __init__(self, code):
                super().__init__("error {}".format(code))
                self.code = code
        self.assertTrue(llm.classify(ReadTimeout("timed out")).retryable)
        self.assertTrue(llm.classify(Err(408)).retryable)
        self.assertIsInstance(llm.classify(Err(401)), llm.StopRun)
        self.assertIsInstance(llm.classify(Err(403)), llm.StopRun)

    def test_the_sdk_client_is_built_with_one_attempt_and_a_timeout(self):
        try:
            from google import genai  # noqa: F401
        except ImportError:
            self.skipTest("google-genai not installed")
        client = llm.GeminiClient("m", "AIzaTESTTESTTESTTESTTESTTESTTEST")   # no request made
        api = client._client._api_client
        self.assertEqual(api._retry.stop.max_attempt_number, 1)
        self.assertEqual(api._http_options.timeout, config.REQUEST_TIMEOUT_S * 1000)


class IsolationTests(unittest.TestCase):
    def test_default_paths_point_into_a_temporary_folder_during_tests(self):
        real = os.path.abspath(REAL_RUNS)
        self.assertFalse(os.path.abspath(run.parse_args([]).log).startswith(real))
        self.assertFalse(os.path.abspath(spotcheck.TO_FILL).startswith(real))
        self.assertFalse(os.path.abspath(config.CALLS_LOG).startswith(real))
