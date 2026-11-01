"""Run the comparison: arm A (template) and arm B (Gemini), both presentations.

DEFAULT IS A DRY RUN. Without --approve nothing is sent and nothing is written:
the plan and the exact payload of every planned LLM call are printed instead.
With --approve the calls are made and one JSONL record per call is appended to
act/runs/calls.jsonl.

    python -m act.run                                  dry run: print every payload
    python -m act.run --approve                        make the calls (resumes automatically)
    python -m act.run --arm template                   arm A only (local, needs no approval)
    python -m act.run --approve --synthetic-only       send synthetic inputs only
    python -m act.run --approve --synthetic-only --only I22,I25 --repeats 2 --workers 2

Resume / cache: a call whose (prompt hash, run index) already has a successful
record of the same kind (pilot or not) is skipped, so an interrupted run
continues where it stopped and nothing is sent twice. Pilot rows (see
act/pilots.json) never count as cache hits for a non-pilot run.

Errors: a per-minute 429, a 5xx, a 408 or a timeout is retried with backoff
(our retries; the SDK's own retrying is pinned off) and the count is logged as
`retries`. A per-day quota error or a 401/403 stops the run. Any other failed
call is recorded and the remaining calls continue.

Concurrency: --workers N sends up to N calls at once. Start times are spaced
by 60/--rpm seconds across all workers, the --rpd budget is shared, and log
lines are written whole under one lock. The log is in completion order; the
report sorts it, so its output does not depend on that order.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import config, llm, presentation, schema, template_arm
from .make_inputs import load_inputs

PACIFIC = ZoneInfo("America/Los_Angeles")      # free-tier RPD resets at midnight Pacific
PILOT_WARNING = ("WARNING: --repeats {} is not the pre-declared {}. This is a PILOT: its rows "
                 "are marked \"pilot\": true and must not be used for reported results.")


class UsageError(ValueError):
    """A command line that cannot be run as asked."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m act.run", description=__doc__.split("\n")[0])
    parser.add_argument("--approve", action="store_true",
                        help="actually send the LLM calls (default: dry run)")
    parser.add_argument("--arm", choices=["both", "template", "llm"], default="both")
    parser.add_argument("--repeats", type=int, default=config.REPEATS,
                        help="LLM runs per (input, presentation); default {} (pre-declared); "
                             "anything else is a pilot".format(config.REPEATS))
    parser.add_argument("--pilot", action="store_true",
                        help="mark this run's rows as pilot even with the default repeats")
    parser.add_argument("--only", default=None,
                        help="comma-separated input IDs to restrict the run to, e.g. I22,I25")
    parser.add_argument("--limit", type=int, default=None, help="at most this many new LLM calls")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="use only the synthetic inputs (no real day leaves the machine)")
    parser.add_argument("--workers", type=int, default=1, help="concurrent LLM calls (default 1)")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--rpm", type=float, default=config.DEFAULT_RPM)
    parser.add_argument("--rpd", type=int, default=config.DEFAULT_RPD)
    parser.add_argument("--inputs", default=config.INPUTS)
    parser.add_argument("--log", default=config.CALLS_LOG)
    return parser.parse_args(argv)


def settings():
    return {"temperature": config.TEMPERATURE, "max_output_tokens": config.MAX_OUTPUT_TOKENS}


# ---------------------------------------------------------------------------
# The log, the pilot convention and the cache
# ---------------------------------------------------------------------------

def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_pilots(path=config.PILOTS_FILE):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("entries", [])


def is_pilot(row, pilots=None):
    """A row is a pilot if it says so, or matches an entry in act/pilots.json."""
    if row.get("pilot") is True:
        return True
    for entry in (load_pilots() if pilots is None else pilots):
        if (row.get("prompt_hash") == entry["prompt_hash"]
                and datetime.fromisoformat(row["date"]) <= datetime.fromisoformat(entry["until"])):
            return True
    return False


def done_keys(records, pilot=False, pilots=None):
    """Keys of calls that already succeeded: the cache.

    LLM rows only count when their pilot status matches this run's, so a pilot
    never stands in for a reported result. Template rows always count.
    """
    keys = set()
    for r in records:
        if r.get("error") is not None:
            continue
        if r["arm"] == "llm" and is_pilot(r, pilots) != pilot:
            continue
        keys.add((r["arm"], r["prompt_hash"], r["run_index"]))
    return keys


def calls_today(records, now):
    today = now.astimezone(PACIFIC).date()
    return sum(1 for r in records if r["arm"] == "llm" and r.get("sent")
               and datetime.fromisoformat(r["date"]).astimezone(PACIFIC).date() == today)


class LogWriter:
    """Appends whole lines under a lock: no interleaving, one write per record."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.written = set()

    def append(self, record):
        key = (record["arm"], record["prompt_hash"], record["run_index"], record.get("pilot"))
        line = json.dumps(record) + "\n"
        with self.lock:
            if record.get("error") is None and key in self.written:
                raise RuntimeError("duplicate record for {}".format(key))
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            if record.get("error") is None:
                self.written.add(key)


# ---------------------------------------------------------------------------
# Selecting inputs and planning calls
# ---------------------------------------------------------------------------

def select(records, only=None, synthetic_only=False):
    """Filter inputs by --only and --synthetic-only; refuse anything ambiguous."""
    if only:
        wanted = [part.strip() for part in only.split(",") if part.strip()]
        bad = [w for w in wanted if not schema.INPUT_ID.match(w)]
        if bad:
            raise UsageError("--only takes input IDs like I22; not valid: {}".format(", ".join(bad)))
        known = {r["payload"]["input_id"]: r for r in records}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise UsageError("--only names inputs that do not exist: {}".format(", ".join(unknown)))
        if synthetic_only:
            real = [w for w in wanted if known[w]["meta"]["source"] != "synthetic"]
            if real:
                raise UsageError("--synthetic-only refuses real inputs: {} {} real. Nothing was "
                                 "sent.".format(", ".join(real), "is" if len(real) == 1 else "are"))
        records = [r for r in records if r["payload"]["input_id"] in set(wanted)]
    else:
        # The exploratory sweep (I35-I40) never joins a default run; name it in --only.
        records = [r for r in records if not r["meta"].get("exploratory")]
    if synthetic_only:
        records = [r for r in records if r["meta"]["source"] == "synthetic"]
    return records


def plan(records, arms, repeats, model):
    """[(arm, presentation, record, run_index, prompt, hash)] in a fixed order."""
    out = []
    for record in records:
        for pres in presentation.PRESENTATIONS:
            prompt = presentation.build_prompt(record["payload"], pres)
            if "template" in arms:
                h = presentation.prompt_hash(prompt, "template", {})
                out.append(("template", pres, record, 1, prompt, h))
            if "llm" in arms:
                h = presentation.prompt_hash(prompt, model, settings())
                for run in range(1, repeats + 1):
                    out.append(("llm", pres, record, run, prompt, h))
    return out


def base_record(arm, pres, record, run, prompt, h, model, sdk, pilot):
    return {
        "arm": arm, "presentation": pres, "input_id": record["payload"]["input_id"],
        "source": record["meta"]["source"], "run_index": run,
        "exploratory": bool(record["meta"].get("exploratory")),
        "pilot": bool(pilot) if arm == "llm" else False,
        "model": model if arm == "llm" else "template (deterministic)",
        "sdk_version": sdk if arm == "llm" else None,
        "settings": settings() if arm == "llm" else None,
        "sdk_retry_attempts": config.SDK_RETRY_ATTEMPTS if arm == "llm" else None,
        "date": datetime.now(timezone.utc).isoformat(),
        "prompt_hash": h, "prompt": prompt, "payload": record["payload"],
        "raw_response": None, "finish_reason": None, "latency_s": None,
        "latency_total_s": None, "retries": 0,
        "tokens": None, "error": None, "sent": False,
    }


def print_plan(records, planned, args, out):
    llm_calls = [p for p in planned if p[0] == "llm"]
    print("PLAN: {} input(s): {}".format(len(records), ", ".join(
        "{} ({})".format(r["payload"]["input_id"], r["meta"]["source"]) for r in records)), file=out)
    print("      presentations: {}; repeats: {}; LLM calls: {}; workers: {}; rpm: {:g}; rpd: {}; "
          "model: {}".format(", ".join(presentation.PRESENTATIONS), args.repeats, len(llm_calls),
                             args.workers, args.rpm, args.rpd, args.model), file=out)
    if args.limit is not None:
        print("      --limit {}: at most {} new calls".format(args.limit, args.limit), file=out)


def dry_run(planned, model, out):
    llm_calls = [p for p in planned if p[0] == "llm"]
    for arm, pres, record, run, prompt, h in llm_calls:
        payload = {"model": model, "config": {**settings(), "system_instruction": prompt["system"]},
                   "contents": prompt["user"]}
        print("=== DRY RUN call: input {} / {} / run {} / hash {}".format(
            record["payload"]["input_id"], pres, run, h[:12]), file=out)
        print(json.dumps(payload, indent=2), file=out)
    print("\nDRY RUN: {} LLM calls planned, nothing sent, nothing written. "
          "Add --approve to send them.".format(len(llm_calls)), file=out)
    return {"planned": len(llm_calls), "sent": 0, "stopped": None}


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

class RateLimiter:
    """Spaces call START times by `interval` seconds across all threads."""

    def __init__(self, rpm, clock, sleep):
        self.interval = 60.0 / rpm
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.next_start = None

    def wait(self):
        with self.lock:
            now = self.clock()
            start = now if self.next_start is None else max(now, self.next_start)
            self.next_start = start + self.interval
        if start > now:
            self.sleep(start - now)


def call_with_backoff(client, prompt, sleep, out):
    """One LLM call; per-minute 429s, 5xx, 408 and timeouts are retried.

    Returns (result, retries). The logged latency_s is the successful attempt
    only; the caller logs latency_total_s including the backoff waits.
    """
    delay = 5.0
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return client.generate(prompt["system"], prompt["user"], settings()), attempt - 1
        except llm.StopRun:
            raise
        except llm.CallFailed as exc:
            exc.retries = attempt - 1
            if not exc.retryable or attempt == config.MAX_RETRIES:
                raise
            wait = exc.retry_after if exc.retry_after else delay
            print("  retry {} in {:.0f} s ({})".format(attempt, wait, type(exc).__name__), file=out)
            sleep(wait)
            delay = min(delay * 2, 120.0)


def run(args, client_factory=None, out=sys.stdout, sleep=time.sleep, clock=time.perf_counter):
    if args.repeats < 1 or args.workers < 1 or args.rpm <= 0:
        raise UsageError("--repeats, --workers and --rpm must be positive")
    pilot = args.pilot or args.repeats != config.REPEATS
    if args.repeats != config.REPEATS:
        print(PILOT_WARNING.format(args.repeats, config.REPEATS), file=out)

    records = select(load_inputs(args.inputs), args.only, args.synthetic_only)
    arms = {"both": ("template", "llm"), "template": ("template",), "llm": ("llm",)}[args.arm]
    planned = plan(records, arms, args.repeats, args.model)
    print_plan(records, planned, args, out)

    if "llm" in arms and not args.approve:
        return dry_run(planned, args.model, out)

    log = read_log(args.log)
    pilots = load_pilots()
    done = done_keys(log, pilot=pilot, pilots=pilots)
    writer = LogWriter(args.log)
    summary = {"planned": len(planned), "skipped_cached": 0, "sent": 0, "template": 0,
               "errors": 0, "stopped": None}

    for arm, pres, record, run_index, prompt, h in planned:
        if arm != "template":
            continue
        if ("template", h, run_index) in done:
            summary["skipped_cached"] += 1
            continue
        rec = base_record(arm, pres, record, run_index, prompt, h, None, None, False)
        start = time.perf_counter()
        rec["raw_response"] = template_arm.feedback(record["payload"])
        rec["latency_s"] = rec["latency_total_s"] = time.perf_counter() - start
        rec["finish_reason"] = "TEMPLATE"
        rec["tokens"] = {"prompt": 0, "output": 0, "thoughts": 0, "total": 0}
        writer.append(rec)
        summary["template"] += 1

    todo = [p for p in planned if p[0] == "llm" and ("llm", p[5], p[3]) not in done]
    summary["skipped_cached"] += sum(1 for p in planned if p[0] == "llm") - len(todo)
    if not todo:
        print("Nothing left to send.", file=out)
        return summary

    factory = client_factory or (lambda model: llm.GeminiClient(model, llm.load_api_key()))
    client = factory(args.model)
    client.check_model()
    sdk = getattr(client, "sdk", llm.sdk_version())
    limiter = RateLimiter(args.rpm, clock, sleep)
    state = {"lock": threading.Lock(), "stop": None, "started": 0,
             "used_today": calls_today(log, datetime.now(timezone.utc))}

    def reserve():
        """Claim a slot for one call, or say why not. Thread-safe."""
        with state["lock"]:
            if state["stop"]:
                return False
            if args.limit is not None and state["started"] >= args.limit:
                state["stop"] = state["stop"] or "limit of {} new calls reached".format(args.limit)
                return False
            if state["used_today"] >= args.rpd:
                state["stop"] = "daily request budget of {} reached (--rpd)".format(args.rpd)
                return False
            state["started"] += 1
            state["used_today"] += 1
            return True

    def one(item):
        arm, pres, record, run_index, prompt, h = item
        if not reserve():
            return "skipped"
        limiter.wait()
        with state["lock"]:
            if state["stop"] and state["stop"].startswith(("quota", "stopped")):
                state["started"] -= 1
                state["used_today"] -= 1
                return "skipped"
        rec = base_record(arm, pres, record, run_index, prompt, h, args.model, sdk, pilot)
        rec["sent"] = True
        start = time.perf_counter()
        try:
            result, retries = call_with_backoff(client, prompt, sleep, out)
            rec.update(raw_response=result["text"], finish_reason=result["finish_reason"],
                       latency_s=result["latency_s"], tokens=result["tokens"], retries=retries,
                       response_json=result.get("response_json"))
            if not result["text"]:
                rec["error"] = "empty response (finish reason {})".format(result["finish_reason"])
        except llm.StopRun as exc:
            rec["error"] = "{}: {}".format(
                "quota exhausted" if isinstance(exc, llm.QuotaExhausted) else "stopped", exc)
            with state["lock"]:
                state["stop"] = ("quota exhausted; rerun after the reset to resume"
                                 if isinstance(exc, llm.QuotaExhausted)
                                 else "stopped: {}".format(str(exc)[:200]))
        except llm.CallFailed as exc:
            rec["error"] = str(exc)
            rec["retries"] = getattr(exc, "retries", 0)
        except Exception as exc:                 # anything unforeseen: record, carry on
            rec["error"] = "unexpected: {}: {}".format(type(exc).__name__, exc)
        rec["latency_total_s"] = time.perf_counter() - start
        writer.append(rec)
        with state["lock"]:
            if rec["error"]:
                summary["errors"] += 1
            else:
                summary["sent"] += 1
            print("{} {} {} run {}{}".format("ok " if not rec["error"] else "ERR",
                                             record["payload"]["input_id"], pres, run_index,
                                             "" if not rec["error"] else ": " + rec["error"][:120]),
                  file=out)
        return "error" if rec["error"] else "ok"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(one, todo))

    summary["stopped"] = state["stop"]
    remaining = sum(1 for o in outcomes if o == "skipped")
    print("\nDone: {} sent ok, {} cached, {} errors, {} not started.{}".format(
        summary["sent"], summary["skipped_cached"], summary["errors"], remaining,
        " Stopped: " + summary["stopped"] if summary["stopped"] else ""), file=out)
    return summary


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (schema.PayloadRejected, UsageError) as exc:
        raise SystemExit("Refused before sending anything: {}".format(exc))


if __name__ == "__main__":
    main()
