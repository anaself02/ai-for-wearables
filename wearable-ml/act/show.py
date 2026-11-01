"""Print logged outputs next to their input payload, for checking by hand.

    python -m act.show --only I22,I25,I28 [--pilot | --no-pilot] [--since 2026-11-01T18:00]
    python -m act.show --unflagged [--no-pilot]

--unflagged prints, for BOTH arms, every output on the inputs where rule 3 does
not apply (I03, I18, I31-I34), with its false-caution result. These outputs are
read by hand (PRE_DECLARED.md Amendment 3, 2026-11-01).

Reads act/runs/calls.jsonl; prints nothing but what is already in it.
"""

import argparse
import json
from datetime import datetime, timezone

from . import config, rubric, schema
from .run import is_pilot, load_pilots, read_log


def select(rows, only=None, arm="llm", unflagged=False, since=None, pilot=None, pilots=()):
    wanted = set(only.split(",")) if only else None
    out = []
    for r in rows:
        if arm != "both" and r["arm"] != arm:
            continue
        if wanted is not None and r["input_id"] not in wanted:
            continue
        if unflagged and schema.caution_required(r["payload"]):
            continue
        if since is not None and datetime.fromisoformat(r["date"]) < since:
            continue
        row_pilot = r["arm"] == "llm" and is_pilot(r, pilots)
        if pilot is not None and row_pilot != pilot:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["input_id"], r["arm"], r["presentation"], r["run_index"], r["date"]))
    return out


def render(r, pilots=()):
    p = {k: v for k, v in r["payload"].items() if k != "input_id"}
    lines = ["=" * 78,
             "{} ({}) / {} / {} / run {} / pilot {} / {}".format(
                 r["input_id"], r["source"], r["arm"], r["presentation"], r["run_index"],
                 r["arm"] == "llm" and is_pilot(r, pilots), r["date"][:19]),
             "payload: " + json.dumps(p),
             "finish_reason: {} | latency_s: {} | latency_total_s: {} | retries: {} | tokens: {}".format(
                 r["finish_reason"], None if r["latency_s"] is None else round(r["latency_s"], 1),
                 None if r.get("latency_total_s") is None else round(r["latency_total_s"], 1),
                 r.get("retries"), r["tokens"])]
    fc, phrases = rubric.false_caution(r["raw_response"], r["payload"])
    if fc is not None:
        lines.append("false caution: {}{}".format("YES " if fc else "no", phrases if fc else ""))
    if r.get("error"):
        lines.append("ERROR: " + r["error"])
    lines.append("raw_response:\n" + (r["raw_response"] or "(none)"))
    return "\n".join(lines)


def main(argv=None, log_path=None, out=None):
    import sys
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="python -m act.show")
    parser.add_argument("--only", default=None)
    parser.add_argument("--arm", choices=["llm", "template", "both"], default=None)
    parser.add_argument("--unflagged", action="store_true",
                        help="both arms, inputs where rule 3 does not apply")
    parser.add_argument("--since", default=None, help="ISO time (UTC if no offset)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pilot", action="store_true", help="pilot rows only")
    group.add_argument("--no-pilot", action="store_true", help="non-pilot rows only")
    args = parser.parse_args(argv)
    since = datetime.fromisoformat(args.since) if args.since else None
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    arm = args.arm or ("both" if args.unflagged else "llm")
    pilots = load_pilots()
    rows = select(read_log(log_path or config.CALLS_LOG), args.only, arm, args.unflagged, since,
                  True if args.pilot else (False if args.no_pilot else None), pilots)
    for r in rows:
        print(render(r, pilots), file=out)
    print("=" * 78, file=out)
    print("{} row(s).".format(len(rows)), file=out)
    return rows


if __name__ == "__main__":
    main()
