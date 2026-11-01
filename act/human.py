"""Human adjudication of false cautions, and a mechanical audit of the labels.

The phrase list (rubric/refusal_phrases.txt) is a SCREEN. The adjudicated
false-caution count comes from rubric/human_labels.csv, one row per LLM output
on an input where rule 3 does not apply:

    input_id, presentation, run_index, human_false_caution (yes/no), note

Definition used for the labels: the output says or clearly implies that the
result cannot be used or relied on, on an input where rule 3 does not apply.

    python -m act.human audit      broader regex over those outputs; every hit
                                   not labelled yes, with its full text
    python -m act.human compare    screen vs human, per input and presentation

Only non-pilot, successful LLM rows are considered. A label is keyed by
(input_id, presentation, run_index); the cache guarantees one successful
non-pilot row per key.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

from . import config, rubric, schema
from .run import is_pilot, load_pilots, read_log

LABELS = os.path.join(config.ACT_DIR, "rubric", "human_labels.csv")
COLUMNS = ["input_id", "presentation", "run_index", "human_false_caution", "note"]
AUDIT = re.compile(r"interpret|reliab|cannot|not be used|caution|trust", re.I)


class LabelError(ValueError):
    pass


def key(row):
    return (row["input_id"], row["presentation"], int(row["run_index"]))


def unflagged_outputs(log, pilots=None, exploratory=False):
    """Successful non-pilot LLM rows on inputs where rule 3 does not apply, sorted.

    exploratory=False (default) is the primary set; True is the boundary sweep only.
    """
    pilots = load_pilots() if pilots is None else pilots
    rows = [r for r in log if r["arm"] == "llm" and r.get("error") is None
            and not is_pilot(r, pilots) and not schema.caution_required(r["payload"])
            and bool(r.get("exploratory")) == exploratory]
    return sorted(rows, key=key)


def load_labels(path=LABELS):
    """{(input_id, presentation, run_index): (bool, note)}; refuses a malformed file."""
    if not os.path.exists(path):
        return {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise LabelError("{} must have the columns {}".format(path, ", ".join(COLUMNS)))
        for line, row in enumerate(reader, 2):
            value = (row["human_false_caution"] or "").strip().lower()
            if value not in ("yes", "no"):
                raise LabelError("line {}: human_false_caution must be yes or no".format(line))
            k = key(row)
            if k in labels:
                raise LabelError("line {}: duplicate label for {}".format(line, k))
            labels[k] = (value == "yes", row["note"])
    return labels


def compare(log, labels, exploratory=False):
    """Screen vs human per (input, presentation), plus mismatches and gaps."""
    outputs = unflagged_outputs(log, exploratory=exploratory)
    cells = defaultdict(lambda: {"outputs": 0, "screen": 0, "human": 0, "unlabelled": 0})
    mismatches, unlabelled = [], []
    seen = set()
    for r in outputs:
        k = key(r)
        seen.add(k)
        cell = cells["{}/{}".format(r["input_id"], r["presentation"])]
        cell["outputs"] += 1
        screen = bool(rubric.false_caution(r["raw_response"], r["payload"])[0])
        cell["screen"] += screen
        if k not in labels:
            cell["unlabelled"] += 1
            unlabelled.append(k)
            continue
        human, note = labels[k]
        cell["human"] += human
        if screen != human:
            mismatches.append({"key": k, "screen": screen, "human": human, "note": note,
                               "text": r["raw_response"]})
    # labels with no matching output (labels for the other set are not orphans)
    other = {key(r) for r in unflagged_outputs(log, exploratory=not exploratory)}
    orphans = sorted(set(labels) - seen - other)
    totals = {name: sum(c[name] for c in cells.values())
              for name in ("outputs", "screen", "human", "unlabelled")}
    return {"cells": dict(sorted(cells.items())), "totals": totals, "mismatches": mismatches,
            "unlabelled": unlabelled, "orphans": orphans}


def audit(log, labels):
    """Outputs matching the broader regex that are NOT labelled yes (primary and sweep)."""
    hits = []
    for r in unflagged_outputs(log) + unflagged_outputs(log, exploratory=True):
        found = sorted({m.group(0).lower() for m in AUDIT.finditer(r["raw_response"] or "")})
        if found and not labels.get(key(r), (False, ""))[0]:
            hits.append({"key": key(r), "matches": found, "label": labels.get(key(r)),
                         "text": r["raw_response"]})
    return hits


def compare_markdown(result):
    lines = ["| input | presentation | outputs | screen (phrase list) | human-adjudicated |",
             "|---|---|---|---|---|"]
    for name, c in result["cells"].items():
        input_id, pres = name.split("/")
        lines.append("| {} | {} | {} | {} | {}{} |".format(
            input_id, pres, c["outputs"], c["screen"], c["human"],
            " ({} unlabelled)".format(c["unlabelled"]) if c["unlabelled"] else ""))
    t = result["totals"]
    lines.append("| **total** | | {} | {} | {} |".format(t["outputs"], t["screen"], t["human"]))
    lines += ["", "Mismatches (screen != human): {}".format(len(result["mismatches"]))]
    for m in result["mismatches"]:
        lines.append("- {} {} run {}: screen {}, human {}. {} Text: {}".format(
            *m["key"], "yes" if m["screen"] else "no", "yes" if m["human"] else "no",
            ("Note: " + m["note"] + ".") if m["note"] else "", m["text"]))
    if result["unlabelled"]:
        lines.append("Unlabelled outputs: {}".format(
            ", ".join("{} {} run {}".format(*k) for k in result["unlabelled"])))
    if result["orphans"]:
        lines.append("Labels with no matching output: {}".format(
            ", ".join("{} {} run {}".format(*k) for k in result["orphans"])))
    return "\n".join(lines)


def main(argv=None, out=None):
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="python -m act.human")
    parser.add_argument("action", choices=["audit", "compare"])
    args = parser.parse_args(argv)
    log, labels = read_log(config.CALLS_LOG), load_labels()
    if args.action == "compare":
        print(compare_markdown(compare(log, labels)), file=out)
        return
    hits = audit(log, labels)
    print("Broader regex /{}/ over {} outputs; hits not labelled yes: {}".format(
        AUDIT.pattern, len(unflagged_outputs(log)), len(hits)), file=out)
    for h in hits:
        print("- {} {} run {} (label: {}) matches {}:\n  {}".format(
            *h["key"], "missing" if h["label"] is None else ("yes" if h["label"][0] else "no"),
            h["matches"], h["text"]), file=out)


if __name__ == "__main__":
    main()
