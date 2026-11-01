"""Blinded human spot-check of usefulness (0, 1 or 2).

    python -m act.spotcheck make [--n 40]    -> runs/spotcheck_to_fill.csv + runs/spotcheck_key.csv
    python -m act.spotcheck aggregate        -> mean usefulness per arm x presentation

`make` draws a seeded random subset of successful outputs, the same number
from each arm x presentation cell, shuffles them, and writes a CSV that shows
only an item number, the day's numbers and the text. Which arm wrote it is in a
separate key file: do not open the key until every row is scored.

Usefulness: 0 = not useful or misleading; 1 = somewhat useful; 2 = useful,
accurate and appropriately cautious.
"""

import argparse
import csv
import os
import random
from collections import defaultdict

from . import config, presentation
from .report import bootstrap, sort_key
from .run import is_pilot, load_pilots, read_log

TO_FILL = os.path.join(config.RUNS_DIR, "spotcheck_to_fill.csv")
KEY = os.path.join(config.RUNS_DIR, "spotcheck_key.csv")


def make(log, n=40, seed=config.SEED, to_fill=TO_FILL, key=KEY):
    pilots = load_pilots()
    ok = [r for r in log if r.get("error") is None and r.get("raw_response")
          and not (r["arm"] == "llm" and is_pilot(r, pilots))]
    ok.sort(key=sort_key)            # the sample must not depend on completion order
    cells = defaultdict(list)
    for r in ok:
        cells[(r["arm"], r["presentation"])].append(r)
    rng = random.Random(seed)
    per_cell = max(1, n // max(1, len(cells)))
    chosen = []
    for cell in sorted(cells):
        chosen += rng.sample(cells[cell], min(per_cell, len(cells[cell])))
    rng.shuffle(chosen)
    os.makedirs(os.path.dirname(to_fill), exist_ok=True)
    with open(to_fill, "w", newline="", encoding="utf-8") as fill, \
            open(key, "w", newline="", encoding="utf-8") as secret:
        fw = csv.writer(fill)
        kw = csv.writer(secret)
        fw.writerow(["item", "numbers", "feedback", "usefulness_0_2", "comment"])
        kw.writerow(["item", "arm", "presentation", "input_id", "run_index"])
        for item, r in enumerate(chosen, 1):
            fw.writerow([item, presentation.as_sentence(r["payload"]), r["raw_response"], "", ""])
            kw.writerow([item, r["arm"], r["presentation"], r["input_id"], r["run_index"]])
    return len(chosen)


def aggregate(to_fill=TO_FILL, key=KEY):
    with open(key, encoding="utf-8") as handle:
        arms = {row["item"]: row for row in csv.DictReader(handle)}
    scores = defaultdict(list)
    missing, invalid = 0, []
    with open(to_fill, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("usefulness_0_2") or "").strip()
            if not value:
                missing += 1
                continue
            if value not in ("0", "1", "2"):
                invalid.append((row["item"], value))
                continue
            k = arms[row["item"]]
            scores["{}/{}".format(k["arm"], k["presentation"])].append(int(value))
    result = {cell: {"n": len(v), "mean_ci": bootstrap(v),
                     "counts": {s: v.count(s) for s in (0, 1, 2)}}
              for cell, v in sorted(scores.items())}
    return result, missing, invalid


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m act.spotcheck")
    parser.add_argument("action", choices=["make", "aggregate"])
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args(argv)
    if args.action == "make":
        n = make(read_log(config.CALLS_LOG), n=args.n)
        print("Wrote {} items to {} (key kept separately in {}).".format(
            n, os.path.relpath(TO_FILL, config.ML_DIR), os.path.relpath(KEY, config.ML_DIR)))
        return
    result, missing, invalid = aggregate()
    for cell, data in result.items():
        mean, lo, hi = data["mean_ci"]
        print("{}: n={} mean usefulness {:.2f} [{:.2f}, {:.2f}] counts {}".format(
            cell, data["n"], mean, lo, hi, data["counts"]))
    print("unscored rows: {}; invalid values: {}".format(missing, invalid or "none"))


if __name__ == "__main__":
    main()
