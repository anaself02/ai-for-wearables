"""Combine Anas's real day summaries with the synthetic cases into inputs.jsonl.

Each record is {"payload": <the allowlisted fields>, "meta": {"source", "case"}}.
Only "payload" is ever sent; "meta" (real or synthetic) stays on this machine.
input_id and day_index are assigned after a seeded shuffle, so neither reveals
whether a day is real, nor when it was.

    python -m act.make_inputs          (wearable-ml venv)
    python -m act.make_inputs --table  which inputs trigger the caution rule (rule 3)
"""

import json
import os
import random

from . import config, schema, synthetic


def load_real(path=config.REAL_DAYS):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build(real_rows, seed=config.SEED, extension=True):
    """Shuffle real + original synthetic cases (I01..), then append the extension.

    The extension cases (synthetic.EXTENSION_CASES) are added after the shuffle,
    so adding them never renumbers or re-hashes the inputs before them.
    """
    items = [({**row["summary"]}, {"source": "real", "case": "complete day"})
             for row in real_rows]
    items += [(dict(values), {"source": "synthetic", "case": case})
              for case, values in synthetic.CASES]
    random.Random(seed).shuffle(items)
    if extension:
        items += [(dict(values), {"source": "synthetic", "case": case, "extension": True})
                  for case, values in synthetic.EXTENSION_CASES]
        items += [(dict(values), {"source": "synthetic", "case": case, "exploratory": True})
                  for case, values in synthetic.EXPLORATORY_CASES]
    records = []
    for index, (values, meta) in enumerate(items, 1):
        payload = {"input_id": "I{:02d}".format(index), "day_index": index, **values}
        payload = {name: payload[name] for name in schema.FIELDS}
        schema.validate_payload(payload)
        records.append({"payload": payload, "meta": meta})
    return records


def load_inputs(path=config.INPUTS):
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        schema.validate_payload(record["payload"])
    return records


def triggers_table(records):
    """Rows of (input_id, source, OOD >= 0.25, degenerate, partial, any trigger)."""
    rows = []
    for r in records:
        p = r["payload"]
        rows.append((p["input_id"], r["meta"]["source"], schema.out_of_distribution(p),
                     p["degenerate"], p["partial"], schema.caution_required(p)))
    return rows


def print_table(records):
    mark = lambda flag: "yes" if flag else "-"
    print("| input | source | OOD >= {:.2f} | degenerate | partial | rule 3 applies |".format(
        config.OOD_FLAG_SHARE))
    print("|---|---|---|---|---|---|")
    for row in triggers_table(records):
        print("| {} | {} | {} | {} | {} | {} |".format(row[0], row[1], *map(mark, row[2:])))
    rows = triggers_table(records)
    print("\nrule 3 applies to {} of {}; it does not apply to: {}".format(
        sum(r[5] for r in rows), len(rows), ", ".join(r[0] for r in rows if not r[5]) or "none"))


def main():
    import sys
    if "--table" in sys.argv[1:]:
        print_table(load_inputs())
        return
    real = load_real()
    records = build(real)
    os.makedirs(config.INPUTS_DIR, exist_ok=True)
    with open(config.INPUTS, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print("inputs: {} ({} real, {} synthetic) -> {}".format(
        len(records), len(real), len(records) - len(real),
        os.path.relpath(config.INPUTS, config.ML_DIR)))


if __name__ == "__main__":
    main()
