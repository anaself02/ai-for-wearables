"""Score every logged output with the rubric and write act/runs/report.md (+ .json).

    python -m act.report                 (pilot rows left out)
    python -m act.report --include-pilot

Per arm x presentation: pass rate per check with a bootstrap 95% CI over
inputs (each input's rate is the mean over its repeats, so an input counts
once however many repeats it has), the arm difference per presentation and
the interaction (B - A on table) - (B - A on sentence), variability across the
repeats, tokens (free tier: $0) and latency, and every failure with its raw text.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

import numpy as np

from . import config, human, rubric, schema
from .run import is_pilot, load_pilots, read_log

N_BOOT = 2000
ALL = "all checks"
METRICS = list(rubric.CHECKS) + [ALL]
#: Separate metric (Amendment 3, 2026-11-01): share of outputs on unflagged inputs
#: that refuse to interpret. Lower is better; not part of "all checks".
FALSE_CAUTION = "false caution"

THREATS = [
    "WESAD, papers about it and common wellness-feedback phrasing may be in the "
    "LLM's training data, so its wording may echo public text rather than the numbers given.",
    "The medical-term and caution-phrase lists are crude keyword rules: a paraphrased "
    "caution can be missed and a harmless word can be flagged. Read the failure list.",
    "The number check accepts any input value at the precision shown, so a number "
    "attached to the wrong quantity still passes.",
    "Arm A was written by the same people who wrote the rubric, so it is expected to pass.",
    "One person's days plus invented cases: nothing here generalises to other people, "
    "devices or models. Free-tier model behaviour can change without notice.",
]


def sort_key(r):
    """Report order never depends on the order calls completed in."""
    return (r["arm"], r["presentation"], r["input_id"], r["run_index"], r["date"])


def select_rows(log, include_pilot=False):
    """Primary rows: no pilots (unless asked) and never the exploratory sweep."""
    pilots = load_pilots()
    rows = [r for r in log if (include_pilot or r["arm"] != "llm" or not is_pilot(r, pilots))
            and not r.get("exploratory")]
    return sorted(rows, key=sort_key)


def exploratory_section(log, labels):
    """The OOD boundary sweep (I35-I40), per input and presentation, kept apart."""
    pilots = load_pilots()
    cells = defaultdict(lambda: {"outputs": 0, "caution_pass": 0, "screen": 0, "human": 0,
                                 "unlabelled": 0})
    meta = {}
    for r in sorted(log, key=sort_key):
        if (r["arm"] != "llm" or not r.get("exploratory") or r.get("error") is not None
                or is_pilot(r, pilots)):
            continue
        name = "{}/{}".format(r["input_id"], r["presentation"])
        flagged = schema.caution_required(r["payload"])
        meta[name] = (r["payload"]["share_out_of_distribution"], flagged)
        c = cells[name]
        c["outputs"] += 1
        if flagged:
            c["caution_pass"] += bool(rubric.check_caution(r["raw_response"], r["payload"])[0])
        else:
            c["screen"] += bool(rubric.false_caution(r["raw_response"], r["payload"])[0])
            label = labels.get((r["input_id"], r["presentation"], r["run_index"]))
            if label is None:
                c["unlabelled"] += 1
            else:
                c["human"] += label[0]
    return {"cells": {k: {**cells[k], "ood_share": meta[k][0], "rule3": meta[k][1]}
                      for k in sorted(cells)},
            "human": human.compare(log, labels, exploratory=True)}


def scored_records(log):
    out = []
    for r in sorted(log, key=sort_key):
        if r.get("error") is not None:
            continue
        checks = rubric.score(r["raw_response"], r["payload"])
        applicable = [v[0] for v in checks.values() if v[0] is not None]
        out.append({**r, "checks": checks, "all": all(applicable),
                    "false_caution": rubric.false_caution(r["raw_response"], r["payload"])})
    return out


def input_rates(records, metric):
    """{input_id: mean pass over that input's runs} for one cell (None = not applicable)."""
    by_input = defaultdict(list)
    for r in records:
        if metric == FALSE_CAUTION:
            value = r["false_caution"][0]
        else:
            value = r["all"] if metric == ALL else r["checks"][metric][0]
        if value is not None:
            by_input[r["input_id"]].append(float(value))
    return {k: float(np.mean(v)) for k, v in by_input.items()}


def bootstrap(values, seed=config.SEED):
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cells(scored):
    grouped = defaultdict(list)
    for r in scored:
        grouped[(r["arm"], r["presentation"])].append(r)
    return grouped


def variability(records):
    """Across the repeats of each (input): unanimity per check and distinct texts."""
    by_input = defaultdict(list)
    for r in records:
        by_input[r["input_id"]].append(r)
    out = {"inputs_with_repeats": sum(1 for v in by_input.values() if len(v) > 1)}
    for metric in METRICS:
        unanimous, sds = [], []
        for runs in by_input.values():
            values = [(r["all"] if metric == ALL else r["checks"][metric][0]) for r in runs]
            values = [float(v) for v in values if v is not None]
            if len(values) > 1:
                unanimous.append(len(set(values)) == 1)
                sds.append(float(np.std(values)))
        out[metric] = {"share_unanimous": float(np.mean(unanimous)) if unanimous else None,
                       "mean_sd": float(np.mean(sds)) if sds else None}
    texts = [len({r["raw_response"] for r in runs}) for runs in by_input.values() if len(runs) > 1]
    out["mean_distinct_texts"] = float(np.mean(texts)) if texts else None
    lengths = [rubric.check_sentences(r["raw_response"])[1] for r in records]
    out["sentences_min_max"] = (min(lengths), max(lengths)) if lengths else None
    return out


def usage(records):
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    tok = defaultdict(int)
    for r in records:
        for k, v in (r.get("tokens") or {}).items():
            tok[k] += v or 0
    return {"calls": len(records),
            "latency_median_s": statistics.median(lat) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "tokens": dict(tok), "cost_usd": 0.0}


def build(log, include_pilot=False, labels=None):
    labels = human.load_labels() if labels is None else labels
    full_log = log
    log = select_rows(log, include_pilot)
    scored = scored_records(log)
    grouped = cells(scored)
    result = {"cells": {}, "differences": {}, "interaction": {}, "by_source": {},
              "errors": [r for r in log if r.get("error") is not None]}
    for (arm, pres), records in sorted(grouped.items()):
        result["cells"]["{}/{}".format(arm, pres)] = {
            "pass": {m: bootstrap(input_rates(records, m).values()) for m in METRICS},
            "variability": variability(records) if arm == "llm" else None,
            "usage": usage(records),
        }
        for source in ("real", "synthetic"):
            subset = [r for r in records if r["source"] == source]
            result["by_source"]["{}/{}/{}".format(arm, pres, source)] = bootstrap(
                input_rates(subset, ALL).values())

    for metric in METRICS:
        rates = {cell: input_rates(records, metric) for cell, records in grouped.items()}
        diffs = {}
        for pres in ("table", "sentence"):
            a, b = rates.get(("template", pres), {}), rates.get(("llm", pres), {})
            common = sorted(set(a) & set(b))
            diffs[pres] = {i: b[i] - a[i] for i in common}
            result["differences"]["{}/{}".format(metric, pres)] = bootstrap(diffs[pres].values())
        common = sorted(set(diffs["table"]) & set(diffs["sentence"]))
        result["interaction"][metric] = bootstrap(
            [diffs["table"][i] - diffs["sentence"][i] for i in common])

    fc_rates = {cell: input_rates(records, FALSE_CAUTION) for cell, records in grouped.items()}
    result["false_caution"] = {"{}/{}".format(*cell): bootstrap(rates.values())
                               for cell, rates in sorted(fc_rates.items())}
    for pres in ("table", "sentence"):
        a, b = fc_rates.get(("template", pres), {}), fc_rates.get(("llm", pres), {})
        result["false_caution"]["B-A/{}".format(pres)] = bootstrap(
            [b[i] - a[i] for i in sorted(set(a) & set(b))])
    result["false_caution_outputs"] = [
        {"arm": r["arm"], "presentation": r["presentation"], "input_id": r["input_id"],
         "run": r["run_index"], "phrases": r["false_caution"][1], "text": r["raw_response"]}
        for r in scored if r["false_caution"][0]]
    # The phrase list is a screen; the adjudicated count is the human labels'.
    result["false_caution_human"] = human.compare(log, labels)
    result["exploratory"] = exploratory_section(full_log, labels)
    result["failures"] = [
        {"arm": r["arm"], "presentation": r["presentation"], "input_id": r["input_id"],
         "source": r["source"], "run": r["run_index"],
         "failed": {k: v[1] for k, v in r["checks"].items() if v[0] is False},
         "text": r["raw_response"]}
        for r in scored if not r["all"]]
    models = sorted({(r["model"], r["sdk_version"]) for r in log if r["arm"] == "llm"})
    result["models"] = [{"model": m, "sdk": s} for m, s in models]
    return result


def fmt_ci(triple):
    mean, lo, hi = triple
    if mean is None:
        return "n/a"
    return "{:.2f} [{:.2f}, {:.2f}]".format(mean, lo, hi)


def markdown(result):
    lines = ["# ACT comparison report", "",
             "Pass rates are means over inputs (each input = mean over its repeats), "
             "with bootstrap 95% CIs over inputs ({} resamples). Arm A ignores the "
             "presentation, so its two columns are the same text.".format(N_BOOT), ""]
    for m in result["models"]:
        lines.append("- LLM: `{}` ({})".format(m["model"], m["sdk"]))
    lines += ["", "## Pass rate per check", "",
              "| cell | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for cell, data in result["cells"].items():
        lines.append("| {} | {} |".format(cell, " | ".join(fmt_ci(data["pass"][m]) for m in METRICS)))
    lines += ["", "## LLM minus template, and the interaction", "",
              "| check | B-A table | B-A sentence | interaction (table - sentence) |",
              "|---|---|---|---|"]
    for m in METRICS:
        lines.append("| {} | {} | {} | {} |".format(
            m, fmt_ci(result["differences"]["{}/table".format(m)]),
            fmt_ci(result["differences"]["{}/sentence".format(m)]),
            fmt_ci(result["interaction"][m])))
    lines += ["", "A difference is only called real when its CI excludes 0.", "",
              "## False caution (separate metric; unflagged inputs only)", "",
              "Share of outputs on inputs where rule 3 does not apply that contain an "
              "interpretation-refusal phrase (rubric/refusal_phrases.txt). Lower is better. "
              "These outputs are also read by hand (python -m act.show --unflagged).", ""]
    for key, value in result["false_caution"].items():
        lines.append("- {}: {}".format(key, fmt_ci(value)))
    for f in result["false_caution_outputs"]:
        lines.append("  - {} / {} / {} run {}: {} -> {}".format(
            f["arm"], f["presentation"], f["input_id"], f["run"], f["phrases"], f["text"]))
    lines += ["", "### Screen vs human-adjudicated false cautions (LLM arm)", "",
              "The phrase list is a screen. The adjudicated count is from "
              "rubric/human_labels.csv (the output says or clearly implies the result "
              "cannot be used or relied on). Non-pilot outputs only.", "",
              human.compare_markdown(result["false_caution_human"])]
    lines += ["", "## EXPLORATORY: out-of-distribution boundary sweep (I35-I40)", "",
              "Added after seeing the I34 results (Amendment 3, 2026-11-01). Inputs identical "
              "to I34 except the OOD share. Excluded from every primary metric above and not "
              "used for any primary claim. Where rule 3 applies (share >= 0.25) the column "
              "shown is the caution check; below it, false cautions (screen / human).", "",
              "| input | presentation | OOD share | rule 3 | outputs | caution present | "
              "false caution: screen | false caution: human |",
              "|---|---|---|---|---|---|---|---|"]
    for name, c in result["exploratory"]["cells"].items():
        input_id, pres = name.split("/")
        human_cell = "n/a" if c["rule3"] else "{}{}".format(
            c["human"], " ({} unlabelled)".format(c["unlabelled"]) if c["unlabelled"] else "")
        lines.append("| {} | {} | {:.2f} | {} | {} | {} | {} | {} |".format(
            input_id, pres, c["ood_share"], "yes" if c["rule3"] else "no", c["outputs"],
            c["caution_pass"] if c["rule3"] else "n/a",
            "n/a" if c["rule3"] else c["screen"], human_cell))
    if not result["exploratory"]["cells"]:
        lines.append("(no sweep outputs logged yet)")
    lines += ["", "## All checks, real vs synthetic inputs", ""]
    for key, value in result["by_source"].items():
        lines.append("- {}: {}".format(key, fmt_ci(value)))
    lines += ["", "## Variability across the repeats (LLM)", ""]
    for cell, data in result["cells"].items():
        v = data["variability"]
        if not v:
            continue
        lines.append("- {}: mean distinct texts per input {}, sentences min/max {}; "
                     "share of inputs unanimous per check: {}".format(
                         cell, None if v["mean_distinct_texts"] is None
                         else round(v["mean_distinct_texts"], 2), v["sentences_min_max"],
                         ", ".join("{} {}".format(m, None if v[m]["share_unanimous"] is None
                                                  else round(v[m]["share_unanimous"], 2))
                                   for m in METRICS)))
    lines += ["", "## Cost and latency", "",
              "| cell | calls | tokens | median latency (s) | p95 latency (s) | cost |",
              "|---|---|---|---|---|---|"]
    for cell, data in result["cells"].items():
        u = data["usage"]
        lines.append("| {} | {} | {} | {} | {} | $0 (free tier) |".format(
            cell, u["calls"], u["tokens"],
            None if u["latency_median_s"] is None else round(u["latency_median_s"], 3),
            None if u["latency_p95_s"] is None else round(u["latency_p95_s"], 3)))
    lines += ["", "## Call errors ({})".format(len(result["errors"])), ""]
    for r in result["errors"]:
        lines.append("- {} {} run {}: {}".format(r["input_id"], r["presentation"],
                                                r["run_index"], r["error"][:300]))
    lines += ["", "## Failures ({}), with the raw text".format(len(result["failures"])), ""]
    for f in result["failures"]:
        lines += ["### {} / {} / {} ({}) run {}".format(f["arm"], f["presentation"],
                                                       f["input_id"], f["source"], f["run"]),
                  "Failed: {}".format(f["failed"]), "", "> " + (f["text"] or "").replace("\n", "\n> "), ""]
    lines += ["## Threats to validity", ""] + ["- " + t for t in THREATS]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m act.report")
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args(argv)
    log = read_log(config.CALLS_LOG)
    if not log:
        raise SystemExit("No calls logged yet: run python -m act.run first.")
    result = build(log, include_pilot=args.include_pilot)
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    with open(os.path.join(config.RUNS_DIR, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    with open(os.path.join(config.RUNS_DIR, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown(result))
    print("Wrote act/runs/report.md and report.json ({} records, {} failures, {} errors).".format(
        len(log), len(result["failures"]), len(result["errors"])))


if __name__ == "__main__":
    main()
