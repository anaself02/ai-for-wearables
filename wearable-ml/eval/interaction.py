"""Interaction analysis: model type x HR input resolution.

Question: does the ranking of the models (SVM, RF, MLP) change when the heart
rate input goes from beat-level (WESAD native) to ~3 s readings (simulated
consumer device)?

    python -m eval.resolution build              # once, needs the WESAD pickles
    python -m eval.interaction plan              # number of fits
    python -m eval.interaction run --go          # every cell, nested LOSO, 5 seeds
    python -m eval.interaction report            # re-analyse the saved fold rows

Every cell uses the frozen protocol of eval/ unchanged (nested.run: outer
leave-one-subject-out, inner subject-grouped tuning on macro-F1, the grids
and seeds in config.py). The only thing that changes between the two columns
is the input table (eval/resolution.py), and both tables hold the same windows.

What is reported, per task and metric:
- each cell: mean and SD over subjects (seeds averaged first), bootstrap 95% CI
- resolution effect per model: native minus 3s, paired per subject
- ranking of the models at each resolution
- interaction contrast per model pair (A, B):
      (A - B at native) - (A - B at 3s), computed per subject, then bootstrapped.
  A CI that excludes 0 is evidence that the gap between A and B depends on the
  resolution. A CI that includes 0 is reported as "no evidence the ranking
  changes", never as "no interaction".
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from . import config, folds, metrics, nested, resolution
from .data import check_path
from .run_eval import log_experiment, pooled_macro_f1, write_csv

ALL = config.BASELINES + config.MODELS
LEVELS = resolution.LEVELS
METRICS = ("macro_f1", "balanced_accuracy", "auroc")
INTERACTION_PAIRS = [("mlp", "svm"), ("mlp", "rf"), ("rf", "svm"),
                     ("svm", "threshold"), ("rf", "threshold"), ("mlp", "threshold")]
RESULTS_DIR = os.path.join(config.EVAL_DIR, "results", "interaction")
PRED_DIR = os.path.join(config.OUT_DIR, "interaction", "predictions")
FOLD_ROWS = os.path.join(RESULTS_DIR, "fold_rows.jsonl")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_levels(in_dir=resolution.OUT_DIR):
    import pandas as pd
    tables = {}
    for level in LEVELS:
        path = check_path(resolution.table_path(level, in_dir))
        if not os.path.exists(path):
            raise SystemExit("missing {} - run `python -m eval.resolution build` first".format(path))
        table = pd.read_csv(path)
        if table.isna().any().any():
            raise ValueError("{} has missing values".format(path))
        tables[level] = table.sort_values(["subject_id", "window_start_s"],
                                          kind="stable").reset_index(drop=True)
    check_paired(tables)
    return tables


def check_paired(tables):
    ids = resolution.ID_COLUMNS
    first = tables[LEVELS[0]][ids]
    for level in LEVELS[1:]:
        if not first.equals(tables[level][ids]):
            raise ValueError("resolution tables do not hold the same windows")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def plan_counts(tasks, seeds, models_, n_subjects=15):
    from .run_eval import fits_per_outer_fold
    per_level = {m: fits_per_outer_fold(m) * n_subjects * len(seeds) * len(tasks) for m in models_}
    return {m: n * len(LEVELS) for m, n in per_level.items()}


def run_all(tables, tasks, seeds, models_, grids=None, log=log_experiment, pred_dir=PRED_DIR,
            inner_k=config.INNER_FOLDS):
    """Every (level, task, model, seed) through nested.run. -> fold rows, failures, pooled."""
    fold_rows, failures, pooled = [], [], []
    for level in LEVELS:
        table = tables[level]
        for task in tasks:
            for model in models_:
                grid = None if grids is None else grids.get(model, {})
                for seed in seeds:
                    started = datetime.now(timezone.utc).isoformat()
                    try:
                        result = nested.run(table, task, model, seed, grid=grid,
                                            inner_k=inner_k)
                    except Exception as exc:                 # recorded, never hidden
                        failures.append({"level": level, "task": task, "model": model,
                                         "seed": seed, "kind": "run failed", "error": str(exc)})
                        log("- {} interaction {} {} {} seed {}: FAILED {}".format(
                            started, level, task, model, seed, exc))
                        continue
                    for row in result["folds"]:
                        row["level"] = level
                    fold_rows += result["folds"]
                    failures += [{"level": level, "task": task, "model": model, "seed": seed, **f}
                                 for f in result["failures"]]
                    pooled.append({"level": level, "task": task, "model": model, "seed": seed,
                                   "pooled_macro_f1": pooled_macro_f1(result["predictions"], task)})
                    if pred_dir:
                        write_csv(os.path.join(pred_dir, "{}_{}_{}_seed{}.csv".format(
                            level, task, model, seed)), result["predictions"])
                    log("- {} interaction {} {} {} seed {}: {} folds, {:.1f} s, "
                        "failures/warnings: {}".format(started, level, task, model, seed,
                                                       len(result["folds"]), result["seconds"],
                                                       result["failures"] or "none"))
    return fold_rows, failures, pooled


# ---------------------------------------------------------------------------
# Analysis (pure functions of the fold rows)
# ---------------------------------------------------------------------------

def per_subject(fold_rows, task, metric):
    """{level: {model: {subject: value, seeds averaged}}}"""
    out = {}
    for level in LEVELS:
        rows = [r for r in fold_rows if r["task"] == task and r["level"] == level]
        out[level] = metrics.subject_table(rows, metric)
    return out


def _paired(a, b):
    common = sorted(set(a) & set(b), key=folds.subject_key)
    return common, np.array([a[s] - b[s] for s in common])


def _ci_row(values):
    mean, lo, hi = metrics.bootstrap_ci(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n_subjects": int(len(values))}


def analyse(fold_rows, task, metric, pairs=INTERACTION_PAIRS):
    table = per_subject(fold_rows, task, metric)
    cells, effects, contrasts, ranking = [], [], [], {}
    for level in LEVELS:
        for model, values in table[level].items():
            vals = [values[s] for s in sorted(values, key=folds.subject_key)]
            mean, lo, hi = metrics.bootstrap_ci(vals)
            cells.append({"task": task, "metric": metric, "level": level, "model": model,
                          "mean": mean, "sd": float(np.std(vals, ddof=1)) if len(vals) > 1
                          else float("nan"), "ci_low": lo, "ci_high": hi,
                          "n_subjects": len(vals)})
        means = {m: float(np.mean(list(v.values()))) for m, v in table[level].items()
                 if m in config.MODELS}
        ranking[level] = sorted(means, key=means.get, reverse=True)
    models_present = set(table[LEVELS[0]]) & set(table[LEVELS[1]])
    for model in sorted(models_present, key=ALL.index):
        _, diff = _paired(table["native"][model], table["3s"][model])
        effects.append({"task": task, "metric": metric, "model": model,
                        "effect": "native - 3s", **_ci_row(diff)})
    for a, b in pairs:
        if not {a, b} <= models_present:
            continue
        subjects, nat = _paired(table["native"][a], table["native"][b])
        _, low = _paired(table["3s"][a], table["3s"][b])
        if len(nat) != len(low):
            raise ValueError("subjects differ between levels for {} vs {}".format(a, b))
        row = {"task": task, "metric": metric, "pair": "{} - {}".format(a, b),
               "gap_native": float(nat.mean()), "gap_3s": float(low.mean()),
               "sign_flips": bool(np.sign(nat.mean()) != np.sign(low.mean())),
               **_ci_row(nat - low)}
        row["contrast"] = "({0} at native) - ({0} at 3s)".format(row["pair"])
        contrasts.append(row)
    return {"cells": cells, "effects": effects, "contrasts": contrasts, "ranking": ranking}


def verdict(result):
    """One plain sentence per task/metric for the report; never claims beyond the CIs."""
    rank = result["ranking"]
    same = rank.get("native") == rank.get("3s")
    lines = ["Model ranking by mean: native {} ; 3s {} ({}).".format(
        " > ".join(rank.get("native", [])), " > ".join(rank.get("3s", [])),
        "same order" if same else "order changes")]
    model_pairs = [c for c in result["contrasts"] if c["pair"].split(" - ")[1] in config.MODELS]
    real = [c for c in model_pairs if c["excludes_zero"]]
    if real:
        lines.append("Interaction supported (95% CI excludes 0) for: {}.".format(
            ", ".join("{} ({:+.3f} [{:+.3f}, {:+.3f}])".format(
                c["pair"], c["mean"], c["ci_low"], c["ci_high"]) for c in real)))
    else:
        lines.append("No model-pair contrast has a CI that excludes 0: no evidence that the "
                     "gap between models depends on resolution with 15 subjects. "
                     "Any change in mean order is within the uncertainty.")
    return lines


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def heatmap(fold_rows, task, metric, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = analyse(fold_rows, task, metric)
    models_ = [m for m in ALL if any(c["model"] == m for c in result["cells"])]
    grid = np.full((len(models_), len(LEVELS)), np.nan)
    text = [["" for _ in LEVELS] for _ in models_]
    for c in result["cells"]:
        i, j = models_.index(c["model"]), LEVELS.index(c["level"])
        grid[i, j] = c["mean"]
        text[i][j] = "{:.3f}\n[{:.2f}, {:.2f}]".format(c["mean"], c["ci_low"], c["ci_high"])
    fig, ax = plt.subplots(figsize=(4.6, 0.75 * len(models_) + 1.4))
    image = ax.imshow(grid, cmap="viridis", aspect="auto")
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    for i in range(len(models_)):
        for j in range(len(LEVELS)):
            light = (grid[i, j] - lo) / ((hi - lo) or 1) < 0.6
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                    color="white" if light else "black")
    ax.set_xticks(range(len(LEVELS)), ["native (beat)", "3 s bins"])
    ax.set_yticks(range(len(models_)), models_)
    ax.set_title("{} - {}\nmean over subjects [95% bootstrap CI]".format(
        task, metric.replace("_", " ")), fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(row):
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(row["mean"], row["ci_low"], row["ci_high"])


def write_report(fold_rows, failures, tasks, out_dir=RESULTS_DIR, build_report=None):
    os.makedirs(out_dir, exist_ok=True)
    all_cells, all_effects, all_contrasts = [], [], []
    md = ["# Interaction analysis: model x HR resolution", "",
          "Generated by `python -m eval.interaction`. Nested leave-one-subject-out, seeds "
          "averaged per subject, then mean over 15 subjects with a 95% percentile bootstrap CI "
          "over subjects ({} resamples). Native = one HR value per beat; 3s = the same beats "
          "averaged in 3 s bins (simulated consumer resolution, not real Fitbit data).".format(
              config.N_BOOT), ""]
    if build_report:
        md += ["Windows: {} (identical at both levels). Classes: {}.".format(
            build_report.get("windows"), build_report.get("class_counts")), ""]
    for task in tasks:
        md += ["## Task: {}".format(task), ""]
        for metric in METRICS:
            if task != "binary" and metric == "auroc":
                continue
            result = analyse(fold_rows, task, metric)
            if not result["cells"]:
                continue
            all_cells += result["cells"]
            all_effects += result["effects"]
            all_contrasts += result["contrasts"]
            png = "heatmap_{}_{}.png".format(task, metric)
            heatmap(fold_rows, task, metric, os.path.join(out_dir, png))
            md += ["### {}".format(metric), "", "![{}]({})".format(png, png), "",
                   "| model | native | 3s | native - 3s |", "|---|---|---|---|"]
            by = {(c["level"], c["model"]): c for c in result["cells"]}
            eff = {e["model"]: e for e in result["effects"]}
            for model in [m for m in ALL if ("native", m) in by]:
                n, t = by[("native", model)], by.get(("3s", model))
                md.append("| {} | {:.3f} [{:.3f}, {:.3f}] | {} | {} |".format(
                    model, n["mean"], n["ci_low"], n["ci_high"],
                    "{:.3f} [{:.3f}, {:.3f}]".format(t["mean"], t["ci_low"], t["ci_high"])
                    if t else "-", _fmt(eff[model]) if model in eff else "-"))
            md += ["", "Interaction contrasts, (A - B at native) - (A - B at 3s):", "",
                   "| pair | gap native | gap 3s | contrast [95% CI] | CI excludes 0 |",
                   "|---|---|---|---|---|"]
            for c in result["contrasts"]:
                md.append("| {} | {:+.3f} | {:+.3f} | {} | {} |".format(
                    c["pair"], c["gap_native"], c["gap_3s"], _fmt(c),
                    "yes" if c["excludes_zero"] else "no"))
            md += [""] + ["- " + line for line in verdict(result)] + [""]
    write_csv(os.path.join(out_dir, "cells.csv"), all_cells)
    write_csv(os.path.join(out_dir, "resolution_effects.csv"), all_effects)
    write_csv(os.path.join(out_dir, "contrasts.csv"), all_contrasts)
    write_csv(os.path.join(out_dir, "failures.csv"), failures)
    md += ["## Failures and warnings", "",
           "{} records (see failures.csv).".format(len(failures)) if failures else "None.", "",
           "## Threats to validity", "",
           "- 3 s binning is a simulation; a real device's filtering differs.",
           "- HR_slope is per reading, so its unit differs between levels (never across a split).",
           "- 15 subjects: wide CIs; a CI that includes 0 is not evidence of no effect.",
           "- Windows overlap 50%; all splits are by subject, but the effective sample is "
           "smaller than the window count.", ""]
    with open(os.path.join(out_dir, "INTERACTION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md))
    return os.path.join(out_dir, "INTERACTION.md")


def save_fold_rows(rows, path=FOLD_ROWS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_fold_rows(path=FOLD_ROWS):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_report():
    path = os.path.join(resolution.OUT_DIR, "build_report.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.interaction")
    parser.add_argument("action", choices=["plan", "run", "report"])
    parser.add_argument("--go", action="store_true", help="required for run")
    parser.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    parser.add_argument("--tasks", default="binary,three")
    parser.add_argument("--models", default=",".join(ALL))
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks, models_ = args.tasks.split(","), args.models.split(",")
    if args.action == "plan":
        counts = plan_counts(tasks, seeds, models_)
        for model, n in counts.items():
            print("{:10s} {:>7,d} fits".format(model, n))
        print("total      {:>7,d} fits (2 resolutions; about twice `eval.run_eval plan`)".format(
            sum(counts.values())))
        return
    if args.action == "report":
        rows = load_fold_rows()
        print(write_report(rows, [], sorted({r["task"] for r in rows}, reverse=True),
                           build_report=_build_report()))
        return
    if not args.go:
        raise SystemExit("run needs --go (after the interaction amendment is applied).")
    tables = load_levels()
    rows, failures, pooled = run_all(tables, tasks, seeds, models_)
    save_fold_rows(rows)
    write_csv(os.path.join(RESULTS_DIR, "pooled_macro_f1.csv"), pooled)
    print(write_report(rows, failures, tasks, build_report=_build_report()))


if __name__ == "__main__":
    main()
