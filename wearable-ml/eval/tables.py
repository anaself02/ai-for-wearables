"""Turn the run's CSVs into eval/RESULTS.md (read-only on the results).

    python -m eval.tables

Reads eval/results/*.csv written by `run_eval run`. Nothing is recomputed from
the data; every number comes from those files.
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import config

RESULTS = os.path.join(config.EVAL_DIR, "results")
OUT = os.path.join(config.EVAL_DIR, "RESULTS.md")
ORDER = ["majority", "threshold", "svm", "rf", "mlp"]
TASK_TITLES = {"binary": "Primary: binary, stress vs non-stress",
               "three": "Secondary: 3-class (baseline / stress / amusement), WESAD only"}
#: Table I (results/scores_hr_only.json): pooled out-of-fold macro-F1, 3-class,
#: heart-rate features only, default hyperparameters, leave-one-subject-out.
TABLE_I_HR_ONLY = {"svm": 0.4994678389427316, "rf": 0.4977246139717865}


def f3(value):
    """Format a number with 3 decimals ('n/a' for None or NaN). Randomness: none."""
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else "{:.3f}".format(value)


def ci(row):
    """'mean [low, high]' from a summary row, 3 decimals. Randomness: none."""
    return "{} [{}, {}]".format(f3(row["mean"]), f3(row["ci_low"]), f3(row["ci_high"]))


def summary_table(summary, metric):
    """Markdown lines of per-model mean, SD and bootstrap CI for one metric.

    Inputs: the summary DataFrame and a metric name. Output: list of lines.
    Numbers are copied from the CSV; nothing is recomputed. Randomness: none.
    """
    rows = summary[summary["metric"] == metric].set_index("model")
    lines = ["| model | mean | SD over subjects | 95% bootstrap CI of the mean |", "|---|---|---|---|"]
    for model in ORDER:
        if model in rows.index:
            r = rows.loc[model]
            lines.append("| {} | {} | {} | [{}, {}] |".format(
                model, f3(r["mean"]), f3(r["sd"]), f3(r["ci_low"]), f3(r["ci_high"])))
    return lines


def difference_table(diffs, metric):
    """Markdown lines of the pairwise differences for one metric.

    Inputs: the differences DataFrame and a metric name. Output: list of lines in
    the order mlp-svm, mlp-rf, rf-svm, then the baseline pairs. Randomness: none.
    """
    rows = diffs[diffs["metric"] == metric]
    lines = ["| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |",
             "|---|---|---|---|"]
    order = {"mlp - svm": 0, "mlp - rf": 1, "rf - svm": 2}
    for _, r in sorted(rows.iterrows(), key=lambda kv: (order.get(kv[1]["pair"], 9),
                                                         kv[1]["pair"])):
        lines.append("| {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {} |".format(
            r["pair"], r["mean"], r["ci_low"], r["ci_high"], "yes" if r["excludes_zero"] else "no"))
    return lines


def chosen_tables(folds, task):
    """Per model: how often each setting was chosen over 15 folds x 5 seeds, and per subject."""
    lines = []
    for model in config.MODELS:
        rows = folds[(folds["model"] == model)]
        if rows.empty:
            continue
        counts = Counter(r["chosen"] for _, r in rows.iterrows())
        lines += ["", "**{}** - chosen setting over {} outer fits (15 subjects x {} seeds):".format(
            model, len(rows), rows["seed"].nunique()), ""]
        lines += ["| setting | times chosen |", "|---|---|"]
        for setting, n in counts.most_common():
            lines.append("| `{}` | {} |".format(setting, n))
        lines += ["", "Per subject (the setting chosen most often over the seeds; in brackets how many "
                  "of the seeds chose it):", "", "| subject | modal choice | seeds agreeing |", "|---|---|---|"]
        for subject, group in rows.groupby("subject", sort=False):
            top, n = Counter(group["chosen"]).most_common(1)[0]
            lines.append("| {} | `{}` | {} of {} |".format(subject, top, n, len(group)))
    return lines


RESOLUTION = os.path.join(RESULTS, "resolution")
CONDITION_ORDER = ["R1s", "R2", "R1stoR2"]


def interaction_section():
    """Markdown lines for the model x HR-resolution stage (PRE_DECLARED.md Amendment 5).

    Input: the CSV and JSON files written by `python -m eval.interaction run --go`
    in eval/results/resolution/ (summary.csv, interaction.csv, rankings.json,
    failures.csv). Output: list of lines, empty if that folder has no summary.csv.
    Every number is copied from those files; nothing is recomputed. Randomness: none.
    """
    summary_path = os.path.join(RESOLUTION, "summary.csv")
    if not os.path.exists(summary_path):
        return []
    summary = pd.read_csv(summary_path)
    summary = summary[summary["metric"] == "macro_f1"]
    contrasts = pd.read_csv(os.path.join(RESOLUTION, "interaction.csv"))
    contrasts = contrasts[contrasts["metric"] == "macro_f1"]
    lines = ["## Interaction: model x heart-rate resolution (binary, Amendment 5)", "",
             "Protocol: `PRE_DECLARED.md` Amendment 5 (frozen before any interaction result). "
             "R1s = beat-level HR, R2 = the same beats averaged into 3.0 s bins (a controlled "
             "change of time resolution, not a device model), R1stoR2 = trained on R1s and tested on R2. "
             "Same 1012 windows in every condition; per-subject macro-F1, 5 seeds averaged first; "
             "95% paired bootstrap CIs over the 15 subjects, **not adjusted for multiplicity**. "
             "Source: `results/resolution/*.csv`.", "",
             "### Macro-F1 per subject by condition (mean [95% CI])", "",
             "| model | R1s | R2 | R1stoR2 |", "|---|---|---|---|"]
    for model in ORDER:
        cells = []
        for condition in CONDITION_ORDER:
            row = summary[(summary["condition"] == condition) & (summary["model"] == model)]
            cells.append(ci(row.iloc[0]) if len(row) else "n/a")
        lines.append("| {} | {} |".format(model, " | ".join(cells)))
    lines += ["", "### Interaction contrasts (the pre-declared test; difference of differences)", "",
              "| contrast | difference at R1s | difference at R2 | contrast (95% CI) | CI excludes 0 |",
              "|---|---|---|---|---|"]
    for _, r in contrasts[contrasts["kind"] == "interaction"].iterrows():
        lines.append("| {} | {:+.3f} | {:+.3f} | {:+.3f} [{:+.3f}, {:+.3f}] | {} |".format(
            r["contrast"], r["diff_at_R1"], r["diff_at_R2"], r["mean"], r["ci_low"], r["ci_high"],
            "yes" if r["excludes_zero"] else "no"))
    lines += ["", "### Resolution effect (R2 minus R1s) and transfer (R1stoR2 minus R1s / R2)", "",
              "| contrast | mean difference [95% CI] | CI excludes 0 |", "|---|---|---|"]
    for _, r in contrasts[contrasts["kind"].isin(["resolution", "transfer"])].iterrows():
        lines.append("| {} | {:+.3f} [{:+.3f}, {:+.3f}] | {} |".format(
            r["contrast"], r["mean"], r["ci_low"], r["ci_high"], "yes" if r["excludes_zero"] else "no"))
    ranking_path = os.path.join(RESOLUTION, "rankings.json")
    if os.path.exists(ranking_path):
        with open(ranking_path, encoding="utf-8") as handle:
            rankings = json.load(handle)
        lines += ["", "### Order of the three models by mean macro-F1 (descriptive; best first)", ""]
        lines += ["- {}: {}".format(c, " > ".join(rankings[c])) for c in CONDITION_ORDER if c in rankings]
    failures_path = os.path.join(RESOLUTION, "failures.csv")
    if os.path.exists(failures_path):
        failures = pd.read_csv(failures_path)
        lines += ["", "MLP convergence (max_iter=300): {} of {} inner fits ran to max_iter and {} of {} "
                  "outer refits did not converge (`results/resolution/failures.csv`).".format(
                      int(failures["inner_fits_not_converged"].sum()), int(failures["inner_fits"].sum()),
                      int((~failures["outer_refit_converged"].astype(bool)).sum()), len(failures))]
    lines += ["", "Limits (stated in Amendment 5): 15 subjects, so the interaction test has low power; "
              "binning does not reproduce a device's smoothing; R1stoR2 also carries a shift in the "
              "feature distributions.", ""]
    return lines


def main():
    """Write eval/RESULTS.md from eval/results/*.csv (read-only on the results).

    Input: none (reads the CSVs written by run_eval; stops if there are none, and
    adds the interaction section if eval/results/resolution/ exists). Output:
    eval/RESULTS.md. Randomness: none.
    """
    tasks = [t for t in ("binary", "three")
             if os.path.exists(os.path.join(RESULTS, "summary_{}.csv".format(t)))]
    if not tasks:
        raise SystemExit("No results yet: run `python -m eval.run_eval run --go` first.")
    pooled = pd.read_csv(os.path.join(RESULTS, "pooled_macro_f1.csv"))
    failures_path = os.path.join(RESULTS, "failures.csv")
    failures = pd.read_csv(failures_path) if os.path.exists(failures_path) else pd.DataFrame()

    lines = ["# Results: SVM, random forest and MLP under one protocol", "",
             "Protocol: `PRE_DECLARED.md` Amendment 4 (frozen before any nested result). Leave-one-subject-out, "
             "15 subjects, inner subject-grouped tuning, 5 seeds averaged per subject first. "
             "Mean and SD are over the 15 subjects; CIs are 95% percentile bootstrap over subjects "
             "(10,000 resamples). Pairwise CIs are **not adjusted for multiplicity**.", ""]
    for task in tasks:
        summary = pd.read_csv(os.path.join(RESULTS, "summary_{}.csv".format(task)))
        diffs = pd.read_csv(os.path.join(RESULTS, "differences_{}.csv".format(task)))
        folds = pd.read_csv(os.path.join(RESULTS, "folds_{}.csv".format(task)))
        lines += ["## {}".format(TASK_TITLES[task]), "", "### Macro-F1 per subject (headline)", ""]
        lines += summary_table(summary, "macro_f1")
        lines += ["", "### Balanced accuracy per subject", ""] + summary_table(summary, "balanced_accuracy")
        if task == "binary":
            lines += ["", "### AUROC per subject", ""] + summary_table(summary, "auroc")
        lines += ["", "### Pairwise differences in macro-F1", ""] + difference_table(diffs, "macro_f1")
        lines += ["", "### Pairwise differences in balanced accuracy", ""] \
            + difference_table(diffs, "balanced_accuracy")
        if task == "binary":
            lines += ["", "### Pairwise differences in AUROC", ""] + difference_table(diffs, "auroc")

        # variability across seeds (stochastic models) and per-class recall
        lines += ["", "### Stability across the 5 seeds (mean macro-F1 over subjects, per seed)", "",
                  "| model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range |", "|---|---|---|---|---|---|---|"]
        for model in ORDER:
            sub = folds[folds["model"] == model]
            if sub.empty:
                continue
            per_seed = sub.groupby("seed")["macro_f1"].mean()
            lines.append("| {} | {} | {} |".format(
                model, " | ".join(f3(per_seed.get(s)) for s in range(5)),
                f3(per_seed.max() - per_seed.min())))
        names = config.TASKS[task]["names"]
        lines += ["", "### Mean per-class recall (over subjects)", "",
                  "| model | " + " | ".join(names[l] for l in config.TASKS[task]["labels"]) + " |",
                  "|---|" + "---|" * len(names)]
        for model in ORDER:
            sub = folds[folds["model"] == model]
            if sub.empty:
                continue
            recalls = defaultdict(list)
            for _, r in sub.iterrows():
                for label, value in json.loads(r["recall"]).items():
                    recalls[int(label)].append(value)
            lines.append("| {} | {} |".format(model, " | ".join(
                f3(float(np.mean(recalls[l]))) for l in config.TASKS[task]["labels"])))

        # pooled macro-F1 next to Table I
        lines += ["", "### Pooled out-of-fold macro-F1 (the Table I quantity)", "",
                  "| model | pooled macro-F1, mean over 5 seeds | range over seeds | Table I (3-class, HR only, default hyperparameters) |",
                  "|---|---|---|---|"]
        sub = pooled[pooled["task"] == task]
        for model in ORDER:
            vals = sub[sub["model"] == model]["pooled_macro_f1"]
            if vals.empty:
                continue
            ref = TABLE_I_HR_ONLY.get(model) if task == "three" else None
            lines.append("| {} | {} | {} to {} | {} |".format(
                model, f3(vals.mean()), f3(vals.min()), f3(vals.max()),
                f3(ref) if ref is not None else "-"))
        lines += ["", "### Chosen hyperparameters"] + chosen_tables(folds, task) + [""]

    lines += ["## Failed or unstable runs", ""]
    if failures.empty:
        lines += ["None recorded.", ""]
    else:
        lines += ["{} record(s) in `results/failures.csv`:".format(len(failures)), ""]
        kinds = failures.groupby(["task", "model", "kind"]).size().reset_index(name="records")
        lines += ["| task | model | kind | records |", "|---|---|---|---|"]
        for _, r in kinds.iterrows():
            lines.append("| {} | {} | {} | {} |".format(r["task"], r["model"], r["kind"], r["records"]))
        conv = failures[failures["kind"] == "convergence warnings"]
        if not conv.empty:
            lines += ["", "MLP convergence (max_iter=300, no early stopping): inner fits that ran to "
                      "max_iter, summed per task:", "",
                      "| task | inner fits not converged | of | outer refits not converged | of outer fits |",
                      "|---|---|---|---|---|"]
            for task in tasks:
                folds = pd.read_csv(os.path.join(RESULTS, "folds_{}.csv".format(task)))
                n_outer = int((folds["model"] == "mlp").sum())
                c = conv[conv["task"] == task]
                lines.append("| {} | {} | {} | {} | {} |".format(
                    task, int(c["inner_fits_not_converged"].sum()), int(c["inner_fits"].sum()),
                    int((~c["outer_refit_converged"].astype(bool)).sum()), n_outer))
        lines.append("")
    lines += interaction_section()
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
