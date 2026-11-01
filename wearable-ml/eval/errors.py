"""Error analysis: where and why the HR-only models fail.

EXPLORATORY. None of this was pre-declared; it describes failures, it does not
test hypotheses, and with 15 subjects every correlation is fragile. Report it
as such.

    # first pass, on the existing default-hyperparameter predictions (3-class)
    python -m eval.errors --predictions results/cv_predictions_hr_only.csv --model svm

    # on the tuned predictions of eval/ or eval/interaction (one model, all seeds)
    python -m eval.errors --predictions "eval/out/predictions/binary_svm_seed*.csv"
    python -m eval.errors --predictions "eval/out/interaction/predictions/native_binary_mlp_seed*.csv"

Inputs: a predictions file (or glob) in either format below, joined to
results/features.csv on (subject, window start) for the context columns
(HR at native resolution, and accelerometer motion, which the HR-only models
never see).

- old format (scripts/03_train_model.py): subject_id, label, window_start_s,
  predicted_<model>  -> task "three" (or --task binary: 3-class predictions
  collapsed to stress vs not; this is NOT a binary model, and is labelled so)
- eval format (eval/nested.py): subject, window_start_s, y_true, y_pred, score,
  model, seed, task. Several seeds: each window's error is the share of seeds
  that got it wrong; "wrong" means wrong in at least half of the seeds.

Analyses (all written to ERRORS.md, CSVs and PNGs in the output folder):
1. per subject: error rate, macro-F1, resting HR (mean HR_mean in baseline),
   stress response (stress minus baseline HR_mean), amusement response;
   Spearman correlation of stress response with macro-F1.
2. per condition: confusion of true condition x predicted class.
3. position: error rate by window position inside its condition block.
4. motion: error rate by accelerometer-motion tertile (ACC_mag_std) per
   condition, and motion of false alarms vs correct non-stress windows.
5. absolute vs person-relative HR: error rate by subject resting-HR tertile,
   and a diagnostic of how the same windows separate after per-subject
   centring (not a model; uses the labels of the subject, so never a result).
6. examples: the most confident errors (eval format, by score) or the windows
   whose HR most contradicts their label (old format), plus HR timelines of
   the two worst subjects with errors marked.
"""

import argparse
import glob
import os

import numpy as np

from . import config
from .data import check_path

OUT_ROOT = os.path.join(config.EVAL_DIR, "results", "errors")
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement"}
CLASS_NAMES = {"three": CONDITIONS, "binary": {0: "non-stress", 1: "stress"}}
KEYS = ["subject_id", "window_start_s"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(pattern, model=None, task=None):
    """-> (table with subject_id, window_start_s, condition, y_true, y_pred,
    error, wrong, score, n_seeds), task, description"""
    import pandas as pd

    paths = sorted(glob.glob(pattern)) or [pattern]
    frames = [pd.read_csv(check_path(p)) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    if "y_pred" in raw.columns:                                   # eval format
        task = task or raw["task"].iat[0]
        if model:
            raw = raw[raw["model"] == model]
        models = sorted(raw["model"].unique())
        if len(models) != 1:
            raise ValueError("predictions hold several models {}; pass --model".format(models))
        raw = raw.rename(columns={"subject": "subject_id"})
        raw["error"] = (raw["y_true"] != raw["y_pred"]).astype(float)
        agg = raw.groupby(KEYS, sort=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", lambda s: s.mode().iat[0]),
            error=("error", "mean"), score=("score", "mean"),
            n_seeds=("seed", "nunique")).reset_index()
        # prediction shown for an error: most common WRONG prediction across the seeds that
        # got the window wrong; a tie goes to the lowest class label (value_counts is sorted
        # by label and idxmax returns the first maximum). Display only: metrics use y_pred.
        shown = (raw[raw["error"] == 1].groupby(KEYS)["y_pred"]
                 .agg(lambda s: s.value_counts().sort_index().idxmax()).rename("y_pred_shown"))
        agg = agg.merge(shown.reset_index(), on=KEYS, how="left")
        desc = "{} ({} task), {} seed(s), from {}".format(
            models[0], task, int(agg["n_seeds"].max()), ", ".join(os.path.basename(p) for p in paths))
    else:                                                          # old format
        column = "predicted_{}".format(model or "svm")
        if column not in raw.columns:
            raise ValueError("no column {} in {}".format(column, pattern))
        agg = raw[KEYS + ["label"]].copy()
        task = task or "three"
        if task == "binary":
            agg["y_true"] = (raw["label"] == config.STRESS).astype(int)
            agg["y_pred"] = (raw[column] == config.STRESS).astype(int)
            note = "3-class predictions collapsed to stress vs not (not a binary model)"
        else:
            agg["y_true"], agg["y_pred"] = raw["label"], raw[column]
            note = "3-class"
        agg["error"] = (agg["y_true"] != agg["y_pred"]).astype(float)
        agg["score"], agg["n_seeds"] = np.nan, 1
        agg["y_pred_shown"] = agg["y_pred"]
        agg = agg.drop(columns="label")
        desc = "{} ({}), default hyperparameters, one run, from {}".format(
            model or "svm", note, os.path.basename(paths[0]))
    agg["wrong"] = agg["error"] >= 0.5
    return agg, task, desc


def attach_context(preds, features_path=config.FEATURES):
    """Join condition, native HR features and accelerometer motion; add position."""
    import pandas as pd

    feats = pd.read_csv(check_path(features_path))
    cols = KEYS + ["label"] + config.HR_FEATURES + ["ACC_mag_std", "ACC_mag_mean"]
    table = preds.merge(feats[cols], on=KEYS, how="left", validate="one_to_one")
    if table["label"].isna().any():
        raise ValueError("{} prediction rows have no matching window in features.csv".format(
            int(table["label"].isna().sum())))
    table["condition"] = table["label"].map(CONDITIONS)
    table = table.sort_values(KEYS).reset_index(drop=True)
    # position of the window inside its contiguous condition block (1 = first)
    block = (table["label"] != table.groupby("subject_id")["label"].shift()).cumsum()
    table["position"] = table.groupby(block).cumcount() + 1
    table["block_length"] = table.groupby(block)["position"].transform("max")
    return table


# ---------------------------------------------------------------------------
# Analyses (pure functions of the joined table)
# ---------------------------------------------------------------------------

def spearman(x, y):
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def per_subject(table, task):
    from sklearn.metrics import f1_score
    import pandas as pd

    labels = sorted(CLASS_NAMES[task])
    rows = []
    for subject, g in table.groupby("subject_id", sort=False):
        hr = g.groupby("label")["HR_mean"].mean()
        rows.append({
            "subject_id": subject, "windows": len(g),
            "error_rate": float(g["error"].mean()),
            "macro_f1": float(f1_score(g["y_true"], g["y_pred"], labels=labels,
                                       average="macro", zero_division=0)),
            "resting_hr": float(hr.get(1, np.nan)),
            "stress_response": float(hr.get(2, np.nan) - hr.get(1, np.nan)),
            "amusement_response": float(hr.get(3, np.nan) - hr.get(1, np.nan)),
            "motion_median": float(g["ACC_mag_std"].median())})
    out = pd.DataFrame(rows).sort_values("macro_f1").reset_index(drop=True)
    return out


def by_condition(table, task):
    import pandas as pd
    names = CLASS_NAMES[task]
    counts = pd.crosstab(table["condition"], table["y_pred"].map(names))
    rates = table.groupby("condition")["error"].agg(["mean", "size"]).rename(
        columns={"mean": "error_rate", "size": "windows"})
    return counts, rates


def by_position(table, edges=(0, 2, 5, 1000)):
    import pandas as pd
    labels = ["windows 1-2", "windows 3-5", "window 6+"]
    bucket = pd.cut(table["position"], list(edges), labels=labels)
    return table.groupby(["condition", bucket], observed=True)["error"].agg(
        ["mean", "size"]).rename(columns={"mean": "error_rate", "size": "windows"}).reset_index()


def by_motion(table):
    import pandas as pd
    tertile = pd.qcut(table["ACC_mag_std"], 3, labels=["low", "mid", "high"])
    rates = table.groupby(["condition", tertile], observed=True)["error"].agg(
        ["mean", "size"]).rename(columns={"mean": "error_rate", "size": "windows"}).reset_index()
    medians = table.groupby("condition")["ACC_mag_std"].median()
    return rates, medians


def false_alarm_motion(table, task):
    """Motion of non-stress windows called stress vs non-stress windows called right."""
    from scipy.stats import mannwhitneyu
    stress_pred = table["y_pred"] == (1 if task == "binary" else config.STRESS)
    calm = table["label"] != config.STRESS
    fa = table.loc[calm & stress_pred, "ACC_mag_std"]
    ok = table.loc[calm & ~table["wrong"], "ACC_mag_std"]
    if len(fa) < 3 or len(ok) < 3:
        return None
    _, p = mannwhitneyu(fa, ok)
    return {"false_alarms": int(len(fa)), "median_motion_false_alarm": float(fa.median()),
            "correct_non_stress": int(len(ok)), "median_motion_correct": float(ok.median()),
            "mann_whitney_p": float(p)}


def by_resting_hr(table, subjects):
    import pandas as pd
    tertile = pd.qcut(subjects["resting_hr"], 3, labels=["low", "mid", "high"])
    lookup = dict(zip(subjects["subject_id"], tertile))
    t = table.assign(resting_tertile=table["subject_id"].map(lookup))
    out = t.groupby(["resting_tertile", "condition"], observed=True)["error"].mean().unstack()
    return out.reindex(["low", "mid", "high"])


def relative_hr_diagnostic(table):
    """How far stress and baseline HR_mean overlap pooled vs after per-subject centring.

    Uses each subject's own labels to centre (baseline mean), so it is an upper
    bound showing how much between-person differences hide; never a model result.
    """
    from sklearn.metrics import roc_auc_score
    sb = table[table["label"].isin([1, 2])]
    y = (sb["label"] == 2).astype(int)
    rest = sb[sb["label"] == 1].groupby("subject_id")["HR_mean"].mean()
    centred = sb["HR_mean"] - sb["subject_id"].map(rest)
    return {"auroc_absolute_hr_mean": float(roc_auc_score(y, sb["HR_mean"])),
            "auroc_subject_centred_hr_mean": float(roc_auc_score(y, centred))}


def examples(table, task, per_type=2):
    """Example errors: the most confident (eval format, by |score|) or, without scores,
    the windows closest to the median HR_mean of each (true condition -> predicted) error
    type, i.e. typical rather than extreme errors."""
    names = CLASS_NAMES[task]
    wrong = table[table["wrong"]].copy()
    wrong["predicted"] = wrong["y_pred_shown"].astype(int).map(names)
    cols = ["subject_id", "window_start_s", "condition", "predicted", "position", "HR_mean",
            "HR_std", "ACC_mag_std"]
    if wrong["score"].notna().any():
        wrong["abs_score"] = wrong["score"].abs()
        picked = wrong.sort_values("abs_score", ascending=False).head(4 * per_type)
        return picked[cols + ["abs_score"]], "largest |score| among wrong windows (most confident errors)"
    picked = []
    for _, group in wrong.groupby(["condition", "predicted"], sort=True):
        distance = (group["HR_mean"] - group["HR_mean"].median()).abs()
        picked.append(group.loc[distance.sort_values().index[:per_type]])
    out = wrong.iloc[0:0] if not picked else __import__("pandas").concat(picked)
    return out[cols], ("for each error type (true condition -> predicted class), the {} windows "
                       "closest to that type's median HR_mean (typical, not extreme, errors)"
                       .format(per_type))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_response(subjects, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(subjects["stress_response"], subjects["macro_f1"], color="#3b6ea8")
    for _, r in subjects.iterrows():
        ax.annotate(r["subject_id"], (r["stress_response"], r["macro_f1"]), fontsize=7,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("stress response: stress - baseline HR_mean (bpm)")
    ax.set_ylabel("per-subject macro-F1")
    ax.set_title("Who is hard to classify?", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_motion(rates, path):
    plt = _plt()
    conds = [c for c in CONDITIONS.values() if c in set(rates["condition"])]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    width = 0.26
    for i, level in enumerate(["low", "mid", "high"]):
        vals = [rates[(rates.condition == c) & (rates.ACC_mag_std == level)]["error_rate"].sum()
                for c in conds]
        ax.bar(np.arange(len(conds)) + (i - 1) * width, vals, width, label=level + " motion",
               color=["#c6dbef", "#6baed6", "#2171b5"][i])
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel("error rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Error rate by wrist-motion tertile", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_timelines(table, subjects_to_plot, path):
    plt = _plt()
    colours = {"baseline": "#7f7f7f", "stress": "#d62728", "amusement": "#2ca02c"}
    fig, axes = plt.subplots(len(subjects_to_plot), 1, figsize=(6.5, 2.3 * len(subjects_to_plot)),
                             squeeze=False)
    for ax, subject in zip(axes[:, 0], subjects_to_plot):
        g = table[table["subject_id"] == subject]
        for cond, colour in colours.items():
            part = g[g["condition"] == cond]
            ax.scatter(part["window_start_s"] / 60, part["HR_mean"], s=12, color=colour, label=cond)
        bad = g[g["wrong"]]
        ax.scatter(bad["window_start_s"] / 60, bad["HR_mean"], s=40, facecolors="none",
                   edgecolors="black", linewidths=0.8, label="misclassified")
        ax.set_title(subject, fontsize=9)
        ax.set_ylabel("HR_mean (bpm)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time in session (min)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=4, frameon=False, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _md_table(frame, floatfmt="{:.3f}"):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                                       for v in row.values) + " |")
    return lines


def run(pattern, model=None, task=None, out_dir=None, features_path=config.FEATURES):
    preds, task, desc = load_predictions(pattern, model, task)
    table = attach_context(preds, features_path)
    name = os.path.splitext(os.path.basename(pattern.replace("*", "all")))[0]
    out_dir = out_dir or os.path.join(OUT_ROOT, "{}_{}_{}".format(name, model or "model", task))
    os.makedirs(out_dir, exist_ok=True)

    subjects = per_subject(table, task)
    rho, p = spearman(subjects["stress_response"], subjects["macro_f1"])
    counts, cond_rates = by_condition(table, task)
    position = by_position(table)
    motion, motion_medians = by_motion(table)
    fa = false_alarm_motion(table, task)
    resting = by_resting_hr(table, subjects)
    rel = relative_hr_diagnostic(table)
    ex, ex_how = examples(table, task)
    worst = list(subjects["subject_id"].head(2))

    fig_response(subjects, os.path.join(out_dir, "subject_response_vs_f1.png"))
    fig_motion(motion, os.path.join(out_dir, "error_by_motion.png"))
    fig_timelines(table, worst + [subjects["subject_id"].iat[-1]],
                  os.path.join(out_dir, "timelines.png"))
    subjects.to_csv(os.path.join(out_dir, "per_subject.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "windows.csv"), index=False)

    md = ["# Error analysis (EXPLORATORY)", "",
          "Predictions: {}. {} windows, {} subjects. Context (native HR, wrist motion) "
          "joined from results/features.csv; the HR-only models never see motion. "
          "Not pre-declared; 15 subjects, so correlations are fragile.".format(
              desc, len(table), table["subject_id"].nunique()), "",
          "Overall window error rate: {:.3f}.".format(table["error"].mean()), "",
          "## 1. Which subjects fail", "",
          "![](subject_response_vs_f1.png)", ""]
    md += _md_table(subjects.round(3))
    md += ["", "Spearman correlation, stress response vs per-subject macro-F1: rho = {:+.2f} "
           "(p = {:.3f}, n = {}).".format(rho, p, len(subjects)), "",
           "## 2. Which conditions fail", ""]
    md += _md_table(counts.reset_index().rename(columns={"condition": "true \\ predicted"}),
                    "{:.0f}")
    md += [""] + _md_table(cond_rates.reset_index())
    md += ["", "## 3. Where inside a condition block", ""] + _md_table(position)
    md += ["", "## 4. Wrist motion (not a model input)", "", "![](error_by_motion.png)", ""]
    md += _md_table(motion.rename(columns={"ACC_mag_std": "motion tertile"}))
    md += ["", "Median ACC_mag_std by condition: " + ", ".join(
        "{} {:.2f}".format(k, v) for k, v in motion_medians.items()) + "."]
    if fa:
        md += ["", "Non-stress windows called stress (n = {false_alarms}) had median motion "
               "{median_motion_false_alarm:.2f} vs {median_motion_correct:.2f} for correct "
               "non-stress windows (n = {correct_non_stress}); Mann-Whitney p = "
               "{mann_whitney_p:.3g} (windows overlap and are not independent, so p is "
               "optimistic).".format(**fa)]
    md += ["", "## 5. Absolute vs person-relative heart rate", "",
           "Error rate by the subject's resting-HR tertile:", ""]
    md += _md_table(resting.reset_index().rename(columns={"resting_tertile": "resting HR"}))
    md += ["", "Diagnostic (uses each subject's own baseline, so NOT a model result): AUROC of "
           "HR_mean for stress vs baseline is {auroc_absolute_hr_mean:.3f} pooled across "
           "people and {auroc_subject_centred_hr_mean:.3f} after subtracting each subject's "
           "resting HR.".format(**rel), "",
           "## 6. Examples", "", "Examples: " + ex_how + ".", ""]
    md += _md_table(ex.round(2))
    md += ["", "![](timelines.png)", "",
           "Timelines: the two lowest-F1 subjects and the highest. Circles = misclassified.", ""]
    with open(os.path.join(out_dir, "ERRORS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(md))
    return out_dir, {"subjects": subjects, "rho": rho, "p": p, "cond_rates": cond_rates,
                     "position": position, "motion": motion, "false_alarm": fa,
                     "resting": resting, "relative": rel}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.errors")
    parser.add_argument("--predictions", required=True, help="file or quoted glob")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", choices=["binary", "three"], default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out_dir, _ = run(args.predictions, args.model, args.task, args.out)
    print("written to", os.path.join(out_dir, "ERRORS.md"))


if __name__ == "__main__":
    main()
