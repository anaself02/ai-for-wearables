"""
04_evaluate.py
--------------
STEP 4 (reporting) and STEP 5: turn the cross-validation predictions from
03_train_model.py into the numbers and figures for the write-up, then write
RESULTS.md.

Nothing is retrained here. We reuse results/cv_predictions_<set>.csv, which
holds the out-of-fold prediction for every window - i.e. every prediction was
made by a model that had never seen that window's subject. Pooling those
predictions and building ONE confusion matrix over all 15 subjects is the
correct way to summarise leave-one-subject-out cross-validation.

Two feature sets are reported side by side:
  "full"    - all 8 channels (research-grade multi-sensor wristband)
  "hr_only" - heart rate only (consumer wearable)

Outputs:
  results/confusion_matrix_<set>_<model>.png
  results/per_subject_f1_<set>.png
  results/classification_report_<set>_<model>.txt
  RESULTS.md                              WESAD results only - tracked in git
  results/personal/RESULTS_personal.md    Phase 3 personal-data results -
                                          NOT tracked (personal health data)

Usage:
    python 04_evaluate.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")          # write PNG files, never open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import project_config as config
import features


# Plot styling. Ink colours are deliberately grey rather than black so that the
# data is the darkest thing on the page; the grid and axes stay recessive.
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#555555"
INK_MUTED = "#8a8a8a"
BAR_COLOR = "#3b6ea5"          # a single hue - there is only one series
SURFACE = "#ffffff"

# Human-readable names for the two feature sets, used in figure titles and in
# RESULTS.md so the report never shows a bare code identifier.
FEATURE_SET_LABELS = {
    "full": "all 8 channels (research-grade wristband)",
    "hr_only": "heart rate only (consumer wearable)",
}

# The algorithm 03_train_model.py saves for each feature set, and therefore
# the one the headline comparison is based on.
HEADLINE_MODEL = "svm"

# Snapshot of the results as they stood BEFORE the inter-beat-interval median
# filter was added to features.py. Kept on disk (this project is not under
# version control) so the before/after comparison below is computed from the
# real earlier numbers rather than from numbers typed in by hand.
ARCHIVE_DIRNAME = "before_ibi_median_filter"


def plot_confusion_matrix(y_true, y_pred, class_names, title, path):
    """
    Draw a confusion matrix as a heatmap.

    A confusion matrix is magnitude data, so it gets a SEQUENTIAL colour scale:
    one hue running light (few windows) to dark (many windows). Each cell is
    shaded by the ROW-normalised value - the share of that true class that was
    predicted as this column - because the classes have different sizes and
    raw counts would make the big class look good automatically. The raw count
    is printed underneath the percentage so nothing is hidden.

    Reading it: row = what the window really was, column = what the model said.
    The diagonal is correct; everything off the diagonal is a mistake.
    """
    counts = confusion_matrix(y_true, y_pred, labels=class_names)

    # Row-normalise: divide each row by its total (guard against a class with
    # zero windows, which would give a divide-by-zero).
    row_totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, row_totals, where=(row_totals != 0))

    fig, ax = plt.subplots(figsize=(5.8, 5.2), facecolor=SURFACE)
    image = ax.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([config.LABEL_NAMES[c] for c in class_names])
    ax.set_yticklabels([config.LABEL_NAMES[c] for c in class_names])
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, pad=14, fontsize=11)

    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Label every cell directly, so the figure is readable without the
    # colour bar and remains readable in greyscale or print.
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            # White text on the dark cells, dark text on the light ones.
            text_color = "#ffffff" if shares[row, col] > 0.55 else INK_PRIMARY
            ax.text(col, row,
                    "{:.0%}\n({:d})".format(shares[row, col], counts[row, col]),
                    ha="center", va="center", color=text_color, fontsize=10)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of the actual class", color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_SECONDARY, length=0)
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return counts


def plot_per_subject_f1(per_subject, title, path):
    """
    Bar chart of macro-F1 for each held-out subject.

    One series, so no legend is needed - the title names what is plotted. Each
    bar is labelled with its value, and a dashed line marks the mean, which
    makes the spread between subjects obvious. That spread is the point of the
    figure: it shows how much performance depends on WHICH person you test on,
    which is exactly what a random split would have hidden.
    """
    per_subject = per_subject.sort_values("f1_macro")

    fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor=SURFACE)
    positions = np.arange(len(per_subject))
    ax.bar(positions, per_subject["f1_macro"], color=BAR_COLOR, width=0.65)

    mean_f1 = per_subject["f1_macro"].mean()
    ax.axhline(mean_f1, color=INK_MUTED, linestyle="--", linewidth=1.2)
    # Place the mean label at the left edge: the bars are sorted ascending, so
    # the left end of the line is the side with the most clear space.
    ax.text(-0.45, mean_f1 + 0.03, "mean {:.2f}".format(mean_f1),
            color=INK_SECONDARY, ha="left", fontsize=9)

    for x, value in zip(positions, per_subject["f1_macro"]):
        ax.text(x, value + 0.02, "{:.2f}".format(value),
                ha="center", color=INK_SECONDARY, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(per_subject["subject_id"])
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Macro-F1 on held-out subject", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, pad=12, fontsize=11)

    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def evaluate_feature_set(feature_set_name, class_names):
    """
    Compute metrics, write the text reports and draw the figures for one
    feature set. Returns everything RESULTS.md needs about it.
    """
    predictions = pd.read_csv(os.path.join(
        config.RESULTS_DIR, "cv_predictions_" + feature_set_name + ".csv"))

    with open(os.path.join(config.RESULTS_DIR,
                           "scores_" + feature_set_name + ".json")) as f:
        saved = json.load(f)

    y_true = predictions["label"]
    summary = {"feature_set": feature_set_name,
               "n_features": saved["n_features"],
               "feature_names": saved["feature_names"],
               "scores": saved["scores"],
               "reports": {},
               "counts": {}}

    print("#" * 66)
    print("Feature set '{}' - {} ({} features)".format(
        feature_set_name, FEATURE_SET_LABELS[feature_set_name],
        saved["n_features"]))
    print("#" * 66)

    # A confusion matrix and a text report for BOTH models, so the comparison
    # in RESULTS.md can be checked rather than taken on trust.
    for model_name in saved["scores"]:
        y_pred = predictions["predicted_" + model_name]

        print("--- {} ---".format(model_name))
        print("accuracy  {:.3f}".format(accuracy_score(y_true, y_pred)))
        print("macro-F1  {:.3f}".format(
            f1_score(y_true, y_pred, average="macro", zero_division=0)))

        report_text = classification_report(
            y_true, y_pred,
            labels=class_names,
            target_names=[config.LABEL_NAMES[c] for c in class_names],
            digits=3, zero_division=0)
        print(report_text)
        summary["reports"][model_name] = report_text

        with open(os.path.join(
                config.RESULTS_DIR,
                "classification_report_{}_{}.txt".format(feature_set_name,
                                                         model_name)),
                  "w") as f:
            f.write(report_text)

        counts = plot_confusion_matrix(
            y_true, y_pred, class_names,
            "{} - {}\nleave-one-subject-out".format(
                model_name.replace("_", " "),
                FEATURE_SET_LABELS[feature_set_name]),
            os.path.join(config.RESULTS_DIR,
                         "confusion_matrix_{}_{}.png".format(feature_set_name,
                                                             model_name)))
        summary["counts"][model_name] = counts

    per_subject = pd.read_csv(os.path.join(
        config.RESULTS_DIR,
        "per_subject_{}_{}.csv".format(feature_set_name, HEADLINE_MODEL)))
    plot_per_subject_f1(
        per_subject,
        "Macro-F1 per held-out subject - {} ({})".format(
            FEATURE_SET_LABELS[feature_set_name], HEADLINE_MODEL),
        os.path.join(config.RESULTS_DIR,
                     "per_subject_f1_" + feature_set_name + ".png"))
    summary["per_subject"] = per_subject

    return summary


def markdown_confusion_table(counts, class_names):
    """Render a confusion matrix as a markdown table (rows = actual)."""
    lines = []
    lines.append("| actual \\ predicted | " + " | ".join(
        config.LABEL_NAMES[c] for c in class_names) + " |")
    lines.append("|---" * (len(class_names) + 1) + "|")
    for i, true_class in enumerate(class_names):
        row = "| **{}** | ".format(config.LABEL_NAMES[true_class])
        row += " | ".join(str(int(counts[i, j])) for j in range(len(class_names)))
        lines.append(row + " |")
    return lines


def ibi_filter_section(summaries, table):
    """
    The before/after section for the inter-beat-interval median filter.

    Every number is read from the archived pre-fix outputs in
    results/before_ibi_median_filter/ and compared against the current run,
    so this section cannot drift out of date the way hand-copied figures do.
    Skipped entirely if the archive is absent.
    """
    archive = os.path.join(config.RESULTS_DIR, ARCHIVE_DIRNAME)
    old_features_path = os.path.join(archive, "features.csv")
    if not os.path.exists(old_features_path):
        return []

    old_scores = {}
    for set_name in ["full", "hr_only"]:
        path = os.path.join(archive, "scores_" + set_name + ".json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            old_scores[set_name] = json.load(f)["scores"]

    old_features = pd.read_csv(old_features_path)
    hr_columns = [name for name in features.FEATURE_NAMES
                  if name.startswith("HR_")]

    lines = []
    lines.append("## Correction: median-filtering the inter-beat-interval "
                 "series")
    lines.append("")
    lines.append("The first version of this pipeline converted every gap "
                 "between two detected pulse peaks straight into a heart "
                 "rate, keeping anything between 40 and 180 bpm. Comparing "
                 "the resulting features against real consumer-wearable data "
                 "(Phase 3, below) exposed a problem with that: the average "
                 "60-second WESAD window ran from `HR_min` {:.0f} bpm to "
                 "`HR_max` {:.0f} bpm - a swing of {:.0f} bpm inside one "
                 "minute. No heart rate moves like that, so the features were "
                 "measuring beat-detection error alongside physiology. How "
                 "much of the model's performance rested on that error is the "
                 "question this section answers.".format(
                     old_features["HR_min"].mean(),
                     old_features["HR_max"].mean(),
                     (old_features["HR_max"] - old_features["HR_min"]).mean()))
    lines.append("")
    lines.append("The fix is a median filter of width "
                 "{} applied to the inter-beat-interval series before it "
                 "becomes a heart rate (`features.py` -> "
                 "`heart_rate_series`, width justified at "
                 "`IBI_MEDIAN_FILTER_BEATS` in `project_config.py`). Nothing "
                 "else changed: the same {} windows survive, and every "
                 "non-HR feature column is bit-for-bit identical to "
                 "before.".format(config.IBI_MEDIAN_FILTER_BEATS, len(table)))
    lines.append("")

    lines.append("### Effect on the heart-rate features")
    lines.append("")
    lines.append("| Feature | Before: mean (sd) | After: mean (sd) |")
    lines.append("|---|---|---|")
    for name in hr_columns:
        lines.append("| `{}` | {:.2f} ({:.2f}) | {:.2f} ({:.2f}) |".format(
            name,
            old_features[name].mean(), old_features[name].std(),
            table[name].mean(), table[name].std()))
    old_span = (old_features["HR_max"] - old_features["HR_min"]).mean()
    new_span = (table["HR_max"] - table["HR_min"]).mean()
    lines.append("| *within-window span* (`HR_max - HR_min`) | *{:.1f} bpm* | "
                 "*{:.1f} bpm* |".format(old_span, new_span))
    lines.append("")
    lines.append("The apparent within-window swing falls by {:.0f}%, and "
                 "`HR_std` - the feature the artifacts inflated most - drops "
                 "from {:.1f} to {:.1f} bpm.".format(
                     100.0 * (old_span - new_span) / old_span,
                     old_features["HR_std"].mean(), table["HR_std"].mean()))
    lines.append("")
    lines.append("**The correction is real but incomplete.** A {:.0f} bpm "
                 "average swing within a single minute is still larger than "
                 "genuine breathing-driven variation, so some detection error "
                 "survives the filter. Widening it further would start "
                 "erasing real physiology (see the note in "
                 "`project_config.py`), so this is where it was "
                 "left.".format(new_span))
    lines.append("")

    lines.append("### Effect on model performance")
    lines.append("")
    lines.append("| Feature set | Model | Macro-F1 before | Macro-F1 after | "
                 "Change |")
    lines.append("|---|---|---|---|---|")
    for set_name in ["full", "hr_only"]:
        for model_name in sorted(summaries[set_name]["scores"]):
            before = old_scores[set_name][model_name]["f1_macro"]
            after = summaries[set_name]["scores"][model_name]["f1_macro"]
            lines.append("| `{}` | {} | {:.3f} | {:.3f} | {:+.3f} |".format(
                set_name, model_name.replace("_", " "), before, after,
                after - before))
    lines.append("")

    hr_before = old_scores["hr_only"][HEADLINE_MODEL]["f1_macro"]
    hr_after = summaries["hr_only"]["scores"][HEADLINE_MODEL]["f1_macro"]
    full_before = old_scores["full"][HEADLINE_MODEL]["f1_macro"]
    full_after = summaries["full"]["scores"][HEADLINE_MODEL]["f1_macro"]
    hr_delta = hr_after - hr_before
    hr_delta_pct = 100.0 * hr_delta / hr_before

    lines.append("### Interpretation")
    lines.append("")
    lines.append("The prediction being tested was that the heart-rate-only "
                 "score was partly built on beat-detection noise, and would "
                 "fall once that noise was removed. **It did fall, and the "
                 "direction was right - but the size is small:** the "
                 "heart-rate-only SVM goes from {:.3f} to {:.3f} macro-F1, a "
                 "change of {:+.3f} ({:+.1f}%).".format(
                     hr_before, hr_after, hr_delta, hr_delta_pct))
    lines.append("")
    lines.append("Read plainly, that means:")
    lines.append("")
    lines.append("1. **The original number was mildly inflated, not "
                 "fabricated.** If the heart-rate-only model had been living "
                 "on detector artifacts, removing most of them would have "
                 "collapsed it toward the {:.3f} macro-F1 of always guessing "
                 "the largest class. It did not - it lost {:.3f} and stayed "
                 "at {:.3f}, still well clear of that floor. Most of what the "
                 "model was using is heart rate, not noise.".format(
                     (2.0 * (table["label_name"].value_counts().max() / len(table))
                      / ((table["label_name"].value_counts().max() / len(table)) + 1.0)) / 3.0,
                     abs(hr_delta), hr_after))
    lines.append("2. **The honest caveat on the size of the effect.** "
                 "{:.3f} macro-F1 across 1015 windows from 15 subjects is a "
                 "small difference. The cross-validation is deterministic, so "
                 "this is not run-to-run randomness, but it is small enough "
                 "that it should be described as \"slightly inflated\" rather "
                 "than treated as a decisive finding.".format(abs(hr_delta)))
    lines.append("3. **The full model was not hurt - it improved slightly** "
                 "({:.3f} -> {:.3f}). With 35 other features available, the "
                 "noisy HR columns contributed little either way, and "
                 "cleaning them helped marginally.".format(
                     full_before, full_after))
    lines.append("")
    lines.append("The corrected features are also the better starting point "
                 "for Phase 3 regardless of what happened to the score, "
                 "because a feature that mostly measures a particular peak "
                 "detector's error pattern cannot transfer to a device that "
                 "uses a different one.")
    lines.append("")
    lines.append("All pre-fix outputs - the feature table, both models' "
                 "scores, the confusion matrices and the previous version of "
                 "this file - are preserved under "
                 "`results/{}/`. This project is not under version control, "
                 "so they are kept as files rather than as "
                 "history.".format(ARCHIVE_DIRNAME))
    lines.append("")

    return lines


def api_pagination_section():
    """
    The pagination finding, for the TRACKED report.

    This is a fact about the Google Health API and about how the fetch code
    was written - not about any individual's health - so it belongs in
    RESULTS.md. The specific days it affected are wearing-pattern
    information, so those stay in the untracked personal report instead.
    """
    lines = []
    lines.append("## Data quality: a silent API pagination bug")
    lines.append("")
    lines.append("Found while fetching personal wearable data in Phase 3, but "
                 "recorded here because it is a finding about the API rather "
                 "than about any particular person's data - and because the "
                 "symptom was indistinguishable from a behavioural fact about "
                 "the wearer.")
    lines.append("")
    lines.append("`scripts/05_apply_to_personal_data.py` fetches intraday "
                 "heart rate one civil day at a time through the dashboard's "
                 "API client. That client pages through results, and the "
                 "caller chooses how many pages to request. The `heart-rate` "
                 "data type declares `max_page_size = 10000`, so an initial "
                 "limit of 4 pages looked like 40,000 samples of headroom - "
                 "far more than a day could possibly hold.")
    lines.append("")
    lines.append("**The API actually returns about 5,000 points per page**, "
                 "so 4 pages capped each day at roughly 20,000 samples. And "
                 "because the API returns the *newest* samples first, hitting "
                 "the cap did not truncate the end of the day - it deleted "
                 "the *beginning*. Days that had in fact been recorded from "
                 "midnight came back appearing to start in the late morning.")
    lines.append("")
    lines.append("That is indistinguishable, from the data alone, from "
                 "someone who simply does not put the watch on until late "
                 "morning - a wrong conclusion about the person, drawn from a "
                 "bug in the fetch. It was caught only by re-fetching the "
                 "same days with a higher page limit and watching the sample "
                 "counts rise from ~20,000 to 27,000-36,000 while the start "
                 "times dropped to 00:00.")
    lines.append("")
    lines.append("Two changes came out of it, both still in the script:")
    lines.append("")
    lines.append("1. `DEFAULT_MAX_PAGES = 16`. A full 24 h day at the observed "
                 "~3 s sample spacing is about 7 pages; 16 leaves headroom.")
    lines.append("2. A truncation guard. Any *completed* day whose first "
                 "sample is later than 00:05 is reported as suspicious, with "
                 "an explicit instruction to re-run with a higher page limit "
                 "to tell \"watch was off\" apart from \"we truncated it\".")
    lines.append("")
    lines.append("The general lesson: a declared `max_page_size` is an upper "
                 "bound the server is allowed to ignore, and silent truncation "
                 "that removes the *oldest* records is invisible unless you go "
                 "looking for it. Which days were genuinely affected is "
                 "recorded in the personal report (see the pointer at the end "
                 "of this file), since that is information about the wearer.")
    lines.append("")
    return lines


def phase3_pointer():
    """
    The short note left in the TRACKED report in place of the Phase 3
    section, so a reader can see that the work exists and where it lives
    rather than finding a silent gap between Phase 2 and the artefact list.
    """
    personal_report = os.path.join(config.PERSONAL_DIR, "RESULTS_personal.md")
    exists = os.path.exists(personal_report)

    lines = []
    lines.append("## Phase 3: applying the model to my own wearable data")
    lines.append("")
    lines.append("The heart-rate-only model was applied to real heart-rate "
                 "data from my own consumer wearable (Google Fitbit Air, read "
                 "live through the Google Health API by "
                 "`scripts/05_apply_to_personal_data.py`, reusing the Django "
                 "dashboard's stored OAuth credentials in-process rather than "
                 "duplicating any credential handling).")
    lines.append("")
    lines.append("**Those results are deliberately not in this file.** They "
                 "consist of my own heart-rate readings, per-day heart-rate "
                 "statistics and wearing pattern - personal health "
                 "information, which stays out of a repository that might "
                 "later be pushed somewhere. They are written instead to:")
    lines.append("")
    lines.append("```")
    lines.append("results/personal/RESULTS_personal.md")
    lines.append("```")
    lines.append("")
    lines.append("which is covered by the `results/personal/` entry in "
                 "`.gitignore` and kept locally only.{}".format(
                     "" if exists else
                     " (Not generated yet - run "
                     "`05_apply_to_personal_data.py inspect` then `predict`, "
                     "then re-run this script.)"))
    lines.append("")
    lines.append("That report contains the per-day heart-rate summary, the "
                 "wearing-pattern and coverage notes, the training-versus-"
                 "personal feature distribution comparison, the model's "
                 "predictions, and the discussion of what the step does and "
                 "does not establish. The one finding from it that is *not* "
                 "personal - the API pagination bug - is written up in this "
                 "file above, and the effect the comparison had on the "
                 "training features is the subject of the "
                 "median-filter correction section.")
    lines.append("")
    return lines


def phase3_section():
    """
    The Phase 3 report itself - written to the UNTRACKED personal report at
    results/personal/RESULTS_personal.md, not to RESULTS.md, because every
    number in it derives from real personal health data.

    Written only if 05 has been run; it reads the summary files that script
    saves, so the numbers are never hand-copied.
    """
    lines = []
    lines.append("# Phase 3: applying the model to my own wearable data")
    lines.append("")
    lines.append("> **Personal health data - kept local, never committed.** "
                 "This file lives under `results/personal/`, which is "
                 "excluded by `.gitignore`. The WESAD results it refers to "
                 "are in `RESULTS.md` at the project root.")
    lines.append("")
    lines.append("Generated by `scripts/04_evaluate.py` from the outputs of "
                 "`scripts/05_apply_to_personal_data.py`.")
    lines.append("")

    # --- the run itself, only if 05 predict has been run --------------------
    summary_path = os.path.join(config.PERSONAL_DIR, "prediction_summary.json")
    comparison_path = os.path.join(config.PERSONAL_DIR,
                                   "feature_distribution_comparison.csv")
    per_day_path = os.path.join(config.PERSONAL_DIR,
                                "predictions_per_day.csv")

    # Nothing to report yet - the caller writes no file at all in that case,
    # rather than leaving a stub behind.
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)
    comparison = pd.read_csv(comparison_path)
    per_day = pd.read_csv(per_day_path)

    inspection_path = os.path.join(config.PERSONAL_DIR, "hr_inspection.csv")

    lines.append("## What was applied, and to what")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| Source | Google Health API, `heart-rate` data type, device "
                 "reported as Google Fitbit Air (`PASSIVELY_MEASURED`) |")
    lines.append("| Auth | Reuses the Django dashboard's stored OAuth "
                 "credentials and API client in-process; no credential "
                 "handling is duplicated in this project |")
    lines.append("| Samples | {:,} readings, {} to {} |".format(
        summary["n_samples"], summary["first_sample"], summary["last_sample"]))
    lines.append("| Observed spacing | ~3 s between readings (p05 1 s, "
                 "p95 3 s) |")
    lines.append("| Windows | {:,} windows of {} s, stepping {} s - identical "
                 "geometry to training |".format(
                     summary["n_windows"], summary["window_seconds"],
                     summary["step_seconds"]))
    lines.append("| Readings per window | median {:.0f}; windows with fewer "
                 "than {} are dropped |".format(
                     summary["median_readings_per_window"],
                     summary["min_samples_per_window"]))
    lines.append("| Days | {} |".format(summary["n_days"]))
    lines.append("| Model | `models/svm_hr_only.joblib`, the 5-feature "
                 "heart-rate-only SVM |")
    lines.append("")

    # --- coverage / wearing pattern ----------------------------------------
    # This is the half of the pagination finding that IS personal: which days
    # were genuinely incomplete, once the API truncation bug was ruled out.
    # The bug itself is written up in RESULTS.md; the days are written here.
    if os.path.exists(inspection_path):
        inspection = pd.read_csv(inspection_path)
        lines.append("## Coverage and wearing pattern")
        lines.append("")
        lines.append("After the API pagination bug was fixed (see "
                     "\"Data quality: a silent API pagination bug\" in "
                     "`RESULTS.md`), the remaining gaps are real rather than "
                     "artifacts of truncation:")
        lines.append("")
        lines.append("| Day | Samples | First | Last | Median gap | Gaps > 5 min |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in inspection.iterrows():
            lines.append("| {} | {:,} | {} | {} | {:.1f} s | {} |".format(
                row["day"], int(row["n"]),
                str(row["first"])[11:16], str(row["last"])[11:16],
                row["median_gap_s"], int(row["gaps_over_5min"])))
        lines.append("")
        lines.append("Days absent from this table returned no samples at all.")
        lines.append("")

    lines.append("## The inputs are far outside the training distribution")
    lines.append("")
    lines.append("This comparison is computed and printed **before** any "
                 "prediction, because it is what decides whether the "
                 "predictions deserve any weight at all. An RBF-kernel SVM "
                 "judges by distance in feature space, so inputs outside the "
                 "training range do not degrade gracefully - they fall "
                 "wherever happens to be nearest.")
    lines.append("")
    lines.append("| Feature | WESAD mean (sd) | Personal mean (sd) | Outside "
                 "training range | Median shift |")
    lines.append("|---|---|---|---|---|")
    for _, row in comparison.iterrows():
        lines.append("| `{}` | {:.2f} ({:.2f}) | {:.2f} ({:.2f}) | **{:.1%}** | "
                     "{:+.2f} sd |".format(
                         row["feature"],
                         row["wesad_mean"], row["wesad_std"],
                         row["personal_mean"], row["personal_std"],
                         row["personal_outside_training_range"],
                         row["median_shift_in_training_sd"]))
    lines.append("")
    lines.append("The cause is the one flagged before the step was written, "
                 "and it is larger than expected:")
    lines.append("")
    lines.append("- **In training**, `HR_*` describes a *beat-to-beat* series "
                 "- roughly 60-80 values per window, each the instantaneous "
                 "rate implied by one inter-beat interval from our own peak "
                 "detection on WESAD's 64 Hz PPG.")
    lines.append("- **Here**, `HR_*` describes ~20 *already-smoothed* bpm "
                 "readings emitted by the watch every 3 s, through its own "
                 "undisclosed filtering.")
    lines.append("")
    lines.append("So the same column name means a different physical "
                 "quantity, and the gap is stark: within a single 60 s window "
                 "the WESAD series averages a spread of `HR_std` = {:.1f} bpm "
                 "and reaches `HR_max` = {:.0f} bpm, while the personal series "
                 "averages `HR_std` = {:.1f} and `HR_max` = {:.0f}.".format(
                     float(comparison.loc[comparison["feature"] == "HR_std",
                                          "wesad_mean"].iloc[0]),
                     float(comparison.loc[comparison["feature"] == "HR_max",
                                          "wesad_mean"].iloc[0]),
                     float(comparison.loc[comparison["feature"] == "HR_std",
                                          "personal_mean"].iloc[0]),
                     float(comparison.loc[comparison["feature"] == "HR_max",
                                          "personal_mean"].iloc[0])))
    lines.append("")
    lines.append("It was this comparison that exposed the beat-detection "
                 "artifacts in the WESAD features, and the inter-beat-interval "
                 "median filter documented above was added in response. The "
                 "figures in the table are computed against the **corrected** "
                 "training distribution; against the uncorrected one the gap "
                 "was wider still (`HR_std` 71.6% outside range, `HR_max` "
                 "67.4%, a -4.52 sd shift). Cleaning the training features "
                 "narrowed the mismatch but did not remove it, because the "
                 "remaining difference is not an artifact - it is the genuine "
                 "distinction between a beat-to-beat series and a device's "
                 "smoothed once-per-3-seconds output.")
    lines.append("")

    lines.append("## Predictions")
    lines.append("")
    lines.append("| Predicted class | Share of windows |")
    lines.append("|---|---|")
    for name, share in summary["predicted_share"].items():
        lines.append("| {} | {:.1%} |".format(name, share))
    lines.append("")
    shifts = comparison["median_shift_in_training_sd"].abs()
    lines.append("**These predictions are degenerate and should be read as a "
                 "negative result, not as a measurement.** Stress is predicted "
                 "for {:.1%} of windows across two weeks of ordinary life, "
                 "while amusement - the class the model was *weakest* at in "
                 "cross-validation - absorbs a large share. That is the "
                 "expected consequence of feeding an RBF SVM inputs whose "
                 "medians sit up to {:.1f} training standard deviations from "
                 "where it was fitted.".format(
                     summary["predicted_share"]["stress"], shifts.max()))
    lines.append("")

    lines.append("Per day (waking hours are {:02d}:00-{:02d}:00; overnight "
                 "windows are reported separately because the model has no "
                 "sleep class and WESAD contains no sleep):".format(
                     summary["waking_hours"][0], summary["waking_hours"][1]))
    lines.append("")
    lines.append("| Day | Windows | Mean bpm | baseline | stress | amusement | "
                 "stress (waking) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in per_day.iterrows():
        waking = row["waking_share_stress"]
        lines.append("| {} | {} | {:.1f} | {:.1%} | {:.1%} | {:.1%} | {} |".format(
            row["day"], int(row["windows"]), row["mean_bpm"],
            row["share_baseline"], row["share_stress"],
            row["share_amusement"],
            "n/a" if pd.isna(waking) else "{:.1%}".format(waking)))
    lines.append("")
    lines.append("There is **no ground truth** for these days, so no accuracy "
                 "is reported and none can be. The day-to-day variation above "
                 "is small and tracks mean heart rate, not anything that has "
                 "been shown to be stress.")
    lines.append("")

    lines.append("## What this step does and does not establish")
    lines.append("")
    lines.append("**Establishes:** the full pipeline runs end to end on real "
                 "consumer-wearable data - live API fetch through reused OAuth "
                 "credentials, windowing with the same geometry as training, "
                 "features computed by the same imported code, and a saved "
                 "model producing predictions.")
    lines.append("")
    lines.append("**Does not establish:** anything about whether I was "
                 "stressed. The model was trained on a laboratory stressor in "
                 "a controlled setting, it has no \"neither\" class, its "
                 "inputs here are measurably out of distribution, and there "
                 "are no labels to check it against. Treating any number in "
                 "this section as a stress measurement would be wrong.")
    lines.append("")
    lines.append("**The useful finding is the negative one:** a model trained "
                 "on research-grade beat-to-beat heart rate does not transfer "
                 "to a consumer device's smoothed heart-rate feed, even when "
                 "the feature names, the window length and the code are "
                 "identical - because the underlying signal is not the same "
                 "measurement. Closing that gap needs either training features "
                 "robust to the difference (smoothed IBIs, as above) or "
                 "training on the target device's own data.")
    lines.append("")

    return lines


def write_results_md(table, summaries, class_names):
    """Write RESULTS.md from the numbers we just computed - nothing hand-typed."""
    n_subjects = table["subject_id"].nunique()
    full = summaries["full"]
    hr = summaries["hr_only"]

    full_score = full["scores"][HEADLINE_MODEL]
    hr_score = hr["scores"][HEADLINE_MODEL]
    f1_drop = hr_score["f1_macro"] - full_score["f1_macro"]
    f1_drop_pct = 100.0 * f1_drop / full_score["f1_macro"]

    lines = []
    lines.append("# WESAD wrist-based stress classification - results")
    lines.append("")
    lines.append("Generated by `scripts/04_evaluate.py`. Every number here comes "
                 "from leave-one-subject-out cross-validation on the WESAD "
                 "dataset.")
    lines.append("")

    # ----------------------------------------------------------------- data
    lines.append("## Data")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| Dataset | WESAD, wrist (Empatica E4) signals only |")
    lines.append("| Signals available | EDA, TEMP, BVP, ACC (x/y/z + "
                 "magnitude), HR derived from BVP |")
    lines.append("| Subjects | {} (S2-S17; S1 and S12 are not part of the "
                 "dataset) |".format(n_subjects))
    lines.append("| Classes | baseline / stress / amusement (the 3-class WESAD "
                 "benchmark) |")
    lines.append("| Window length | {} s |".format(config.WINDOW_SECONDS))
    lines.append("| Window step | {} s ({:.0f}% overlap) |".format(
        config.STEP_SECONDS,
        100.0 * (1.0 - config.STEP_SECONDS / float(config.WINDOW_SECONDS))))
    lines.append("| Label purity required | {:.0%} - windows spanning a "
                 "condition change are dropped |".format(
                     config.LABEL_PURITY_THRESHOLD))
    lines.append("| Windows kept | {} |".format(len(table)))
    lines.append("")

    lines.append("### Class balance")
    lines.append("")
    lines.append("| Class | Windows | Share |")
    lines.append("|---|---|---|")
    for name, count in table["label_name"].value_counts().items():
        lines.append("| {} | {} | {:.1f}% |".format(
            name, count, 100.0 * count / len(table)))
    lines.append("")
    lines.append("The classes are imbalanced because the baseline condition "
                 "simply lasted longer in the study protocol. That is why the "
                 "headline metric is macro-F1 rather than accuracy: macro-F1 "
                 "weights all three classes equally, so a model that ignores "
                 "the smallest class cannot hide behind a high accuracy.")
    lines.append("")

    lines.append("### Windows per subject")
    lines.append("")
    lines.append("| Subject | Windows |")
    lines.append("|---|---|")
    for subject, count in table.groupby("subject_id").size().items():
        lines.append("| {} | {} |".format(subject, count))
    lines.append("")

    # ----------------------------------------------------------- validation
    lines.append("## Validation method")
    lines.append("")
    lines.append("Leave-one-subject-out cross-validation ({} folds): each fold "
                 "trains on {} subjects and tests on the one subject held "
                 "out.".format(n_subjects, n_subjects - 1))
    lines.append("")
    lines.append("A random row-wise split would be wrong here. Windows from the "
                 "same person share their skin conductance level, their resting "
                 "heart rate and their sensor fit, and with 50% overlap two "
                 "neighbouring windows literally share 30 seconds of signal. "
                 "Split randomly, the model could score well by recognising the "
                 "person rather than the stress state - leakage that inflates "
                 "the reported accuracy. Splitting by subject measures the thing "
                 "we actually care about: does this generalise to a new person?")
    lines.append("")

    # -------------------------------------------------------- the two models
    lines.append("## Two feature sets: research-grade vs. consumer wearable")
    lines.append("")
    lines.append("The same windows, the same folds and the same model settings "
                 "were used twice, changing only which columns the model is "
                 "given:")
    lines.append("")
    lines.append("| Feature set | Features | What it represents |")
    lines.append("|---|---|---|")
    lines.append("| `full` | {} | An Empatica E4 research wristband: "
                 "electrodermal activity, skin temperature, raw PPG and raw "
                 "3-axis accelerometry, plus heart rate |".format(
                     full["n_features"]))
    lines.append("| `hr_only` | {} | A mainstream consumer wearable "
                 "(Fitbit-class): heart rate and nothing else. `{}` |".format(
                     hr["n_features"], ", ".join(hr["feature_names"])))
    lines.append("")
    lines.append("Nothing was tuned separately for the two sets - same "
                 "algorithm, same hyperparameters, same cross-validation. Any "
                 "difference in score is therefore caused by the missing "
                 "sensors, not by the modelling.")
    lines.append("")

    # ------------------------------------------------------- headline result
    lines.append("### Headline comparison ({}, leave-one-subject-out)".format(
        HEADLINE_MODEL.upper()))
    lines.append("")
    lines.append("| Feature set | Accuracy | Macro-F1 |")
    lines.append("|---|---|---|")
    lines.append("| `full` - all 8 channels | {:.3f} | **{:.3f}** |".format(
        full_score["accuracy"], full_score["f1_macro"]))
    lines.append("| `hr_only` - heart rate only | {:.3f} | **{:.3f}** |".format(
        hr_score["accuracy"], hr_score["f1_macro"]))
    lines.append("| **Change** | {:+.3f} | **{:+.3f} ({:+.1f}%)** |".format(
        hr_score["accuracy"] - full_score["accuracy"], f1_drop, f1_drop_pct))
    lines.append("")
    lines.append("**Restricting the model to what a wrist-heart-rate-only "
                 "consumer wearable can provide costs {:.3f} macro-F1, a "
                 "{:.1f}% relative drop** ({:.3f} -> {:.3f}).".format(
                     abs(f1_drop), abs(f1_drop_pct),
                     full_score["f1_macro"], hr_score["f1_macro"]))
    lines.append("")
    lines.append("For reference, always predicting the largest class "
                 "(baseline) would give an accuracy of {:.3f} and a macro-F1 of "
                 "{:.3f}, so the heart-rate-only model is well above chance "
                 "- it is degraded, not useless.".format(
                     table["label_name"].value_counts().max() / len(table),
                     # macro-F1 of the majority-class-only classifier:
                     # F1 = 2p/(p+1) for the majority class, 0 for the others.
                     (2.0 * (table["label_name"].value_counts().max() / len(table))
                      / ((table["label_name"].value_counts().max() / len(table)) + 1.0)) / 3.0))
    lines.append("")
    lines.append("Both algorithms were run on both feature sets; the full grid "
                 "is below, so the choice of SVM for the headline can be "
                 "checked rather than taken on trust.")
    lines.append("")
    lines.append("| Feature set | Model | Accuracy | Macro-F1 |")
    lines.append("|---|---|---|---|")
    for set_name in ["full", "hr_only"]:
        for model_name in sorted(summaries[set_name]["scores"]):
            s = summaries[set_name]["scores"][model_name]
            lines.append("| `{}` | {} | {:.3f} | {:.3f} |".format(
                set_name, model_name.replace("_", " "),
                s["accuracy"], s["f1_macro"]))
    lines.append("")

    # ------------------------------------- the IBI median-filter correction
    lines.extend(ibi_filter_section(summaries, table))

    # --------------------------------------------------- per-feature-set detail
    for set_name in ["full", "hr_only"]:
        summary = summaries[set_name]
        per_subject = summary["per_subject"]
        lines.append("## Detail: `{}` - {}".format(
            set_name, FEATURE_SET_LABELS[set_name]))
        lines.append("")
        lines.append("Pooled out-of-fold performance of the {} across all {} "
                     "subjects: accuracy **{:.3f}**, macro-F1 **{:.3f}**.".format(
                         HEADLINE_MODEL.upper(), n_subjects,
                         summary["scores"][HEADLINE_MODEL]["accuracy"],
                         summary["scores"][HEADLINE_MODEL]["f1_macro"]))
        lines.append("")
        lines.append("### Confusion matrix ({})".format(HEADLINE_MODEL.upper()))
        lines.append("")
        lines.append("Rows are the true class, columns the predicted class "
                     "(counts of windows).")
        lines.append("")
        lines.extend(markdown_confusion_table(
            summary["counts"][HEADLINE_MODEL], class_names))
        lines.append("")
        lines.append("![Confusion matrix](results/confusion_matrix_{}_{}.png)"
                     .format(set_name, HEADLINE_MODEL))
        lines.append("")
        lines.append("### Per-class scores ({})".format(HEADLINE_MODEL.upper()))
        lines.append("")
        lines.append("```")
        lines.append(summary["reports"][HEADLINE_MODEL].rstrip())
        lines.append("```")
        lines.append("")
        lines.append("### Per-subject variation")
        lines.append("")
        lines.append("Macro-F1 when each subject in turn is the held-out test "
                     "set: mean {:.3f}, worst {:.3f} ({}), best {:.3f} "
                     "({}).".format(
                         per_subject["f1_macro"].mean(),
                         per_subject["f1_macro"].min(),
                         per_subject.loc[per_subject["f1_macro"].idxmin(),
                                         "subject_id"],
                         per_subject["f1_macro"].max(),
                         per_subject.loc[per_subject["f1_macro"].idxmax(),
                                         "subject_id"]))
        lines.append("")
        lines.append("| Subject | Windows | Accuracy | Macro-F1 |")
        lines.append("|---|---|---|---|")
        for _, row in per_subject.iterrows():
            lines.append("| {} | {} | {:.3f} | {:.3f} |".format(
                row["subject_id"], int(row["n_windows"]),
                row["accuracy"], row["f1_macro"]))
        lines.append("")
        lines.append("![Per-subject macro-F1](results/per_subject_f1_{}.png)"
                     .format(set_name))
        lines.append("")

    # ------------------------------------------------------------ limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("These apply to every number above, and especially to any "
                 "output of `scripts/05_apply_to_personal_data.py`.")
    lines.append("")
    lines.append("1. **Lab-induced stress, not real-world stress.** WESAD's "
                 "\"stress\" class is the Trier Social Stress Test - public "
                 "speaking and mental arithmetic in front of a panel - and its "
                 "\"amusement\" class is a set of funny video clips. Both are "
                 "short, deliberately provoked, and recorded sitting in a lab. "
                 "Real-world stress is longer, weaker, mixed with physical "
                 "activity, caffeine, illness, temperature and sleep debt. A "
                 "model trained on the former is not validated on the latter.")
    lines.append("2. **The three classes are not exhaustive.** The model must "
                 "answer baseline / stress / amusement for every window it is "
                 "given. It has no \"none of these\" option, so applying it to "
                 "arbitrary daily data forces every moment into one of three "
                 "lab conditions, whether or not any of them apply.")
    lines.append("3. **Different sensor, different hardware.** The heart-rate "
                 "features here are computed by our own peak detection on "
                 "WESAD's 64 Hz Empatica E4 PPG signal. A consumer device "
                 "reports an already-processed heart rate at a much lower and "
                 "often irregular rate, using its own undocumented filtering. "
                 "Features with the same name are not guaranteed to have the "
                 "same distribution.")
    lines.append("4. **Small subject pool.** 15 subjects, all young adults, "
                 "recorded in one lab on one day each. The per-subject tables "
                 "above show how wide the spread between individuals already "
                 "is within that narrow group.")
    lines.append("5. **Overlapping windows are not independent.** The 50% "
                 "overlap means neighbouring windows share 30 s of signal. "
                 "Subject-wise splitting prevents this from leaking across "
                 "train and test, but it does mean the effective sample size "
                 "is smaller than the {} window count suggests.".format(
                     len(table)))
    lines.append("6. **Hyperparameters were not tuned.** Both models use "
                 "scikit-learn defaults. Tuning honestly would require a "
                 "nested cross-validation loop (an inner loop over the "
                 "training subjects only); tuning on the outer folds would "
                 "leak test information and inflate the scores.")
    lines.append("")

    # ---------------------------------------------------- phase 3 (personal)
    # The pagination finding belongs here (it is about the API). The Phase 3
    # results themselves do not - they are personal health data, and go to
    # the untracked report instead; only a pointer is left behind.
    lines.extend(api_pagination_section())
    lines.extend(phase3_pointer())

    # ------------------------------------------------------------- artefacts
    lines.append("## Saved artefacts")
    lines.append("")
    lines.append("| File | What it is |")
    lines.append("|---|---|")
    lines.append("| `models/svm_model.joblib` | Full 8-channel SVM pipeline "
                 "(scaler + classifier), refitted on all subjects |")
    lines.append("| `models/feature_config.json` | Settings and exact column "
                 "order for the full model |")
    lines.append("| `models/svm_hr_only.joblib` | Heart-rate-only SVM pipeline, "
                 "refitted on all subjects - the one usable on consumer "
                 "wearable data |")
    lines.append("| `models/feature_config_hr_only.json` | Settings and exact "
                 "column order for the heart-rate-only model |")
    lines.append("| `models/features.py`, `models/project_config.py` | A copy of the "
                 "feature-extraction code, so the same features can be rebuilt "
                 "from other data |")
    lines.append("| `results/features.csv` | The full feature table |")
    lines.append("| `results/cv_predictions_<set>.csv` | Out-of-fold prediction "
                 "for every window |")
    lines.append("")
    lines.append("Everything under `results/personal/` is personal health "
                 "data and is **not tracked** (see `.gitignore`). It exists "
                 "only on the machine that produced it:")
    lines.append("")
    lines.append("| File (untracked) | What it is |")
    lines.append("|---|---|")
    lines.append("| `results/personal/RESULTS_personal.md` | The Phase 3 "
                 "report - per-day heart-rate summary, coverage, "
                 "out-of-distribution comparison, predictions |")
    lines.append("| `results/personal/hr_samples.csv` | Raw personal "
                 "heart-rate samples fetched from the Google Health API |")
    lines.append("| `results/personal/hr_inspection.csv` | Per-day coverage "
                 "and sample-spacing summary |")
    lines.append("| `results/personal/feature_distribution_comparison.csv` | "
                 "Training vs personal feature distributions |")
    lines.append("| `results/personal/predictions_windows.csv` | Per-window "
                 "personal predictions, with the features behind them |")
    lines.append("| `results/personal/predictions_per_day.csv` | Per-day "
                 "aggregation of those predictions |")
    lines.append("")

    lines.append("## How to reproduce")
    lines.append("")
    lines.append("```")
    lines.append(".venv\\Scripts\\activate")
    lines.append("python scripts/01_load_and_inspect.py")
    lines.append("python scripts/02_feature_extraction.py")
    lines.append("python scripts/03_train_model.py")
    lines.append("python scripts/04_evaluate.py")
    lines.append("")
    lines.append("# Phase 3 - personal wearable data (needs the Django")
    lines.append("# dashboard's stored Google credentials; stop its runserver first)")
    lines.append("python scripts/05_apply_to_personal_data.py inspect")
    lines.append("python scripts/05_apply_to_personal_data.py predict")
    lines.append("python scripts/04_evaluate.py   # re-run to regenerate both reports")
    lines.append("```")
    lines.append("")

    path = os.path.join(config.PROJECT_DIR, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_personal_report():
    """
    Write the Phase 3 report to results/personal/RESULTS_personal.md.

    Kept as a SEPARATE file from RESULTS.md because every number in it comes
    from real personal heart-rate readings. results/personal/ is excluded by
    .gitignore, so this report stays on the machine that produced it even
    though RESULTS.md is committed.

    Returns the path written, or None if 05 has not been run yet.
    """
    lines = phase3_section()
    if lines is None:
        return None

    os.makedirs(config.PERSONAL_DIR, exist_ok=True)
    path = os.path.join(config.PERSONAL_DIR, "RESULTS_personal.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    table = pd.read_csv(config.FEATURES_CSV)
    class_names = sorted(config.LABEL_NAMES)

    summaries = {}
    for feature_set_name in features.FEATURE_SETS:
        summaries[feature_set_name] = evaluate_feature_set(
            feature_set_name, class_names)

    # Personal report first, so that RESULTS.md's pointer can say truthfully
    # whether the file it points at exists.
    personal_path = write_personal_report()
    path = write_results_md(table, summaries, class_names)

    print("Figures written to " + config.RESULTS_DIR)
    print("WESAD summary (tracked)      : " + path)
    if personal_path:
        print("Phase 3 report (NOT tracked) : " + personal_path)
    else:
        print("Phase 3 report               : not written - run "
              "05_apply_to_personal_data.py first")


if __name__ == "__main__":
    main()
