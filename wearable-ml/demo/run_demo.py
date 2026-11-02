"""
run_demo.py - end-to-end walkthrough for the mid-project demo.

    python demo/run_demo.py
    python demo/run_demo.py --day 2026-10-29 --mask-date
    python demo/run_demo.py --save          # also writes demo/out/ (date masked)
    python demo/run_demo.py --with-rf --takeout "<path to the Heart Rate folder>"

What it does, live (nothing is hard-coded or read from a results file, except
the saved scores it compares against):

  1. Header: environment versions, file fingerprints, anything not reproducible.
  2. Class balance per subject from the cached WESAD feature table.
  3. Leave-one-subject-out evaluation, for the binary task (primary) and the
     3-class task (secondary), with a majority baseline, a one-threshold rule
     baseline and the repo's default SVM.
  4. One local day of Fitbit heart rate, scored with the WESAD-trained models,
     with day-level statistics and an out-of-distribution check.

What it will NOT do: write any file (except the --save copy under demo/out/), call script 05's predict step, load the
17 GB WESAD pickles, tune anything, or open any Takeout file other than
heart_rate_YYYY-MM-DD.csv.

Model trained on WESAD only; personal data is only scored.
"""

import argparse
import contextlib
import glob
import datetime
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True  # a demo must not drop files anywhere

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEMO_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
for _path in (SCRIPTS_DIR, DEMO_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import demo_personal as personal  # noqa: E402
import demo_wesad as wesad  # noqa: E402
from demo_guard import FitGuard  # noqa: E402
import features  # noqa: E402
import project_config as config  # noqa: E402

BANNER = "Model trained on WESAD only; personal data is only scored."
def amendment_text(masked):
    """Amendment 1, with the first demo run's date hidden when output is masked."""
    first_run = ("the first demo run's output (day hidden)" if masked
                 else "the 2026-10-30 demo output")
    return ("Amendment 1 (PRE_DECLARED.md, 2026-10-31) was made AFTER seeing "
            + first_run + ": the 60% class-dominance flag was mis-specified for "
            "the binary task (non-stress is 70% of the training prior), so for "
            "binary it is dropped and only the out-of-distribution flag applies. "
            "The 3-class dominance rule is unchanged. Nothing was tuned to that "
            "output.")


COMPARISON_NOTE = (
    "SVM vs rule comparison: both are single-run point estimates with NO "
    "confidence interval (bootstrap intervals over subjects are in eval/). This demo does "
    "not claim either one is better than the other.")
OUT_DIR = os.path.join(DEMO_DIR, "out")
EXPECTED_SKLEARN = "1.5.2"  # the version that wrote the saved model
DEFAULT_TZ = "America/Toronto"
PINNED = ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "joblib"]

problems = []  # everything that could not be reproduced, listed at the end


def say(text=""):
    print(text)


def rule(title):
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def not_reproducible(reason):
    problems.append(reason)
    say("  NOT REPRODUCIBLE: " + reason)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. header
# ---------------------------------------------------------------------------

def read_pins():
    pins = {}
    path = os.path.join(PROJECT_DIR, "requirements.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
            if match:
                pins[match.group(1).lower()] = match.group(2)
    return pins


def show_header():
    rule("1. ENVIRONMENT AND FILES")
    say(BANNER)
    say("Python {}".format(sys.version.split()[0]))
    pins = read_pins()
    for name in PINNED:
        installed = metadata.version(name)
        pinned = pins.get(name)
        verdict = ("ok" if installed == pinned
                   else "differs from pinned {}".format(pinned))
        say("  {:14s} {:10s} {}".format(name, installed, verdict))
        if installed != pinned:
            not_reproducible("{} {} != pinned {}".format(name, installed, pinned))

    say("File fingerprints (first 12 hex of sha256):")
    for rel in ("results/features.csv", "models/svm_hr_only.joblib",
                "models/feature_config_hr_only.json",
                "models/hr_training_distribution.json"):
        path = os.path.join(PROJECT_DIR, rel)
        say("  {:40s} {}".format(rel, sha(path) if os.path.exists(path) else "MISSING"))
        if not os.path.exists(path):
            not_reproducible(rel + " is missing")


# ---------------------------------------------------------------------------
# 2. class balance
# ---------------------------------------------------------------------------

def show_class_balance(table):
    rule("2. CLASS BALANCE PER SUBJECT (WESAD, wrist, 60 s windows, 30 s step)")
    counts = pd.crosstab(table["subject_id"], table["label_name"])[
        ["baseline", "stress", "amusement"]]
    counts["total"] = counts.sum(axis=1)
    counts["stress share"] = (counts["stress"] / counts["total"]).map("{:.2f}".format)
    say(counts.to_string())
    total = counts[["baseline", "stress", "amusement", "total"]].sum()
    say("ALL    baseline {}  stress {}  amusement {}  total {}  ({} subjects)".format(
        total["baseline"], total["stress"], total["amusement"], total["total"],
        len(counts)))
    say("Binary task merges baseline + amusement into 'non-stress': stress is "
        "{:.1%} of windows.".format(total["stress"] / total["total"]))


# ---------------------------------------------------------------------------
# 3. leave-one-subject-out
# ---------------------------------------------------------------------------

def load_saved_scores(fs_name):
    path = os.path.join(config.RESULTS_DIR, "scores_{}.json".format(fs_name))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["scores"]


def show_loso(table, with_rf):
    rule("3. LEAVE-ONE-SUBJECT-OUT (15 folds, pooled out-of-fold predictions)")
    say("Default hyperparameters, one run, NO tuning (nested tuning and 5 seeds "
        "are in eval/, PRE_DECLARED.md Amendment 4).")
    say("The binary and 3-class tables are different problems; their macro-F1 "
        "values are not directly comparable.")
    say(COMPARISON_NOTE)

    feature_sets = {"hr_only": features.FEATURE_SETS["hr_only"],
                    "full": features.FEATURE_SETS["full"]}
    results = {}

    for task in wesad.TASKS:
        y = wesad.task_labels(table, task["key"])
        ref_mean, ref_sd = wesad.shuffled_reference(y)
        say()
        say("-" * 78)
        say(task["title"])
        say("-" * 78)
        say("Shuffled-label reference (200 permutations): macro-F1 {:.3f} "
            "(SD {:.3f}). Majority-class balanced accuracy is 1/{}.".format(
                ref_mean, ref_sd, len(task["labels"])))

        t0 = time.perf_counter()
        results[task["key"]] = wesad.evaluate_task(
            table, task, feature_sets, with_rf=with_rf)
        say("(computed in {:.1f} s)".format(time.perf_counter() - t0))

        for fs_name, models in results[task["key"]].items():
            say()
            say("Feature set: {} ({} features)".format(
                fs_name, len(feature_sets[fs_name])))
            say("  {:26s} {:>9s} {:>9s} {:>8s}   {:>20s}".format(
                "model", "macro-F1", "bal.acc", "acc", "per-subject macro-F1"))
            saved = load_saved_scores(fs_name) if task["key"] == "three" else None
            for name, s in models.items():
                check = ""
                if task["key"] == "three" and name in wesad.SAVED_KEYS:
                    if saved is None:
                        check = "  (no saved reference)"
                    else:
                        ref = saved[wesad.SAVED_KEYS[name]]
                        same = (abs(ref["f1_macro"] - s["macro_f1"]) < 1e-9
                                and abs(ref["accuracy"] - s["accuracy"]) < 1e-9)
                        check = "  matches saved scores_{}.json".format(fs_name) if same \
                            else "  MISMATCH vs saved (macro-F1 {:.4f})".format(ref["f1_macro"])
                        if not same:
                            not_reproducible("{} {} differs from saved".format(fs_name, name))
                say("  {:26s} {:9.3f} {:9.3f} {:8.3f}   {:>9.3f} (SD {:.3f}){}".format(
                    name, s["macro_f1"], s["balanced_accuracy"], s["accuracy"],
                    s["per_subject_mean"], s["per_subject_sd"], check))

            say("  Per-class precision / recall:")
            for name, s in models.items():
                parts = ["{} P{:.2f} R{:.2f}".format(
                    task["names"][label], s["precision"][label], s["recall"][label])
                    for label in task["labels"]]
                say("    {:26s} {}".format(name, " | ".join(parts)))

            svm = models["SVM (RBF, default)"]
            say("  Confusion matrix, SVM (rows = actual, columns = predicted):")
            matrix = svm["confusion"].copy()
            matrix.index = [task["names"][i] for i in matrix.index]
            matrix.columns = [task["names"][i] for i in matrix.columns]
            for line in matrix.to_string().splitlines():
                say("    " + line)

            if fs_name == "hr_only" and task["key"] == "three":
                value = svm["macro_f1"]
                if value > ref_mean + 3 * ref_sd:
                    say("  Reading: hr_only 3-class SVM macro-F1 {:.3f} is weak but "
                        "above chance (shuffled labels {:.3f}).".format(value, ref_mean))
                else:
                    say("  Reading: hr_only 3-class SVM macro-F1 {:.3f} is NOT "
                        "distinguishable from shuffled labels ({:.3f}).".format(
                            value, ref_mean))
                full_value = results["three"]["full"]["SVM (RBF, default)"]["macro_f1"] \
                    if "full" in results["three"] else None
                if full_value is not None:
                    say("  Restricting to heart rate changes macro-F1 by {:+.3f} "
                        "vs the full 8-channel set ({:.3f}).".format(
                            value - full_value, full_value))
    return results


# ---------------------------------------------------------------------------
# 4. one Fitbit day
# ---------------------------------------------------------------------------

def find_takeout_folder(explicit):
    if explicit:
        return explicit
    pattern = os.path.join(
        os.path.dirname(PROJECT_DIR), "health dashboard", "data-samples", "*",
        "Takeout", "Google Health", "Physical Activity_GoogleData", "Heart Rate")
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def show_personal(table, args):
    rule("4. SCORING ONE DAY OF FITBIT DATA (scored only - never trained on)")
    say(BANNER)
    say("There is no ground truth for any personal day, so NO accuracy or F1 is "
        "claimed here.")
    say(amendment_text(args.mask_date))

    folder = find_takeout_folder(args.takeout)
    if not folder:
        return not_reproducible(
            "Takeout heart-rate folder not found (use --takeout <folder>).")

    try:
        if args.day:
            day = personal.date.fromisoformat(args.day)
            samples, opened = personal.load_local_day(folder, day, args.tz)
        else:
            day, samples, opened = personal.choose_day(folder, args.tz)
    except (personal.TakeoutError, ValueError) as exc:
        return not_reproducible(str(exc))

    windows = personal.build_windows(samples, day, args.tz)
    if windows.empty:
        return not_reproducible("the chosen day has no scorable windows.")

    # --- load the models (fitted on WESAD only), then guard against any fit ---
    import sklearn
    with open(os.path.join(config.MODELS_DIR, "feature_config_hr_only.json"),
              encoding="utf-8") as f:
        feature_names = json.load(f)["feature_names_in_order"]
    with open(os.path.join(config.MODELS_DIR, "hr_training_distribution.json"),
              encoding="utf-8") as f:
        distribution = json.load(f)

    saved_model = None
    if sklearn.__version__ == EXPECTED_SKLEARN:
        saved_model = joblib.load(os.path.join(config.MODELS_DIR, "svm_hr_only.joblib"))
    else:
        not_reproducible("scikit-learn {} cannot be trusted to load the saved model "
                         "(written by {}); skipping it.".format(
                             sklearn.__version__, EXPECTED_SKLEARN))

    binary_model = wesad.fit_binary_hr_model(table, feature_names)  # WESAD only

    label = "Day #1 (date hidden)" if args.mask_date else "Day {}".format(day)
    spacing = personal.reading_spacing(samples)
    waking = windows["waking"].values

    say()
    say("{} (local time, {}). Files opened: {} (heart_rate_YYYY-MM-DD.csv only).".format(
        label, args.tz, opened))
    say("Device(s) named in the file's own 'data source' column: {}".format(
        ", ".join("{} ({:,} readings)".format(k, v)
                  for k, v in personal.device_names(samples).items())))
    say("Readings: {:,}. Spacing between readings: median {:.1f} s, p05 {:.1f} s, "
        "p95 {:.1f} s; gaps over 5 min: {}.".format(
            len(samples), spacing["median"], spacing["p05"], spacing["p95"],
            spacing["gaps_over_5min"]))
    say("Windows (60 s, step 30 s, >= {} readings): {:,} scorable, {:,} in waking "
        "hours (08-23). Median readings per window: {:.0f}.".format(
            personal.MIN_SAMPLES_PER_WINDOW, len(windows), int(waking.sum()),
            float(np.median(windows["n_samples"]))))
    say("Training (WESAD) heart rate is beat-to-beat, about 60-80 values per window. "
        "This device gives one smoothed value every ~3 s.")

    mean_hr = distribution["features"]["HR_mean"]["mean"]
    per_window = float(np.median(windows["n_samples"]))
    say()
    say("Resolution mismatch (heart-rate values per 60 s window):")
    say("  WESAD training : beat-to-beat, one value per heartbeat, about {:.2f} s "
        "apart, about {:.0f} values per window".format(60.0 / mean_hr, mean_hr))
    say("                   (implied by the training mean heart rate of {:.1f} bpm; "
        "not measured from the raw signal here)".format(mean_hr))
    say("  This Fitbit day: device-smoothed, one value every {:.1f} s (median; p05 "
        "{:.1f} s, p95 {:.1f} s), {:.0f} values per window".format(
            spacing["median"], spacing["p05"], spacing["p95"], per_window))
    say("  => about {:.1f}x fewer values per window, each already smoothed by the "
        "device.".format(mean_hr / per_window))

    ood = personal.ood_report(windows, distribution, feature_names)
    say()
    say("Out-of-distribution check (WESAD training range):")
    say("  {:10s} {:>20s} {:>24s}".format(
        "feature", "windows outside range", "median shift (train SD)"))
    for row in ood:
        say("  {:10s} {:>19.1%} {:>24.2f}".format(
            row["feature"], row["share_outside"], row["median_shift_sd"]))
    worst = max(row["share_outside"] for row in ood)

    # --- predictions: nothing below may learn anything ----------------------
    with FitGuard():
        say()
        say("[Model: BINARY SVM, trained in memory on WESAD hr_only features "
            "(stress vs non-stress), default hyperparameters C=1.0, gamma=scale, "
            "class_weight=balanced; not saved]")
        binary_names = wesad.TASKS[0]["names"]
        binary_pred = personal.predict_only(binary_model, windows, feature_names)
        shares_b = personal.class_shares(binary_pred, binary_names)
        shares_b_w = personal.class_shares(binary_pred, binary_names, waking)
        say("  all windows : " + ", ".join("{} {:.1%}".format(k, v)
                                           for k, v in shares_b.items()))
        say("  waking hours: " + ", ".join("{} {:.1%}".format(k, v)
                                           for k, v in shares_b_w.items()))
        for note in personal.notes_for(shares_b, worst, dominance=False):
            say("  NOTE: " + note)

        if saved_model is not None:
            say()
            say("[Model: SAVED 3-CLASS SVM, models/svm_hr_only.joblib, trained on "
                "WESAD only (baseline / stress / amusement), default hyperparameters]")
            names3 = wesad.TASKS[1]["names"]
            pred3 = personal.predict_only(saved_model, windows, feature_names)
            shares3 = personal.class_shares(pred3, names3)
            shares3_w = personal.class_shares(pred3, names3, waking)
            say("  all windows : " + ", ".join("{} {:.1%}".format(k, v)
                                               for k, v in shares3.items()))
            say("  waking hours: " + ", ".join("{} {:.1%}".format(k, v)
                                               for k, v in shares3_w.items()))
            for note in personal.notes_for(shares3, worst):
                say("  NOTE: " + note)

    say()
    say("Reminder: these are class shares from a model with no 'none of these' "
        "class, applied to a different sensor, with no labels. They are not a "
        "measurement of stress.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class Tee:
    """Write to the terminal and to an in-memory copy at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="WESAD stress model demo")
    parser.add_argument("--takeout", help="folder holding heart_rate_YYYY-MM-DD.csv")
    parser.add_argument("--day", help="local day to score, YYYY-MM-DD "
                                      "(default: newest day with >= 1000 windows)")
    parser.add_argument("--tz", default=DEFAULT_TZ, help="timezone of the local day")
    parser.add_argument("--with-rf", action="store_true",
                        help="also run the random forest (optional extra, ~15 s)")
    parser.add_argument("--mask-date", action="store_true",
                        help="print 'Day #1' instead of the date")
    parser.add_argument("--no-personal", action="store_true",
                        help="skip section 4")
    parser.add_argument("--save", action="store_true",
                        help="also write a copy of this output to demo/out/ "
                             "(gitignored). Implies --mask-date.")
    return parser.parse_args(argv)


def run(args, saved_name=None):
    problems.clear()
    started = time.perf_counter()

    show_header()

    features_csv = config.FEATURES_CSV
    if not os.path.exists(features_csv):
        not_reproducible("results/features.csv is missing; nothing to evaluate.")
        return 1
    table = pd.read_csv(features_csv)

    show_class_balance(table)
    show_loso(table, args.with_rf)
    if not args.no_personal:
        show_personal(table, args)

    rule("SUMMARY")
    say(BANNER)
    written = ("demo/out/{} (this --save copy only)".format(saved_name)
               if saved_name else "none")
    say("Total runtime: {:.1f} s. Files written: {}.".format(
        time.perf_counter() - started, written))
    if problems:
        say("NOT REPRODUCIBLE items ({}):".format(len(problems)))
        for item in problems:
            say("  - " + item)
        return 1
    say("Everything above was computed live; no reproducibility problems found.")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if not args.save:
        return run(args)

    # --save: keep a copy under demo/out/. The date is always masked in it.
    args.mask_date = True
    name = "demo_output_{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    copy = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, copy)):
        code = run(args, saved_name=name)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(copy.getvalue())
    return code


if __name__ == "__main__":
    sys.exit(main())
