"""
03_train_model.py
-----------------
STEP 4: train and cross-validate classical models on the feature table.

Two models (both are Week 3 syllabus topics):
  - Random Forest
  - Support Vector Machine (RBF kernel)

...trained on TWO feature sets (defined in features.py):
  - "full"    : all 40 features, 8 channels - what a research-grade
                multi-sensor wristband (Empatica E4) provides
  - "hr_only" : the 5 heart-rate features - what a consumer wearable
                (Fitbit-class) actually reports

Both feature sets go through the SAME loop in this script: the same windows,
the same folds, the same model settings, the same metrics. The only thing
that differs is which columns are handed to the model. That is what makes the
comparison honest - nothing is tuned per feature set, so any difference in
score is caused by the missing sensors and nothing else.

WHY SUBJECT-WISE CROSS-VALIDATION, NOT A RANDOM SPLIT
-----------------------------------------------------
Each subject contributes roughly 40-60 windows, and those windows are highly
correlated with one another: they come from the same person, the same skin,
the same sensor fit, the same day. Two windows from subject S5 look far more
like each other than like any window from S11 - before you even consider
whether the person was stressed.

If we shuffled all the rows and split them randomly, most subjects would have
some of their windows in the training set and some in the test set. The model
could then score well simply by recognising the person ("this is S5's
baseline skin conductance level") instead of recognising the stress state.
That is data leakage, and it inflates the reported accuracy. Our 50% window
overlap makes it worse still: a randomly split pair of neighbouring windows
would literally share 30 seconds of the same signal across train and test.

So we use LeaveOneGroupOut with subject_id as the group. Every fold trains on
14 subjects and tests on the 1 held-out subject, whose data the model has
never seen in any form. That measures what we actually care about: does this
generalise to a NEW person? It gives lower numbers than a random split, and
those lower numbers are the honest ones.

WHY A PIPELINE WITH A SCALER
----------------------------
An SVM with an RBF kernel compares samples by distance, so a feature measured
in thousands (BVP amplitude) would drown out one measured in single digits
(EDA microsiemens). StandardScaler puts every feature on a comparable scale.
Crucially, the scaler sits INSIDE the pipeline, so in each CV fold it is
fitted on the training subjects only - the held-out subject's statistics never
leak into the preprocessing. (Random Forest does not need scaling - it splits
on one feature at a time - but keeping the same pipeline shape for both models
costs nothing and makes them directly comparable.)

WHICH MODEL GETS SAVED
----------------------
For each feature set we save the SVM. Not because SVM always wins on every
metric, but because fixing the algorithm is what makes the full vs. hr_only
comparison a like-for-like one: same algorithm, same protocol, only the
inputs differ. The Random Forest numbers are still computed and reported, so
the choice can be checked rather than taken on trust.

Outputs, per feature set:
  results/cv_predictions_<set>.csv       out-of-fold prediction per window
  results/per_subject_<set>_<model>.csv  per-held-out-subject scores
  results/scores_<set>.json              pooled scores for both models
  models/<...>.joblib                    the SVM, refitted on all subjects
  models/<...>.json                      the settings needed to rebuild the
                                         features for that model

Usage:
    python 03_train_model.py
"""

import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import project_config as config
import features


# Where each feature set's saved model and feature config end up. Keeping this
# as a plain lookup means the filenames are visible in one place rather than
# being assembled by string-building somewhere in the middle of the code.
OUTPUT_FILENAMES = {
    "full":    {"model": "svm_model.joblib",
                "feature_config": "feature_config.json"},
    "hr_only": {"model": "svm_hr_only.joblib",
                "feature_config": "feature_config_hr_only.json"},
}

# The algorithm we save for each feature set (see "WHICH MODEL GETS SAVED").
MODEL_TO_SAVE = "svm"


def build_models():
    """
    Define the two candidate models.

    These settings are identical for both feature sets - deliberately. If we
    tuned the hr_only model differently from the full model, any difference in
    their scores would be partly caused by the tuning rather than purely by
    the missing sensors, and the comparison would no longer answer the
    question we are asking.

    Random Forest:
      - 300 trees: enough for the scores to stop moving between runs.
      - class_weight="balanced": our classes are imbalanced (roughly 55%
        baseline / 30% stress / 15% amusement, because the baseline condition
        simply lasted longer in the protocol). This reweights each class by
        the inverse of its frequency, so the model is not rewarded for
        ignoring amusement and always guessing baseline.

    SVM:
      - RBF kernel, the standard non-linear default.
      - C=1.0 and gamma="scale" are scikit-learn's defaults; we keep them
        rather than tuning, because tuning honestly would need a nested
        cross-validation loop (an inner loop over the training subjects only)
        and that is beyond the scope of this step.
      - class_weight="balanced" for the same reason as above.
    """
    random_forest = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1)),
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=config.RANDOM_STATE)),
    ])

    return {"random_forest": random_forest, "svm": svm}


def cross_validate_by_subject(model, X, y, groups):
    """
    Run leave-one-subject-out cross-validation.

    Returns:
      predictions - one prediction per row of X, each made by a model that
                    had never seen that row's subject ("out-of-fold"
                    predictions). Pooling them lets us build one confusion
                    matrix over all 15 subjects.
      per_subject - accuracy and macro-F1 for each held-out subject, so we
                    can see how much performance varies from person to person.
    """
    splitter = LeaveOneGroupOut()
    predictions = np.empty(len(y), dtype=y.dtype)
    per_subject = []

    for train_index, test_index in splitter.split(X, y, groups):
        held_out = groups.iloc[test_index].unique()[0]

        model.fit(X.iloc[train_index], y.iloc[train_index])
        fold_predictions = model.predict(X.iloc[test_index])
        predictions[test_index] = fold_predictions

        per_subject.append({
            "subject_id": held_out,
            "n_windows": len(test_index),
            "accuracy": accuracy_score(y.iloc[test_index], fold_predictions),
            "f1_macro": f1_score(y.iloc[test_index], fold_predictions,
                                 average="macro", zero_division=0),
        })

    return predictions, pd.DataFrame(per_subject)


def save_feature_config(path, feature_set_name, feature_names):
    """
    Write down every setting that shaped the feature table for this model.

    This matters because a saved model is only meaningful when fed features
    built exactly this way, in exactly this column order. When the same
    pipeline is later applied to personal wearable data (see
    05_apply_to_personal_data.py), this file is the specification to follow.
    """
    settings = {
        "feature_set": feature_set_name,
        "window_seconds": config.WINDOW_SECONDS,
        "step_seconds": config.STEP_SECONDS,
        "label_purity_threshold": config.LABEL_PURITY_THRESHOLD,
        "target_labels": config.LABEL_NAMES,
        "sampling_rates_hz": {
            "wrist_ACC": config.FS_WRIST_ACC,
            "wrist_BVP": config.FS_WRIST_BVP,
            "wrist_EDA": config.FS_WRIST_EDA,
            "wrist_TEMP": config.FS_WRIST_TEMP,
        },
        "heart_rate_from_bvp": {
            "bandpass_low_hz": config.BVP_BANDPASS_LOW_HZ,
            "bandpass_high_hz": config.BVP_BANDPASS_HIGH_HZ,
            "butterworth_order": config.BVP_FILTER_ORDER,
            "min_peak_distance_s": config.BVP_MIN_PEAK_DISTANCE_S,
            "peak_prominence": "0.5 * std of the filtered window",
            "plausible_bpm_range": [config.HR_MIN_BPM, config.HR_MAX_BPM],
        },
        "feature_names_in_order": feature_names,
    }
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def save_training_distribution(table, path):
    """
    Write the heart-rate features' training distribution to a small JSON file
    next to the saved models.

    WHY THIS FILE EXISTS. Two other places need to know what the HR features
    looked like during training, so they can say how far a new set of
    readings falls outside it:

      * scripts/05_apply_to_personal_data.py, when comparing personal
        wearable data against training;
      * the Django health dashboard's arousal card, which runs the
        heart-rate-only model against live data.

    Both could recompute these numbers from results/features.csv, but that is
    a 640 KB file that belongs to the training run, not to the saved model.
    Writing the summary here instead means the model ships with everything
    needed to interpret it, in a few hundred bytes, and there is exactly one
    definition of "the training distribution" rather than two.
    """
    hr_columns = [name for name in features.FEATURE_NAMES
                  if name.startswith("HR_")]

    distribution = {
        "description": ("Distribution of the heart-rate features across the "
                        "WESAD training windows. Used to measure how far new "
                        "data falls outside what the model was fitted on."),
        "source": "WESAD wrist BVP, heart rate derived by features.py",
        "n_windows": int(len(table)),
        "n_subjects": int(table["subject_id"].nunique()),
        "window_seconds": config.WINDOW_SECONDS,
        "ibi_median_filter_beats": config.IBI_MEDIAN_FILTER_BEATS,
        "features": {
            name: {
                "mean": float(table[name].mean()),
                "std": float(table[name].std()),
                "min": float(table[name].min()),
                "median": float(table[name].median()),
                "max": float(table[name].max()),
            }
            for name in hr_columns
        },
    }

    with open(path, "w") as f:
        json.dump(distribution, f, indent=2)


def save_model_card(table, all_scores, path):
    """
    Write a model card: what this model was trained on, how it was validated,
    how well it scored, and what it cannot be trusted to do.

    A model card is the standard way to ship that context WITH a model
    instead of leaving it in a report the model gets separated from. Here it
    has a concrete job: the Django dashboard shows a "how this works" page
    next to its prediction, and that page renders this file. Without it the
    dashboard would have to hard-code performance figures, which would then
    quietly disagree with RESULTS.md the first time anything is retrained.
    """
    majority_share = (table["label_name"].value_counts().max() / len(table))

    card = {
        "model_file": OUTPUT_FILENAMES["hr_only"]["model"],
        "algorithm": "SVM (RBF kernel) inside a StandardScaler pipeline",
        "trained_on": {
            "dataset": "WESAD",
            "signals": "wrist (Empatica E4) heart rate, derived from BVP",
            "n_subjects": int(table["subject_id"].nunique()),
            "n_windows": int(len(table)),
            "window_seconds": config.WINDOW_SECONDS,
            "step_seconds": config.STEP_SECONDS,
            "class_balance": {
                name: int(count)
                for name, count in table["label_name"].value_counts().items()
            },
        },
        "classes": {str(k): v for k, v in config.LABEL_NAMES.items()},
        "validation": "leave-one-subject-out cross-validation ({} folds)".format(
            table["subject_id"].nunique()),
        "scores": all_scores,
        "always_guess_largest_class": {
            "accuracy": float(majority_share),
            # F1 of the majority class is 2p/(p+1); the other two score 0.
            "f1_macro": float((2.0 * majority_share / (majority_share + 1.0)) / 3.0),
        },
        "limitations": [
            "Trained on laboratory-induced stress (the Trier Social Stress "
            "Test) and amusement (funny video clips), not on real-world "
            "daily stress.",
            "Has no 'none of these' class. Every window it is given comes "
            "back as baseline, stress or amusement, whether or not any of "
            "those describe what was happening.",
            "Restricted to wrist heart rate, it loses {:.3f} macro-F1 "
            "against the full 8-channel sensor model ({:.3f} -> {:.3f}), "
            "because electrodermal activity and skin temperature carry much "
            "of the signal.".format(
                all_scores["full"][MODEL_TO_SAVE]["f1_macro"]
                - all_scores["hr_only"][MODEL_TO_SAVE]["f1_macro"],
                all_scores["full"][MODEL_TO_SAVE]["f1_macro"],
                all_scores["hr_only"][MODEL_TO_SAVE]["f1_macro"]),
            "Its heart rate is beat-to-beat, computed by our own peak "
            "detection on a 64 Hz PPG signal. A consumer wearable reports an "
            "already-smoothed rate every few seconds, so the same feature "
            "names describe different quantities and the inputs sit measurably "
            "outside the training distribution.",
            "15 subjects, all young adults, one lab, one session each.",
            "Applied to personal data there is no ground truth, so no "
            "accuracy can be claimed for any prediction it makes there.",
        ],
        "honest_summary": (
            "A demonstration that a wrist-heart-rate stress classifier can be "
            "trained, validated subject-wise and applied end to end. Not a "
            "validated real-world stress detector."),
    }

    with open(path, "w") as f:
        json.dump(card, f, indent=2)


def train_one_feature_set(feature_set_name, table):
    """
    Train and cross-validate every model on ONE feature set, save the results,
    and return the pooled scores.
    """
    feature_names = features.FEATURE_SETS[feature_set_name]

    # X = the features for this set. y = the class. groups = who it came from.
    X = table[feature_names]
    y = table["label"]
    groups = table["subject_id"]

    print("=" * 66)
    print("Feature set '{}': {} features".format(
        feature_set_name, len(feature_names)))
    print("  " + ", ".join(feature_names))
    print("=" * 66)

    results = {}
    for model_name, model in build_models().items():
        print("Training " + model_name + " ...")
        predictions, per_subject = cross_validate_by_subject(model, X, y, groups)

        overall_accuracy = accuracy_score(y, predictions)
        overall_f1 = f1_score(y, predictions, average="macro", zero_division=0)

        results[model_name] = {
            "model": model,
            "predictions": predictions,
            "per_subject": per_subject,
            "accuracy": overall_accuracy,
            "f1_macro": overall_f1,
        }

        print("  pooled out-of-fold accuracy : {:.3f}".format(overall_accuracy))
        print("  pooled out-of-fold macro-F1 : {:.3f}".format(overall_f1))
        print("  per-subject macro-F1        : mean {:.3f}, worst {:.3f} ({}), "
              "best {:.3f} ({})".format(
                  per_subject["f1_macro"].mean(),
                  per_subject["f1_macro"].min(),
                  per_subject.loc[per_subject["f1_macro"].idxmin(), "subject_id"],
                  per_subject["f1_macro"].max(),
                  per_subject.loc[per_subject["f1_macro"].idxmax(), "subject_id"]))
        print()

    # --- save the out-of-fold predictions ---------------------------------
    # 04_evaluate.py builds its confusion matrices from these, so it never has
    # to retrain anything.
    predictions_table = table[["subject_id", "label", "label_name",
                               "window_start_s"]].copy()
    for model_name in results:
        predictions_table["predicted_" + model_name] = \
            results[model_name]["predictions"]
    predictions_table.to_csv(
        os.path.join(config.RESULTS_DIR,
                     "cv_predictions_" + feature_set_name + ".csv"),
        index=False)

    for model_name in results:
        results[model_name]["per_subject"].to_csv(
            os.path.join(config.RESULTS_DIR,
                         "per_subject_{}_{}.csv".format(feature_set_name,
                                                        model_name)),
            index=False)

    # --- save the model ----------------------------------------------------
    # Refit on ALL subjects. The cross-validation above has already given us
    # the honest performance estimate; the model we ship should use every
    # scrap of data available.
    saved_model = results[MODEL_TO_SAVE]["model"]
    saved_model.fit(X, y)

    model_path = os.path.join(config.MODELS_DIR,
                              OUTPUT_FILENAMES[feature_set_name]["model"])
    joblib.dump(saved_model, model_path)
    print("Saved {} (refitted on all {} subjects) -> {}".format(
        MODEL_TO_SAVE, groups.nunique(), model_path))

    config_path = os.path.join(
        config.MODELS_DIR, OUTPUT_FILENAMES[feature_set_name]["feature_config"])
    save_feature_config(config_path, feature_set_name, feature_names)
    print("Saved feature config -> " + config_path)
    print()

    scores = {model_name: {"accuracy": results[model_name]["accuracy"],
                           "f1_macro": results[model_name]["f1_macro"]}
              for model_name in results}
    scores_path = os.path.join(config.RESULTS_DIR,
                               "scores_" + feature_set_name + ".json")
    with open(scores_path, "w") as f:
        json.dump({"feature_set": feature_set_name,
                   "n_features": len(feature_names),
                   "feature_names": feature_names,
                   "saved_model": MODEL_TO_SAVE,
                   "scores": scores}, f, indent=2)

    return scores


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    table = pd.read_csv(config.FEATURES_CSV)

    print("Feature table : {} windows, {} subjects".format(
        len(table), table["subject_id"].nunique()))
    print("Class balance :", dict(table["label_name"].value_counts()))
    print("Cross-validation: leave-one-subject-out ({} folds)".format(
        table["subject_id"].nunique()))
    print()

    all_scores = {}
    for feature_set_name in features.FEATURE_SETS:
        all_scores[feature_set_name] = train_one_feature_set(
            feature_set_name, table)

    # The heart-rate training distribution, for anything that needs to judge
    # whether new data resembles what the model was fitted on.
    distribution_path = os.path.join(config.MODELS_DIR,
                                     "hr_training_distribution.json")
    save_training_distribution(table, distribution_path)
    print("Saved HR training distribution -> " + distribution_path)
    print()

    # The feature-extraction code itself travels with the models, so the same
    # features can be rebuilt later from other data.
    shutil.copy(os.path.join(config.SCRIPTS_DIR, "features.py"),
                os.path.join(config.MODELS_DIR, "features.py"))
    shutil.copy(os.path.join(config.SCRIPTS_DIR, "project_config.py"),
                os.path.join(config.MODELS_DIR, "project_config.py"))

    # --- the headline comparison -------------------------------------------
    print("=" * 66)
    print("Full sensor set vs. heart-rate only ({}, leave-one-subject-out)"
          .format(MODEL_TO_SAVE))
    print("=" * 66)
    print("{:10s} {:>9s} {:>10s}".format("set", "accuracy", "macro-F1"))
    for feature_set_name in features.FEATURE_SETS:
        s = all_scores[feature_set_name][MODEL_TO_SAVE]
        print("{:10s} {:9.3f} {:10.3f}".format(
            feature_set_name, s["accuracy"], s["f1_macro"]))

    # The model card needs the scores, so it is written once both feature
    # sets have been cross-validated.
    card_path = os.path.join(config.MODELS_DIR, "model_card.json")
    save_model_card(table, all_scores, card_path)
    print("Saved model card -> " + card_path)
    print()

    full = all_scores["full"][MODEL_TO_SAVE]
    hr = all_scores["hr_only"][MODEL_TO_SAVE]
    print()
    print("Macro-F1 change when restricted to heart rate alone: "
          "{:+.3f} ({:+.1f}%)".format(
              hr["f1_macro"] - full["f1_macro"],
              100.0 * (hr["f1_macro"] - full["f1_macro"]) / full["f1_macro"]))


if __name__ == "__main__":
    main()
