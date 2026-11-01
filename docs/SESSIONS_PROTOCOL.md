# Sessions: protocol and pre-declared labelling rule

Written 2026-10-31, **before** any session was recorded and before the scoring
code existed. If this file and a later result disagree about what was planned,
this file wins and the difference is reported. The same rule is appended to
`wearable-ml/PRE_DECLARED.md` as Amendment 2.

## Purpose

A session is a span of time a person labels themselves (for example "seated
rest" or "a timed arithmetic task"). Sessions are the only ground truth this
project has for personal data. They are used **only to grade** the predictions
of models trained on WESAD. They are never used to train, tune, calibrate, set
a threshold or adapt anything.

## How to run a session (only on yourself)

1. 10 minutes of seated rest.
2. 10 minutes of timed mental arithmetic, for example counting backwards aloud
   from 1,022 in steps of 13 and restarting at 1,022 after a mistake.
3. About 20 minutes of seated recovery.

Run it only on yourself, never on anyone else. Stop at any time if it becomes
uncomfortable. Enter the times in America/Toronto local time.

## Session kinds

| Kind | Primary score | Label |
|---|---|---|
| `rest` | yes | non-stress |
| `stress_task` | yes | stress |
| `recovery` | no, reported separately | none |
| `natural_stress` | no, reported separately | none |

## Labelling rule

- Windows are the same as everywhere else in this project: 60 s long, 30 s
  step, anchored to local midnight in the person's timezone, at least 10
  readings, features `HR_mean`, `HR_std`, `HR_min`, `HR_max`, `HR_slope`.
- A window gets a label only if it lies **fully inside one session** and starts
  **at least 60 s after the session start** (label-lag buffer). A window that
  crosses a session boundary is dropped.
- The primary score uses `rest` windows (non-stress) against `stress_task`
  windows (stress). `recovery` and `natural_stress` windows are not part of it;
  for those only the share of windows predicted stress is reported.

## Minimum for a headline score

At least **30 labelled windows per class** (non-stress and stress). Below that,
only the window counts are shown, with the statement that there is not enough
labelled data to score.

## What is compared, on the same windows

- Majority class: the more frequent binary class in the WESAD training table
  (non-stress), predicted for every window.
- Rule: a single `HR_mean` threshold, fitted on WESAD training windows only to
  maximise balanced accuracy. Stress if `HR_mean` is above it.
- Binary SVM (RBF, default hyperparameters as in the wearable-ml demo), trained
  in memory on the WESAD `hr_only` features in `results/features.csv`. Never
  saved.

Metrics: macro-F1, balanced accuracy, per-class precision and recall, the
confusion matrix, window counts, and per session the share of windows
predicted stress.

## Limits, stated with every result

- Labels are subjective: a "stress task" may not stress, and rest may not rest.
- Label lag: the body does not change state at the second a session starts;
  the 60 s buffer reduces this but does not remove it.
- Induction is uneven: the same task stresses different people, and the same
  person on different days, by different amounts.
- With this few sessions **no confidence interval is possible**, and data from
  one person do not generalise to anyone else.
- The out-of-distribution evidence (share of windows outside the WESAD range,
  the device, the reading spacing) is printed with every score.

## Privacy

Sessions belong to the person who recorded them. The free-text note is never
logged, never exported and never sent to any external service. The export
replaces dates with "Session 1..N" and minutes relative to the first session.
