# Pre-declared analysis plan

Written 2026-10-31, **before** any new computation for the binary task, the
resolution factor, the MLP, or the demo. If this file and a later result
disagree about what was planned, this file wins and the difference is reported.

## Research question

> Does a WESAD-trained stress model transfer to a consumer wrist wearable?

(Confirm against the course proposal file if one is added; if they conflict,
ask before changing this.)

## What had already been computed (so it is not mistaken for pre-registered)

All of the following are **3-class**, were seen before this file was written,
and are therefore **exploratory**:

- `results/scores_*.json` (default hyperparameters, leave-one-subject-out,
  one seed): hr_only SVM macro-F1 0.499, full SVM 0.717, and the RF numbers.
- Shuffled-label reference for 3-class macro-F1: 0.332 (SD 0.014).
- Within-subject HR_mean differences (stress vs baseline, amusement vs
  baseline) and a diagnostic with per-subject z-scoring (0.509).

No binary (stress vs non-stress) result existed before this file.

## Primary and secondary tasks

- **Primary task for personal-data transfer: binary stress vs non-stress.**
  Positive class = stress (WESAD label 2). Non-stress = baseline (1) +
  amusement (3), merged.
- **Secondary: 3-class (baseline / stress / amusement), WESAD-only.** It is not
  used to make claims about personal data.
- I do **not** claim the binary task is better than the 3-class task unless a
  run shows it. The two are different problems and their macro-F1 values are
  not directly comparable.

## Metrics (all tasks)

Macro-F1, balanced accuracy, per-class precision and recall, confusion matrix.
Accuracy may be shown but is never the headline. Pooled out-of-fold predictions
over all 15 subjects, plus per-subject macro-F1 mean and SD.

## Wording rule

The hr_only 3-class result (0.499) is described as "weak but above chance
(shuffled labels 0.332)". It is never described as being at chance.

## Baselines

- Majority class (fit on the training subjects of each fold).
- Rule: a single `HR_mean` threshold fit on the training folds only. Binary:
  predict stress if `HR_mean` > threshold, threshold chosen to maximise
  balanced accuracy on the training windows. 3-class variant: the same
  threshold, predicting baseline or stress, **never amusement** (so its
  amusement recall is 0 by construction).

## Factors (Stage 2)

- **Model type:** classical = SVM (RBF); small neural network =
  `MLPClassifier` (one hidden layer, 16 units). Random forest is an optional
  extra only.
- **Input resolution:** native beat-level heart rate vs about 3 s.
  The 3 s level is **simulated** by averaging the beat-level series in 3 s bins.
  It is a simulation of a consumer device, not Fitbit's own (undisclosed)
  smoothing, so a gap to real Fitbit data is expected to remain.

## Validation and tuning

- Outer: leave-one-subject-out, 15 folds. No random-window splits.
- Tuning (Stage 2): nested. Inner tuning uses the training subjects of each
  outer fold only. Each outer test subject is predicted once, after the
  hyperparameters are frozen for that fold.
- **Stage 1 demo uses default hyperparameters with no tuning**, and says so on
  screen. It is a walkthrough, not the evaluation.
- Seeds: **5** for stochastic models (MLP, RF). The SVM is deterministic and is
  reported as such. Bootstrap confidence intervals over subjects (Stage 2).

## Personal (Fitbit) data

- It is **only scored**. Nothing is trained, tuned or fine-tuned on it.
- There is no ground truth for any personal day, so **no accuracy or F1 is
  claimed** for it. Output is day-level statistics only.
- Reading rules: windows 60 s, step 30 s, at least 10 readings per window,
  anchored to local midnight; timezone `America/Toronto` (taken from the Takeout
  profile, not yet confirmed by the owner); waking hours 08:00-23:00 local.
- Display rules, fixed now: a day is flagged **out of distribution** when the
  largest share of windows outside the training range on any heart-rate feature
  is >= 0.25. Predictions are flagged **degenerate** when any single class takes
  >= 60% of windows. In either case the class shares are printed as not
  interpretable.
- Only `heart_rate_YYYY-MM-DD.csv` files are opened. No other Takeout file is
  read.

## Disclosure

The inter-beat-interval median filter width of 5 was chosen **after** comparing
WESAD features with my own Fitbit data (distribution comparison only; no labels
were used and no model was fitted on that data). This is a preprocessing
decision informed by personal-data distributions and is reported as such.

## Reported even if they fail

Tuned vs default (even if no gain), the MLP underperforming the SVM, the
resolution factor collapsing toward chance-level reference, an interaction
whose confidence interval includes zero, the binary variant, and personal-data
scoring staying out of distribution.

---

# Amendments and updates (append-only; the text above is unchanged)

## Amendment 1 - 2026-10-31

**Made after seeing the 2026-10-30 demo output**, not before. That output
flagged the binary model's personal-day predictions as "DEGENERATE" because one
class (non-stress) took 98.5% of windows.

**What was wrong with the original rule.** The display rule above ("degenerate
when any single class takes >= 60% of windows") was mis-specified for the binary
task. Non-stress is 70.2% of the WESAD training prior (713 of 1,015 windows), so
a model that is behaving normally would already sit above 60% most of the time.
The rule cannot tell a degenerate model from a correct one on that task.

**What changes.**

- **Binary task:** the dominance flag is dropped. The out-of-distribution flag
  is the only warning. It depends on the input features, not on the
  predictions: the share of windows outside the training range on a heart-rate
  feature (flag threshold unchanged at >= 0.25), with the median shift in
  training SD printed beside it. No threshold is set on the median shift.
- **3-class task:** unchanged. The 60% dominance rule stays.
- No other rule, threshold or metric is changed, and nothing was tuned against
  this output. The demo prints this amendment, with its date, on screen.

## Update 1 - 2026-10-31 (confirmation, not a rule change)

- The owner confirmed the timezone `America/Toronto`.
- The owner confirmed that default day 2026-10-30 is acceptable.

## Update 2 - 2026-10-31 (planned for Stage 2)

When script 02 is re-run in Stage 2, it will include a **label-to-sampling
alignment check** per subject (signal lengths against label length at the
documented sampling rates, and where each condition starts and ends in
seconds), processing one subject at a time. This closes the one item the
audit could only check indirectly.

## Update 3 - 2026-10-31 (demo reporting rules, from the owner)

- The demo labels which model produced each personal-day result: the saved
  3-class joblib, or the binary SVM trained in memory on WESAD only with default
  hyperparameters.
- The demo states that the SVM vs rule comparison is single-run point estimates
  with no confidence interval, and does not claim either is better.
- The demo prints the Fitbit reading spacing next to the WESAD beat-level
  resolution.
- A `--save` copy of the output is written under `demo/out/` (gitignored), with
  the date masked.

## Amendment 2 - 2026-10-31 (personal labelled sessions)

**Made before any session was recorded and before the scoring code existed.**
The full protocol is in `health dashboard/docs/SESSIONS_PROTOCOL.md`.

- Session labels are used **only to grade** predictions. They are never used to
  train, tune, calibrate, set a threshold or adapt any model.
- Primary scoring: `rest` windows (non-stress) vs `stress_task` windows
  (stress). `recovery` and `natural_stress` are reported separately and are not
  part of the primary score.
- Windows as everywhere else (60 s, 30 s step, local-midnight anchor, at least
  10 readings, the five `HR_` features). A window is labelled only if it lies
  fully inside one session and starts at least 60 s after the session start
  (label-lag buffer). Windows crossing a session boundary are dropped.
- A headline score needs at least 30 labelled windows per class; otherwise only
  the counts are shown, with a statement that there is not enough labelled data.
- Compared on the same windows: majority class and the `HR_mean` threshold rule
  (both fitted on WESAD training data only) and the binary SVM (default
  hyperparameters, trained in memory on WESAD `hr_only` features).
- Metrics: macro-F1, balanced accuracy, per-class precision and recall,
  confusion matrix, window counts, and per session the share of windows
  predicted stress. No confidence intervals (too few sessions); results from
  one person do not generalise.

## Amendment 3 - 2026-11-01 (ACT role: LLM feedback vs template)

**Written before any LLM call was made and before the comparison code was run.**
Code and full rules: `act/` (README there).

- **Question.** Does a free-tier LLM (Gemini, a Flash-family model chosen from
  the API's model list; the exact model string is recorded in every log record)
  write short daily feedback from model outputs that is as faithful and as
  cautious as a deterministic template?
- **Arms.** A: fixed templates, 2-3 sentences, with a mandatory caution sentence
  when the day is out of distribution (share >= 0.25, the existing rule),
  degenerate or partial. B: the LLM with one fixed system prompt (use only the
  given numbers, no medical claims, say when the result cannot be interpreted);
  temperature and max output tokens fixed and logged.
- **Factor.** Input presentation: (1) a plain table of the numbers, (2) one
  summarised sentence with the same numbers in the same format. The interaction
  reported is (B - A on table) - (B - A on sentence), bootstrap CI over inputs.
- **Inputs.** About 30: Anas's complete days (summary statistics from the
  dashboard's arousal results, no dates) plus synthetic edge cases (normal,
  high stress share, out of distribution, degenerate, partial, low coverage).
  Real vs synthetic is recorded locally and never sent. Never Rahil's data.
- **Repeats.** 5 runs per (input, presentation) for arm B; arm A is
  deterministic and run once.
- **Automated rubric (per output).** (1) every number in the text matches an
  input value at the precision shown (shares may appear as percentages; the
  two rule constants named in the prompt, 25% and 20 hours, are also allowed);
  (2) no word from `act/rubric/medical_terms.txt`, unless negated within the
  three preceding words; (3) a caution sentence (`act/rubric/caution_phrases.txt`)
  on every out-of-distribution, degenerate or partial input, not scored on
  others; (4) at most 3 sentences.
- **Human check.** Usefulness 0-2 on a random blinded subset (arm hidden,
  order shuffled), filled in by the owner.
- **Reported even if unflattering.** Pass rate per check per arm and
  presentation with bootstrap 95% CIs over inputs, variability across the 5
  repeats, tokens (free tier, so $0) and latency, and every failure with its raw
  text. No claim that either arm is better unless the CI excludes zero.
- **Threats to validity.** WESAD and similar public material, and common
  feedback phrasing, may be in the LLM's training data; the keyword and phrase
  lists are crude; one person's days do not generalise.
- **Privacy.** Only the ten allowlisted de-identified summary fields leave the
  machine, after an explicit `--approve`. Free-tier prompts may be used by
  Google to improve its products and may be read by human reviewers.
- 2026-11-01: 5-call pilot on synthetic I03 with system prompt v1; outputs faithful, no truncation, latency 24 to 51 s per call.
- 2026-11-01: 12-call pilot (2 repeats) on synthetic I22, I25, I28 with system prompt v1; all 12 STOP, all passed the automated checks, latency 6.8 to 30.9 s per call.
- 2026-11-01: before the full run, added 4 synthetic in-distribution inputs (I31-I34: OOD share below 0.25, not partial, not degenerate; one with a high stress share) so that false cautions can be measured. I01-I30 unchanged. A false caution is a sentence containing an interpretation-refusal phrase ("cannot be interpreted", "not interpretable", "unreliable", "not reliable", "cannot be relied", "should not be relied", "should not be used", "not meaningful", "inconclusive"; case-insensitive) on an input where rule 3 does not apply; it is reported separately and does not change the pre-declared caution check. Every output of both arms on the inputs where rule 3 does not apply (I03, I18, I31-I34) is read by hand. Full run: 34 inputs, 340 LLM calls.
- 2026-11-01: synthetic-only run, 120 LLM calls (12 synthetic inputs × 2 presentations × 5 repeats; gemini-3.5-flash-lite, google-genai 2.34.0, system prompt v1): 120 STOP, 0 errors, 0 retries, no failures on the four automated checks. False caution is human-adjudicated (`act/rubric/human_labels.csv`; the output says or clearly implies the result cannot be used or relied on, on an input where rule 3 does not apply); the refusal-phrase list is a screen only and is unchanged. Of the 60 LLM outputs on I03, I18 and I31–I34: human 5 (I34 sentence, runs 1–5), screen 4. The screen missed 1 of the 5 on I34 (run 5, "cannot be reliably interpreted"). I34 table 0 of 5, all other inputs 0 of 50, template arm 0 of 12. A broader regex audit found no further candidates.
- 2026-11-01 (EXPLORATORY, added after seeing the I34 results): boundary sweep on the out-of-distribution share at 0.15, 0.20, 0.22, 0.23, 0.25 and 0.26 (I35–I40, synthetic, otherwise identical to I34). Reported separately from the primary results and not used for any primary claim.

## Amendment 4 - 2026-11-01 (two models under one protocol; frozen before any nested result)

**Written and applied before the first nested run. Code: `wearable-ml/eval/`.**

- **Input.** `results/features.csv` as it is (sha256 d0d72f86b8a66c09ff0d3258bbba1fbf480f3f92cd85e606af0289d0aaaf2c36; 1015 windows, 15 subjects, 60 s windows with a 30 s step), the five heart-rate features `HR_mean, HR_std, HR_min, HR_max, HR_slope` only. The table is not rebuilt.
- **Tasks.** Primary: binary, stress (label 2) vs non-stress (baseline 1 + amusement 3). Secondary: 3-class (baseline / stress / amusement), WESAD only. The two tasks are different problems and their metrics are not directly comparable.
- **Baselines.** (a) Majority class, fitted on the training subjects. (b) One `HR_mean` threshold, chosen to **maximise macro-F1 on the training subjects only** (binary: stress above the cut, non-stress below; 3-class: stress above, baseline below, never amusement, so amusement F1 is 0 by construction and macro-F1 is over the three classes; ties go to the lowest cut). For this evaluation this replaces the balanced-accuracy criterion named under "Baselines" above; that change was made before any result of this protocol. The demo and dashboard keep their own rule.
- **Models.** SVM (RBF) and random forest (300 trees), both with balanced class weights; MLP (`MLPClassifier`) trained for a fixed `max_iter=300` with **no early stopping and no validation split inside fit**. Inputs are standardised by a scaler fitted on the training rows of each fit only. The MLP is balanced by random oversampling of the minority class(es) to the majority count, done inside fit on the rows that fit receives: during inner tuning these are the inner-training rows only, never inner-validation rows. Whether an MLP fit reached `max_iter` (the case in which scikit-learn warns of non-convergence) is recorded for every inner fit and every outer refit and reported in `eval/EXPERIMENTS.md`; `max_iter` is not changed after results are seen.
- **Grids (fixed).** SVM: C in {0.1, 1, 10, 100} x gamma in {scale, 0.01, 0.1}. RF: max_depth in {None, 10, 20} x min_samples_leaf in {1, 5}. MLP: hidden_layer_sizes in {(32,), (64, 32)} x alpha in {1e-4, 1e-2}.
- **Validation.** Outer: leave-one-subject-out, 15 folds, each subject tested once. Inner: the 14 training subjects are shuffled with the seed and dealt into 7 subject-grouped folds of 2; each grid point is fitted on 6 folds and scored by macro-F1 on the 7th; the best mean inner macro-F1 wins (ties: first grid point in the order above); the winner is refitted on all 14 training subjects and predicts the test subject once. Nothing about the test subject is used before that prediction.
- **Seeds.** 0, 1, 2, 3, 4 (inner fold assignment, RF, MLP initialisation and oversampling). The SVM and the baselines are deterministic given the folds. The grid search runs 2 fits in parallel; results do not depend on that number.
- **Metrics (per held-out subject).** Macro-F1, balanced accuracy, per-class precision and recall, confusion matrix, and AUROC (binary only; majority class scores 0.5). Per-window predictions of every run are saved (subject, condition, fold, seed, model, score).
- **Summaries.** For each subject the 5 seeds are averaged first; then the mean and SD over the 15 subjects. 95% percentile bootstrap CIs of the mean over subjects (10,000 resamples, seed 6322), and for paired differences (resampling subjects). The headline is the mean per-subject macro-F1. The pooled out-of-fold macro-F1 (the quantity in Table I) is also reported, for comparison only.
- **Pre-declared comparisons.** MLP minus SVM, MLP minus RF, RF minus SVM, and each of the three models minus each baseline (6 more): 9 comparisons per task and metric. The CIs are **not adjusted for multiplicity**. Nothing outside this list is claimed as a model difference.
- **Claims.** A difference is called real only when its CI excludes 0. No winner is named when the CIs overlap. The MLP is said to beat the classical models only if both MLP-minus-SVM and MLP-minus-RF exclude 0 in its favour.
- **No retuning.** Grids, seeds, folds, features, metrics and the bootstrap are not changed after test results have been seen. Every run, failures and non-convergence included, is recorded in `eval/EXPERIMENTS.md`.
- **Later stage, not part of this amendment.** The interaction analysis (model x heart-rate resolution, beat-level vs about 3 s) and the window-length grid.
