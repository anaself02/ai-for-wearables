# eval/ - two models under one protocol (design)

Written 2026-11-01 (Part 1: code and tests only, no experiment run). The frozen
choices live in `config.py`; the draft Amendment 4 restates them for
`PRE_DECLARED.md`. No `eval/DESIGN.md` existed before this file.

## Input (audited read-only, reused as-is, never rebuilt)

`results/features.csv`, sha256 `167f323d...207a` of the content with CRLF read as LF
(pinned in `config.py`, Amendment 6; the Windows CRLF copy hashes to `d0d72f86...2c36`;
`run` refuses if it changes). 1015 windows, 15 subjects (S2-S17; S1 and S12 do not
exist in WESAD), 60 s windows with a 30 s step, kept only when 100% inside one
condition. Labels 1 baseline (557), 2 stress (302), 3 amusement (156). Heart
rate is computed from the wrist BVP at 64 Hz (0.7-3.7 Hz Butterworth, order 3,
filtfilt; peaks at least 0.4 s apart with prominence 0.5 x window SD;
inter-beat intervals median-filtered over 5 beats; 40-180 bpm kept), one value
per beat. Features: `HR_mean, HR_std` (population SD), `HR_min, HR_max,
HR_slope` (least-squares slope per beat). No missing values. Only the five HR
features are used; the 40-feature set is used only for the reproduction check.

## Tasks

- Primary: binary, stress (2) vs non-stress (1 + 3). Non-stress is 70.2%.
- Secondary: 3-class (baseline / stress / amusement), WESAD only.

## Models and baselines

| name | estimator | balancing | grid (inner-tuned) |
|---|---|---|---|
| majority | DummyClassifier(most_frequent) | - | - |
| threshold | one HR_mean cut maximising macro-F1 on the training subjects only (Amendment 4; the demo and dashboard keep balanced accuracy); 3-class predicts stress / baseline, never amusement | - | - |
| svm | StandardScaler + SVC(rbf) | class_weight="balanced" | C {0.1, 1, 10, 100} x gamma {scale, 0.01, 0.1} |
| rf | StandardScaler + RandomForest(300 trees) | class_weight="balanced" | max_depth {None, 10, 20} x min_samples_leaf {1, 5} |
| mlp | StandardScaler + MLPClassifier(max_iter=300, NO early stopping, no validation split inside fit) | random oversampling of minority classes to the majority count, inside fit only, on the rows fit receives (inner-training rows during tuning); scikit-learn 1.5.2's MLP has no sample weights | hidden {(32,), (64, 32)} x alpha {1e-4, 1e-2} |

The scaler is part of each pipeline, so it is fitted only on the rows a model
is fitted on (inner: 12 subjects; outer refit: 14). The scaler on the RF
changes nothing (trees are scale-invariant) and is kept for a uniform pipeline.

## Validation

- Outer: leave-one-subject-out, 15 folds, each subject tested once.
- Inner: the 14 training subjects are shuffled with the seed and dealt into 7
  folds of 2 subjects; each grid point is fitted on 6 folds (12 subjects) and
  scored on the 7th; selection on mean inner macro-F1 (ties: first grid
  point); refit on all 14; one prediction of the test subject.
- Seeds 0-4. They set the inner fold assignment, the RF and the MLP
  (initialisation, oversampling). The SVM and the
  baselines are deterministic given the folds.
- `folds.json` records every outer split and every seed's inner groups.
- The grid search runs 2 fits in parallel (processes, `GRID_JOBS`); results do
  not depend on it (a test compares `n_jobs=1` and `2`). Convergence of every
  inner fit and of each outer refit is read from the fitted MLP (it ran to
  `max_iter` = the case where scikit-learn warns) and written per fold to
  `EXPERIMENTS.md` and `results/failures.csv`.
- Each finished (task, model, seed) is saved to `out/partial/`, so an
  interrupted run resumes; partials from other code are refused.

## Metrics and summaries

Per outer fold (= per subject): macro-F1, balanced accuracy, per-class
precision and recall, confusion matrix, AUROC (binary; score = SVM decision
function, RF / MLP stress probability, threshold distance HR_mean - cut,
majority constant = 0.5). Summary: per subject, average the 5 seeds first;
then mean and SD over the 15 subjects; 95% percentile bootstrap CI of the mean
over subjects (10,000 resamples, seed 6322). Differences (MLP - SVM, MLP - RF,
RF - SVM, each model - each baseline) are paired per subject and bootstrapped
the same way. A difference is only called real when its CI excludes 0.

Note: Table I's 0.499 / 0.717 are POOLED macro-F1 over all out-of-fold
predictions, with default hyperparameters. This protocol reports the MEAN of
per-subject macro-F1 with tuned hyperparameters, so its numbers are not the
same quantity; the pooled value is also written for comparison.

## Outputs (Part 2)

`folds.json`, `manifest.json` (versions, seeds, grids, sha256 of the feature
table and of every eval/*.py), `EXPERIMENTS.md` (every run, failures
included), `results/summary_{task}.csv`, `results/differences_{task}.csv`,
`results/folds_{task}.csv` (per-fold metrics and chosen hyperparameters),
`results/pooled_macro_f1.csv` (pooled out-of-fold macro-F1 per run, Table I style),
`results/failures.csv`, and `out/predictions/{task}_{model}_seed{n}.csv`
(per window: subject, condition, fold, seed, model, y_true, y_pred, score;
git-ignored, regenerable).

## Known leakage risks (none reaches the test subject)

1. The MLP no longer uses scikit-learn's internal early-stopping split (which
   could put copies of oversampled rows, and 50%-overlapping windows of one
   subject, on both sides of a split). It trains a fixed 300 epochs; the only
   tuning is in the subject-grouped inner folds.
2. The features were designed and the IBI filter width chosen before this
   protocol, partly while looking at WESAD 3-class results (disclosed in
   PRE_DECLARED.md). The grids are fixed before any nested result.
3. Overlapping windows are only a problem across a split; every split here is
   by subject.
4. Pairwise CIs are not adjusted for multiplicity (9 pre-declared comparisons
   per task and metric); nothing outside that list is claimed.
