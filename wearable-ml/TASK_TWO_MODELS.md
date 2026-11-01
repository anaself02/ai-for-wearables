Task: Two ML models under one protocol (SVM and RF as Model A, MLP as Model B). NO API calls, NO commits. Do not modify demo/, act/, the dashboard, or any existing file before Monday's presentation. New code only under wearable-ml/eval/. If eval/DESIGN.md from an earlier stage already exists, reuse it and say so.

Part 1 (do now, no experiments run):
1. Audit read-only: how results/features.csv was built (HR source and resolution, window 60 s step 30 s, features HR_mean, HR_std, HR_min, HR_max, HR_slope, label mapping, subject ids). Reuse it as the single input for this task so results match the current Table I numbers; do not rebuild features.
2. Build eval/ code for the following protocol:
   - Primary task: binary stress vs non-stress (baseline + amusement). Secondary: 3-class, WESAD only.
   - Baselines: majority class; HR_mean threshold fitted on training subjects only.
   - Models: SVM (RBF), random forest (300 trees), MLP (sklearn MLPClassifier, early stopping, standardised inputs). Class balancing where the model supports it; for the MLP use oversampling of the minority class inside the training fold only, or sample weights, and say which.
   - Outer validation: leave-one-subject-out, 15 folds, each test subject used once. Inner tuning: subject-grouped folds over the 14 training subjects, selection on macro-F1, refit on all 14, test once. Feature scaler fitted on training subjects only.
   - Fixed grids: SVM C in {0.1,1,10,100} x gamma in {scale,0.01,0.1}; RF max_depth in {None,10,20} x min_samples_leaf in {1,5}; MLP hidden in {(32,),(64,32)} x alpha in {1e-4,1e-2}.
   - Seeds 0 to 4 for the stochastic parts (RF, MLP, inner fold shuffling).
   - Per fold metrics: macro-F1, balanced accuracy, per-class precision and recall, confusion matrix, AUROC (binary). Summaries: mean and SD over subjects (seeds averaged first); bootstrap CIs over subjects (10000 resamples) for each model, and for pairwise differences (MLP minus SVM, MLP minus RF, RF minus SVM) and each model minus each baseline.
   - Outputs: eval/folds.json (outer and inner subject ids), per-window predictions for every run (subject, condition, fold, seed, model, score), results CSVs, eval/EXPERIMENTS.md (every run including failures), and a run manifest (library versions, seeds, file hashes).
3. Tests: no subject in both train and test; scaler fitted on training data only; inner folds grouped by subject; determinism with a fixed seed; bootstrap sanity on synthetic data; eval/ never reads data-samples or Takeout files; the SVM full-feature LOSO with default hyperparameters reproduces the existing 0.717 macro-F1 (3-class) within a small tolerance, and explain any difference.
4. Draft Amendment 4 for PRE_DECLARED.md (show it, do NOT apply yet): freezes tasks, models, grids, seeds, metrics, bootstrap procedure, and the rule that nothing is retuned after seeing test results. Also state that the interaction analysis (model x HR resolution) and window-length grid are a later stage.
5. Estimate the run time of the full nested run for all three models and 5 seeds, and say what you would cut if it is too slow (for example fewer grid points, never fewer seeds below 3).

Report: files created, test count, the draft amendment, the run-time estimate, and any leakage risk you see. Then STOP and wait for my go.

Part 2 (only after I say go): apply the amendment, run the full evaluation, write the results tables, and report: the table for the primary and secondary tasks with mean, SD and bootstrap CIs, the pairwise differences with CIs, the chosen hyperparameters per fold, any failed or unstable runs, and a plain statement of whether the MLP beats the classical models beyond the uncertainty. Do not claim a winner if the CIs overlap.
