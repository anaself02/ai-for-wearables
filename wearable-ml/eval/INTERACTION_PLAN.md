# Interaction analysis: model x HR resolution (plan and how to run)

Owner: Rahil. Code: `eval/resolution.py` (inputs), `eval/interaction.py` (runs,
analysis, heatmaps, report), tests in `eval/tests/test_interaction.py`
(synthetic data only, no WESAD needed).

## Draft Amendment 5 for PRE_DECLARED.md (NOT applied; agree with Anas first)

Append this text **before** running `python -m eval.interaction run --go`, and
date it. It must be in the file before any interaction result is seen.

> ## Amendment 5 - <date> (interaction analysis: model x HR resolution)
>
> **Written before any interaction result was computed.**
>
> - **Factors.** Model type (majority, threshold, SVM, RF, MLP) x HR input
>   resolution (native beat-level vs 3 s bins). Window length (30/60/120 s) is
>   **not** run; the guidelines require two factors and this is the one in the
>   research question. The proposal's 1D-CNN is replaced by the MLP already
>   frozen in Amendment 4; this is stated in the interim report.
> - **3 s level.** The beat-level series from `heart_rate_series` (unchanged),
>   averaged in 3 s bins anchored at the window start; empty bins dropped; a
>   window needs at least 10 non-empty bins. It is a simulation, not Fitbit's
>   own smoothing.
> - **Pairing.** Only windows valid at both levels are kept, so both levels use
>   identical windows. The native level is checked against
>   `results/features.csv` (HR columns must match on shared windows).
> - **Protocol.** Each cell uses Amendment 4 unchanged: nested LOSO, the same
>   grids, seeds 0-4, inner selection on macro-F1, bootstrap over subjects
>   (10,000 resamples, seed 6322). Nothing is retuned after seeing results.
> - **Tasks.** Binary (primary) and 3-class (secondary).
> - **Reported.** Every cell (mean, SD, 95% CI); native minus 3s per model;
>   model ranking at each level; interaction contrast per pair
>   `(A - B at native) - (A - B at 3s)`, paired per subject, bootstrap CI.
> - **Decision rule.** An interaction is claimed only when the contrast's CI
>   excludes 0. Otherwise: "no evidence the ranking changes with resolution
>   (15 subjects)". Reported even if every CI includes 0.
> - **Alignment check** (Update 2) runs during the rebuild and is reported.

## Steps on your machine (from `wearable-ml/`, venv active, WESAD in `data/WESAD`)

```
python -m pytest -q eval/tests                     # all tests, about 1-2 min
python -m eval.resolution build                    # ~ minutes; reads each 1 GB pickle once
#   check the printout: "all subjects aligned: True", and
#   "max |diff|" for every HR_ column should be 0 (native == features.csv).
#   If not, STOP and tell Anas before running anything.
python -m eval.interaction plan                    # number of fits
#   apply Amendment 5 to PRE_DECLARED.md now
python -m eval.interaction run --go                # the long run
python -m eval.interaction report                  # rebuild the report from saved rows
```

Faster first pass for the interim report (state it as such):
`python -m eval.interaction run --go --seeds 0,1,2 --tasks binary`
(the guidelines' minimum is 3 seeds).

Outputs in `eval/results/interaction/`: `INTERACTION.md` (tables + verdict),
`heatmap_<task>_<metric>.png`, `cells.csv`, `resolution_effects.csv`,
`contrasts.csv`, `fold_rows.jsonl`, `failures.csv`, `pooled_macro_f1.csv`.
Per-window predictions go to `eval/out/interaction/predictions/` (git-ignored,
used later by the error analysis). The inputs are in
`eval/results/resolution/` with `build_report.json`.

## Known issue found while building this

`eval/config.FEATURES_SHA256` is the hash of a Windows (CRLF) copy of
`results/features.csv`. The repo's `.gitattributes` checks text files out with
LF on Mac/Linux, so there `run_eval run --go` refuses and the test
`test_the_feature_table_is_the_one_the_protocol_froze` fails. The file content
is the same (converting LF to CRLF reproduces the frozen hash exactly). Fix
suggestion for Anas: hash the file with line endings normalised. Not changed
here because `config.py` is frozen.
