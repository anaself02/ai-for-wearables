# eval/ - two-model evaluation and the HR-resolution stage

What it does: trains and evaluates SVM, random forest and MLP (plus a majority and an `HR_mean`
threshold baseline) with nested leave-one-subject-out validation on `results/features.csv`
(PRE_DECLARED.md Amendment 4), then asks whether the model ranking depends on the time resolution
of the heart-rate input (Amendment 5), and analyses the errors (exploratory).

| entry point | what it does | main outputs |
|---|---|---|
| `python -m eval.run_eval plan / benchmark / reproduce / run --go` | plan, time one fit, reproduce Table I, full nested run (about 1.15 h) | `results/{summary,differences,folds}_*.csv`, `pooled_macro_f1.csv`, `failures.csv`, `folds.json`, `manifest.json`, `EXPERIMENTS.md`, `out/predictions/` |
| `python -m eval.tables` | writes `RESULTS.md` from the CSVs (read-only on the results) | `RESULTS.md` |
| `python -m eval.interaction parity / plan / run --go` | model x resolution stage (about 52 min) | `results/resolution/*`, `resolution_manifest.json` |
| `python -m eval.errors --predictions GLOB --out DIR` | exploratory error analysis | `ERRORS.md`, `per_subject.csv`, figures |
| `python -m eval.splits [--check]` | writes the exact splits as CSV/JSON | `results/splits/*` |

Seeds: 0-4 (inner folds, RF, MLP, oversampling), bootstrap seed 6322, set in `config.py`. The saved splits
are `folds.json` and `results/splits/`. Design notes: `DESIGN.md`, `DESIGN_RESOLUTION.md`; every run is
logged in `EXPERIMENTS.md`; the first draft of Amendment 5 is kept in `DRAFT_AMENDMENT_5.md` (superseded).

Run everything from `wearable-ml/` with its `.venv`. Tests:

    python -m unittest discover -s eval/tests -t .

## Where the raw WESAD data is

Only the resolution stage reads raw WESAD (`S*/S*.pkl`, about 1 GB each, one at a
time). No path is written into the code. The folder is, in this order:

1. the command-line argument `--wesad-dir DIR`
2. the environment variable `WESAD_DIR`
3. `data/WESAD` inside `wearable-ml/` (git-ignored)

    python -m eval.interaction parity --wesad-dir /path/to/WESAD
    # or
    export WESAD_DIR=/path/to/WESAD        (PowerShell: $env:WESAD_DIR = "...")
    python -m eval.resolution parity

`DIR` must contain `S2/S2.pkl`, `S3/S3.pkl`, ... (S1 and S12 do not exist). The
rebuild is checked against `results/features.csv` (parity); the stage stops if
they differ. The first evaluation (`python -m eval.run_eval`) needs only
`results/features.csv`. Nothing under `eval/` reads personal data.
