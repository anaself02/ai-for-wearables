# wearable-ml

**Does a stress classifier trained on WESAD transfer to a consumer wrist wearable that reports only heart rate?**
SVM, random forest and MLP are trained on 5 heart-rate features (60 s windows, 30 s step) from 15 WESAD
subjects with leave-one-subject-out validation. Team: Anas El Fali and Rahil Shah, course COEN 6322 / 498.
The protocol was pre-declared in [`PRE_DECLARED.md`](PRE_DECLARED.md) before the results existed.

## Headline results

Binary task (stress vs non-stress): mean per-subject macro-F1 over 15 subjects, 5 seeds averaged first,
95% bootstrap CI over subjects.

| model | macro-F1 [95% CI] |
|---|---|
| majority class | 0.413 [0.411, 0.415] |
| HR_mean threshold | 0.689 [0.594, 0.778] |
| SVM | 0.731 [0.639, 0.818] |
| random forest | 0.747 [0.669, 0.826] |
| MLP | 0.760 [0.679, 0.833] |

Source: `eval/results/summary_binary.csv` (rows `metric = macro_f1`).

Random forest loses about 0.031 macro-F1 at 3 s resolution (CI [-0.046, -0.016]); SVM and MLP are not clearly
affected. We name no winner among the models (source: `eval/out/report_facts.md` section 6, from `eval/results/resolution/interaction.csv`).

## Repository layout

```
wearable-ml/
├── PRE_DECLARED.md   pre-declaration, append-only
├── RESULTS.md        first WESAD results (3-class, default hyperparameters)
├── scripts/          data prep: 01 inspect, 02 features, 03 train, 04 evaluate, 05 personal data
├── models/           shared feature code, saved SVMs, model card
├── results/          features.csv (the input of eval/) and outputs of scripts 02-04
├── eval/             SVM / RF / MLP nested evaluation, interaction stage, error analysis, tests
├── act/              ACT: LLM feedback vs deterministic template (rubric, runs, tests)
├── demo/             demo: LOSO tables and one scored Fitbit day
├── docs/             ARCHITECTURE.md
├── reproduce.ps1 / reproduce.sh
└── requirements.txt
```

## Setup

Python 3.12 (results were produced with 3.12.10). Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the commands below from `wearable-ml/` with the venv active (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).

## Data

- **WESAD**, Kaggle mirror: <https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset>.
  Place it so that `data/WESAD/S2/S2.pkl` ... `data/WESAD/S17/S17.pkl` exist (15 subjects; S1 and S12 do not exist).
  `data/` is git-ignored. Only scripts 01-02 and `eval.interaction parity` read these pickles; the main
  evaluation needs only `results/features.csv`, which is in the repo.
- **Private, not in the repo:** personal Fitbit data and the dashboard sessions (labelled sessions live in the separate
  `../health dashboard/` project).
- **Check `results/features.csv`:** the hash ignores line endings. `PRE_DECLARED.md` Amendment 6 gives both values:
  `d0d72f86b8a66c09ff0d3258bbba1fbf480f3f92cd85e606af0289d0aaaf2c36` (CRLF, Windows) and
  `167f323da96a6e3c034c32f8cf05c8d506084c0dbf9518c36cd7a7341fab207a` (CRLF read as LF, pinned in `eval/config.py`).

  ```bash
  python -c "from eval import config, data; print(data.features_sha256() == config.FEATURES_SHA256)"   # expect True
  ```

## Reproduce

Run times are from the authors' Windows machine (`eval/out/report_facts.md`, sections 4 and 6).

```bash
python scripts/02_feature_extraction.py      # optional, not run by reproduce.ps1 or reproduce.sh; needs raw WESAD -> results/features.csv (hash check must still pass)
python -m eval.splits --check                # compare splits with eval/folds.json -> eval/results/splits/
python -m eval.run_eval plan                 # prints the plan
python -m eval.run_eval run --go             # ~69 min (4152 s) -> eval/results/{summary,differences,folds}_{binary,three}.csv,
                                             #   pooled_macro_f1.csv, failures.csv, eval/folds.json, eval/manifest.json, eval/EXPERIMENTS.md
python -m eval.tables                        # -> eval/RESULTS.md
python -m eval.interaction parity            # needs raw WESAD -> eval/out/resolution/parity.json
python -m eval.interaction plan
python -m eval.interaction run --go          # ~52 min (3103 s) -> eval/results/resolution/*.csv, rankings.json, eval/resolution_manifest.json
python -m eval.tables                        # adds the interaction section to eval/RESULTS.md
python -m eval.errors --predictions "eval/out/predictions/binary_svm_seed*.csv" --out eval/results/errors_tuned/binary_svm
```

The last line is one of three error-analysis calls; `binary_mlp` and `three_svm` use the same flags with their own prediction glob and `--out`. `reproduce.ps1` / `reproduce.sh` run the steps from `splits` onward in this order, plus setup, test and demo (the optional `scripts/02` step is not in them). They print only by default; `-Run` / `--run` executes and `-Long` / `--long` adds the long steps.

## Tests

```bash
python -m unittest discover -s eval/tests -t .
python -m unittest discover -s act/tests -t .
python -m unittest discover -s demo/tests
```

## Pre-declaration

`PRE_DECLARED.md` is append-only: earlier text is never edited.

- Amendment 1: for the binary task the 60% dominance ("degenerate") flag is dropped and the out-of-distribution flag is the only warning; the 3-class rule is unchanged.
- Amendment 2: a headline personal score needs at least 30 labelled windows per class; otherwise only the counts are shown, with a statement that there is not enough labelled data.
- Amendment 3: ACT, LLM feedback vs template.
- Amendment 4: SVM, RF, MLP under one nested leave-one-subject-out protocol.
- Amendment 5: model x heart-rate-resolution interaction.
- Amendment 6: feature-table hash made line-ending independent (no data change).

## Demo and ACT

- Demo: `python demo/run_demo.py` (options: `--takeout`, `--day`, `--tz`, `--with-rf`, `--mask-date`, `--no-personal`, `--save`).
- A headline personal score is shown only with at least 30 labelled windows per class (Amendment 2); below that only the window counts are shown, with a statement that there is not enough labelled data.
- All 197 LLM calls so far used synthetic inputs (`eval/out/report_facts.md` section 8).
- API key: `act/llm.py` reads `GEMINI_API_KEY` from `wearable-ml/.env` (git-ignored; never commit it). A process environment variable is not read.

## Limitations

- 15 subjects: wide CIs.
- WESAD stress is lab-induced, not everyday stress.
- The MLP did not converge: it hit `max_iter=300` in nearly every fit (4196 of 4200 inner fits in the interaction run).
- The 3 s condition is a simulation (beat-level HR averaged into 3 s bins), not a device.
- The error analysis is exploratory.
- No multiplicity adjustment on CIs.
