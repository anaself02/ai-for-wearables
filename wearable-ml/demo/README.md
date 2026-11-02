# demo/ - mid-project walkthrough

Shows, live, what the project does: class balance per subject, leave-one-subject-out
(LOSO) tables with a majority baseline, a one-threshold rule and the repo's default SVM
(binary and 3-class), and optionally one local day of Fitbit heart rate scored with
WESAD-trained models plus an out-of-distribution check. Default hyperparameters, one run,
no tuning (the tuned 5-seed evaluation is `eval/`).

| file | role |
|---|---|
| `run_demo.py` | entry point: `python demo/run_demo.py [--day YYYY-MM-DD --takeout DIR] [--mask-date] [--no-personal] [--with-rf] [--save]` |
| `demo_wesad.py` | WESAD-only half (reads `results/features.csv`; learns from WESAD only) |
| `demo_personal.py` | personal half: reads only `heart_rate_YYYY-MM-DD.csv` from a Takeout folder, never learns |
| `demo_guard.py` | makes any fit or save call raise during personal scoring |
| `tests/` | `python -m unittest discover -s demo/tests` |

Outputs: console text. `--save` also writes a date-masked copy to `demo/out/` (git-ignored: day-level
personal statistics). Personal data is never read unless `--takeout` is given; the Takeout folder is not
part of the repository. Seeds: the shuffled-label reference uses seed 0 (`demo_wesad.shuffled_reference`);
the models use `random_state=42` from `scripts/project_config.py`.
