# eval/ - two-model evaluation and the HR-resolution stage

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
