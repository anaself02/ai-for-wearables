"""Write the exact data splits of the evaluation to plain files.

    python -m eval.splits            write eval/results/splits/*
    python -m eval.splits --check    also compare with eval/folds.json (exit 1 if they differ)

Nothing here is random on its own: the inner folds come from `folds.inner_folds`
(`random.Random(seed)`, seeds 0-4 from `config.SEEDS`), the outer folds are
leave-one-subject-out in subject order, and the window table is read from
`results/features.csv` (hash-checked, `config.FEATURES_SHA256`). The files are
a readable copy of what the code produces, so the splits can be inspected and
diffed without running a model.

Files written to `eval/results/splits/`:
    windows.csv       one row per window: subject_id, window_start_s, label,
                      label_name, binary_label (stress = 1)   [1015 rows]
    subjects.csv      per subject: windows per class
    outer_folds.csv   the 15 leave-one-subject-out folds (test subject, 14 train subjects)
    inner_folds.csv   for every seed and outer fold, the 7 inner validation groups
                      of 2 subjects (training subjects only)
    seeds.json        seeds, bootstrap seed and size, inner fold count, feature-table hash
"""

import argparse
import csv
import json
import os

from . import config, folds
from .data import check_path, features_sha256, load_table

SPLITS_DIR = os.path.join(config.EVAL_DIR, "results", "splits")
LABEL_NAMES = config.TASKS["three"]["names"]


def build(features_path=config.FEATURES, seeds=config.SEEDS, k=config.INNER_FOLDS):
    """Splits as plain Python data.

    Inputs: the feature table path, the seeds (default 0-4) and the number of
    inner folds (default 7). Output: dict with `windows`, `subjects`, `outer`,
    `inner` (lists of row dicts) and `seeds_info`. Deterministic: the only
    random source is `folds.inner_folds`, seeded by each value in `seeds`.
    """
    table = load_table(check_path(features_path))
    subjects = sorted(table["subject_id"].unique(), key=folds.subject_key)
    windows = [{"subject_id": r.subject_id, "window_start_s": r.window_start_s,
                "label": int(r.label), "label_name": LABEL_NAMES[int(r.label)],
                "binary_label": int(r.label == config.STRESS)}
               for r in table.itertuples()]
    per_subject = []
    for s in subjects:
        sub = table[table["subject_id"] == s]
        per_subject.append({"subject_id": s, "windows": len(sub),
                            **{LABEL_NAMES[c]: int((sub["label"] == c).sum()) for c in LABEL_NAMES}})
    outer, inner = [], []
    for number, (test, train) in enumerate(folds.outer_folds(subjects), 1):
        outer.append({"fold": number, "test_subject": test, "train_subjects": " ".join(train)})
        for seed in seeds:
            for group, held in enumerate(folds.inner_folds(train, seed, k), 1):
                inner.append({"seed": seed, "outer_test_subject": test, "inner_fold": group,
                              "validation_subjects": " ".join(held)})
    info = {"seeds": list(seeds), "inner_folds": k, "n_subjects": len(subjects),
            "subjects": subjects, "n_windows": len(windows),
            "bootstrap_seed": config.BOOT_SEED, "n_boot": config.N_BOOT,
            "features_sha256_crlf_to_lf": features_sha256(features_path)}
    return {"windows": windows, "subjects": per_subject, "outer": outer, "inner": inner,
            "seeds_info": info}


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write(out_dir=SPLITS_DIR, features_path=config.FEATURES):
    """Write the five split files to `out_dir`; returns the list of paths.

    Inputs: output folder and feature table path. Seeds: `config.SEEDS`
    (inner folds) and `config.BOOT_SEED` (recorded only).
    """
    data = build(features_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, key in (("windows.csv", "windows"), ("subjects.csv", "subjects"),
                      ("outer_folds.csv", "outer"), ("inner_folds.csv", "inner")):
        path = os.path.join(out_dir, name)
        _write_csv(path, data[key])
        paths.append(path)
    path = os.path.join(out_dir, "seeds.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data["seeds_info"], handle, indent=1)
        handle.write("\n")
    paths.append(path)
    return paths


def differences_with_folds_json(features_path=config.FEATURES, path=config.FOLDS_FILE):
    """Names of the parts of `eval/folds.json` that differ from the folds the code makes now.

    Inputs: feature table and folds file. Output: list of strings, empty when
    the saved file equals `folds.all_folds` for the same subjects and seeds.
    """
    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    subjects = sorted(load_table(features_path)["subject_id"].unique(), key=folds.subject_key)
    fresh = json.loads(json.dumps(folds.all_folds(subjects)))
    return [part for part in ("outer", "inner", "inner_k") if saved.get(part) != fresh.get(part)]


def main(argv=None):
    """Command line; see the module docstring. Seeds: `config.SEEDS`, `config.BOOT_SEED`."""
    parser = argparse.ArgumentParser(prog="python -m eval.splits")
    parser.add_argument("--check", action="store_true", help="compare with eval/folds.json")
    parser.add_argument("--out", default=SPLITS_DIR)
    args = parser.parse_args(argv)
    for path in write(args.out):
        print("wrote", os.path.relpath(path, config.ML_DIR))
    if args.check:
        bad = differences_with_folds_json()
        print("eval/folds.json matches the splits made by the code" if not bad
              else "eval/folds.json DIFFERS in: " + ", ".join(bad))
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
