"""Command line for the two-model evaluation (from wearable-ml/, its .venv).

    python -m eval.run_eval plan                 what would run, and how many fits
    python -m eval.run_eval benchmark            time one inner fit per model -> estimate
    python -m eval.run_eval reproduce            the Table I check (full SVM, 0.717)
    python -m eval.run_eval run --go             the full nested run (Part 2 only)

`run` refuses without --go, and refuses if results/features.csv no longer has
the hash recorded in config.FEATURES_SHA256. Memory: one (task, model, seed)
is held at a time; the feature table is about 1,000 rows.
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

from . import config, folds, metrics, nested
from .data import load_table, sha256

ALL = config.BASELINES + config.MODELS


def fits_per_outer_fold(model):
    grid = config.GRIDS.get(model)
    if not grid:
        return 1
    points = 1
    for values in grid.values():
        points *= len(values)
    return points * config.INNER_FOLDS + 1


def plan_counts(tasks=("binary", "three"), seeds=config.SEEDS, n_subjects=15):
    out = {}
    for model in ALL:
        out[model] = fits_per_outer_fold(model) * n_subjects * len(seeds) * len(tasks)
    return out


def benchmark(table):
    """Mean seconds per fit on 12 subjects (an inner-fold sized fit), over every grid point."""
    import numpy as np
    from sklearn.base import clone

    from . import models
    from .data import labels

    subjects = sorted(set(table["subject_id"]), key=folds.subject_key)[:12]
    rows = np.isin(table["subject_id"], subjects)
    X = table.loc[rows, config.HR_FEATURES].to_numpy(dtype=float)
    out = {}
    for task in ("binary", "three"):
        y = labels(table, task)[rows]
        for model in config.MODELS:
            grid = config.GRIDS[model]
            times = []
            for params in _grid_points(grid):
                est = clone(models.make(model, task, 0)).set_params(**params)
                started = time.perf_counter()
                est.fit(X, y)
                times.append(time.perf_counter() - started)
            out["{}/{}".format(task, model)] = sum(times) / len(times)
    return out


def _grid_points(grid):
    import itertools
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def estimate(per_fit, n_subjects=15, seeds=config.SEEDS):
    """Seconds for the full run from per-fit timings (outer refits ~ +17% rows, ignored)."""
    total = 0.0
    lines = []
    for key, seconds in sorted(per_fit.items()):
        task, model = key.split("/")
        fits = fits_per_outer_fold(model) * n_subjects * len(seeds)
        total += fits * seconds
        lines.append((key, fits, seconds, fits * seconds))
    return total, lines


def manifest(seeds, extra=None):
    import numpy, pandas, scipy, sklearn
    data = {
        "date": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
                     "pandas": pandas.__version__, "scipy": scipy.__version__},
        "seeds": list(seeds), "inner_folds": config.INNER_FOLDS, "n_boot": config.N_BOOT,
        "boot_seed": config.BOOT_SEED, "grids": {k: {p: [list(v) if isinstance(v, tuple) else v
                                                        for v in vs] for p, vs in g.items()}
                                                 for k, g in config.GRIDS.items()},
        "files": {"results/features.csv": sha256(config.FEATURES)},
        "code": {name: sha256(os.path.join(config.EVAL_DIR, name))
                 for name in sorted(os.listdir(config.EVAL_DIR)) if name.endswith(".py")},
    }
    data.update(extra or {})
    return data


def log_experiment(text, path=config.EXPERIMENTS):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if new:
            handle.write("# Experiments\n\nEvery run, including failures, appended in order.\n\n")
        handle.write(text.rstrip() + "\n\n")


def write_csv(path, rows, columns=None):
    if not rows:
        return
    columns = columns or list(rows[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                             for k, v in row.items()})


def pooled_macro_f1(predictions, task):
    """Macro-F1 over all out-of-fold predictions of one run (the Table I quantity)."""
    from sklearn.metrics import f1_score
    if not predictions:
        return None
    return float(f1_score([p["y_true"] for p in predictions], [p["y_pred"] for p in predictions],
                          labels=config.TASKS[task]["labels"], average="macro", zero_division=0))


def run_all(tasks, seeds, models_, out_dir=config.OUT_DIR, table=None):
    table = load_table() if table is None else table
    subjects = sorted(set(table["subject_id"]), key=folds.subject_key)
    folds.write_folds(subjects, seeds=seeds)
    all_folds, all_failures, pooled = [], [], []
    for task in tasks:
        for model in models_:
            for seed in seeds:
                started = datetime.now(timezone.utc).isoformat()
                try:
                    result = nested.run(table, task, model, seed)
                except Exception as exc:
                    log_experiment("- {} {} {} seed {}: FAILED {}: {}".format(
                        started, task, model, seed, type(exc).__name__, exc))
                    all_failures.append({"task": task, "model": model, "seed": seed,
                                         "kind": "run failed", "error": str(exc)})
                    continue
                write_csv(os.path.join(out_dir, "predictions",
                                       "{}_{}_seed{}.csv".format(task, model, seed)),
                          result["predictions"])
                all_folds += result["folds"]
                pooled.append({"task": task, "model": model, "seed": seed,
                               "pooled_macro_f1": pooled_macro_f1(result["predictions"], task)})
                all_failures += [{"task": task, "model": model, "seed": seed, **f}
                                 for f in result["failures"]]
                log_experiment("- {} {} {} seed {}: {} folds, {:.1f} s, failures/warnings: {}"
                               .format(started, task, model, seed, len(result["folds"]),
                                       result["seconds"], result["failures"] or "none"))
    return all_folds, all_failures, pooled


def write_results(fold_rows, failures, tasks, out_dir=config.EVAL_DIR, pooled=()):
    results_dir = os.path.join(out_dir, "results")
    for task in tasks:
        rows = [r for r in fold_rows if r["task"] == task]
        summary = []
        for metric in ("macro_f1", "balanced_accuracy", "auroc"):
            for model, s in metrics.summarise(rows, metric).items():
                summary.append({"task": task, "metric": metric, "model": model, **s})
        write_csv(os.path.join(results_dir, "summary_{}.csv".format(task)), summary)
        diffs = []
        for metric in ("macro_f1", "balanced_accuracy", "auroc"):
            for pair, d in metrics.paired_differences(rows, metric).items():
                diffs.append({"task": task, "metric": metric, "pair": pair, **d})
        write_csv(os.path.join(results_dir, "differences_{}.csv".format(task)), diffs)
        write_csv(os.path.join(results_dir, "folds_{}.csv".format(task)), rows)
    write_csv(os.path.join(results_dir, "failures.csv"), failures)
    write_csv(os.path.join(results_dir, "pooled_macro_f1.csv"), list(pooled))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval")
    parser.add_argument("action", choices=["plan", "benchmark", "reproduce", "run"])
    parser.add_argument("--go", action="store_true", help="required for run")
    parser.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    parser.add_argument("--tasks", default="binary,three")
    parser.add_argument("--models", default=",".join(ALL))
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",")
    models_ = args.models.split(",")

    if args.action == "plan":
        counts = plan_counts(tasks, seeds)
        for model, n in counts.items():
            print("{:10s} {:>7,d} fits".format(model, n))
        print("total      {:>7,d} fits".format(sum(counts.values())))
        return
    if args.action == "reproduce":
        from .reproduce import EXPECTED_FULL_SVM_MACRO_F1, full_svm_loso
        value, n = full_svm_loso()
        print("full SVM ({} features) pooled macro-F1 {:.6f}; Table I {:.6f}; diff {:+.2e}".format(
            n, value, EXPECTED_FULL_SVM_MACRO_F1, value - EXPECTED_FULL_SVM_MACRO_F1))
        return
    if args.action == "benchmark":
        per_fit = benchmark(load_table())
        total, lines = estimate(per_fit, seeds=seeds)
        for key, fits, seconds, subtotal in lines:
            print("{:14s} {:>6,d} fits x {:.3f} s = {:>7.0f} s".format(key, fits, seconds, subtotal))
        print("estimated total: {:.0f} s ({:.1f} h), single process".format(total, total / 3600))
        return
    if not args.go:
        raise SystemExit("run needs --go (Part 2, after the amendment is applied).")
    if sha256(config.FEATURES) != config.FEATURES_SHA256:
        raise SystemExit("results/features.csv changed since the protocol was frozen.")
    fold_rows, failures, pooled = run_all(tasks, seeds, models_)
    write_results(fold_rows, failures, tasks, pooled=pooled)
    with open(config.MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest(seeds, {"tasks": tasks, "models": models_}), handle, indent=1)
    print("done: {} fold rows, {} failure/warning records".format(len(fold_rows), len(failures)))


if __name__ == "__main__":
    main()
