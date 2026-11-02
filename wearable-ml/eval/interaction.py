"""Model x HR-resolution interaction (Stage 2D, Amendment 5 of PRE_DECLARED.md; binary task only).

    python -m eval.interaction plan            what would run, fits, time estimate
    python -m eval.interaction parity          rebuild R1s/R2 from raw WESAD, compare R1s with features.csv
    python -m eval.interaction run --go        the full run (Amendment 5 is applied; about 52 min at n_jobs=2)


Conditions (per held-out subject, seeds averaged first):
    R1s      train and tune on R1s, test on R1s
    R2      train and tune on R2, test on R2
    R1stoR2  train and tune on R1s, test on R2 (the transfer-relevant diagnostic)
R1s and R1stoR2 come from the same fits (one model, predicting two test tables).
Both resolutions use the common window set (windows with >= 10 readings at R2).
Grids, seeds, folds, metrics and bootstrap are those of Amendment 4, unchanged.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from . import config, cross, folds, metrics, resolution
from .data import features_sha256, sha256
from .run_eval import convergence_text, fits_per_outer_fold, log_experiment, manifest, \
    pooled_macro_f1, write_csv

TASK = "binary"
CONDITIONS = ["R1s", "R2", "R1stoR2"]
ALL = config.BASELINES + config.MODELS
MODEL_PAIRS = [("mlp", "svm"), ("mlp", "rf"), ("rf", "svm")]
OUT = os.path.join(config.OUT_DIR, "resolution")
RESULTS = os.path.join(config.EVAL_DIR, "results", "resolution")
MANIFEST = os.path.join(config.EVAL_DIR, "resolution_manifest.json")
METRICS = ("macro_f1", "balanced_accuracy", "auroc")
#: seconds for one (model, seed) over 15 outer folds, from the Amendment 4 binary run
#: (eval/out/run.log: SVM 11-16, RF 137-176, MLP 151-285 with n_jobs=2).
SECONDS_PER_RUN = {"majority": 1, "threshold": 1, "svm": 13, "rf": 157, "mlp": 237}


def protocol_hash():
    """Hash of everything that defines these results; partials from other code are refused."""
    import hashlib

    digest = hashlib.sha256()
    for name in ("config.py", "data.py", "folds.py", "metrics.py", "models.py", "nested.py",
                 "cross.py", "resolution.py", "interaction.py"):
        digest.update(sha256(os.path.join(config.EVAL_DIR, name)).encode())
    for path in (resolution.R1S_CSV, resolution.R2_CSV):
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def plan(seeds=config.SEEDS, models_=ALL):
    """-> (fit counts, estimated seconds). R1s and R1stoR2 share fits, so two training
    tables are fitted: R1s (also scored on R2) and R2."""
    fits = {m: fits_per_outer_fold(m) * 15 * len(seeds) * 2 for m in models_}
    seconds = sum(SECONDS_PER_RUN[m] * len(seeds) * 2 for m in models_)
    return fits, seconds


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def run_all(r1, r2, seeds, models_, out_dir=OUT, fresh=False, say=print, n_jobs=config.GRID_JOBS):
    """Run every (training resolution, model, seed) of the interaction stage.

    Inputs: the row-aligned R1s and R2 window tables, seeds, model names, output
    folder (finished units are saved under partial/ and reused unless fresh=True),
    a progress callback and the number of parallel grid fits. Output: (fold rows,
    failure records, pooled values). R1s-trained models are scored on R1s and on R2
    (R1stoR2); R2-trained models on R2. Seeds: each seed goes to cross.run (inner
    folds, RF, MLP, oversampling); SVM and baselines are deterministic.
    """
    os.makedirs(os.path.join(out_dir, "partial"), exist_ok=True)
    code = protocol_hash()
    fold_rows, failures, pooled = [], [], []
    plan_ = [("R1s", r1, {"R1s": r1, "R1stoR2": r2}), ("R2", r2, {"R2": r2})]
    total = len(plan_) * len(models_) * len(seeds)
    done = 0
    for train_name, train_table, eval_tables in plan_:
        for model in models_:
            for seed in seeds:
                done += 1
                tag = "binary_train{}_{}_seed{}".format(train_name, model, seed)
                path = os.path.join(out_dir, "partial", tag + ".json")
                if os.path.exists(path) and not fresh:
                    with open(path, encoding="utf-8") as handle:
                        saved = json.load(handle)
                    if saved["protocol_hash"] != code:
                        raise SystemExit("{} was made by different code; use --fresh.".format(path))
                    fold_rows += saved["folds"]
                    failures += saved["failures"]
                    pooled += saved["pooled"]
                    say("[{}/{}] {} resumed from partial".format(done, total, tag))
                    continue
                started = datetime.now(timezone.utc).isoformat()
                try:
                    result = cross.run(train_table, eval_tables, model, seed, n_jobs=n_jobs)
                except Exception as exc:
                    log_experiment("- {} resolution {} {} seed {}: RUN FAILED {}: {}".format(
                        started, train_name, model, seed, type(exc).__name__, exc))
                    failures.append({"model": model, "seed": seed, "train": train_name,
                                     "kind": "run failed", "error": str(exc)})
                    say("[{}/{}] {} RUN FAILED: {}".format(done, total, tag, exc))
                    continue
                run_folds, run_pooled = [], []
                for name, res in result["results"].items():
                    write_csv(os.path.join(out_dir, "predictions", "{}_{}_{}_seed{}.csv".format(
                        name, TASK, model, seed)), res["predictions"])
                    run_folds += res["folds"]
                    run_pooled.append({"condition": name, "model": model, "seed": seed,
                                       "pooled_macro_f1": pooled_macro_f1(res["predictions"], TASK)})
                run_folds = json.loads(json.dumps(run_folds))
                run_failures = [{"model": model, "seed": seed, "train": train_name, **f}
                                for f in result["failures"]]
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({"protocol_hash": code, "folds": run_folds,
                               "failures": run_failures, "pooled": run_pooled}, handle)
                fold_rows += run_folds
                failures += run_failures
                pooled += run_pooled
                kinds = sorted({f["kind"] for f in result["failures"]})
                detail = convergence_text(result["failures"])
                log_experiment("- {} resolution (trained on {}) {} seed {}: {} folds, {:.1f} s, "
                               "pooled macro-F1 {}, failures/warnings: {}{}".format(
                                   started, train_name, model, seed, len(run_folds) // len(eval_tables),
                                   result["seconds"],
                                   ", ".join("{} {:.3f}".format(p["condition"], p["pooled_macro_f1"])
                                             for p in run_pooled),
                                   ", ".join(kinds) if kinds else "none",
                                   ("\n" + detail) if detail else ""))
                say("[{}/{}] {} done in {:.0f} s ({})".format(
                    done, total, tag, result["seconds"], ", ".join(kinds) if kinds else "no failures"))
    return fold_rows, failures, pooled


# ---------------------------------------------------------------------------
# statistics (all from saved fold rows; Amendment 4 procedure)
# ---------------------------------------------------------------------------

def _per_subject(fold_rows, condition, metric):
    rows = [r for r in fold_rows if r["condition"] == condition]
    return metrics.subject_table(rows, metric)


def _ci_row(values):
    mean, lo, hi = metrics.bootstrap_ci(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "n_subjects": len(values)}


def summaries(fold_rows, metric):
    """Per condition and model: mean, SD over subjects, bootstrap CI."""
    out = []
    for condition in CONDITIONS:
        rows = [r for r in fold_rows if r["condition"] == condition]
        for model, s in metrics.summarise(rows, metric).items():
            out.append({"condition": condition, "metric": metric, "model": model, **s})
    return out


def within_condition_differences(fold_rows, metric):
    """Amendment 4's pairs (model pairs and each model minus each baseline), per condition."""
    out = []
    for condition in CONDITIONS:
        rows = [r for r in fold_rows if r["condition"] == condition]
        for pair, d in metrics.paired_differences(rows, metric).items():
            out.append({"condition": condition, "metric": metric, "pair": pair, **d})
    return out


def interaction_contrasts(fold_rows, metric="macro_f1"):
    """Pre-declared. Per subject (seeds averaged first):
       interaction  (a-b at R1s) - (a-b at R2)           for the three model pairs
       resolution   model at R2 - model at R1s            for each model
       transfer     model at R1stoR2 - model at R1s, and   R1stoR2 - R2,  for each model
    each with a paired bootstrap over subjects."""
    t = {c: _per_subject(fold_rows, c, metric) for c in CONDITIONS}
    out = []

    def subjects(*tables):
        return sorted(set.intersection(*(set(x) for x in tables)))

    for a, b in MODEL_PAIRS:
        if not all(m in t["R1s"] and m in t["R2"] for m in (a, b)):
            continue
        common = subjects(t["R1s"][a], t["R1s"][b], t["R2"][a], t["R2"][b])
        d1 = [t["R1s"][a][s] - t["R1s"][b][s] for s in common]
        d2 = [t["R2"][a][s] - t["R2"][b][s] for s in common]
        row = _ci_row([x - y for x, y in zip(d1, d2)])
        out.append({"kind": "interaction", "metric": metric,
                    "contrast": "({0}-{1} at R1s) - ({0}-{1} at R2)".format(a, b),
                    "diff_at_R1": float(np.mean(d1)), "diff_at_R2": float(np.mean(d2)), **row})
    for model in config.BASELINES + config.MODELS:
        for kind, hi_c, lo_c in (("resolution", "R2", "R1s"), ("transfer", "R1stoR2", "R1s"),
                                 ("transfer", "R1stoR2", "R2")):
            if model not in t[hi_c] or model not in t[lo_c]:
                continue
            common = subjects(t[hi_c][model], t[lo_c][model])
            out.append({"kind": kind, "metric": metric,
                        "contrast": "{} at {} - {} at {}".format(model, hi_c, model, lo_c),
                        **_ci_row([t[hi_c][model][s] - t[lo_c][model][s] for s in common])})
    return out


def rankings(fold_rows, metric="macro_f1"):
    """Models ordered by mean per-subject macro-F1, per condition (descriptive)."""
    out = {}
    for condition in CONDITIONS:
        means = {m: s["mean"] for m, s in metrics.summarise(
            [r for r in fold_rows if r["condition"] == condition], metric).items()
            if m in config.MODELS}
        out[condition] = sorted(means, key=lambda m: -means[m])
    return out


def pooled_table(pooled):
    """Pooled out-of-fold macro-F1 per condition and model (mean, min, max over seeds).

    Input: list of pooled values from run_all. Output: list of row dicts.
    Randomness: none.
    """
    out = {}
    for p in pooled:
        out.setdefault((p["condition"], p["model"]), []).append(p["pooled_macro_f1"])
    return [{"condition": c, "model": m, "pooled_macro_f1_mean": float(np.mean(v)),
             "min": float(np.min(v)), "max": float(np.max(v))} for (c, m), v in sorted(out.items())]


def heatmap(summary_rows, path):
    """One figure: rows = models + baselines, columns = R1s, R2; cell = mean per-subject
    macro-F1 (95% bootstrap CI printed in the cell)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [m for m in ("majority", "threshold", "svm", "rf", "mlp")]
    cols = ["R1s", "R2"]
    cell = {(r["model"], r["condition"]): r for r in summary_rows if r["metric"] == "macro_f1"}
    grid = np.array([[cell[(m, c)]["mean"] if (m, c) in cell else np.nan for c in cols]
                     for m in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(grid, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cols)),
                  ["R1s (beat-level)", "R2 (3.0 s bins)"])
    ax.set_yticks(range(len(rows)), ["majority", "HR_mean threshold", "SVM", "RF", "MLP"])
    for i, m in enumerate(rows):
        for j, c in enumerate(cols):
            if (m, c) in cell:
                r = cell[(m, c)]
                ax.text(j, i, "{:.3f}\n[{:.3f}, {:.3f}]".format(r["mean"], r["ci_low"], r["ci_high"]),
                        ha="center", va="center", fontsize=8,
                        color="white" if r["mean"] < 0.62 else "black")
    ax.axhline(1.5, color="white", linewidth=2)
    ax.set_title("Binary stress: macro-F1 per subject, mean [95% CI]\n"
                 "(baselines above the line; 15 subjects, 5 seeds)", fontsize=9)
    fig.colorbar(image, ax=ax, label="macro-F1")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_results(fold_rows, failures, pooled, results_dir=RESULTS):
    """Write the interaction-stage CSVs to eval/results/resolution/.

    Inputs: fold rows, failure records, pooled values, output folder. Output:
    summary, differences, interaction, folds, failures, pooled_macro_f1 CSVs.
    Seed: the paired bootstrap uses config.BOOT_SEED (6322), 10,000 resamples.
    """
    summary = [r for m in METRICS for r in summaries(fold_rows, m)]
    write_csv(os.path.join(results_dir, "summary.csv"), summary)
    write_csv(os.path.join(results_dir, "differences.csv"),
              [r for m in METRICS for r in within_condition_differences(fold_rows, m)])
    write_csv(os.path.join(results_dir, "interaction.csv"),
              [r for m in METRICS for r in interaction_contrasts(fold_rows, m)])
    write_csv(os.path.join(results_dir, "folds.csv"), fold_rows)
    write_csv(os.path.join(results_dir, "failures.csv"), failures)
    write_csv(os.path.join(results_dir, "pooled_macro_f1.csv"), pooled_table(pooled))
    with open(os.path.join(results_dir, "rankings.json"), "w", encoding="utf-8") as handle:
        json.dump(rankings(fold_rows), handle, indent=1)
    heatmap(summary, os.path.join(results_dir, "heatmap_macro_f1.png"))


def main(argv=None):
    """Command line: python -m eval.interaction plan | parity | run --go [--fresh] [--wesad-dir DIR] [--seeds].

    `plan` prints fits and a time estimate; `parity` rebuilds R1s/R2 from raw WESAD
    and checks them against results/features.csv; `run` needs --go and a proven
    parity. Seeds: config.SEEDS (0-4) unless --seeds is given.
    """
    parser = argparse.ArgumentParser(prog="python -m eval.interaction")
    parser.add_argument("action", choices=["plan", "parity", "run"])
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--wesad-dir", default=None,
                        help="folder holding S2/S2.pkl ...; default $WESAD_DIR, else data/WESAD")
    parser.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.action == "plan":
        fits, seconds = plan(seeds)
        for m, n in fits.items():
            print("{:10s} {:>7,d} fits".format(m, n))
        print("estimated {:.0f} s ({:.0f} min) with n_jobs={}".format(seconds, seconds / 60,
                                                                       config.GRID_JOBS))
        return
    if args.action == "parity":
        result = resolution.write_all(say=lambda m: print(m, flush=True),
                                      wesad_dir=args.wesad_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "n_dropped_at_R2"}, indent=1))
        sys.exit(0 if result["ok"] else 1)
    if args.action == "run":
        if not args.go:
            raise SystemExit("run needs --go (a long run: about 52 min at n_jobs=2).")
        if features_sha256(config.FEATURES) != config.FEATURES_SHA256:
            raise SystemExit("results/features.csv changed since the protocol was frozen.")
        r1, r2 = resolution.load_tables()                 # refuses unless R1s parity is proven
        fold_rows, failures, pooled = run_all(r1, r2, seeds, ALL, fresh=args.fresh,
                                              say=lambda m: print(m, flush=True))
        write_results(fold_rows, failures, pooled)
        with open(MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest(seeds, {
                "stage": "2D resolution interaction", "conditions": CONDITIONS,
                "bin_seconds": resolution.BIN_SECONDS, "min_readings": resolution.MIN_READINGS,
                "n_windows_common": int(len(r1)),
                "files": {"results/features.csv": features_sha256(config.FEATURES),
                          "windows_R1s.csv": sha256(resolution.R1S_CSV),
                          "windows_R2.csv": sha256(resolution.R2_CSV)}}), handle, indent=1)
        print("done: {} fold rows, {} failure/warning records".format(len(fold_rows), len(failures)))
        return


if __name__ == "__main__":
    main()
