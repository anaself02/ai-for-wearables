"""Daily summary numbers for Anas's complete days, from the dashboard's arousal results.

Run with the DASHBOARD's Python (it needs Django and the stored readings):

    "../health dashboard/.venv/Scripts/python.exe" -m act.build_real_inputs

(from the wearable-ml folder). It reads the dashboard database through its own
code (people.repository / health.arousal), one day at a time, scores each chosen
day with the saved 3-class model, and writes act/inputs/real_days.jsonl
(gitignored) holding ONLY allowlisted summary numbers: no date, no timestamp, no
reading. Nothing is fitted; nothing is sent anywhere.

share_predicted_stress is the share of windows the 3-class model labels like
the WESAD stress condition; degenerate is the 3-class rule (one class >= 60%).
Only Anas's profile is read (schema.require_allowed_person).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(HERE)
DASHBOARD = os.path.join(os.path.dirname(ML_DIR), "health dashboard")


def setup_django():
    sys.path.insert(0, DASHBOARD)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()


def pick_evenly(items, n):
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def summary(assessment):
    """The allowlisted numbers of one scored day (no input_id / day_index yet)."""
    shares = assessment.shares
    return {
        "n_windows": int(assessment.n_windows),
        "mean_bpm": round(float(assessment.mean_bpm), 1),
        "share_predicted_stress": round(float(shares.get(2, 0.0)), 3),
        "share_out_of_distribution": round(float(assessment.worst_ood_share), 3),
        "degenerate": bool(max(shares.values()) >= 0.60),
        "partial": bool(assessment.coverage["partial"]),
        "coverage_hours": round(float(assessment.coverage["hours"]), 1),
        "median_spacing_s": round(float(assessment.spacing["median"]), 1),
    }


def main():
    sys.path.insert(0, ML_DIR)
    from act import config, schema

    setup_django()
    from health import arousal
    from people.models import Person

    person = Person.objects.get(slug=schema.require_allowed_person("anas"))
    days = [d for d in arousal.scorable_days(person) if d["scorable"] and d["complete"]]
    days.sort(key=lambda d: d["day"])
    chosen = pick_evenly(days, config.N_REAL)

    rows = []
    for order, entry in enumerate(chosen, 1):
        assessment = arousal.assess(person, entry["day"])     # one day in memory at a time
        if not assessment.available:
            continue
        rows.append({"order": order, "summary": summary(assessment)})

    os.makedirs(config.INPUTS_DIR, exist_ok=True)
    with open(config.REAL_DAYS, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print("complete scorable days: {}; written: {} (no dates) -> {}".format(
        len(days), len(rows), os.path.relpath(config.REAL_DAYS, ML_DIR)))


if __name__ == "__main__":
    main()
