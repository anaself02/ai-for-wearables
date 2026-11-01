"""The fixed system prompt and the two ways of presenting one day's numbers.

Both presentations carry the same nine numbers in the same format (shares as
fractions to three decimals, the rest as stored), so the factor is the layout
only: a plain table, or one summarised sentence. input_id is never shown.
"""

import hashlib
import json

from . import schema
from .config import COMPLETE_DAY_HOURS, OOD_FLAG_SHARE

PRESENTATIONS = ("table", "sentence")

SYSTEM_PROMPT = (
    "You write short feedback about one day of wrist heart-rate data that a "
    "stress model trained on laboratory data (WESAD) has scored.\n"
    "Rules:\n"
    "1. Use only the numbers given. Do not invent, estimate or compute new numbers.\n"
    "2. Make no medical or diagnostic claims and give no health advice.\n"
    "3. If share_out_of_distribution is at least {ood:.2f}, or degenerate is true, "
    "or partial is true (fewer than {hours} hours with data), say plainly that the "
    "result cannot be interpreted and why.\n"
    "4. Write at most 3 sentences of plain text, no lists or headings."
).format(ood=OOD_FLAG_SHARE, hours=COMPLETE_DAY_HOURS)

LABELS = [
    ("day_index", "day"),
    ("n_windows", "windows scored"),
    ("mean_bpm", "mean heart rate (bpm)"),
    ("share_predicted_stress", "share_predicted_stress"),
    ("share_out_of_distribution", "share_out_of_distribution"),
    ("degenerate", "degenerate"),
    ("partial", "partial"),
    ("coverage_hours", "hours with data"),
    ("median_spacing_s", "median spacing between readings (s)"),
]


def fmt(name, value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if name.startswith("share_"):
        return "{:.3f}".format(value)
    if isinstance(value, int):
        return str(value)
    return "{:.1f}".format(value)


def as_table(payload):
    rows = ["| field | value |", "|---|---|"]
    rows += ["| {} | {} |".format(label, fmt(name, payload[name])) for name, label in LABELS]
    return "\n".join(rows)


def as_sentence(payload):
    p = {name: fmt(name, payload[name]) for name, _ in LABELS}
    return ("Day {day_index}: {n_windows} windows scored, mean heart rate {mean_bpm} bpm, "
            "share_predicted_stress {share_predicted_stress}, share_out_of_distribution "
            "{share_out_of_distribution}, degenerate {degenerate}, partial {partial}, "
            "{coverage_hours} hours with data, median spacing between readings "
            "{median_spacing_s} s.").format(**p)


def build_prompt(payload, presentation):
    """{'system', 'user'} for one call. Refuses anything off the allowlist."""
    schema.validate_payload(payload)
    if presentation not in PRESENTATIONS:
        raise ValueError("presentation must be one of {}".format(PRESENTATIONS))
    body = as_table(payload) if presentation == "table" else as_sentence(payload)
    user = "Here are the numbers for one day:\n\n{}\n\nWrite the feedback.".format(body)
    schema.assert_safe_text(SYSTEM_PROMPT)
    schema.assert_safe_text(user)
    return {"system": SYSTEM_PROMPT, "user": user}


def prompt_hash(prompt, model, settings):
    blob = json.dumps({"prompt": prompt, "model": model, "settings": settings},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
