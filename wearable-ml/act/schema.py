"""The allowlist: the only data that may leave this machine.

A payload is a dict with EXACTLY these ten fields, each of a fixed type and
range. Anything else - an extra field, a date, a name, a note, a list of raw
readings, a timestamp - is refused before a prompt is even built, and the
finished prompt text is checked again (`assert_safe_text`).
"""

import math
import re

FIELDS = ["input_id", "day_index", "n_windows", "mean_bpm",
          "share_predicted_stress", "share_out_of_distribution",
          "degenerate", "partial", "coverage_hours", "median_spacing_s"]

INPUT_ID = re.compile(r"^I\d{2}$")          # I01..I99: carries no meaning

#: (type, min, max) for every numeric field.
NUMERIC = {
    "day_index": (int, 1, 99),
    "n_windows": (int, 0, 3000),
    "mean_bpm": (float, 20.0, 250.0),
    "share_predicted_stress": (float, 0.0, 1.0),
    "share_out_of_distribution": (float, 0.0, 1.0),
    "coverage_hours": (float, 0.0, 25.0),
    "median_spacing_s": (float, 0.0, 3600.0),
}
BOOLEAN = ("degenerate", "partial")

#: Only these two people exist in the project, and only Anas's days may be sent.
ALLOWED_PERSON = "anas"
PARTICIPANT_NAMES = ("anas", "rahil")

DATE_LIKE = re.compile(
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"
    r"|\b\d{1,2}:\d{2}\b|T\d{2}:\d{2}"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b(january|february|march|april|june|july|august|september|october|"
    r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I)


class PayloadRejected(ValueError):
    """A payload or prompt that must not leave the machine."""


def require_allowed_person(slug):
    if slug != ALLOWED_PERSON:
        raise PayloadRejected(
            "Only {}'s days may be used here (got {!r}); never Rahil's.".format(
                ALLOWED_PERSON.title(), slug))
    return slug


def validate_payload(payload):
    """Return the payload unchanged if it is exactly the allowlist, else raise."""
    if not isinstance(payload, dict):
        raise PayloadRejected("A payload must be a dict.")
    extra = sorted(set(payload) - set(FIELDS))
    missing = sorted(set(FIELDS) - set(payload))
    if extra:
        raise PayloadRejected("Fields not on the allowlist: {}".format(", ".join(extra)))
    if missing:
        raise PayloadRejected("Missing fields: {}".format(", ".join(missing)))

    input_id = payload["input_id"]
    if not isinstance(input_id, str) or not INPUT_ID.match(input_id):
        raise PayloadRejected("input_id must look like I01, I02, ... (got {!r})".format(input_id))
    for name in BOOLEAN:
        if not isinstance(payload[name], bool):
            raise PayloadRejected("{} must be true or false".format(name))
    for name, (kind, lo, hi) in NUMERIC.items():
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PayloadRejected("{} must be a number, not {!r}".format(name, type(value).__name__))
        if kind is int and not isinstance(value, int):
            raise PayloadRejected("{} must be a whole number".format(name))
        if not math.isfinite(value) or not lo <= value <= hi:
            raise PayloadRejected("{} = {} is outside {}..{}".format(name, value, lo, hi))
    return payload


def assert_safe_text(text):
    """The final prompt: no date, clock time, weekday, month or participant name."""
    match = DATE_LIKE.search(text)
    if match:
        raise PayloadRejected("Prompt contains date- or time-like text: {!r}".format(match.group(0)))
    lowered = text.lower()
    for name in PARTICIPANT_NAMES:
        if re.search(r"\b{}\b".format(name), lowered):
            raise PayloadRejected("Prompt contains a participant name.")
    return text


def out_of_distribution(payload):
    from .config import OOD_FLAG_SHARE
    return payload["share_out_of_distribution"] >= OOD_FLAG_SHARE


def caution_required(payload):
    return out_of_distribution(payload) or payload["degenerate"] or payload["partial"]
