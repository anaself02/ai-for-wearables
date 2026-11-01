"""Automated rubric. Each check returns (passed, detail); passed is None when
the check does not apply to that input.

NUMBERS: every number written in the text must equal an input value at the
precision shown. A token with d decimals matches a value v when
|token - v| <= 0.5 * 10**-d (+1e-9), i.e. the token is v correctly rounded (so
"18%" and "18.3%" both match 0.183, "72" matches 72.4, "2,879" matches 2879).
A token followed by "%" is compared with 100 x each share field. Also allowed:
the two rule constants named in the system prompt (0.25 / 25% and 20 hours).
Everything else (an invented, recomputed or wrongly rounded number) fails.

MEDICAL: no term from medical_terms.txt, unless negated (see that file).

CAUTION: on inputs that are out of distribution, degenerate or partial, at
least one sentence contains a phrase from caution_phrases.txt.

SENTENCES: at most 3. Sentences end at . ! or ? followed by whitespace or the
end; a decimal point inside a number does not end one.

FALSE CAUTION (separate metric, added 2026-11-01 before the full run; not one
of the four checks above and not part of "all checks"): on an input where
rule 3 does NOT apply, the text contains a phrase from refusal_phrases.txt.
`false_caution` returns True (a false caution), False, or None when rule 3
applies and the metric is not defined.
"""

import os
import re

from .. import schema

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SENTENCES = 3
CHECKS = ("numbers", "medical", "caution", "sentences")

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*%)?")
NEGATIONS = {"not", "no", "never", "cannot", "can't", "isn't", "without", "nor"}
RULE_CONSTANTS = [(0.25, False), (25.0, True), (20.0, False)]


def _read_list(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle
                if line.strip() and not line.lstrip().startswith("#")]


MEDICAL_TERMS = _read_list("medical_terms.txt")
CAUTION_PHRASES = _read_list("caution_phrases.txt")
REFUSAL_PHRASES = _read_list("refusal_phrases.txt")


def numbers_in(text):
    """[(value, decimals, is_percent, raw)] for every number written in the text."""
    found = []
    for match in NUMBER.finditer(text):
        whole, frac, percent = match.group(1), match.group(2) or "", match.group(3)
        value = float(whole.replace(",", "") + frac)
        found.append((value, max(0, len(frac) - 1), bool(percent), match.group(0).strip()))
    return found


def _candidates(payload):
    plain, percent = [], []
    for name in schema.NUMERIC:
        plain.append(float(payload[name]))
        if name.startswith("share_"):
            percent.append(100.0 * float(payload[name]))
    plain += [v for v, is_pct in RULE_CONSTANTS if not is_pct]
    percent += [v for v, is_pct in RULE_CONSTANTS if is_pct]
    return plain, percent


def check_numbers(text, payload):
    plain, percent = _candidates(payload)
    unmatched = []
    for value, decimals, is_percent, raw in numbers_in(text):
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        pool = percent if is_percent else plain
        if not any(abs(value - v) <= tolerance for v in pool):
            unmatched.append(raw)
    return (not unmatched), unmatched


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def check_medical(text):
    words = _words(text)
    hits = []
    for i, word in enumerate(words):
        for term in MEDICAL_TERMS:
            hit = word.startswith(term[:-1]) if term.endswith("*") else word == term
            if hit and not NEGATIONS & set(words[max(0, i - 3):i]):
                hits.append(word)
    return (not hits), hits


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def check_caution(text, payload):
    if not schema.caution_required(payload):
        return None, "not required"
    lowered = [s.lower() for s in sentences(text)]
    found = [s for s in lowered if any(phrase in s for phrase in CAUTION_PHRASES)]
    return bool(found), (found[0] if found else "no caution sentence")


def false_caution(text, payload):
    """(True if an interpretation-refusal phrase appears on an unflagged input, found)."""
    if schema.caution_required(payload):
        return None, "rule 3 applies"
    lowered = (text or "").lower()
    found = [phrase for phrase in REFUSAL_PHRASES if phrase in lowered]
    return bool(found), found


def check_sentences(text):
    n = len(sentences(text))
    return (0 < n <= MAX_SENTENCES), n


def score(text, payload):
    """{check: (passed or None, detail)} for one output."""
    text = text or ""
    return {
        "numbers": check_numbers(text, payload),
        "medical": check_medical(text),
        "caution": check_caution(text, payload),
        "sentences": check_sentences(text),
    }
