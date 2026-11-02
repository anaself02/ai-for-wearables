# act/ - LLM feedback vs a template (the ACT role)

A free-tier LLM (Gemini) writes short feedback about one day from the model's
daily outputs. It is compared with a deterministic template, both scored with
the same rubric. The plan was fixed first: `../PRE_DECLARED.md`, Amendment 3
(2026-11-01). Nothing here trains or tunes a model.

## What leaves the machine

Only the ten allowlisted, de-identified daily summary fields (`schema.py`):
`input_id, day_index, n_windows, mean_bpm, share_predicted_stress,
share_out_of_distribution, degenerate, partial, coverage_hours,
median_spacing_s`. An extra field, a date or clock time, a weekday or month, a
participant name, a note, a list of readings or a timestamp is refused before
a prompt is built, and the finished prompt text is checked again. `input_id`
is never sent. Only Anas's days and synthetic cases are used; Rahil's never.

**Free tier and Google's use of prompts.** For unpaid services, Google's terms
(ai.google.dev/gemini-api/terms, read 2026-11-01) say content "is used to
provide, improve, and develop Google products", that "human reviewers may read,
annotate, and process your API input and output", and "Do not submit
sensitive, confidential, or personal information to the Unpaid Services". The
pricing page lists "Used to improve our products: Yes" for every free tier. In
Canada (outside the EEA, Switzerland and the UK) this applies. Decide before
`--approve` whether Anas's de-identified day summaries may go; `--synthetic-only`
sends only the invented cases.

## Model choice (checked 2026-11-01)

- Listed through the API (`client.models.list()`): stable Flash-family models
  with `generateContent` include `gemini-3.8-flash`, `gemini-3.7-flash`,
  `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`,
  `gemini-3.1-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, plus
  `-latest` aliases and previews.
- Pricing page (updated 2026-10-30): free tier "Free of charge" for 3.8 Flash,
  3.6 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 2.5 Flash, 2.5 Flash-Lite.
- **Chosen: `gemini-3.5-flash-lite`** (`config.MODEL`): stable, a pinned
  name rather than an alias, on the free tier, and the free-tier daily quota is
  reported to be large enough for one full run in a day. Override with `--model`.
- **Rate limits are not published on the docs page any more**: it says they
  "can be viewed in Google AI Studio", are "applied per project, not per API
  key", and RPD resets "at midnight Pacific time". A third-party page gives
  5 RPM / 20 RPD for 3.8 Flash and 10 RPM / 500 RPD for 3.5 Flash-Lite; this is
  **not verified**. Check your numbers at aistudio.google.com and pass them as
  `--rpm` / `--rpd` (defaults 8 and 450, deliberately below that figure).

## Size and time of a full run

34 inputs (22 real, 12 synthetic; I31-I34 appended 2026-11-01) x 2 presentations
x 5 repeats = **340 LLM calls** (arm A adds 68 local outputs): 120 synthetic,
220 real. With `--workers 4 --rpm 8` the run is capped at 8 calls a minute while
calls take up to about 30 s, so about 43 minutes; at 45 s a call 4 workers give
about 5.3 calls a minute, so about 64 minutes. It fits in one day only if the
daily quota is at least 340 (otherwise the run stops at `--rpd` and resumes the
next day).

## Status (2026-11-01, from `../eval/out/report_facts.md` section 8)

Only synthetic inputs have been sent so far: 197 LLM calls to `gemini-3.5-flash-lite`
(125 primary, 60 exploratory sweep, 12 pilot), 0 with a real day; no API errors. The
planned real-day calls (220) have not been made. The API key is read from `wearable-ml/.env`
(`GEMINI_API_KEY`, git-ignored), not from an environment variable. LLM sampling uses
temperature 0.7 and is not seeded (the API call passes no seed), so a rerun of the LLM arm
gives different text; the cache in `runs/calls.jsonl` is what makes a finished run repeatable.
The input shuffle, the bootstrap and the spot-check draw use `SEED = 6322` (`config.py`).

## How to run (from `wearable-ml/`, with its `.venv`)

```
.venv\Scripts\pip install -r act\requirements.txt

# 1. inputs (once): Anas's complete days via the dashboard's Python, then combine
"..\health dashboard\.venv\Scripts\python.exe" -m act.build_real_inputs
.venv\Scripts\python.exe -m act.make_inputs

# 2. dry run (the default): prints every payload, sends and writes nothing
.venv\Scripts\python.exe -m act.run

# 3. approve: send the calls. Same command resumes after an interruption.
.venv\Scripts\python.exe -m act.run --approve --rpm 8 --rpd 450
.venv\Scripts\python.exe -m act.run --approve --synthetic-only   # no real day sent
.venv\Scripts\python.exe -m act.run --approve --limit 5           # a small trial

# 4. report, then the blinded human check
.venv\Scripts\python.exe -m act.report            # -> act/runs/report.md, report.json
.venv\Scripts\python.exe -m act.spotcheck make    # -> runs/spotcheck_to_fill.csv (+ key)
#    fill usefulness_0_2 (0/1/2) without opening spotcheck_key.csv, then:
.venv\Scripts\python.exe -m act.spotcheck aggregate

# tests (fake client, no network)
.venv\Scripts\python.exe -m unittest discover -s act/tests -t .
```

`--arm template` runs arm A alone; it is local and needs no approval.

**Options for trials.**
- `--only I22,I25,I28` restricts the run to those inputs. With `--synthetic-only`
  any real ID is refused before anything is sent. Combines with `--limit`.
- `--repeats N` (default 5, pre-declared). Anything else prints a warning and
  marks the rows `"pilot": true`; they must not be used for reported results.
  `--pilot` marks a 5-repeat run as pilot too.
- `--workers N` (default 1) sends up to N calls at once. Start times stay at
  least 60/`--rpm` s apart across all workers and `--rpd` is shared. Log lines
  are written whole under a lock; the report sorts rows, so completion order does
  not matter. A failed call is logged and the others continue; a per-day quota
  error or a 401/403 stops new calls.
- `python -m act.make_inputs --table` shows which inputs trigger rule 3.
- `python -m act.show --only I22,I25,I28 --pilot` prints logged outputs with
  their payload, latency, retries, tokens and finish reason.

**Pilot convention.** A row is a pilot when it has `"pilot": true`, or matches
an entry in `act/pilots.json` (same prompt hash, dated at or before `until`;
used for the 5-call I03 pilot run before the field existed). Rows are never
edited. Pilot rows are never cache hits for a non-pilot run, and `act.report`
and `act.spotcheck` leave them out (`act.report --include-pilot` to see them).

**Retries and latency.** google-genai 2.34.0, by default, makes ONE attempt
(`HttpOptions.retry_options` is None, so tenacity stops after 1 attempt) and
sets NO timeout (`HttpOptions.timeout` None, httpx timeout None). Its optional
retry policy (5 attempts, 1 s doubling to 60 s, on 408/429/5xx) is only active
when `retry_options` is set. The client here pins one attempt explicitly and a
180 s timeout. All retries are ours: `retries` per row counts them,
`latency_s` is the successful attempt only, `latency_total_s` includes backoff.

**Resume and cache.** Every successful call is keyed by (prompt hash, run
index); the hash covers the model, settings and full prompt. A rerun skips
those, so an interrupted or quota-stopped run continues where it left off and
nothing is paid for twice. A per-minute 429 or a 5xx is retried with backoff
(the server's `retryDelay` when given); a per-day quota error or any other
error stops the run and prints what is left.

## Arms, factor, rubric

- **Arm A** (`template_arm.py`): 2-3 fixed sentences; a mandatory "cannot be
  interpreted" sentence when the day is out of distribution (share >= 0.25),
  degenerate or partial.
- **Arm B** (`llm.py`, `presentation.SYSTEM_PROMPT`): one system prompt, temperature
  0.7, max 400 output tokens, 5 repeats per (input, presentation).
- **Factor**: the same numbers as a table or as one sentence. Arm A ignores it.
- **Rubric** (`rubric/__init__.py`, rules documented there): numbers match the
  input at the precision shown (shares may be percentages; the prompt's two
  constants allowed), no word from `rubric/medical_terms.txt` unless negated,
  a caution sentence (`rubric/caution_phrases.txt`) on flagged inputs, at most 3
  sentences. Note: 28 of the 34 inputs need a caution sentence (all 22 real days
  are out of distribution), so the caution check dominates.
- **False caution** (separate metric, frozen in Amendment 3 on 2026-11-01): on
  the 6 inputs where rule 3 does not apply (I03, I18, I31-I34), an output with a
  phrase from `rubric/refusal_phrases.txt`. Not part of "all checks". Every
  output on those inputs, both arms, is read by hand:
  `python -m act.show --unflagged --no-pilot`. The phrase list is a screen
  only; the adjudicated count comes from `rubric/human_labels.csv` (yes/no per
  LLM output). `python -m act.human compare` shows screen vs human per input and
  presentation with every mismatch (also in the report); `python -m act.human
  audit` runs a broader regex and lists every hit not labelled yes.
- **EXPLORATORY boundary sweep** (I35-I40, added 2026-11-01 after seeing the
  I34 results): I34 with the OOD share at 0.15, 0.20, 0.22, 0.23, 0.25, 0.26.
  Rows carry `"exploratory": true`; they never join a default or
  `--synthetic-only` run (name them with `--only`), are excluded from every
  primary metric and the primary false-caution counts, and have their own
  report section. Human labels for I35-I38 go in the same `human_labels.csv`.

## Logs

`runs/calls.jsonl`, one record per call: arm, presentation, input_id, source,
run index, model string, SDK version, date (UTC), settings, prompt hash, full
prompt, payload, raw response, response JSON, finish reason, latency, token
counts, error. The API key is never logged (errors are redacted). `runs/` and
the input files are git-ignored (`act/.gitignore`): they hold a real person's
day-level numbers.
