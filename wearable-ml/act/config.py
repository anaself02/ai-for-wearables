"""Every setting of the comparison in one place. Each is written into the logs."""

import os

ACT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(ACT_DIR)
INPUTS_DIR = os.path.join(ACT_DIR, "inputs")
RUNS_DIR = os.path.join(ACT_DIR, "runs")
REAL_DAYS = os.path.join(INPUTS_DIR, "real_days.jsonl")
INPUTS = os.path.join(INPUTS_DIR, "inputs.jsonl")
CALLS_LOG = os.path.join(RUNS_DIR, "calls.jsonl")
ENV_FILE = os.path.join(ML_DIR, ".env")
API_KEY_NAME = "GEMINI_API_KEY"

# Chosen on 2026-11-01 from the API's model list (README: "Model choice").
# A pinned stable name, not a "-latest" alias, so the model cannot change
# under the experiment. Override with --model; the string used is logged.
MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.7          # > 0 on purpose: the repeats measure variability
MAX_OUTPUT_TOKENS = 400
REPEATS = 5

# Free-tier limits are shown only in AI Studio (README). These defaults are
# conservative guesses; set --rpm / --rpd to the numbers AI Studio shows.
DEFAULT_RPM = 8
DEFAULT_RPD = 450
MAX_RETRIES = 5             # our own retries (logged per row as "retries")

# The SDK's own retrying is pinned OFF (one attempt), so every retry is ours and
# is counted. google-genai 2.34.0 already defaults to one attempt and to NO
# timeout; the timeout below stops a hung request from blocking a worker forever.
SDK_RETRY_ATTEMPTS = 1
REQUEST_TIMEOUT_S = 180
PILOTS_FILE = os.path.join(ACT_DIR, "pilots.json")

# Pre-declared rules (PRE_DECLARED.md): out of distribution at a share >= 0.25;
# a complete day has >= 20 hours with data.
OOD_FLAG_SHARE = 0.25
COMPLETE_DAY_HOURS = 20

N_REAL = 22
N_TOTAL = 34               # 22 real + 8 synthetic, then I31-I34 appended (Amendment 3)
SEED = 6322
