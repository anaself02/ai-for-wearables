#!/usr/bin/env bash
# Reproduce the results of wearable-ml, in the order of README.md ("Reproduce the results").
#
#   ./reproduce.sh                 print every step (dry run, nothing is executed)
#   ./reproduce.sh --run           execute the quick steps (setup, test, splits, plan, tables, demo)
#   ./reproduce.sh --run --long    also execute the long steps: main evaluation (~1.15 h), interaction run (~52 min), errors
#   ./reproduce.sh --run splits    execute only the named steps (setup test splits plan eval tables interaction errors demo)
#
# Seeds are fixed in eval/config.py (0-4, bootstrap 6322). Needs Python 3.12 and, for the `interaction`
# step's parity check, the raw WESAD folder (data/WESAD or $WESAD_DIR).
set -euo pipefail
cd "$(dirname "$0")"

RUN=0; LONG=0; TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --run) RUN=1 ;;
    --long) LONG=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) TARGETS+=("$arg") ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(setup test splits plan eval tables interaction errors demo)

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

# step NAME LONGFLAG -- command...
step() {
  local name="$1" is_long="$2"; shift 3
  local wanted=0
  for t in "${TARGETS[@]}"; do [ "$t" = "$name" ] && wanted=1; done
  [ "$wanted" -eq 1 ] || return 0
  echo "[$name] $*"
  if [ "$RUN" -eq 1 ]; then
    if [ "$is_long" = "long" ] && [ "$LONG" -eq 0 ]; then echo "    (long step skipped: add --long)"; return 0; fi
    "$@"
  fi
}

step setup       short -- bash -c "$(command -v python3.12 || command -v python) -m venv .venv && ( [ -x .venv/Scripts/python.exe ] && .venv/Scripts/python.exe -m pip install -r requirements.txt || .venv/bin/python -m pip install -r requirements.txt )"
step test        short -- "$PY" -m unittest discover -s eval/tests -t .
step splits      short -- "$PY" -m eval.splits --check
step plan        short -- "$PY" -m eval.run_eval plan
step eval        long  -- "$PY" -m eval.run_eval run --go
step tables      short -- "$PY" -m eval.tables
step interaction long  -- bash -c "$PY -m eval.interaction parity && $PY -m eval.interaction plan && $PY -m eval.interaction run --go && $PY -m eval.tables"
step errors      long  -- bash -c "for r in binary_svm binary_mlp three_svm; do $PY -m eval.errors --predictions \"eval/out/predictions/\${r}_seed*.csv\" --out eval/results/errors_tuned/\$r; done"
step demo        short -- "$PY" demo/run_demo.py --no-personal

[ "$RUN" -eq 1 ] || echo "(dry run: nothing was executed; add --run)"
