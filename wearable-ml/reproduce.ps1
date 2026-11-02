# Reproduce the results of wearable-ml, in the order of README.md ("Reproduce the results").
#
#   .\reproduce.ps1                     print every step (dry run, nothing is executed)
#   .\reproduce.ps1 -Run                execute the quick steps (setup, test, splits, plan, tables, demo)
#   .\reproduce.ps1 -Run -Long          also execute the long steps: main evaluation (~1.15 h), interaction run (~52 min), errors
#   .\reproduce.ps1 -Run -Target splits execute only the named steps
#                                       (setup test splits plan eval tables interaction errors demo)
#
# Seeds are fixed in eval\config.py (0-4, bootstrap 6322). Needs Python 3.12 and, for the `interaction`
# step's parity check, the raw WESAD folder (data\WESAD or $env:WESAD_DIR).
param(
    [switch]$Run,
    [switch]$Long,
    [string[]]$Target = @("setup", "test", "splits", "plan", "eval", "tables", "interaction", "errors", "demo")
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { ".venv/bin/python" }

function Step([string]$Name, [bool]$IsLong, [scriptblock]$Command) {
    if ($Target -notcontains $Name) { return }
    Write-Host "[$Name] $Command"
    if (-not $Run) { return }
    if ($IsLong -and -not $Long) { Write-Host "    (long step skipped: add -Long)"; return }
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "step $Name failed (exit code $LASTEXITCODE)" }
}

Step "setup"       $false { python -m venv .venv; & $py -m pip install -r requirements.txt }
Step "test"        $false { & $py -m unittest discover -s eval/tests -t . }
Step "splits"      $false { & $py -m eval.splits --check }
Step "plan"        $false { & $py -m eval.run_eval plan }
Step "eval"        $true  { & $py -m eval.run_eval run --go }
Step "tables"      $false { & $py -m eval.tables }
Step "interaction" $true  { & $py -m eval.interaction parity; & $py -m eval.interaction plan; & $py -m eval.interaction run --go; & $py -m eval.tables }
Step "errors"      $true  { foreach ($r in "binary_svm", "binary_mlp", "three_svm") { & $py -m eval.errors --predictions "eval/out/predictions/${r}_seed*.csv" --out "eval/results/errors_tuned/$r" } }
Step "demo"        $false { & $py demo/run_demo.py --no-personal }

if (-not $Run) { Write-Host "(dry run: nothing was executed; add -Run)" }
