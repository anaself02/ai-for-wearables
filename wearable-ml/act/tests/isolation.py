"""Keep tests away from the real act/runs/ and act/inputs/.

Each test module does `from act.tests.isolation import setUpModule, tearDownModule`.
For the whole module every default path in act.config (and the spot-check
files) points into a temporary folder, and tearDownModule fails if anything
under the real act/runs/ or act/inputs/ changed while the module ran.
"""

import os
import tempfile
from unittest import mock

from act import config, spotcheck

REAL_RUNS = os.path.join(config.ACT_DIR, "runs")
REAL_INPUTS = os.path.join(config.ACT_DIR, "inputs")
_STATE = {}


def snapshot():
    found = {}
    for folder in (REAL_RUNS, REAL_INPUTS):
        for base, _dirs, names in os.walk(folder):
            for name in names:
                path = os.path.join(base, name)
                stat = os.stat(path)
                found[path] = (stat.st_size, stat.st_mtime_ns)
    return found


def setUpModule():
    _STATE["before"] = snapshot()
    _STATE["tmp"] = tempfile.TemporaryDirectory()
    root = _STATE["tmp"].name
    _STATE["patches"] = [
        mock.patch.object(config, "RUNS_DIR", os.path.join(root, "runs")),
        mock.patch.object(config, "INPUTS_DIR", os.path.join(root, "inputs")),
        mock.patch.object(config, "CALLS_LOG", os.path.join(root, "runs", "calls.jsonl")),
        mock.patch.object(config, "INPUTS", os.path.join(root, "inputs", "inputs.jsonl")),
        mock.patch.object(config, "REAL_DAYS", os.path.join(root, "inputs", "real_days.jsonl")),
        mock.patch.object(spotcheck, "TO_FILL", os.path.join(root, "runs", "fill.csv")),
        mock.patch.object(spotcheck, "KEY", os.path.join(root, "runs", "key.csv")),
    ]
    for patch in _STATE["patches"]:
        patch.start()


def tearDownModule():
    for patch in _STATE["patches"]:
        patch.stop()
    _STATE["tmp"].cleanup()
    if snapshot() != _STATE["before"]:
        raise AssertionError("A test changed files under act/runs or act/inputs.")
