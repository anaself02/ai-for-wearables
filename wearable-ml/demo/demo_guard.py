"""
demo_guard.py
-------------
A context manager that makes any attempt to train, fit or save a model raise.

The demo wraps the whole personal-data scoring step in it. If a code path in
that step ever tried to learn from personal data (fit, fit_transform,
partial_fit) or to save a model, it would stop with an AssertionError instead
of quietly succeeding.

Predict and transform calls are untouched, so scoring still works.
"""

from contextlib import ExitStack
from unittest import mock

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MESSAGE = "FIT GUARD: attempted to {} while scoring personal data. " \
          "Personal data is only scored, never used to train."

_TARGETS = [
    (Pipeline, "fit"), (Pipeline, "fit_transform"),
    (StandardScaler, "fit"), (StandardScaler, "partial_fit"),
    (StandardScaler, "fit_transform"),
    (SVC, "fit"),
    (RandomForestClassifier, "fit"),
    (MLPClassifier, "fit"), (MLPClassifier, "partial_fit"),
    (DummyClassifier, "fit"),
]


class FitGuard:
    """with FitGuard(): ...  # any fit/dump inside raises AssertionError"""

    def __enter__(self):
        self._stack = ExitStack()
        for cls, name in _TARGETS:
            self._stack.enter_context(mock.patch.object(
                cls, name,
                side_effect=AssertionError(MESSAGE.format(
                    "{}.{}".format(cls.__name__, name)))))
        self._stack.enter_context(mock.patch.object(
            joblib, "dump",
            side_effect=AssertionError(MESSAGE.format("joblib.dump"))))
        return self

    def __exit__(self, *exc_info):
        self._stack.close()
        return False
