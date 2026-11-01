"""Test helper: make any attempt to train, fit or save a model raise.

Wrap the whole upload and scoring flow in `fit_guard()`. If a code path ever
tried to learn from uploaded data (fit, fit_transform, partial_fit) or to save a
model, the test would fail with an AssertionError instead of passing quietly.
Predict and transform calls are left alone, so scoring still works.

Imported only by tests; scikit-learn is imported lazily so the application
itself never needs it for this.
"""

from contextlib import ExitStack
from unittest import mock

MESSAGE = ("FIT GUARD: attempted to {} on uploaded data. Models are trained on "
           "WESAD only; uploaded data is only scored.")


def fit_guard():
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    targets = [(Pipeline, "fit"), (Pipeline, "fit_transform"),
               (StandardScaler, "fit"), (StandardScaler, "partial_fit"),
               (StandardScaler, "fit_transform"), (SVC, "fit"),
               (RandomForestClassifier, "fit"), (MLPClassifier, "fit"),
               (MLPClassifier, "partial_fit"), (DummyClassifier, "fit")]
    stack = ExitStack()
    for cls, name in targets:
        stack.enter_context(mock.patch.object(
            cls, name,
            side_effect=AssertionError(MESSAGE.format("{}.{}".format(cls.__name__, name)))))
    stack.enter_context(mock.patch.object(
        joblib, "dump", side_effect=AssertionError(MESSAGE.format("joblib.dump"))))
    return stack
