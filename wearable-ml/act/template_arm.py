"""Arm A: deterministic feedback from fixed templates.

Two or three sentences. Every number is copied from the input at the same
precision the rubric accepts. When the day is out of distribution, degenerate
or partial, the second sentence is a mandatory caution sentence. The template
ignores the presentation: it reads the numbers, not the prompt.
"""

from . import schema
from .presentation import fmt


def pct(share):
    return "{:.1f}%".format(100 * share)


def feedback(payload):
    schema.validate_payload(payload)
    p = payload
    first = ("Day {}: the model labelled {} of {} windows as stress-like, with a mean "
             "heart rate of {} bpm.").format(p["day_index"], pct(p["share_predicted_stress"]),
                                             p["n_windows"], fmt("mean_bpm", p["mean_bpm"]))
    if schema.caution_required(p):
        reasons = []
        if schema.out_of_distribution(p):
            reasons.append("{} of windows fall outside the training data".format(
                pct(p["share_out_of_distribution"])))
        if p["degenerate"]:
            reasons.append("one class takes most windows")
        if p["partial"]:
            reasons.append("only {} hours of the day have data".format(
                fmt("coverage_hours", p["coverage_hours"])))
        second = "This result cannot be interpreted: {}.".format("; ".join(reasons))
        third = "Readings were {} s apart on average.".format(
            fmt("median_spacing_s", p["median_spacing_s"]))
        return " ".join([first, second, third])
    second = ("Readings were {} s apart on average and covered {} hours, and {} of windows "
              "fall outside the training data.").format(
        fmt("median_spacing_s", p["median_spacing_s"]),
        fmt("coverage_hours", p["coverage_hours"]), pct(p["share_out_of_distribution"]))
    return " ".join([first, second])
