from django.urls import path, register_converter

from . import views


class DateConverter:
    """YYYY-MM-DD in the URL, for the single-day drill-down."""

    regex = r"\d{4}-\d{2}-\d{2}"

    def to_python(self, value):
        return value

    def to_url(self, value):
        return str(value)


register_converter(DateConverter, "isodate")

app_name = "health"

urlpatterns = [
    path("dashboard/", views.index, name="index"),
    # Everything the dashboard derives rather than measures, including
    # the arousal card, which is an interpretation and not a reading.
    path("insights/", views.insights, name="insights"),
    # Neither of these needs the live API on every load: the fitness page
    # reads stored values only, and the calculator falls back to the
    # textbook cycle when no nights can be fetched.
    path("fitness/", views.fitness, name="fitness"),
    path("sleep-calculator/", views.sleep_calculator, name="sleep_calculator"),
    # The arousal card: a POST computes today's assessment, and the
    # methodology page explains what it can and cannot mean.
    path("arousal/compute/", views.arousal_compute, name="arousal_compute"),
    path(
        "arousal/how-this-works/",
        views.arousal_methodology,
        name="arousal_methodology",
    ),
    path("metric/<slug:slug>/", views.metric_detail, name="metric"),
    # Registered before the generic day route so "sleep" reaches the hypnogram
    # rather than the intraday-sample view, which sleep has no extractor for.
    path("metric/sleep/night/<isodate:day>/", views.sleep_night, name="sleep_night"),
    path("metric/<slug:slug>/day/<isodate:day>/", views.metric_day, name="metric_day"),
    path("raw/", views.raw_index, name="raw_index"),
    path("raw/<slug:slug>/", views.raw_data_points, name="raw"),
]
