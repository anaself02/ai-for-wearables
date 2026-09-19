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
    path("metric/<slug:slug>/", views.metric_detail, name="metric"),
    path("metric/<slug:slug>/day/<isodate:day>/", views.metric_day, name="metric_day"),
    path("raw/", views.raw_index, name="raw_index"),
    path("raw/<slug:slug>/", views.raw_data_points, name="raw"),
]
