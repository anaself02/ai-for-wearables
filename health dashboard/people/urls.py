from datetime import date

from django.urls import path, register_converter

from . import views, views_import, views_sessions


class DateConverter:
    """YYYY-MM-DD in the URL. An impossible date (2026-13-45) is a 404."""

    regex = r"\d{4}-\d{2}-\d{2}"

    def to_python(self, value):
        return date.fromisoformat(value)   # ValueError -> Django treats as no match

    def to_url(self, value):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)


register_converter(DateConverter, "isodate")

app_name = "people"

urlpatterns = [
    path("", views.home, name="home"),
    path("p/<slug:slug>/", views.person_home, name="person"),
    path("p/<slug:slug>/heart-rate/", views.heart_rate, name="heart_rate"),
    path("p/<slug:slug>/heart-rate/day/<isodate:day>/", views.heart_rate_day,
         name="heart_rate_day"),
    path("p/<slug:slug>/arousal/", views.arousal_view, name="arousal"),
    path("p/<slug:slug>/arousal/how-this-works/", views.arousal_methodology,
         name="arousal_methodology"),
    path("p/<slug:slug>/import/", views_import.import_data, name="import_data"),
    path("p/<slug:slug>/import/confirm/", views_import.import_confirm,
         name="import_confirm"),
    path("p/<slug:slug>/import/cancel/", views_import.import_cancel,
         name="import_cancel"),
    path("p/<slug:slug>/sessions/", views_sessions.sessions_page, name="sessions"),
    path("p/<slug:slug>/sessions/<int:pk>/edit/", views_sessions.session_edit,
         name="session_edit"),
    path("p/<slug:slug>/sessions/<int:pk>/delete/", views_sessions.session_delete,
         name="session_delete"),
    path("p/<slug:slug>/sessions/export.csv", views_sessions.sessions_export,
         name="sessions_export"),
    path("p/<slug:slug>/raw/", views.raw_index, name="raw_index"),
    path("p/<slug:slug>/raw/<isodate:day>/", views.raw_day, name="raw_day"),
]
