from datetime import date

from django.urls import path, register_converter

from . import views, views_upload


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
    path("p/<slug:slug>/upload/", views_upload.upload, name="upload"),
    path("p/<slug:slug>/upload/confirm/", views_upload.upload_confirm,
         name="upload_confirm"),
    path("p/<slug:slug>/upload/cancel/", views_upload.upload_cancel,
         name="upload_cancel"),
    path("p/<slug:slug>/raw/", views.raw_index, name="raw_index"),
    path("p/<slug:slug>/raw/<isodate:day>/", views.raw_day, name="raw_day"),
]
