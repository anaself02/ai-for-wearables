from django.urls import path

from . import views

app_name = "people"

urlpatterns = [
    path("", views.home, name="home"),
    path("p/<slug:slug>/", views.person_home, name="person"),
]
