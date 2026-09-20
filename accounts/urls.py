from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("oauth/login/", views.google_login, name="google_login"),
    path("oauth/callback/", views.oauth_callback, name="oauth_callback"),
    path("oauth/logout/", views.sign_out, name="sign_out"),
    path("settings/", views.profile_settings, name="settings"),
]
