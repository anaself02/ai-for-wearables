from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("oauth/login/", views.google_login, name="google_login"),
    path("oauth/callback/", views.oauth_callback, name="oauth_callback"),
    path("oauth/logout/", views.sign_out, name="sign_out"),
    path("settings/", views.profile_settings, name="settings"),
    # Shown once, straight after the OAuth round trip.
    path("welcome/", views.welcome, name="welcome"),
    path("welcome/warm/", views.welcome_warm, name="welcome_warm"),
]
