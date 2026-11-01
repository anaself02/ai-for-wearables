from django.urls import path

from . import views

app_name = "members"

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("waiting/", views.waiting, name="waiting"),
    path("claim/", views.claim, name="claim"),
]
