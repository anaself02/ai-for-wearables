"""Sign up, log in, log out, and the page an unlinked account sees.

Nothing here logs an email address or a password, and nothing is ever emailed.
Sign-up creates an account and a PENDING claim for a profile; an administrator
must approve the link before any data is visible.
"""

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST

from people import links

from . import throttle
from .forms import ClaimForm, LoginForm, SignupForm, normalise_email

NOT_AVAILABLE = "Email or password is incorrect."
LOCKED = ("Too many failed attempts for this email. Wait 5 minutes and try again.")
SIGNUP_NOTICE = ("No email is sent or verified. An administrator must approve the "
                 "link between your account and the profile before any data is "
                 "visible to you.")


def safe_next(request):
    target = request.POST.get("next") or request.GET.get("next") or ""
    if target and url_has_allowed_host_and_scheme(
            target, allowed_hosts={request.get_host()}, require_https=False):
        return target
    return "/"


@never_cache
def signup(request):
    if request.user.is_authenticated:
        return redirect("/")
    form = SignupForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(
            request, "Account created. An administrator must approve the link to "
                     "your profile before any data is visible. You can log in now "
                     "to see its status.")
        return redirect("members:login")
    return render(request, "members/signup.html",
                  {"form": form, "notice": SIGNUP_NOTICE,
                   "no_profiles": not links.unlinked_people().exists()})


@never_cache
def login_view(request):
    if request.user.is_authenticated:
        return redirect("/")
    form = LoginForm(request.POST or None)
    error = ""
    if request.method == "POST" and form.is_valid():
        email = normalise_email(form.cleaned_data["email"])
        if throttle.is_locked(email):
            error = LOCKED
        else:
            user = authenticate(request, username=email,
                                password=form.cleaned_data["password"])
            if user is None:
                error = LOCKED if throttle.record_failure(email) else NOT_AVAILABLE
            else:
                throttle.reset(email)
                login(request, user)
                return redirect(safe_next(request))
    elif request.method == "POST":
        error = NOT_AVAILABLE      # an empty field gets the same one message
    return render(request, "members/login.html",
                  {"form": form, "error": error,
                   "next": request.GET.get("next", "")})


@require_POST
def logout_view(request):
    logout(request)
    return redirect("members:login")


@never_cache
@login_required
def waiting(request):
    """What an account with no linked profile sees: its own claim status, no data."""
    if links.linked_person(request.user) is not None:
        return redirect("/")
    claim = (request.user.claims.order_by("-created_at").first())
    form = ClaimForm() if links.pending_claim(request.user) is None else None
    return render(request, "members/waiting.html", {"claim": claim, "form": form})


@never_cache
@login_required
@require_POST
def claim(request):
    """Ask for a profile from the waiting page (after a rejection or an unlink)."""
    form = ClaimForm(request.POST)
    if form.is_valid():
        try:
            links.create_claim(request.user, form.cleaned_data["person"])
            messages.success(request, "Request sent. An administrator must approve it.")
        except links.LinkError as exc:
            messages.error(request, str(exc))
    else:
        messages.error(request, " ".join(form.errors.get("person", [])
                                         or ["Choose which profile you are."]))
    return redirect("members:waiting")
