"""Sign in with Google: start the flow, handle the callback, sign out."""

import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.utils import timezone
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from .forms import ProfileForm
from .greeting import display_name, greeting_for
from .models import GoogleCredential, Profile
from .oauth import authorization_url, build_flow, verify_id_token

logger = logging.getLogger(__name__)
User = get_user_model()

SESSION_STATE_KEY = "google_oauth_state"
SESSION_VERIFIER_KEY = "google_oauth_code_verifier"


def landing(request):
    """Signed-out home page: nothing but a Sign in with Google button."""
    if request.user.is_authenticated:
        return redirect("health:index")
    return render(request, "accounts/landing.html")


@require_http_methods(["GET", "POST"])
def google_login(request):
    """Send the user to Google's consent screen."""
    url, state, code_verifier = authorization_url()
    request.session[SESSION_STATE_KEY] = state
    # PKCE: the verifier behind the code_challenge we just sent Google has to be
    # replayed at the token exchange, which happens on the callback request.
    request.session[SESSION_VERIFIER_KEY] = code_verifier
    return redirect(url)


def oauth_callback(request):
    """Handle Google's redirect back: exchange the code, sign the user in."""
    error = request.GET.get("error")
    if error:
        logger.warning("Google returned an OAuth error: %s", error)
        return render(
            request,
            "accounts/oauth_error.html",
            {"detail": f"Google reported: {error}"},
            status=400,
        )

    expected_state = request.session.pop(SESSION_STATE_KEY, None)
    code_verifier = request.session.pop(SESSION_VERIFIER_KEY, None)
    received_state = request.GET.get("state")
    if not expected_state or expected_state != received_state:
        # Mismatched state means this callback was not started by this session.
        return render(
            request,
            "accounts/oauth_error.html",
            {"detail": "OAuth state did not match. Start the sign-in again."},
            status=400,
        )

    if not code_verifier:
        # The session lost the PKCE verifier - usually a server restart with
        # DEBUG reloading, or cookies cleared mid-flow.
        return render(
            request,
            "accounts/oauth_error.html",
            {"detail": "The sign-in session expired. Start the sign-in again."},
            status=400,
        )

    flow = build_flow(state=expected_state, code_verifier=code_verifier)

    # Rebuild the callback URL from settings rather than from the request, so a
    # visit to 127.0.0.1 instead of localhost cannot cause a redirect_uri
    # mismatch against the value registered in Google Cloud Console.
    authorization_response = (
        f"{settings.GOOGLE_OAUTH_REDIRECT_URI}?{request.META.get('QUERY_STRING', '')}"
    )

    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as exc:  # oauthlib raises a wide family of errors here
        logger.exception("Token exchange failed")
        return render(
            request,
            "accounts/oauth_error.html",
            {"detail": f"Token exchange failed: {exc}"},
            status=400,
        )

    credentials = flow.credentials

    if not credentials.refresh_token:
        logger.warning("Google did not return a refresh token")

    try:
        claims = verify_id_token(credentials.id_token)
    except Exception as exc:
        logger.exception("ID token verification failed")
        return render(
            request,
            "accounts/oauth_error.html",
            {"detail": f"Could not verify Google identity: {exc}"},
            status=400,
        )

    user = _get_or_create_user(claims)

    credential, _ = GoogleCredential.objects.get_or_create(
        user=user, defaults={"google_sub": claims["sub"]}
    )
    # Google owns the display name, and _get_or_create_user only writes it
    # when the row is new, so a name changed later would never arrive.
    given = claims.get("given_name", "")[:150]
    if given and given != user.first_name:
        user.first_name = given
        user.save(update_fields=["first_name"])

    credential.google_sub = claims["sub"]
    credential.email = claims.get("email", "")
    credential.update_from_credentials(credentials)

    login(request, user)
    messages.success(request, "Google account connected.")
    return redirect("accounts:welcome")


def _get_or_create_user(claims: dict) -> User:
    """Find the Django user behind a set of verified Google ID-token claims.

    'sub' is Google's stable account id and is matched first; email is only a
    fallback for a user created before this credential row existed.
    """
    existing = GoogleCredential.objects.filter(google_sub=claims["sub"]).select_related("user").first()
    if existing:
        return existing.user

    email = claims.get("email", "")
    user, created = User.objects.get_or_create(
        username=email or claims["sub"],
        defaults={
            "email": email,
            "first_name": claims.get("given_name", "")[:150],
            "last_name": claims.get("family_name", "")[:150],
        },
    )
    if created:
        # Sign-in is via Google only; there is no local password to log in with.
        user.set_unusable_password()
        user.save(update_fields=["password"])
    return user


@require_http_methods(["POST"])
def sign_out(request):
    logout(request)
    return redirect(reverse("accounts:landing"))


@login_required
@require_http_methods(["GET", "POST"])
def profile_settings(request):
    """Date of birth and sex, which the fitness norms cannot work without.

    Nothing else lives here. The dashboard reads everything else from the API,
    and asking for more than the comparison needs would be collecting personal
    data for no stated purpose.
    """
    profile, _ = Profile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Saved.")
            return redirect("health:fitness")
    else:
        form = ProfileForm(instance=profile)

    return render(request, "accounts/settings.html", {"form": form, "profile": profile})


#: How long the greeting stays up at minimum, in milliseconds.
#:
#: This is a legibility floor, not a loading time. When the cache is already
#: warm the fetch behind this screen finishes in a few hundred milliseconds,
#: and at that speed the greeting is a flicker on the way to somewhere else -
#: which is worse than not greeting anybody. Two seconds is a full cycle of
#: the loader plus enough to read a short line.
#:
#: It is worth being straight about the trade: on a warm cache this is time
#: the visitor waits for the sake of reading the screen rather than for work.
#: That is a different thing from a timer dressed up as loading, which is what
#: the ceiling below and the fetch itself exist to avoid.
WELCOME_FLOOR_MS = 2000

#: The longest this screen may hold somebody when the fetch is slow or failing.
WELCOME_CEILING_MS = 8000


@login_required
def welcome(request):
    """The moment between signing in and the dashboard being ready.

    Straight after the OAuth round trip the cache is cold, and the overview is
    one API call per metric. This screen covers work that is genuinely
    happening rather than a timer pretending something is: the page renders
    immediately, then asks for the fetch to be done, then leaves.
    """
    return render(
        request,
        "accounts/welcome.html",
        {
            "greeting": greeting_for(timezone.localtime()),
            "name": display_name(request.user),
            "floor_ms": WELCOME_FLOOR_MS,
            "ceiling_ms": WELCOME_CEILING_MS,
        },
    )


@login_required
def welcome_warm(request):
    """Fill the cache the dashboard is about to read. Always answers 204.

    Failure here is not worth reporting to the visitor: every error this can
    raise is one the dashboard already handles per card, and the only thing a
    failure changes is that the next page is slow rather than fast. What it
    must never do is strand somebody on an interstitial.
    """
    from health.views import OVERVIEW_WINDOW_DAYS, _load_window

    try:
        _load_window(request, OVERVIEW_WINDOW_DAYS)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Warming the dashboard cache failed: %s", exc)

    return HttpResponse(status=204)
