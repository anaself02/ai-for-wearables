"""Helpers around the Google OAuth 2.0 authorization-code flow.

The token exchange itself is done by google-auth-oauthlib rather than by hand,
so redirect validation, PKCE-less code exchange and ID-token verification all
follow Google's own implementation.
"""

import google.auth.transport.requests
from django.conf import settings
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow

from .models import GOOGLE_TOKEN_URI

AUTH_URI = "https://accounts.google.com/o/oauth2/auth"

# Allow a little clock drift between this machine and Google when checking the
# ID token's iat/exp claims.
CLOCK_SKEW_SECONDS = 10


def _client_config() -> dict:
    """The in-memory equivalent of a downloaded client_secret.json."""
    return {
        "web": {
            "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
            "auth_uri": AUTH_URI,
            "token_uri": GOOGLE_TOKEN_URI,
            "redirect_uris": [settings.GOOGLE_OAUTH_REDIRECT_URI],
        }
    }


def build_flow(state: str | None = None, code_verifier: str | None = None) -> Flow:
    """Construct a Flow.

    ``code_verifier`` must be supplied when rebuilding the flow to handle the
    callback: the verifier generated for the authorization request has to be
    replayed at the token exchange, and a fresh Flow object does not have it.
    """
    return Flow.from_client_config(
        _client_config(),
        scopes=settings.GOOGLE_OAUTH_SCOPES,
        redirect_uri=settings.GOOGLE_OAUTH_REDIRECT_URI,
        state=state,
        code_verifier=code_verifier,
    )


def authorization_url() -> tuple[str, str, str]:
    """Return (url, state, code_verifier) for Google's consent screen.

    access_type=offline is what makes Google issue a refresh token at all.
    prompt=consent forces a refresh token on *every* run: without it Google
    returns one only on the very first authorization for this client, which
    makes the app impossible to re-test after the database is reset.

    google-auth-oauthlib performs PKCE by default: authorization_url() generates
    a code_verifier on the flow and sends its S256 challenge to Google. Both the
    state and that verifier have to outlive this request, because the callback
    arrives on a different request with a different Flow object.
    """
    flow = build_flow()
    url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return url, state, flow.code_verifier


def verify_id_token(raw_id_token: str) -> dict:
    """Validate the ID token's signature and audience, returning its claims."""
    return id_token.verify_oauth2_token(
        raw_id_token,
        google.auth.transport.requests.Request(),
        audience=settings.GOOGLE_OAUTH_CLIENT_ID,
        clock_skew_in_seconds=CLOCK_SKEW_SECONDS,
    )
