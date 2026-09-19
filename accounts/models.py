"""Per-user Google OAuth credentials."""

from datetime import timezone as dt_timezone

from django.conf import settings
from django.db import models
from django.utils import timezone
from google.oauth2.credentials import Credentials

from .fields import EncryptedTextField

GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Refresh a little early rather than racing the expiry on a slow request.
EXPIRY_SKEW_SECONDS = 60


class GoogleCredential(models.Model):
    """OAuth tokens for one Django user's Google account.

    One row per user, so adding more users later needs no schema change.
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="google_credential",
    )
    google_sub = models.CharField(
        max_length=255,
        unique=True,
        help_text="Google's stable account identifier ('sub' claim).",
    )
    email = models.EmailField(blank=True)

    access_token = EncryptedTextField(blank=True, default="")
    refresh_token = EncryptedTextField(blank=True, default="")
    expires_at = models.DateTimeField(null=True, blank=True)
    scopes = models.JSONField(default=list, blank=True)

    needs_reconnect = models.BooleanField(
        default=False,
        help_text=(
            "Set when Google rejects the refresh token. While the OAuth consent "
            "screen is in Testing mode this happens every 7 days."
        ),
    )
    last_refreshed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Google credential"

    def __str__(self):
        return f"Google credential for {self.user}"

    # -- state -------------------------------------------------------------

    @property
    def is_usable(self) -> bool:
        """True if we hold a refresh token and Google has not rejected it."""
        return bool(self.refresh_token) and not self.needs_reconnect

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return True
        return timezone.now() >= self.expires_at

    # -- conversion to/from google-auth ------------------------------------

    def to_credentials(self) -> Credentials:
        """Build a google-auth Credentials object that can refresh itself."""
        expiry = None
        if self.expires_at:
            # google-auth compares expiry against naive UTC datetimes.
            expiry = self.expires_at.astimezone(dt_timezone.utc).replace(tzinfo=None)

        return Credentials(
            token=self.access_token or None,
            refresh_token=self.refresh_token or None,
            token_uri=GOOGLE_TOKEN_URI,
            client_id=settings.GOOGLE_OAUTH_CLIENT_ID,
            client_secret=settings.GOOGLE_OAUTH_CLIENT_SECRET,
            scopes=self.scopes or settings.GOOGLE_OAUTH_SCOPES,
            expiry=expiry,
        )

    def update_from_credentials(self, credentials: Credentials, *, save=True):
        """Copy a fresh access token (and refresh token, if re-issued) back in."""
        self.access_token = credentials.token or ""

        # Google omits refresh_token on refresh responses; keep the one we hold.
        if credentials.refresh_token:
            self.refresh_token = credentials.refresh_token

        if credentials.expiry:
            self.expires_at = credentials.expiry.replace(tzinfo=dt_timezone.utc)
        else:
            self.expires_at = None

        if credentials.scopes:
            self.scopes = list(credentials.scopes)

        self.needs_reconnect = False
        self.last_refreshed_at = timezone.now()
        if save:
            self.save()

    def mark_needs_reconnect(self):
        self.needs_reconnect = True
        self.access_token = ""
        self.save(update_fields=["needs_reconnect", "access_token", "updated_at"])
