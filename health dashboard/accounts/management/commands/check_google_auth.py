"""Verify the configured OAuth client credentials actually work.

Useful after rotating the client secret: it forces a real token refresh against
Google and then makes one live API call, so a wrong secret is caught in a couple
of seconds instead of during a browser sign-in.

    .venv\\Scripts\\python.exe manage.py check_google_auth

Nothing secret is printed - values are masked.
"""

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request

from accounts.models import GoogleCredential
from health import datatypes
from health.client import GoogleHealthClient, GoogleHealthError, GoogleReconnectRequired


def mask(value: str, keep: int = 6) -> str:
    if not value:
        return "(unset)"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}...{'*' * 6} ({len(value)} chars)"


class Command(BaseCommand):
    help = "Check that the configured Google OAuth client credentials still work."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            help="Username to check. Defaults to the only user with stored credentials.",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("Configuration"))
        self.stdout.write(f"  client id      {mask(settings.GOOGLE_OAUTH_CLIENT_ID, 24)}")
        self.stdout.write(f"  client secret  {mask(settings.GOOGLE_OAUTH_CLIENT_SECRET)}")
        self.stdout.write(f"  redirect uri   {settings.GOOGLE_OAUTH_REDIRECT_URI}")
        self.stdout.write(f"  time zone      {settings.TIME_ZONE}")
        self.stdout.write(f"  force ipv4     {settings.FORCE_IPV4}")

        credentials = GoogleCredential.objects.select_related("user")
        if options["user"]:
            credentials = credentials.filter(user__username=options["user"])
        record = credentials.first()
        if record is None:
            raise CommandError("No stored Google credentials. Sign in through the app first.")

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Stored credential for {record.user}"))
        self.stdout.write(f"  refresh token  {mask(record.refresh_token)}")
        self.stdout.write(f"  expires at     {record.expires_at}")
        self.stdout.write(f"  reconnect flag {record.needs_reconnect}")

        # -- force a refresh ------------------------------------------------
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Refreshing the access token"))
        google_credentials = record.to_credentials()
        try:
            google_credentials.refresh(Request())
        except RefreshError as exc:
            detail = str(exc)
            if "invalid_client" in detail:
                raise CommandError(
                    "Google rejected the client credentials (invalid_client).\n"
                    "  The GOOGLE_OAUTH_CLIENT_SECRET in .env does not match this client.\n"
                    "  Note that runserver's autoreloader does not re-read .env - restart it."
                )
            if "invalid_grant" in detail:
                record.mark_needs_reconnect()
                raise CommandError(
                    "Google rejected the refresh token (invalid_grant).\n"
                    "  Expected roughly weekly while the consent screen is in Testing mode.\n"
                    "  Sign in again through the app to get a fresh one."
                )
            raise CommandError(f"Refresh failed: {exc}")

        record.update_from_credentials(google_credentials)
        self.stdout.write(self.style.SUCCESS("  OK - new access token issued"))
        self.stdout.write(f"  expires at     {record.expires_at}")

        # -- one live API call ----------------------------------------------
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Calling the Google Health API"))
        data_type = datatypes.get("daily-resting-heart-rate")
        today = timezone.localdate()
        try:
            points = GoogleHealthClient(record).list_data_points(
                data_type, start=today - timedelta(days=7), end=today + timedelta(days=1)
            )
        except (GoogleHealthError, GoogleReconnectRequired) as exc:
            raise CommandError(f"API call failed: {exc}")

        self.stdout.write(
            self.style.SUCCESS(
                f"  OK - {data_type.slug} returned {len(points)} data point"
                f"{'' if len(points) == 1 else 's'}"
            )
        )
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Credentials are working."))
