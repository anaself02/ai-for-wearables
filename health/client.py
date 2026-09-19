"""Thin client over the Google Health API v4 REST surface.

google-auth's AuthorizedSession attaches the bearer token and refreshes it
using the stored refresh token when it has expired, so nothing here performs a
token exchange by hand. Whatever new access token a refresh produces is written
straight back to the database.
"""

import logging
import threading
from datetime import date

from django.conf import settings
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import AuthorizedSession

from .datatypes import DataType

logger = logging.getLogger(__name__)

# Separate connect and read budgets. A scalar timeout applies to BOTH, and the
# connect half is spent per candidate address, so a host with several dead
# addresses can stall for a multiple of it before any read begins.
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)


class GoogleReconnectRequired(Exception):
    """Google rejected the stored refresh token; the user must consent again.

    Expected roughly weekly while the OAuth consent screen sits in Testing
    mode, because Google expires those refresh tokens after 7 days.
    """


class GoogleHealthError(Exception):
    """The API returned an error response."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _civil(day: date) -> dict:
    """A CivilDateTime covering the start of the given calendar day."""
    return {"date": {"year": day.year, "month": day.month, "day": day.day}}


class GoogleHealthClient:
    def __init__(self, credential):
        if not credential.is_usable:
            raise GoogleReconnectRequired(
                "No usable Google refresh token is stored for this user."
            )
        self.credential = credential
        self._credentials = credential.to_credentials()
        self._session = AuthorizedSession(self._credentials)
        self._last_seen_token = self._credentials.token
        # The dashboard fetches several metrics in parallel over one client.
        # AuthorizedSession already locks its own refresh; this guards the
        # database write that mirrors the new token back.
        self._token_lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    def _persist_refreshed_token(self):
        with self._token_lock:
            if self._credentials.token != self._last_seen_token:
                logger.info("Access token refreshed for %s", self.credential.user)
                self.credential.update_from_credentials(self._credentials)
                self._last_seen_token = self._credentials.token

    def _call(self, method: str, path: str, **kwargs):
        url = f"{settings.GOOGLE_HEALTH_API_BASE}{path}"
        try:
            response = self._session.request(
                method, url, timeout=REQUEST_TIMEOUT, **kwargs
            )
        except RefreshError as exc:
            logger.warning("Refresh token rejected for %s: %s", self.credential.user, exc)
            self.credential.mark_needs_reconnect()
            raise GoogleReconnectRequired(
                "Google rejected the stored refresh token."
            ) from exc
        finally:
            self._persist_refreshed_token()

        if response.status_code == 401:
            self.credential.mark_needs_reconnect()
            raise GoogleReconnectRequired("Google returned 401 for this access token.")

        if not response.ok:
            raise GoogleHealthError(
                f"{method} {path} returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text[:2000],
            )

        return response.json()

    # -- reads -------------------------------------------------------------

    def list_data_points(
        self,
        data_type: DataType,
        *,
        start: date | None = None,
        end: date | None = None,
        page_size: int | None = None,
        max_pages: int = 1,
    ) -> list[dict]:
        """GET .../dataTypes/{slug}/dataPoints, following up to max_pages pages."""
        params = {
            "pageSize": min(page_size or data_type.max_page_size, data_type.max_page_size)
        }
        if start and end:
            params["filter"] = data_type.time_filter(start, end)

        path = f"/users/me/dataTypes/{data_type.slug}/dataPoints"
        points: list[dict] = []
        page_token = None

        for _ in range(max_pages):
            if page_token:
                params["pageToken"] = page_token
            payload = self._call("GET", path, params=params)
            points.extend(payload.get("dataPoints", []))
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return points

    def daily_roll_up(
        self,
        data_type: DataType,
        *,
        start: date,
        end: date,
        window_size_days: int = 1,
    ) -> list[dict]:
        """POST .../dataPoints:dailyRollUp - server-side aggregation per civil day.

        The range is closed-open, so `end` should be the day after the last one
        you want back. Preferred over summing intraday points client-side.
        """
        if not data_type.supports_daily_rollup:
            raise ValueError(f"{data_type.slug} does not support dailyRollUp")

        span_days = (end - start).days
        if span_days > data_type.rollup_max_days:
            raise ValueError(
                f"{data_type.slug} allows at most {data_type.rollup_max_days} days "
                f"per dailyRollUp request; asked for {span_days}."
            )

        body = {
            "range": {"start": _civil(start), "end": _civil(end)},
            "windowSizeDays": window_size_days,
        }
        path = f"/users/me/dataTypes/{data_type.slug}/dataPoints:dailyRollUp"
        payload = self._call("POST", path, json=body)
        return payload.get("rollupDataPoints", [])
