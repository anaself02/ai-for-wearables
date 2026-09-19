"""Short-lived cache in front of the Google Health API.

Without this the overview page issues ten API calls on *every* load, including a
plain browser refresh. Nothing in a health feed changes second to second, so a
few minutes of staleness costs nothing and removes almost all of the traffic.

Two rules matter here:

* **Keys are per user.** These are personal health readings; a key collision
  would show one person another's data. The user id is part of every key.
* **Errors are never cached.** Only a value that was returned successfully is
  written back, so a failed call is retried on the next request rather than
  being remembered for the rest of the TTL.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime

from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from . import metrics

logger = logging.getLogger(__name__)

#: Bump when a cached payload's shape changes, to strand old entries.
CACHE_VERSION = 1


@dataclass(frozen=True)
class Fetched:
    """Points plus where they came from, so the UI can show freshness."""

    points: list
    fetched_at: datetime
    from_cache: bool

    @property
    def age_seconds(self) -> float:
        return (timezone.now() - self.fetched_at).total_seconds()


def _key(*parts) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"health:v{CACHE_VERSION}:{digest}"


def _load(key, ttl, produce) -> Fetched:
    if ttl <= 0:  # caching disabled
        return Fetched(produce(), timezone.now(), False)

    payload = cache.get(key)
    if payload is not None:
        return Fetched(payload["points"], payload["fetched_at"], True)

    points = produce()
    now = timezone.now()
    # Only reached when produce() did not raise, so failures stay uncached.
    cache.set(key, {"points": points, "fetched_at": now}, ttl)
    return Fetched(points, now, False)


def fetch_points(client, metric, start: date, end: date, *, user, force=False) -> Fetched:
    """Cached wrapper around metrics.fetch_points."""
    key = _key("metric", user.pk, metric.slug, metric.source, start, end)
    if force:
        cache.delete(key)
    return _load(
        key,
        settings.HEALTH_CACHE_SECONDS,
        lambda: metrics.fetch_points(client, metric, start, end),
    )


#: Pages of samples to request for one civil day.
#:
#: This was 4, and 4 silently truncated every heart-rate day. The heart-rate
#: data type declares ``max_page_size = 10000``, which made 4 pages look like
#: 40000 samples of headroom. The API actually returns about 5000 points per
#: page, so 4 pages capped a day at roughly 20000 samples - and because the
#: API returns the NEWEST samples first, the cap deleted the BEGINNING of the
#: day. Days recorded from midnight came back appearing to start in the late
#: morning, which reads as "the watch was not being worn" rather than as a
#: bug. Measured: the same days return 27000-36000 samples once the limit is
#: raised, starting at 00:00.
#:
#: A full 24 h of heart rate at the observed ~3 s spacing is about 7 pages;
#: 16 leaves headroom without being so large that a runaway costs many
#: requests. Sparser intraday types (HRV, SpO2) never came close to the old
#: limit, so this only ever bit heart rate - but the fix belongs at the
#: default, not at one call site.
DEFAULT_INTRADAY_MAX_PAGES = 16


def fetch_intraday(
    client, data_type, day: date, *, user, force=False,
    max_pages=DEFAULT_INTRADAY_MAX_PAGES,
) -> Fetched:
    """Cached wrapper around a single civil day of raw samples.

    A day that has already ended cannot gain new samples, so it is held far
    longer than today's still-accumulating data.
    """
    from datetime import timedelta

    key = _key("day", user.pk, data_type.slug, day)
    if force:
        cache.delete(key)

    ttl = (
        settings.HEALTH_CACHE_SECONDS_PAST_DAY
        if day < timezone.localdate()
        else settings.HEALTH_CACHE_SECONDS
    )
    return _load(
        key,
        ttl,
        lambda: client.list_data_points(
            data_type, start=day, end=day + timedelta(days=1), max_pages=max_pages
        ),
    )


def wants_refresh(request) -> bool:
    """?refresh=1 forces a re-fetch, for when the watch has just synced."""
    return request.GET.get("refresh") in {"1", "true", "yes"}


def oldest(fetches) -> Fetched | None:
    """The least fresh of several fetches - what the page as a whole is worth."""
    real = [f for f in fetches if f is not None]
    return min(real, key=lambda f: f.fetched_at) if real else None
