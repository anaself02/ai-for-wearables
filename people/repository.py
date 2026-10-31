"""Reading a person's stored heart rate. Every function takes a Person.

This module is the only way views and the arousal code reach stored readings,
and it is built so that "whose data is this?" is never implicit: every function
requires a `Person` instance and filters on it. Passing a slug, an id or
nothing raises, so a forgotten argument fails loudly instead of quietly
returning someone else's rows.

Local days are cut in the person's own timezone from real midnights, so a
daylight-saving day is 23 or 25 hours long rather than being assumed to be 24.
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .models import SECONDS_PER_DAY, HeartRateDay, Person


def require_person(person):
    if not isinstance(person, Person):
        raise TypeError(
            "A Person is required (got {!r}). Readings are never read without "
            "naming whose they are.".format(person))
    return person


# ---------------------------------------------------------------------------
# local-day arithmetic
# ---------------------------------------------------------------------------

def local_day_bounds(day, tz_name):
    """(start_epoch, end_epoch) in UTC seconds for one local civil day."""
    tz = ZoneInfo(tz_name)
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    nxt = day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=tz)
    return int(start.timestamp()), int(end.timestamp())


def utc_dates_for(day, tz_name):
    """The UTC dates whose stored days can hold readings of this local day."""
    start, end = local_day_bounds(day, tz_name)
    first = datetime.fromtimestamp(start, timezone.utc).date()
    last = datetime.fromtimestamp(end - 1, timezone.utc).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def local_day_readings(person, day):
    """(epoch seconds, bpm) of one local day, sorted. Empty arrays if none."""
    require_person(person)
    start, end = local_day_bounds(day, person.timezone)
    rows = HeartRateDay.objects.filter(
        person=person, utc_date__in=utc_dates_for(day, person.timezone))

    epochs, values = [], []
    for row in rows:
        e, b = row.unpack()
        epochs.append(e)
        values.append(b)
    if not epochs:
        return np.array([], dtype="int64"), np.array([], dtype=float)

    epochs, values = np.concatenate(epochs), np.concatenate(values)
    order = np.argsort(epochs, kind="stable")
    epochs, values = epochs[order], values[order]
    keep = (epochs >= start) & (epochs < end)
    return epochs[keep], values[keep]


def local_dates_with_data(person):
    """Local dates (in the person's timezone) that hold at least one reading."""
    require_person(person)
    tz = ZoneInfo(person.timezone)
    dates = set()
    for first, last in person.heart_rate_days.values_list("first_utc", "last_utc"):
        d1, d2 = first.astimezone(tz).date(), last.astimezone(tz).date()
        d = d1
        while d <= d2:
            dates.add(d)
            d += timedelta(days=1)
    return sorted(dates)


def spacing(epochs):
    """Seconds between consecutive readings: median, p05, p95, gaps over 5 min."""
    gaps = np.diff(epochs)
    if len(gaps) == 0:
        return {"median": None, "p05": None, "p95": None, "gaps_over_5min": 0}
    return {"median": float(np.median(gaps)),
            "p05": float(np.percentile(gaps, 5)),
            "p95": float(np.percentile(gaps, 95)),
            "gaps_over_5min": int(np.sum(gaps > 300))}


def summarise_local_day(person, day):
    """Computed-from-readings summary of one local day, or None if empty."""
    epochs, bpm = local_day_readings(person, day)
    if len(epochs) == 0:
        return None
    tz = ZoneInfo(person.timezone)
    return {
        "day": day,
        "n": int(len(epochs)),
        "mean": float(np.mean(bpm)),
        "min": float(np.min(bpm)),
        "max": float(np.max(bpm)),
        "first_local": datetime.fromtimestamp(int(epochs[0]), timezone.utc).astimezone(tz),
        "last_local": datetime.fromtimestamp(int(epochs[-1]), timezone.utc).astimezone(tz),
        "spacing": spacing(epochs),
    }


def recent_day_summaries(person, limit=14):
    """Summaries of the newest `limit` local days that have data, newest first."""
    require_person(person)
    days = sorted(local_dates_with_data(person), reverse=True)[:limit]
    summaries = [summarise_local_day(person, d) for d in days]
    return [s for s in summaries if s]


def totals(person):
    """{'utc_days', 'readings', 'devices'} for the chooser and headers."""
    from django.db.models import Sum

    require_person(person)
    agg = person.heart_rate_days.aggregate(readings=Sum("n"))
    devices = {}
    for sources in person.heart_rate_days.values_list("sources", flat=True):
        for name, count in sources.items():
            devices[name] = devices.get(name, 0) + count
    return {"utc_days": person.heart_rate_days.count(),
            "readings": agg["readings"] or 0,
            "devices": devices}
