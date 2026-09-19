"""Dashboard views.

The overview page needs one API call per metric. Done sequentially that is
about ten seconds of page load, so the calls are fanned out over a thread pool
sharing a single client - google-auth's AuthorizedSession locks its own token
refresh, and GoogleHealthClient locks the database write that mirrors it.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone

from . import cache as health_cache
from . import datatypes, metrics
from .client import GoogleHealthClient, GoogleHealthError, GoogleReconnectRequired
from .models import HistoricalMetric

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
ALLOWED_WINDOWS = [7, 30, 90]
MAX_WINDOW_DAYS = 90

# Overview cards only ever show a short trend; the detail page does the long ones.
OVERVIEW_WINDOW_DAYS = 7

# A week of `steps` is ~2.8 MB of pretty-printed JSON, which is enough to make a
# browser give up on the <pre> block. The raw pages exist to prove the pipeline,
# so they show a representative slice and report the real count.
MAX_RENDERED_POINTS = 50

MAX_PARALLEL_FETCHES = 8


def _window(request, default=DEFAULT_WINDOW_DAYS):
    """Read ?days=N, clamped. Half-open: end is exclusive, so today is included.

    `timezone.localdate()` resolves "today" in settings.TIME_ZONE, which is the
    same zone the API stamps its civil dates in. `date.today()` would use the
    host clock instead, and the two disagree for several hours every night.
    """
    try:
        days = int(request.GET.get("days", default))
    except (TypeError, ValueError):
        days = default
    days = max(1, min(days, MAX_WINDOW_DAYS))
    end = timezone.localdate() + timedelta(days=1)
    return days, end - timedelta(days=days), end


def _credential_or_none(request):
    return getattr(request.user, "google_credential", None)


def _latest_historical(user):
    """Newest imported reading per historical metric type, keyed by type.

    These are one-time Takeout imports, never live sync, and the templates always
    label them as such so they cannot be read as current data.
    """
    wanted = {m.historical_metric_type for m in metrics.METRICS if m.historical_metric_type}
    return {
        metric_type: HistoricalMetric.latest_for(user, metric_type)
        for metric_type in wanted
    }


def _historical_series(readings, label, unit):
    return metrics.chart_data(
        [reading.date.isoformat() for reading in readings],
        [{"label": label, "data": [reading.value for reading in readings], "unit": unit}],
    )


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------


@login_required
def index(request):
    credential = _credential_or_none(request)
    if credential is None or not credential.is_usable:
        return render(
            request,
            "health/reconnect.html",
            {"detail": "No usable Google token is stored for this account."},
        )

    days, start, end = _window(request, OVERVIEW_WINDOW_DAYS)

    try:
        client = GoogleHealthClient(credential)
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})

    force = health_cache.wants_refresh(request)

    def load(metric):
        try:
            fetched = health_cache.fetch_points(
                client, metric, start, end, user=request.user, force=force
            )
            return metric, fetched, None
        except GoogleReconnectRequired:
            raise
        except GoogleHealthError as exc:
            logger.warning("%s failed: %s", metric.slug, exc)
            return metric, None, str(exc)

    targets = metrics.fetchable()
    try:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as pool:
            results = list(pool.map(load, targets))
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})

    loaded = {
        metric.slug: (fetched.points if fetched else [], error)
        for metric, fetched, error in results
    }
    # The page is only as fresh as its least fresh card.
    freshness = health_cache.oldest([fetched for _, fetched, _ in results])
    historical = _latest_historical(request.user)

    cards = []
    for metric in metrics.METRICS:
        card = {"metric": metric, "error": None, "value": None, "value_label": None}

        if metric.group == metrics.UNAVAILABLE:
            # No live data by definition; fall back to a labelled legacy reading.
            card["historical"] = historical.get(metric.historical_metric_type)
            cards.append(card)
            continue

        points, error = loaded.get(metric.slug, ([], None))
        card["error"] = error
        series = metric.series(points) if metric.series else None
        card["series"] = series
        card["series_json"] = json.dumps(series) if series else None
        card["count"] = len(points)

        if metric.slug == "exercise":
            card["value"] = len(points)
            card["value_label"] = f"in {days} days"
        elif metric.slug == "sleep":
            rows = metric.rows(points) if metric.rows else []
            card["value"] = rows[0]["asleep_hm"] if rows else None
            card["value_label"] = rows[0]["date"] if rows else None
        else:
            value, when = metrics.latest_value(series)
            card["value"] = value
            card["value_label"] = when

        cards.append(card)

    return render(
        request,
        "health/dashboard.html",
        {
            "cards": cards,
            "days": days,
            "start": start,
            "end": end,
            "credential": credential,
            "freshness": freshness,
        },
    )


# ---------------------------------------------------------------------------
# per-metric detail
# ---------------------------------------------------------------------------


@login_required
def metric_detail(request, slug):
    metric = metrics.get(slug)
    if metric is None:
        raise Http404(f"Unknown metric {slug!r}")

    context = {
        "metric": metric,
        "windows": ALLOWED_WINDOWS,
        "data_type": metric.data_type if metric.has_detail else None,
    }

    if metric.group == metrics.UNAVAILABLE:
        # Confirmed empty rather than broken, so this renders without any API call.
        # Anything imported from Takeout is shown here, clearly marked as legacy.
        readings = HistoricalMetric.series_for(request.user, metric.historical_metric_type)
        series = (
            _historical_series(readings, "VO2 max (legacy)", readings[0].unit)
            if readings
            else None
        )
        context.update(
            {
                "historical": readings[-1] if readings else None,
                "historical_count": len(readings),
                "historical_first": readings[0] if readings else None,
                "series": series,
                "series_json": json.dumps(series) if series else None,
            }
        )
        return render(request, "health/metric_unavailable.html", context)

    days, start, end = _window(request)
    context.update({"days": days, "start": start, "end": end})

    credential = _credential_or_none(request)
    if credential is None:
        return redirect("accounts:landing")

    try:
        client = GoogleHealthClient(credential)
        fetched = health_cache.fetch_points(
            client,
            metric,
            start,
            end,
            user=request.user,
            force=health_cache.wants_refresh(request),
        )
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})
    except GoogleHealthError as exc:
        context["error"] = str(exc)
        context["error_body"] = exc.body
        return render(request, "health/metric_detail.html", context)

    points = fetched.points
    context["freshness"] = fetched
    series = metric.series(points) if metric.series else None
    value, when = metrics.latest_value(series)

    context.update(
        {
            "count": len(points),
            "series": series,
            "series_json": json.dumps(series) if series else None,
            "rows": metric.rows(points) if metric.rows else None,
            "value": value,
            "value_label": when,
            "day_options": _day_options(series) if metric.intraday_slug else None,
        }
    )
    return render(request, "health/metric_detail.html", context)


def _day_options(series):
    """Most recent days first, for the drill-down picker."""
    if not series:
        return []
    return list(reversed(series.get("labels", [])))[:30]


# ---------------------------------------------------------------------------
# single-day intraday drill-down
# ---------------------------------------------------------------------------


@login_required
def metric_day(request, slug, day):
    metric = metrics.get(slug)
    if metric is None or not metric.intraday_slug:
        raise Http404(f"No intraday view for {slug!r}")

    try:
        target = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError:
        raise Http404("Expected a date as YYYY-MM-DD")

    credential = _credential_or_none(request)
    if credential is None:
        return redirect("accounts:landing")

    data_type = metric.intraday_type
    extract = metrics.INTRADAY_EXTRACTORS.get(metric.intraday_slug)

    context = {
        "metric": metric,
        "day": target,
        "prev_day": target - timedelta(days=1),
        "next_day": target + timedelta(days=1),
        "data_type": data_type,
    }

    try:
        client = GoogleHealthClient(credential)
        # One civil day, full resolution: several pages of samples. A day that
        # has already ended is cached for much longer than today.
        fetched = health_cache.fetch_intraday(
            client,
            data_type,
            target,
            user=request.user,
            force=health_cache.wants_refresh(request),
        )
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})
    except GoogleHealthError as exc:
        context["error"] = str(exc)
        return render(request, "health/metric_day.html", context)

    points = fetched.points
    context["freshness"] = fetched
    series = extract(points) if extract else None
    context["count"] = len(points)
    context["series"] = series
    context["series_json"] = json.dumps(series) if series else None
    return render(request, "health/metric_day.html", context)


# ---------------------------------------------------------------------------
# raw JSON pages (kept from step 2 as a debugging aid)
# ---------------------------------------------------------------------------


@login_required
def raw_index(request):
    return render(
        request,
        "health/raw_index.html",
        {"data_types": datatypes.all_types(), "window_days": DEFAULT_WINDOW_DAYS},
    )


@login_required
def raw_data_points(request, slug):
    data_type = datatypes.get(slug)
    if data_type is None:
        raise Http404(f"Unknown data type {slug!r}")

    days, start, end = _window(request)

    credential = _credential_or_none(request)
    if credential is None:
        return redirect("accounts:landing")

    context = {
        "data_type": data_type,
        "days": days,
        "start": start,
        "end": end,
        "filter_expression": data_type.time_filter(start, end),
        "request_url": (
            f"GET https://health.googleapis.com/v4"
            f"/users/me/dataTypes/{data_type.slug}/dataPoints"
        ),
    }

    try:
        points = GoogleHealthClient(credential).list_data_points(
            data_type, start=start, end=end
        )
    except GoogleReconnectRequired as exc:
        logger.info("Reconnect required for %s: %s", request.user, exc)
        return render(request, "health/reconnect.html", {"detail": str(exc)})
    except GoogleHealthError as exc:
        context["error"] = str(exc)
        context["error_body"] = exc.body
        return render(request, "health/raw.html", context)

    shown = points[:MAX_RENDERED_POINTS]
    context["count"] = len(points)
    context["shown"] = len(shown)
    context["truncated"] = len(points) > len(shown)
    context["payload"] = json.dumps(shown, indent=2, ensure_ascii=False)
    return render(request, "health/raw.html", context)
