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

from . import arousal as arousal_module
from . import cache as health_cache
from . import datatypes, insights as insights_module, metrics
from . import fitness as fitness_module
from . import sleep_cycles
from .client import GoogleHealthClient, GoogleHealthError, GoogleReconnectRequired
from .models import HistoricalMetric

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
ALLOWED_WINDOWS = [7, 30, 90]
MAX_WINDOW_DAYS = 90

# The overview fetches one long window and slices it: the cards' sparklines want
# a week, the insight layer wants a month, and issuing both would double the API
# calls for the same data.
OVERVIEW_WINDOW_DAYS = 30
SPARKLINE_POINTS = 7

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


def _load_window(request, window_days):
    """Fetch every metric for one window, honouring the cache.

    The overview and the insights page want the same month of points and share
    a cache key, so whichever of the two is opened second costs no API calls.

    Returns (data, response). Exactly one of them is not None: a rendered
    reconnect page when the stored credential cannot be used, otherwise the
    loaded points.
    """
    credential = _credential_or_none(request)
    if credential is None or not credential.is_usable:
        return None, render(
            request,
            "health/reconnect.html",
            {"detail": "No usable Google token is stored for this account."},
        )

    days, start, end = _window(request, window_days)

    try:
        client = GoogleHealthClient(credential)
    except GoogleReconnectRequired as exc:
        return None, render(request, "health/reconnect.html", {"detail": str(exc)})

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
        return None, render(request, "health/reconnect.html", {"detail": str(exc)})

    return {
        "credential": credential,
        "days": days,
        "start": start,
        "end": end,
        "loaded": {
            metric.slug: (fetched.points if fetched else [], error)
            for metric, fetched, error in results
        },
        # The page is only as fresh as its least fresh card.
        "freshness": health_cache.oldest([fetched for _, fetched, _ in results]),
    }, None


@login_required
def index(request):
    """The overview: one card per metric, and nothing that was inferred.

    Everything on this page was measured by the watch and returned by the API.
    What the dashboard derives from those measurements lives on the insights
    page, where it cannot be mistaken for a reading.
    """
    data, response = _load_window(request, OVERVIEW_WINDOW_DAYS)
    if response is not None:
        return response

    loaded = data["loaded"]
    days = data["days"]
    historical = _latest_historical(request.user)

    cards = []
    for metric in metrics.METRICS:
        card = {"metric": metric, "error": None, "value": None, "value_label": None}

        if not metric.is_api_backed:
            # UNAVAILABLE (endpoint exists, returns nothing) and LEGACY (no
            # endpoint at all) both render from stored history, never a call.
            card["historical"] = historical.get(metric.historical_metric_type)
            cards.append(card)
            continue

        points, error = loaded.get(metric.slug, ([], None))
        card["error"] = error
        series = metric.series(points) if metric.series else None
        # The card shows a week; the same fetch feeds the month-long insights.
        spark = metrics.tail(series, SPARKLINE_POINTS)
        card["series"] = spark
        card["series_json"] = json.dumps(spark) if spark else None
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
            "start": data["start"],
            "end": data["end"],
            "credential": data["credential"],
            "freshness": data["freshness"],
        },
    )


@login_required
def insights(request):
    """Everything the dashboard infers rather than measures.

    Deliberately not on the overview. A reading and an interpretation of a
    reading are different kinds of claim, and putting them in one grid invites
    the second to borrow the authority of the first. The arousal signal lives
    here for the same reason.
    """
    data, response = _load_window(request, OVERVIEW_WINDOW_DAYS)
    if response is not None:
        return response

    # Derived observations over the whole fetched window. Each insight decides
    # for itself whether it has enough data, so one thin metric never blanks
    # the page.
    points_by_slug = {slug: points for slug, (points, _) in data["loaded"].items()}
    computed = insights_module.build(metrics.insight_series(points_by_slug))

    return render(
        request,
        "health/insights.html",
        {
            "days": data["days"],
            "start": data["start"],
            "end": data["end"],
            "freshness": data["freshness"],
            "insights": insights_module.available(computed),
            "insights_pending": insights_module.pending(computed),
            "methodology": insights_module.methodology(computed),
            # Read from the cache only - never computed here. Running the
            # model costs several pages of samples plus a few thousand
            # windows of feature extraction, which is not something to do
            # behind every page load. The card offers a button instead.
            "arousal": arousal_module.cached_assessment(
                request.user, timezone.localdate()),
            "arousal_day": timezone.localdate(),
        },
    )


# ---------------------------------------------------------------------------
# the arousal card: running the WESAD-trained model against live data
# ---------------------------------------------------------------------------


@login_required
def arousal_compute(request):
    """Compute today's arousal assessment on demand, then show the dashboard.

    POST only: it spends real work and writes a cache entry, so it should not
    be triggered by a link preview or a refresh of a GET.
    """
    if request.method != "POST":
        return redirect("health:insights")

    credential = _credential_or_none(request)
    if credential is None or not credential.is_usable:
        return render(
            request,
            "health/reconnect.html",
            {"detail": "No usable Google token is stored for this account."},
        )

    try:
        client = GoogleHealthClient(credential)
        arousal_module.compute_assessment(
            client,
            request.user,
            timezone.localdate(),
            force=health_cache.wants_refresh(request),
        )
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})
    except GoogleHealthError as exc:
        # The card renders the reason rather than the page failing.
        logger.warning("Arousal fetch failed: %s", exc)
        arousal_module.store_unavailable(
            request.user, timezone.localdate(),
            "Heart-rate data could not be fetched: {}".format(exc))

    return redirect("health:insights")


@login_required
def arousal_methodology(request):
    """The honesty layer: what the model is, and what it cannot tell you.

    Rendered from the model's own card, written by the training project, so
    the figures here cannot drift away from that project's RESULTS.md.
    """
    return render(
        request,
        "health/arousal_methodology.html",
        {
            "card": arousal_module.load_model_card(),
            "window_seconds": arousal_module.WINDOW_SECONDS,
            "step_seconds": arousal_module.STEP_SECONDS,
            "min_samples": arousal_module.MIN_SAMPLES_PER_WINDOW,
            "assessment": arousal_module.cached_assessment(
                request.user, timezone.localdate()),
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
        "data_type": metric.data_type,
    }

    if not metric.is_api_backed:
        # Two cases, both rendered from stored history alone and never from a
        # live call: UNAVAILABLE means the endpoint exists but returns nothing
        # for this user; LEGACY means no endpoint exists at all.
        readings = HistoricalMetric.series_for(request.user, metric.historical_metric_type)
        series = (
            _historical_series(
                readings, f"{metric.label} (legacy)", readings[0].unit
            )
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
        template = (
            "health/metric_legacy.html"
            if metric.group == metrics.LEGACY
            else "health/metric_unavailable.html"
        )
        return render(request, template, context)

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
# one night's sleep-stage timeline
# ---------------------------------------------------------------------------


@login_required
def sleep_night(request, day):
    """Hypnogram for a single night: one band per stage across the night.

    Inherently per-night, so it lives on its own page rather than on the
    multi-day sleep view, which links into it.
    """
    try:
        target = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError:
        raise Http404("Expected a date as YYYY-MM-DD")

    credential = _credential_or_none(request)
    if credential is None:
        return redirect("accounts:landing")

    metric = metrics.get("sleep")
    context = {
        "metric": metric,
        "day": target,
        "prev_day": target - timedelta(days=1),
        "next_day": target + timedelta(days=1),
    }

    # A night is keyed by the civil date it ends on, so fetching that single day
    # is enough; a two-day window guards a session that straddles the boundary.
    try:
        client = GoogleHealthClient(credential)
        fetched = health_cache.fetch_points(
            client,
            metric,
            target,
            target + timedelta(days=1),
            user=request.user,
            force=health_cache.wants_refresh(request),
        )
    except GoogleReconnectRequired as exc:
        return render(request, "health/reconnect.html", {"detail": str(exc)})
    except GoogleHealthError as exc:
        context["error"] = str(exc)
        return render(request, "health/sleep_night.html", context)

    context["freshness"] = fetched
    nights = [
        row
        for row in metric.rows(fetched.points)
        if row["date"] == target.isoformat() and not row.get("is_nap")
    ]
    context["summary"] = nights[0] if nights else None

    # Pick the night's main session: the longest non-nap one ending on this date.
    candidates = [
        point
        for point in fetched.points
        if metrics._sleep_night_date(point) == target.isoformat()
        and not ((point.get("sleep") or {}).get("metadata") or {}).get("nap")
    ]
    candidates.sort(
        key=lambda p: metrics.as_int(
            ((p.get("sleep") or {}).get("summary") or {}).get("minutesAsleep")
        )
        or 0,
        reverse=True,
    )

    # Rendered as positioned HTML rather than a canvas, so no JSON payload and
    # no charting library is needed on this page.
    chart = metrics.hypnogram(candidates[0]) if candidates else None
    context["hypnogram"] = chart
    if candidates and chart is None:
        # A `classic` sleep record has no stage breakdown to draw.
        context["no_stages"] = True
        context["sleep_type"] = (candidates[0].get("sleep") or {}).get("type")

    return render(request, "health/sleep_night.html", context)


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


# ---------------------------------------------------------------------------
# fitness: one VO2 max figure against published norms
# ---------------------------------------------------------------------------


@login_required
def fitness(request):
    """Where the stored VO2 max sits in the population norms for age and sex.

    Needs no API call. The three VO2 max metrics are all in the UNAVAILABLE
    group - the live endpoint only produces a figure from GPS-tracked outdoor
    runs, of which this account has none - so the only figure that exists is
    the one imported from Takeout, and that is already in the database.
    """
    profile = getattr(request.user, "profile", None)
    reading = HistoricalMetric.latest_for(request.user, "vo2-max")

    source = reading.source if reading else ""
    # A demographic estimate is modelled from age, sex, weight and resting
    # heart rate. Anything else is derived from actual exercise, and only that
    # can carry an age inversion. See health/fitness.py.
    measured = bool(source) and "demographic" not in source.lower()

    assessment = fitness_module.assess(
        reading.value if reading else None,
        source=source,
        measured=measured,
        born=profile.born if profile else None,
        sex=profile.sex if profile else "",
        today=timezone.localdate(),
    )

    return render(
        request,
        "health/fitness.html",
        {
            "assessment": assessment,
            "reading": reading,
            "profile": profile,
            "norms": fitness_module.NORMS.get(assessment.sex or "male", {}),
            "percentile_columns": fitness_module.PERCENTILES,
            "why_circular": fitness_module.WHY_CIRCULAR,
            "why_the_gap": fitness_module.WHY_THE_GAP_IS_THE_PART_TO_READ,
        },
    )


# ---------------------------------------------------------------------------
# the sleep/wake calculator
# ---------------------------------------------------------------------------


DEFAULT_WAKE_TIME = "07:00"
CYCLE_WINDOW_DAYS = 30


def _parse_wake_time(raw):
    """'07:00' -> time(7, 0). Anything unparseable falls back to the default."""
    from datetime import datetime as _datetime

    for text in (raw, DEFAULT_WAKE_TIME):
        try:
            return _datetime.strptime(str(text).strip(), "%H:%M").time()
        except (TypeError, ValueError):
            continue
    return None


def _personal_cycle(request):
    """Measure this sleeper's cycle length, or explain why it could not be.

    One metric, one cached fetch. A failure here is never fatal: the page falls
    back to the textbook figure and says that it did.
    """
    credential = _credential_or_none(request)
    if credential is None or not credential.is_usable:
        return sleep_cycles.Estimate(
            reason="Not signed in to Google, so no recorded nights could be read."
        )

    days, start, end = _window(request, CYCLE_WINDOW_DAYS)
    try:
        client = GoogleHealthClient(credential)
        fetched = health_cache.fetch_points(
            client,
            metrics.get("sleep"),
            start,
            end,
            user=request.user,
            force=health_cache.wants_refresh(request),
        )
    except (GoogleHealthError, GoogleReconnectRequired) as exc:
        logger.warning("Sleep fetch for the cycle estimate failed: %s", exc)
        return sleep_cycles.Estimate(
            reason="The recorded nights could not be fetched, so only the standard figure is available."
        )

    return sleep_cycles.estimate(fetched.points)


@login_required
def sleep_calculator(request):
    """Bedtimes for a target wake time, by whole sleep cycles.

    Shows the textbook 90-minute cycle and, when enough nights support one, a
    cycle length measured from this sleeper's own stage transitions - side by
    side, never silently substituted.
    """
    wake_time = _parse_wake_time(request.GET.get("wake", DEFAULT_WAKE_TIME))
    personal = _personal_cycle(request)

    standard = sleep_cycles.bedtimes(wake_time, sleep_cycles.TEXTBOOK_CYCLE_MINUTES)
    measured = (
        sleep_cycles.bedtimes(wake_time, personal.cycle_minutes)
        if personal.cycle_minutes
        else []
    )

    return render(
        request,
        "health/sleep_calculator.html",
        {
            "wake": wake_time.strftime("%H:%M"),
            "standard": standard,
            "measured": measured,
            "personal": personal,
            "cycle_minutes": sleep_cycles.TEXTBOOK_CYCLE_MINUTES,
            "latency_minutes": sleep_cycles.FALL_ASLEEP_MINUTES,
            "window_days": CYCLE_WINDOW_DAYS,
        },
    )
