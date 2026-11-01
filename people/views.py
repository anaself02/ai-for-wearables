"""The home page, and every page about one person's data.

Every page about data resolves the person from the URL first, then asks the
repository for that person's readings only. `person_page` does the resolving and
`render_person` puts the person into every page's context, so the top bar can
show their name and device on all of them and no view can forget to.

Values shown here are either exactly as in the uploaded file (reading counts,
minimum and maximum bpm, raw rows) or computed by this app from those rows
(daily averages, per-minute means, spacing). The second kind carries the
"derived" class, which the stylesheet draws dashed.
"""

from datetime import date
from functools import wraps

from django.contrib import messages
from django.shortcuts import get_object_or_404, render
from django.views.decorators.cache import never_cache

from django.http import Http404

from health import arousal as arousal_module
from members.access import person_or_404

from . import repository
from .models import Person
from .sources import get_adapter

EXPECTED_AFTER_PRESENTATION = (
    "No data uploaded yet. Expected after the mid-project presentation.")
NO_DATA_YET = "No data uploaded yet."

WINDOWS = [7, 30, 90]
DEFAULT_WINDOW = 30
RAW_ROWS_SHOWN = 100


def empty_message(person):
    """The honest empty state. Never a placeholder, sample or invented number."""
    if not get_adapter(person.adapter_key).implemented:
        return EXPECTED_AFTER_PRESENTATION
    return NO_DATA_YET


def person_page(view):
    """Resolve <slug> to a Person (404 if unknown) and never serve from cache."""

    @never_cache
    @wraps(view)
    def wrapper(request, slug, *args, **kwargs):
        person = get_object_or_404(Person, slug=slug)
        person_or_404(request.user, person)     # an account sees only its own profile
        return view(request, person, *args, **kwargs)

    return wrapper


def render_person(request, template, person, context=None):
    """Render with the person in context, so the top bar always names them."""
    context = dict(context or {})
    context["person"] = person
    if not person.has_data:
        context["empty_message"] = empty_message(person)
    return render(request, template, context)


def chart_series(summaries):
    """Chart.js payload of daily average / min / max, from computed summaries."""
    return {
        "labels": [s["day"].isoformat() for s in summaries],
        "datasets": [
            {"label": "Average", "data": [round(s["mean"], 1) for s in summaries],
             "unit": "bpm"},
            {"label": "Min", "data": [s["min"] for s in summaries],
             "unit": "bpm", "faint": True},
            {"label": "Max", "data": [s["max"] for s in summaries],
             "unit": "bpm", "faint": True},
        ],
    }


# ---------------------------------------------------------------------------
# home
# ---------------------------------------------------------------------------

@never_cache
def home(request):
    """The person chooser: staff only. Everyone else is sent to their own profile."""
    if not request.user.is_staff:
        raise Http404("No such page.")   # the access rules never let them get here
    rows = []
    for person in Person.objects.all():
        stats = repository.totals(person)
        rows.append({
            "person": person,
            "range": person.local_range(),
            "utc_days": stats["utc_days"],
            "readings": stats["readings"],
            "empty_message": "" if person.has_data else empty_message(person),
        })
    return render(request, "people/home.html", {"rows": rows})


# ---------------------------------------------------------------------------
# one person
# ---------------------------------------------------------------------------

@person_page
def person_home(request, person):
    """Overview: the last 30 days at a glance, from this person's rows only."""
    if not person.has_data:
        return render_person(request, "people/person_home.html", person)

    stats = repository.totals(person)
    summaries = repository.daily_summaries(person, days=DEFAULT_WINDOW)
    return render_person(request, "people/person_home.html", person, {
        "range": person.local_range(),
        "readings": stats["readings"],
        "devices": stats["devices"],
        "recent": list(reversed(summaries))[:14],
        "series_json": chart_series(summaries),
        "window": DEFAULT_WINDOW,
    })


@person_page
def heart_rate(request, person):
    """Heart rate over a window of days ending at the person's latest day."""
    try:
        days = int(request.GET.get("days", DEFAULT_WINDOW))
    except ValueError:
        days = DEFAULT_WINDOW
    days = days if days in WINDOWS else DEFAULT_WINDOW

    if not person.has_data:
        return render_person(request, "people/heart_rate.html", person,
                             {"windows": WINDOWS, "days": days})

    summaries = repository.daily_summaries(person, days=days)
    return render_person(request, "people/heart_rate.html", person, {
        "windows": WINDOWS,
        "days": days,
        "summaries": list(reversed(summaries)),
        "series_json": chart_series(summaries),
        "day_options": [s["day"].isoformat() for s in reversed(summaries)],
        "readings": sum(s["n"] for s in summaries),
    })


@person_page
def heart_rate_day(request, person, day):
    """One local day, as per-minute means (computed from the stored readings)."""
    if not person.has_data:
        return render_person(request, "people/heart_rate_day.html", person,
                             {"day": day})

    labels, values, count = repository.minute_means(person, day)
    previous, following = repository.neighbours(person, day)
    context = {"day": day, "count": count, "prev_day": previous, "next_day": following}
    if count:
        summary = repository.summarise_local_day(person, day)
        context.update({
            "summary": summary,
            "devices": repository.device_names_for_day(person, day),
            "series_json": {"labels": labels,
                            "datasets": [{"label": "Per-minute mean", "data": values,
                                          "unit": "bpm"}]},
        })
    return render_person(request, "people/heart_rate_day.html", person, context)


@person_page
def raw_index(request, person):
    """The files loaded for this person, and the days of rows they hold."""
    if not person.has_data:
        return render_person(request, "people/raw_index.html", person)

    days = repository.daily_summaries(person)
    return render_person(request, "people/raw_index.html", person, {
        "recordings": person.recordings.all(),
        "days": list(reversed(days)),
    })


@person_page
def raw_day(request, person, day):
    """The stored rows of one local day, as loaded from the uploaded files."""
    if not person.has_data:
        return render_person(request, "people/raw_day.html", person, {"day": day})

    rows, total = repository.raw_rows(person, day, limit=RAW_ROWS_SHOWN)
    previous, following = repository.neighbours(person, day)
    return render_person(request, "people/raw_day.html", person, {
        "day": day, "rows": rows, "total": total, "shown": len(rows),
        "prev_day": previous, "next_day": following,
    })


# ---------------------------------------------------------------------------
# arousal: the WESAD-trained model, applied to one person's chosen day
# ---------------------------------------------------------------------------

CONDITION_NAMES = {1: "baseline condition", 2: "stress condition (TSST)",
                   3: "amusement condition"}


def share_rows(assessment):
    """[(condition, share of all windows, share of waking windows)]."""
    return [(CONDITION_NAMES[label], assessment.shares.get(label, 0.0),
             assessment.waking_shares.get(label, 0.0))
            for label in CONDITION_NAMES]


@person_page
def arousal_view(request, person):
    """List the scorable days and score one: ?date=YYYY-MM-DD, default newest.

    The model was trained on WESAD only. The chosen day is only scored; nothing
    is fitted or saved. The person, device and reading spacing are shown with
    the result so a Fitbit day and an EmotiBit day can never look alike.
    """
    context = {"min_windows": arousal_module.MIN_WINDOWS_TO_SCORE,
               "scored_only": arousal_module.WESAD_ONLY_NOTICE,
               "window_seconds": arousal_module.WINDOW_SECONDS,
               "step_seconds": arousal_module.STEP_SECONDS,
               "min_samples": arousal_module.MIN_SAMPLES_PER_WINDOW}
    if not person.has_data:
        return render_person(request, "people/arousal.html", person, context)

    days = arousal_module.scorable_days(person)
    default = arousal_module.default_day(days)
    chosen = default

    raw = request.GET.get("date")
    if raw:
        try:
            chosen = date.fromisoformat(raw)
        except ValueError:
            messages.error(request, "That is not a valid date; showing the most "
                                    "recent scorable day instead.")

    context.update({"days": days, "default_day": default, "chosen": chosen})
    if chosen is None:
        context["no_scorable_day"] = True
    else:
        assessment = arousal_module.assess(person, chosen)
        context["assessment"] = assessment
        if assessment.available:
            context["share_rows"] = share_rows(assessment)
            context["spacing"] = assessment.spacing
    return render_person(request, "people/arousal.html", person, context)


@person_page
def arousal_methodology(request, person):
    """What the model is, how it was tested, and what it cannot tell you."""
    return render_person(request, "people/arousal_methodology.html", person, {
        "card": arousal_module.load_model_card(),
        "window_seconds": arousal_module.WINDOW_SECONDS,
        "step_seconds": arousal_module.STEP_SECONDS,
        "min_samples": arousal_module.MIN_SAMPLES_PER_WINDOW,
        "min_windows": arousal_module.MIN_WINDOWS_TO_SCORE,
        "scored_only": arousal_module.WESAD_ONLY_NOTICE,
    })
