"""The home page (who is being viewed) and one person's own page.

Every page about data resolves the person from the URL first, then asks the
repository for that person's readings only. The person's name and device are in
the context of every such page, so the top bar can show them on all of them.
"""

from django.shortcuts import get_object_or_404, render
from django.views.decorators.cache import never_cache

from . import repository
from .models import Person
from .sources import get_adapter

EXPECTED_AFTER_PRESENTATION = (
    "No data uploaded yet. Expected after the mid-project presentation.")
NO_DATA_YET = "No data uploaded yet."


def empty_message(person):
    """The honest empty state. Never a placeholder, sample or invented number."""
    if not get_adapter(person.adapter_key).implemented:
        return EXPECTED_AFTER_PRESENTATION
    return NO_DATA_YET


@never_cache
def home(request):
    """Lists everyone with device, date range and status. No sign-in."""
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


@never_cache
def person_home(request, slug):
    """One person's page: their own readings and nothing of anyone else's."""
    person = get_object_or_404(Person, slug=slug)
    context = {"person": person}

    if not person.has_data:
        context["empty_message"] = empty_message(person)
        return render(request, "people/person_home.html", context)

    stats = repository.totals(person)
    context.update({
        "range": person.local_range(),
        "utc_days": stats["utc_days"],
        "readings": stats["readings"],
        "devices": stats["devices"],
        "recent": repository.recent_day_summaries(person, limit=14),
    })
    return render(request, "people/person_home.html", context)
