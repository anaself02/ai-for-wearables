"""The Sessions tab: a person's own labelled sessions and how the WESAD models
grade against them.

Same access as every other person page (the owner and staff; anyone else gets
404). A session is always looked up together with the person in the URL, so one
person's session can never be opened, edited or exported under another's name.
Nothing here logs or exports the note.
"""

import csv

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST

from health import arousal
from health.wesad_binary import MODEL_LABEL

from . import sessions
from .models import Session
from .session_forms import SessionForm
from .views import person_page, render_person


def _context(person, form, editing=None):
    result = sessions.score(person)
    return {
        "form": form, "editing": editing,
        "rows": sessions.coverage_rows(person),
        "score": result,
        "label_lag": sessions.LABEL_LAG_SECONDS,
        "min_per_class": sessions.MIN_WINDOWS_PER_CLASS,
        "coverage_warning": sessions.COVERAGE_WARNING,
        "window_seconds": arousal.WINDOW_SECONDS,
        "step_seconds": arousal.STEP_SECONDS,
        "min_samples": arousal.MIN_SAMPLES_PER_WINDOW,
        "model_label": MODEL_LABEL,
        "not_enough": sessions.NOT_ENOUGH,
        "no_intervals": sessions.NO_INTERVALS,
        "ood_flag": arousal.OOD_FLAG_SHARE,
    }


@person_page
def sessions_page(request, person):
    """The list, the add form (POST adds), and the grading."""
    form = SessionForm(request.POST or None, person=person)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Session added.")
        return redirect("people:sessions", person.slug)
    return render_person(request, "people/sessions.html", person, _context(person, form))


@person_page
def session_edit(request, person, pk):
    session = get_object_or_404(Session, pk=pk, person=person)
    form = SessionForm(request.POST or None, person=person, instance=session)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Session updated.")
        return redirect("people:sessions", person.slug)
    return render_person(request, "people/sessions.html", person,
                         _context(person, form, editing=session))


@person_page
@require_POST
def session_delete(request, person, pk):
    get_object_or_404(Session, pk=pk, person=person).delete()
    messages.success(request, "Session deleted.")
    return redirect("people:sessions", person.slug)


@person_page
def sessions_export(request, person):
    """Session-level results as CSV: "Session 1..N", relative minutes, no notes."""
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="sessions-{}.csv"'.format(
        person.slug)
    writer = csv.DictWriter(response, fieldnames=sessions.EXPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(sessions.export_rows(sessions.score(person)))
    return response
