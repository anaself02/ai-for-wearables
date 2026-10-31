"""Uploading a person's files: choose, validate, then confirm.

Nothing is stored by the first request. It parses the files, shows a validation
report (what was read, what was skipped, row counts, the UTC and local date
range, the timezone assumed, the device names found), and parks the files in a
spool under a token. Only the second request, from the report page's Confirm
button, stores anything, and it re-parses the parked files rather than trusting
the first pass.

Uploaded data is only stored and, later, scored by a model that was trained on
WESAD alone. Nothing here trains, fits or tunes anything.
"""

import os

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect
from django.views.decorators.http import require_POST

from . import ingest, spool
from .sources import SourceError, SourceFile, get_adapter
from .views import person_page, render_person

SCORED_ONLY_NOTICE = "Model trained on WESAD only; uploaded data is only scored."
MAX_LISTED_SKIPPED = 20


def _base_name(name):
    return os.path.basename((name or "").replace("\\", "/"))


def collect(adapter, uploads):
    """Accepted SourceFiles and skipped names from the uploaded files.

    A name the adapter does not accept is never read. A zip is opened, and only
    members the adapter accepts are read out of it.
    """
    files, skipped_names, skipped_count, total = [], [], 0, 0
    for upload in uploads:
        name = _base_name(upload.name)
        total += upload.size
        if total > settings.MAX_UPLOAD_BYTES:
            raise SourceError(
                "That upload is too large to load here. Upload only the Heart "
                "Rate folder (the heart_rate_YYYY-MM-DD.csv files).")
        if name.lower().endswith(".zip"):
            got, skipped = ingest.files_from_zip(adapter, upload.read())
            files.extend(got)
            skipped_count += skipped
        elif adapter.accepts(name):
            files.append(SourceFile(name, upload.read()))
        else:
            skipped_count += 1
            skipped_names.append(name)
    return files, skipped_names, skipped_count


@person_page
def upload(request, person):
    """The upload form (GET) and the validation step (POST)."""
    adapter = get_adapter(person.adapter_key)
    context = {"adapter": adapter, "scored_only": SCORED_ONLY_NOTICE}

    if request.method != "POST":
        return render_person(request, "people/upload.html", person, context)

    if not adapter.implemented:
        plan = ingest.build_plan(person, [])
        context["report"] = plan.report
        return render_person(request, "people/upload_report.html", person, context)

    uploads = request.FILES.getlist("files")
    if not uploads:
        context["error"] = "Choose at least one file."
        return render_person(request, "people/upload.html", person, context)

    try:
        files, skipped_names, skipped_count = collect(adapter, uploads)
    except SourceError as exc:
        context["error"] = str(exc)
        return render_person(request, "people/upload.html", person, context)

    plan = ingest.build_plan(person, files, skipped_other=skipped_count)
    context.update({
        "report": plan.report,
        "skipped_names": skipped_names[:MAX_LISTED_SKIPPED],
        "skipped_more": max(0, len(skipped_names) - MAX_LISTED_SKIPPED),
        "needs_person_check": bool(plan.report.warnings),
    })
    if plan.report.can_commit:
        context["token"] = spool.save(person, files, skipped_names)
    return render_person(request, "people/upload_report.html", person, context)


@person_page
@require_POST
def upload_confirm(request, person):
    """Store a validated upload. Re-parses the parked files first."""
    adapter = get_adapter(person.adapter_key)
    context = {"adapter": adapter, "scored_only": SCORED_ONLY_NOTICE}
    token = request.POST.get("token", "")

    try:
        files, skipped_names = spool.load(person, token)
    except spool.BadToken as exc:
        context["error"] = str(exc)
        return render_person(request, "people/upload.html", person, context)

    plan = ingest.build_plan(person, files, skipped_other=len(skipped_names))
    if plan.report.warnings and not request.POST.get("confirm_person"):
        context.update({"report": plan.report, "token": token,
                        "needs_person_check": True,
                        "error": "Tick the box to confirm this is the right person."})
        return render_person(request, "people/upload_report.html", person, context)

    try:
        result = ingest.commit_plan(plan)
    except SourceError as exc:
        context["error"] = str(exc)
        return render_person(request, "people/upload.html", person, context)

    spool.discard(person, token)
    messages.success(
        request,
        "Loaded {:,} readings from {} file(s) for {}: {} new UTC day(s), {} merged "
        "into existing days.".format(result.rows_loaded, result.files_loaded,
                                     person.display_name, result.days_new,
                                     result.days_merged))
    return redirect("people:person", slug=person.slug)


@person_page
@require_POST
def upload_cancel(request, person):
    spool.discard(person, request.POST.get("token", ""))
    messages.success(request, "Nothing was loaded.")
    return redirect("people:person", slug=person.slug)
