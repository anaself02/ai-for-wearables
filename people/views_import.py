"""Import data: choose files, validate, then confirm.

Nothing is stored by the first request. It parses the files, shows a validation
report (what was read, what was skipped and why, row counts, the UTC and local
range, the timezone assumed, the device names found) and parks the PARSED readings
under a token. The uploaded files themselves are not kept: they are parsed and
discarded when the request ends. Only the second request, from the report page's
Confirm button, stores anything.

What is accepted:

  * a loose .csv or .xlsx (first sheet), accepted by its HEADER: the three Fitbit
    heart-rate columns, exactly. Any other column refuses the file, with a plain
    message, before a data row is read. No other layout is guessed.
  * a .zip, whose members are accepted only if they are named
    heart_rate_YYYY-MM-DD.csv. Every other member is skipped and listed, never
    opened. A zip is read in memory and never extracted to disk.

Uploaded data is only stored and, later, scored by a model that was trained on
WESAD alone. Nothing here trains, fits or tunes anything.

Django's cap of 100 files per request (DATA_UPLOAD_MAX_NUMBER_FILES) is raised in
settings. It is a global setting, not a per-route one, which is harmless because
this is the only route that accepts files.
"""

import os

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect
from django.views.decorators.http import require_POST

from . import ingest, spool
from .sources import SourceError, SourceFile, get_adapter
from .views import person_page, render_person
from .xlsx import xlsx_to_csv_bytes

SCORED_ONLY_NOTICE = "Model trained on WESAD only; uploaded data is only scored."
MAX_LISTED_SKIPPED = 20
MAX_FILE_BYTES = ingest.MAX_MEMBER_BYTES
ACCEPTED = "a .csv or .xlsx file, or a .zip of heart_rate_YYYY-MM-DD.csv files"


def _base_name(name):
    return os.path.basename((name or "").replace("\\", "/"))


def collect(adapter, uploads):
    """(SourceFiles, skipped names, skipped count) from the uploaded files.

    A zip is opened and only members named like the adapter's files are read out of
    it. A loose .csv or .xlsx is accepted for its header to be checked, whatever it
    is called. Anything else is skipped without being read.
    """
    files, skipped_names, skipped_count, total = [], [], 0, 0
    for upload in uploads:
        name = _base_name(upload.name)
        total += upload.size
        if total > settings.MAX_UPLOAD_BYTES:
            raise SourceError(
                "That upload is too large to load here. Upload only the Heart "
                "Rate folder (the heart_rate_YYYY-MM-DD.csv files).")
        lower = name.lower()

        if lower.endswith(".zip"):
            got, skipped = ingest.files_from_zip(adapter, upload.read())
            files.extend(got)
            skipped_count += skipped
        elif lower.endswith((".csv", ".xlsx")):
            if upload.size > MAX_FILE_BYTES:
                raise SourceError("{} is larger than {} MB, which is too large to "
                                  "load here.".format(name, MAX_FILE_BYTES // (1024 * 1024)))
            data = upload.read()
            if lower.endswith(".xlsx"):
                data = xlsx_to_csv_bytes(name, data)
            files.append(SourceFile(name, data))
        else:
            skipped_count += 1
            skipped_names.append(name)
    return files, skipped_names, skipped_count


def _context(adapter):
    return {"adapter": adapter, "scored_only": SCORED_ONLY_NOTICE, "accepted": ACCEPTED}


@person_page
def import_data(request, person):
    """The import form (GET) and the validation step (POST)."""
    adapter = get_adapter(person.adapter_key)
    context = _context(adapter)

    if request.method != "POST":
        return render_person(request, "people/import.html", person, context)

    if not adapter.implemented:
        # Whatever was chosen, nothing is read or saved: say why, and what is needed.
        plan = ingest.build_plan(person, [])
        context["report"] = plan.report
        return render_person(request, "people/import_report.html", person, context)

    uploads = request.FILES.getlist("files")
    if not uploads:
        context["error"] = "Choose at least one file."
        return render_person(request, "people/import.html", person, context)

    try:
        files, skipped_names, skipped_count = collect(adapter, uploads)
    except SourceError as exc:
        context["error"] = str(exc)
        return render_person(request, "people/import.html", person, context)

    plan = ingest.build_plan(person, files, skipped_other=skipped_count)
    context.update({
        "report": plan.report,
        "skipped_names": skipped_names[:MAX_LISTED_SKIPPED],
        "skipped_more": max(0, len(skipped_names) - MAX_LISTED_SKIPPED),
        "needs_person_check": bool(plan.report.warnings),
    })
    if plan.report.can_commit:
        # The files are discarded after this request; only the parsed readings wait.
        context["token"] = spool.save(person, plan.loads, skipped_names)
    return render_person(request, "people/import_report.html", person, context)


@person_page
@require_POST
def import_confirm(request, person):
    """Store a validated import from the parked readings."""
    adapter = get_adapter(person.adapter_key)
    context = _context(adapter)
    token = request.POST.get("token", "")

    try:
        items, skipped_names = spool.load(person, token)
    except spool.BadToken as exc:
        context["error"] = str(exc)
        return render_person(request, "people/import.html", person, context)

    plan = ingest.plan_from_parsed(person, items, skipped_other=len(skipped_names))
    if plan.report.warnings and not request.POST.get("confirm_person"):
        context.update({"report": plan.report, "token": token,
                        "needs_person_check": True,
                        "error": "Tick the box to confirm this is the right person."})
        return render_person(request, "people/import_report.html", person, context)

    try:
        result = ingest.commit_plan(plan)
    except SourceError as exc:
        context["error"] = str(exc)
        return render_person(request, "people/import.html", person, context)

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
def import_cancel(request, person):
    spool.discard(person, request.POST.get("token", ""))
    messages.success(request, "Nothing was loaded.")
    return redirect("people:person", slug=person.slug)
