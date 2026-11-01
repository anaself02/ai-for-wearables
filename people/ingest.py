"""Loading a person's files: parse, report, de-duplicate, store.

One code path serves both the `import_heart_rate` command and the upload page,
so what the command loads and what the browser loads can never differ.

The flow is two steps so that nothing is saved before it has been shown:

    plan = build_plan(person, files)     # parse everything, write nothing
    plan.report                          # what was found, what would happen
    commit_plan(plan)                    # only now are rows written

Nothing here trains, fits or tunes a model. Uploaded data is only stored, and
later only scored by a model that was trained on WESAD alone.

De-duplication is on two levels:

  * file level: a file with the same SHA-256 already loaded for this person is
    reported as "already loaded" and skipped, so re-uploading changes nothing;
  * row level: readings are merged by timestamp into the UTC day they belong to,
    so two different exports that overlap never produce a duplicate reading.
"""

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from django.db import transaction

from .models import SECONDS_PER_DAY, HeartRateDay, Person, Recording
from .sources import SourceError, SourceFile, SourceNotImplemented, get_adapter

#: Guards for zip uploads. The Heart Rate folder of a real export is ~100 daily files
#: and ~110 MB. A zip is read in memory, never extracted to disk.
MAX_ZIP_ENTRIES = 5000                      # names in the archive, read or not
MAX_ZIP_MEMBERS_READ = 400                  # files actually read
MAX_UNCOMPRESSED_BYTES = 600 * 1024 * 1024  # total of the members read
MAX_MEMBER_BYTES = 50 * 1024 * 1024         # any one file


# ---------------------------------------------------------------------------
# getting files in
# ---------------------------------------------------------------------------

def sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def files_from_folder(adapter, folder):
    """(files the adapter accepts, count of names skipped) from a folder tree.

    Only files whose NAME the adapter accepts are ever opened. Everything else
    in the tree (a Takeout export holds hundreds of unrelated files, some of
    them personal profile, location and security data) is only counted.
    """
    if not os.path.isdir(folder):
        raise SourceError("Folder not found: {}".format(folder))
    accepted, skipped = [], 0
    for base, _dirs, names in os.walk(folder):
        for name in sorted(names):
            if adapter.accepts(name):
                with open(os.path.join(base, name), "rb") as handle:
                    accepted.append(SourceFile(name, handle.read()))
            else:
                skipped += 1
    return accepted, skipped


def files_from_zip(adapter, data):
    """(files the adapter accepts, count of members skipped) from zip bytes.

    Members are matched by base name only, and only accepted members are read.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise SourceError("That file is not a valid zip archive.")

    accepted, skipped, total = [], 0, 0
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise SourceError(
                "That zip holds too many files ({}). Upload only the Heart Rate folder "
                "(the heart_rate_YYYY-MM-DD.csv files), not the whole export."
                .format(len(entries)))
        for info in entries:
            if info.is_dir():
                continue
            name = os.path.basename(info.filename.replace("\\", "/"))
            if not adapter.accepts(name):
                skipped += 1
                continue
            total += info.file_size
            if (len(accepted) >= MAX_ZIP_MEMBERS_READ or total > MAX_UNCOMPRESSED_BYTES
                    or info.file_size > MAX_MEMBER_BYTES):
                raise SourceError(
                    "That zip is too large to load here. Upload only the Heart "
                    "Rate folder (the heart_rate_YYYY-MM-DD.csv files), not the "
                    "whole export.")
            accepted.append(SourceFile(name, archive.read(info)))
    return accepted, skipped


# ---------------------------------------------------------------------------
# the plan and its report
# ---------------------------------------------------------------------------

@dataclass
class FileOutcome:
    name: str
    status: str                 # "will load" | "already loaded" | "error"
    detail: str = ""
    rows: int = 0
    first_utc: datetime = None
    last_utc: datetime = None


@dataclass
class IngestReport:
    person_name: str
    device_label: str
    timezone_assumed: str
    fatal: str = ""
    files: list = field(default_factory=list)
    skipped_other: int = 0
    rows_to_load: int = 0
    first_utc: datetime = None
    last_utc: datetime = None
    first_local: object = None
    last_local: object = None
    days_new: int = 0
    days_merged: int = 0
    rows_already_present: int = 0
    devices: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def n_error(self):
        return sum(1 for f in self.files if f.status == "error")

    @property
    def n_will_load(self):
        return sum(1 for f in self.files if f.status == "will load")

    @property
    def n_already(self):
        return sum(1 for f in self.files if f.status == "already loaded")

    @property
    def can_commit(self):
        """Only when nothing failed and there is something new to store."""
        return not self.fatal and self.n_error == 0 and self.n_will_load > 0

    def refusal(self):
        """Why the commit is refused, or ''."""
        if self.fatal:
            return self.fatal
        if self.n_error:
            return ("{} file(s) could not be read. Nothing was loaded; fix or "
                    "remove them and try again.".format(self.n_error))
        if self.n_will_load == 0:
            if self.n_already:
                return "Everything in this upload is already loaded. Nothing to do."
            return "No file in this upload is one the {} reader accepts.".format(
                self.device_label)
        return ""


@dataclass
class Plan:
    person: Person
    report: IngestReport
    loads: list = field(default_factory=list)   # [(sha256, file name, ParsedFile)]


def build_plan(person, files, skipped_other=0):
    """Parse everything; write nothing. Returns a Plan with its report."""
    adapter = get_adapter(person.adapter_key)
    report = IngestReport(person_name=person.display_name,
                          device_label=person.device_label,
                          timezone_assumed=person.timezone,
                          skipped_other=skipped_other)
    plan = Plan(person=person, report=report)

    if not adapter.implemented:
        report.fatal = adapter.not_implemented_message
        return plan

    known = set(person.recordings.values_list("sha256", flat=True))
    seen_in_upload = set()
    per_day = {}   # utc_date ordinal -> arrays gathered from this upload

    for source_file in files:
        digest = sha256_of(source_file.data)
        if digest in known or digest in seen_in_upload:
            report.files.append(FileOutcome(
                source_file.name, "already loaded",
                "same content (SHA-256) as a file already loaded for {}".format(
                    person.display_name)))
            continue
        seen_in_upload.add(digest)

        try:
            parsed = adapter.parse_file(source_file)
        except SourceNotImplemented as exc:
            report.fatal = str(exc)
            return plan
        except SourceError as exc:
            report.files.append(FileOutcome(source_file.name, "error", str(exc)))
            continue

        r = parsed.report
        extra = []
        if r.dropped_unparseable:
            extra.append("{} unreadable row(s) dropped".format(r.dropped_unparseable))
        if r.dropped_duplicate:
            extra.append("{} duplicate row(s) dropped".format(r.dropped_duplicate))
        report.files.append(FileOutcome(
            source_file.name, "will load", "; ".join(extra), r.rows,
            r.first_utc, r.last_utc))
        plan.loads.append((digest, source_file.name, parsed))
        for name, count in parsed.sources.items():
            report.devices[name] = report.devices.get(name, 0) + count

    _summarise(plan, person)
    return plan


def plan_from_parsed(person, items, skipped_other=0):
    """Rebuild a Plan from files that were parsed earlier, without their bytes.

    `items` is [(sha256, file name, ParsedFile)]. The uploaded files are not kept
    after they are parsed; only these parsed readings wait for the confirmation.
    Duplicates are checked again NOW against what is stored, in case something was
    loaded in the meantime.
    """
    adapter = get_adapter(person.adapter_key)
    report = IngestReport(person_name=person.display_name,
                          device_label=person.device_label,
                          timezone_assumed=person.timezone,
                          skipped_other=skipped_other)
    plan = Plan(person=person, report=report)
    if not adapter.implemented:
        report.fatal = adapter.not_implemented_message
        return plan

    known = set(person.recordings.values_list("sha256", flat=True))
    for digest, name, parsed in items:
        if digest in known:
            report.files.append(FileOutcome(
                name, "already loaded",
                "same content (SHA-256) as a file already loaded for {}".format(
                    person.display_name)))
            continue
        r = parsed.report
        report.files.append(FileOutcome(name, "will load", "", r.rows,
                                        r.first_utc, r.last_utc))
        plan.loads.append((digest, name, parsed))
        for source, count in parsed.sources.items():
            report.devices[source] = report.devices.get(source, 0) + count

    _summarise(plan, person)
    return plan


def _summarise(plan, person):
    report = plan.report
    if not plan.loads:
        return

    tz = ZoneInfo(person.timezone)
    firsts = [p.report.first_utc for _, _, p in plan.loads]
    lasts = [p.report.last_utc for _, _, p in plan.loads]
    report.first_utc, report.last_utc = min(firsts), max(lasts)
    report.first_local = report.first_utc.astimezone(tz).date()
    report.last_local = report.last_utc.astimezone(tz).date()
    report.rows_to_load = sum(p.report.rows for _, _, p in plan.loads)

    days = set()
    for _, _, parsed in plan.loads:
        days.update(int(o) for o in np.unique(parsed.epochs // SECONDS_PER_DAY))
    existing = {d.toordinal() - _EPOCH_ORDINAL
                for d in person.heart_rate_days.values_list("utc_date", flat=True)}
    report.days_merged = len(days & existing)
    report.days_new = len(days - existing)

    label = person.device_label.lower().split()[0]
    foreign = [n for n in report.devices if label not in n.lower()]
    if foreign:
        report.warnings.append(
            "The files name the device {} but this person's device is {}. "
            "Check this is the right person before loading.".format(
                ", ".join(repr(n) for n in foreign), repr(person.device_label)))


_EPOCH_ORDINAL = datetime(1970, 1, 1).toordinal()


# ---------------------------------------------------------------------------
# storing
# ---------------------------------------------------------------------------

def day_stats(epochs, bpm):
    gaps = np.diff(epochs)
    return {
        "n": int(len(epochs)),
        "first_utc": datetime.fromtimestamp(int(epochs[0]), timezone.utc),
        "last_utc": datetime.fromtimestamp(int(epochs[-1]), timezone.utc),
        "bpm_min": float(np.min(bpm)),
        "bpm_max": float(np.max(bpm)),
        "bpm_mean": float(np.mean(bpm)),
        "gap_median": float(np.median(gaps)) if len(gaps) else None,
        "gap_p05": float(np.percentile(gaps, 5)) if len(gaps) else None,
        "gap_p95": float(np.percentile(gaps, 95)) if len(gaps) else None,
    }


@dataclass
class CommitResult:
    files_loaded: int = 0
    rows_loaded: int = 0
    rows_already_present: int = 0
    days_new: int = 0
    days_merged: int = 0


def commit_plan(plan):
    """Write a plan to the database. All or nothing."""
    reason = plan.report.refusal()
    if reason:
        raise SourceError(reason)

    person = plan.person
    result = CommitResult()

    # Group every reading of this upload by UTC day, remembering source names.
    by_day = {}
    for _, _, parsed in plan.loads:
        ordinals = parsed.epochs // SECONDS_PER_DAY
        for ordinal in np.unique(ordinals):
            mask = ordinals == ordinal
            names = np.array(parsed.source_names, dtype=object)[parsed.source_codes[mask]]
            by_day.setdefault(int(ordinal), []).append(
                (parsed.epochs[mask], parsed.bpm[mask], names))

    with transaction.atomic():
        for ordinal, chunks in sorted(by_day.items()):
            epochs = np.concatenate([c[0] for c in chunks])
            bpm = np.concatenate([c[1] for c in chunks])
            names = np.concatenate([c[2] for c in chunks])
            utc_date = datetime.fromtimestamp(ordinal * SECONDS_PER_DAY,
                                              timezone.utc).date()

            existing = HeartRateDay.objects.filter(person=person,
                                                   utc_date=utc_date).first()
            sources = {}
            if existing is not None:
                old_epochs, old_bpm = existing.unpack()
                # Existing rows win on a clash; only genuinely new rows are added.
                new_mask = ~np.isin(epochs, old_epochs)
                result.rows_already_present += int((~new_mask).sum())
                epochs, bpm, names = epochs[new_mask], bpm[new_mask], names[new_mask]
                sources = dict(existing.sources)
                if len(epochs) == 0:
                    continue
                epochs = np.concatenate([old_epochs, epochs])
                bpm = np.concatenate([old_bpm, bpm])
                result.days_merged += 1
            else:
                result.days_new += 1

            fresh, counts = np.unique(names, return_counts=True)
            for name, count in zip(fresh, counts):
                sources[str(name)] = sources.get(str(name), 0) + int(count)

            order = np.argsort(epochs, kind="stable")
            epochs, bpm = epochs[order], bpm[order]
            offsets, packed = HeartRateDay.pack(epochs, bpm)
            fields = day_stats(epochs, bpm)
            fields.update(offsets=offsets, bpm_x10=packed, sources=sources)
            HeartRateDay.objects.update_or_create(
                person=person, utc_date=utc_date, defaults=fields)
            result.rows_loaded += int(len(epochs)) - (
                0 if existing is None else existing.n)

        for digest, file_name, parsed in plan.loads:
            r = parsed.report
            Recording.objects.create(
                person=person, original_filename=file_name, sha256=digest,
                row_count=r.rows, first_utc=r.first_utc, last_utc=r.last_utc,
                device_seen=", ".join(parsed.source_names)[:255],
                timezone_assumed=person.timezone)
            result.files_loaded += 1

    return result


# ---------------------------------------------------------------------------
# plain-text rendering, for the command line
# ---------------------------------------------------------------------------

def render_report(report):
    lines = ["Validation report for {} ({})".format(report.person_name,
                                                     report.device_label)]
    if report.fatal:
        lines.append("  CANNOT LOAD: " + report.fatal)
        return "\n".join(lines)
    lines.append("  Timezone assumed for local days: " + report.timezone_assumed)
    lines.append("  Files: {} to load, {} already loaded, {} with errors; "
                 "{} other file name(s) skipped without being opened".format(
                     report.n_will_load, report.n_already, report.n_error,
                     report.skipped_other))
    for f in report.files:
        if f.status == "will load":
            continue
        lines.append("    [{}] {}: {}".format(f.status, f.name, f.detail))
    if report.rows_to_load:
        lines.append("  Readings to load: {:,}".format(report.rows_to_load))
        lines.append("  UTC range  : {} to {}".format(report.first_utc, report.last_utc))
        lines.append("  Local range: {} to {} ({})".format(
            report.first_local, report.last_local, report.timezone_assumed))
        lines.append("  UTC days: {} new, {} merged into existing".format(
            report.days_new, report.days_merged))
        lines.append("  Device name(s) in the files: " + ", ".join(
            "{} ({:,})".format(k, v) for k, v in report.devices.items()))
    for warning in report.warnings:
        lines.append("  WARNING: " + warning)
    refusal = report.refusal()
    lines.append("  " + ("WILL NOT LOAD: " + refusal if refusal else "Ready to load."))
    return "\n".join(lines)
