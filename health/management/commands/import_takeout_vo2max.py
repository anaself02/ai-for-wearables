"""One-time import of the Fitbit-era passive VO2 max estimate from Google Takeout.

    manage.py import_takeout_vo2max "<path to Takeout folder, or the json file>"

The file lives at:
    Takeout/Google Health/Global Export Data/demographic_vo2_max-<date>.json

and each record looks like:

    {"dateTime": "07/15/26 00:00:00",
     "value": {"demographicVO2Max": 53.14916,
               "demographicVO2MaxError": 3.0,
               "filteredDemographicVO2Max": 53.14916,
               "filteredDemographicVO2MaxError": 3.0}}

`filteredDemographicVO2Max` is the smoothed series Fitbit surfaced as the Cardio
Fitness Score - its error term shrinks as readings accumulate - so that is what
gets stored. The raw estimate is kept in `extra` rather than thrown away.

This reads local files only. It never contacts Google.
"""

import json
from datetime import datetime
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from health.models import HistoricalMetric

User = get_user_model()

SOURCE = "Fitbit demographic estimate"
UNIT = "ml/kg/min"
# Real exports are named demographic_vo2_max-<date>.json; the leading wildcard
# keeps a renamed or prefixed copy discoverable.
FILE_GLOB = "*demographic_vo2_max*.json"

# Takeout writes US-style MM/DD/YY. Confirmed against a real export whose records
# run 07/15/26 -> 09/11/26 one day apart: read as DD/MM that sequence is impossible.
DATE_FORMAT = "%m/%d/%y %H:%M:%S"


def parse_date(raw: str):
    return datetime.strptime(raw, DATE_FORMAT).date()


def read_records(path: Path) -> list[dict]:
    """Return (date, value, raw, error) tuples from one export file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise CommandError(f"{path.name}: expected a JSON array, got {type(payload).__name__}")

    readings = []
    for index, record in enumerate(payload):
        value = (record or {}).get("value") or {}
        filtered = value.get("filteredDemographicVO2Max")
        raw = value.get("demographicVO2Max")
        best = filtered if filtered is not None else raw

        if best is None or "dateTime" not in record:
            # Skip rather than abort: one malformed row should not lose the export.
            continue
        try:
            day = parse_date(record["dateTime"])
        except ValueError:
            raise CommandError(
                f"{path.name} record {index}: cannot parse dateTime "
                f"{record['dateTime']!r} as {DATE_FORMAT!r}"
            )

        readings.append(
            {
                "date": day,
                "value": round(float(best), 2),
                "extra": {
                    "raw_estimate": raw,
                    "error": value.get("filteredDemographicVO2MaxError"),
                    "source_file": path.name,
                },
            }
        )
    return readings


def find_files(target: Path) -> list[Path]:
    """Accept the json file, its folder, or the Takeout root."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise CommandError(f"No such file or directory: {target}")

    matches = sorted(target.rglob(FILE_GLOB))
    if not matches:
        raise CommandError(
            f"No {FILE_GLOB} found under {target}.\n"
            f"  Expected it in: Takeout/Google Health/Global Export Data/"
        )
    return matches


class Command(BaseCommand):
    help = "Import historical VO2 max readings from a Google Takeout export."

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            help="The demographic_vo2_max json file, its folder, or the Takeout root.",
        )
        parser.add_argument("--user", help="Username to import for.")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse and report without writing anything.",
        )

    def handle(self, *args, **options):
        user = self._resolve_user(options.get("user"))
        files = find_files(Path(options["path"]).expanduser())

        self.stdout.write(self.style.MIGRATE_HEADING("Files"))
        for path in files:
            self.stdout.write(f"  {path}")

        readings = {}
        for path in files:
            for reading in read_records(path):
                # Later files win when exports overlap.
                readings[reading["date"]] = reading

        if not readings:
            raise CommandError("No usable VO2 max readings found.")

        ordered = sorted(readings.values(), key=lambda r: r["date"])
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Parsed"))
        self.stdout.write(f"  readings   {len(ordered)}")
        self.stdout.write(f"  range      {ordered[0]['date']} to {ordered[-1]['date']}")
        self.stdout.write(f"  first      {ordered[0]['value']} {UNIT}")
        self.stdout.write(f"  latest     {ordered[-1]['value']} {UNIT}")
        self.stdout.write(f"  user       {user}")

        if options["dry_run"]:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Dry run - nothing written."))
            return

        created, updated = self._write(user, ordered)

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Written"))
        self.stdout.write(f"  created    {created}")
        self.stdout.write(f"  updated    {updated}")
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {len(ordered)} historical VO2 max readings for {user}."
            )
        )

    @transaction.atomic
    def _write(self, user, readings):
        created = updated = 0
        for reading in readings:
            _, was_created = HistoricalMetric.objects.update_or_create(
                user=user,
                metric_type=HistoricalMetric.MetricType.VO2_MAX,
                date=reading["date"],
                defaults={
                    "value": reading["value"],
                    "unit": UNIT,
                    "source": SOURCE,
                    "extra": reading["extra"],
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1
        return created, updated

    def _resolve_user(self, username):
        if username:
            try:
                return User.objects.get(username=username)
            except User.DoesNotExist:
                raise CommandError(f"No user named {username!r}.")

        users = User.objects.all()
        if users.count() == 1:
            return users.first()
        if not users:
            raise CommandError("No users exist. Sign in through the app first.")
        raise CommandError(
            "Several users exist - pass --user with one of: "
            + ", ".join(u.username for u in users[:10])
        )
