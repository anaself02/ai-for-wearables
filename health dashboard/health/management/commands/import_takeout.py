"""Import every supported legacy metric from a Google Takeout export.

    manage.py import_takeout "<path to the Takeout folder>"
    manage.py import_takeout "<path>" --only sleep-score --dry-run

A metric is only supported here if the live API cannot supply it - see
health/takeout.py for what is deliberately excluded and why.

This is a one-time manual import. It reads local files and never contacts Google.
"""

from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from health import takeout
from health.models import HistoricalMetric

User = get_user_model()


def resolve_user(username):
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


def run_import(command, path, username, only, dry_run):
    """Shared by this command and the older vo2max-only alias."""
    user = resolve_user(username)
    root = Path(path).expanduser()
    if not root.exists():
        raise CommandError(f"No such file or directory: {root}")

    if only and only not in takeout.IMPORTS:
        raise CommandError(
            f"Unknown metric {only!r}. Known: {', '.join(sorted(takeout.IMPORTS))}"
        )

    try:
        collected = takeout.collect(root, only=only)
    except takeout.TakeoutParseError as exc:
        raise CommandError(str(exc))

    if not collected:
        expected = "\n".join(
            f"  {spec.label:12} {spec.glob:34} in {spec.expected_in}"
            for spec in takeout.IMPORTS.values()
            if not only or spec.metric_type == only
        )
        raise CommandError(f"Nothing found under {root}. Looked for:\n{expected}")

    command.stdout.write(command.style.MIGRATE_HEADING("Parsed"))
    for metric_type, readings in collected.items():
        spec = takeout.IMPORTS[metric_type]
        command.stdout.write(
            f"  {spec.label:12} {len(readings):>5} readings  "
            f"{readings[0]['date']} to {readings[-1]['date']}  "
            f"latest {readings[-1]['value']}{spec.unit}"
        )
    command.stdout.write(f"  {'user':12} {user}")

    if dry_run:
        command.stdout.write("")
        command.stdout.write(command.style.WARNING("Dry run - nothing written."))
        return

    created, updated = write(user, collected)

    command.stdout.write("")
    command.stdout.write(command.style.MIGRATE_HEADING("Written"))
    command.stdout.write(f"  created    {created}")
    command.stdout.write(f"  updated    {updated}")
    command.stdout.write("")
    command.stdout.write(
        command.style.SUCCESS(
            f"Imported {created + updated} historical readings for {user}."
        )
    )


@transaction.atomic
def write(user, collected):
    created = updated = 0
    for metric_type, readings in collected.items():
        spec = takeout.IMPORTS[metric_type]
        for reading in readings:
            _, was_created = HistoricalMetric.objects.update_or_create(
                user=user,
                metric_type=metric_type,
                date=reading["date"],
                defaults={
                    "value": reading["value"],
                    "unit": spec.unit,
                    "source": spec.source,
                    "extra": reading["extra"],
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1
    return created, updated


class Command(BaseCommand):
    help = "Import legacy metrics (sleep score, VO2 max) from a Google Takeout export."

    def add_arguments(self, parser):
        parser.add_argument(
            "path", help="The Takeout folder, a subfolder, or a single export file."
        )
        parser.add_argument("--user", help="Username to import for.")
        parser.add_argument(
            "--only",
            help="Import a single metric: " + ", ".join(sorted(takeout.IMPORTS)),
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Parse and report without writing."
        )

    def handle(self, *args, **options):
        run_import(
            self,
            options["path"],
            options.get("user"),
            options.get("only"),
            options["dry_run"],
        )
