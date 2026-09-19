"""VO2 max only. Kept as an alias now that `import_takeout` handles every metric.

    manage.py import_takeout_vo2max "<path>"

Equivalent to `manage.py import_takeout "<path>" --only vo2-max`.
"""

from django.core.management.base import BaseCommand

from health.management.commands.import_takeout import run_import


class Command(BaseCommand):
    help = "Import historical VO2 max readings from a Google Takeout export."

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            help="The demographic_vo2_max json file, its folder, or the Takeout root.",
        )
        parser.add_argument("--user", help="Username to import for.")
        parser.add_argument(
            "--dry-run", action="store_true", help="Parse and report without writing."
        )

    def handle(self, *args, **options):
        run_import(
            self,
            options["path"],
            options.get("user"),
            "vo2-max",
            options["dry_run"],
        )
