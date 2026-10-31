"""Load one person's heart-rate files from a folder.

    manage.py import_heart_rate anas "<path to the Heart Rate folder>"
    manage.py import_heart_rate anas "<path>" --dry-run

This is the same code path as the upload page: files are parsed into a plan, a
validation report is printed, and only then is anything stored. Only files the
person's device reader accepts BY NAME are opened (for a Fitbit Takeout that is
heart_rate_YYYY-MM-DD.csv); every other file in the folder tree is only counted.
Nothing is trained, fitted or tuned on what is loaded.
"""

from django.core.management.base import BaseCommand, CommandError

from people import ingest
from people.models import Person
from people.sources import SourceError, get_adapter


class Command(BaseCommand):
    help = "Load a person's heart-rate files from a folder (same path as the upload page)."

    def add_arguments(self, parser):
        parser.add_argument("slug", help="the person's slug, e.g. anas")
        parser.add_argument("folder", help="folder holding the heart-rate files")
        parser.add_argument("--dry-run", action="store_true",
                            help="parse and report without storing anything")

    def handle(self, *args, **options):
        try:
            person = Person.objects.get(slug=options["slug"])
        except Person.DoesNotExist:
            known = ", ".join(Person.objects.values_list("slug", flat=True))
            raise CommandError("No person {!r}. Known: {}.".format(
                options["slug"], known))

        try:
            adapter = get_adapter(person.adapter_key)
            files, skipped = ingest.files_from_folder(adapter, options["folder"])
            plan = ingest.build_plan(person, files, skipped_other=skipped)
        except SourceError as exc:
            raise CommandError(str(exc))

        self.stdout.write(ingest.render_report(plan.report))

        if options["dry_run"]:
            self.stdout.write("Dry run: nothing was stored.")
            return
        if not plan.report.can_commit:
            self.stdout.write("Nothing stored.")
            return

        result = ingest.commit_plan(plan)
        self.stdout.write(
            "Stored {:,} readings from {} file(s): {} new UTC day(s), {} merged. "
            "{:,} reading(s) were already present.".format(
                result.rows_loaded, result.files_loaded, result.days_new,
                result.days_merged, result.rows_already_present))
