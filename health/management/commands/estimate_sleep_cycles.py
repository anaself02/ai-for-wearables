"""Run the cycle estimator against real nights and print what it found.

The point of this command is to look at the evidence before trusting the
number the calculator page shows. It prints every night it could read, the
cycles measured on each, and what was discarded, so the spread can be judged
by eye rather than by a single median.

    manage.py estimate_sleep_cycles --days 60
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from datetime import timedelta

from health import metrics, sleep_cycles
from health.cache import fetch_points
from health.client import GoogleHealthClient, GoogleHealthError, GoogleReconnectRequired


class Command(BaseCommand):
    help = "Measure sleep cycle length from recorded stage transitions."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Window to read.")
        parser.add_argument("--user", help="Username, if more than one exists.")

    def handle(self, *args, **options):
        User = get_user_model()
        if options["user"]:
            user = User.objects.filter(username=options["user"]).first()
        else:
            user = User.objects.first()
        if user is None:
            raise CommandError("No user to read nights for.")

        credential = getattr(user, "google_credential", None)
        if credential is None or not credential.is_usable:
            raise CommandError(f"No usable Google credential for {user}.")

        days = options["days"]
        end = timezone.localdate() + timedelta(days=1)
        start = end - timedelta(days=days)

        try:
            client = GoogleHealthClient(credential)
            fetched = fetch_points(
                client, metrics.get("sleep"), start, end, user=user, force=True
            )
        except (GoogleHealthError, GoogleReconnectRequired) as exc:
            raise CommandError(f"Could not fetch nights: {exc}")

        points = fetched.points
        self.stdout.write(f"{len(points)} sleep record(s) over {days} days\n")

        for point in points:
            night = sleep_cycles.night_cycles(point)
            if not night["has_stages"]:
                continue
            date = (
                ((point.get("sleep") or {}).get("interval") or {}).get("startTime", "?")
            )[:10]
            cycles = ", ".join(f"{length:.0f}" for length in night["cycles"]) or "-"
            self.stdout.write(
                f"  {date}  REM periods {night['rem_periods']}  "
                f"cycles [{cycles}]  dropped {night['dropped']}"
            )

        result = sleep_cycles.estimate(points)
        self.stdout.write("")
        self.stdout.write(f"nights with stages   {result.nights_with_stages}")
        self.stdout.write(f"nights contributing  {result.nights_contributing}")
        self.stdout.write(f"cycles measured      {result.cycles_measured}")
        self.stdout.write(f"cycles dropped       {result.cycles_dropped}")

        if result.cycle_minutes is None:
            self.stdout.write(self.style.WARNING(f"\n{result.reason}"))
            return

        self.stdout.write(
            f"median               {result.cycle_minutes} min "
            f"(+/- {result.spread_minutes}, range {result.shortest:.0f} to {result.longest:.0f})"
        )
        if result.reliable:
            self.stdout.write(self.style.SUCCESS("\nPasses the gates; the page will use it."))
        else:
            self.stdout.write(self.style.WARNING(f"\n{result.reason}"))
