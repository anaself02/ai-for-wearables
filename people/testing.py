"""Synthetic data for tests. Contains no real personal data.

Files use the real Takeout column names (structure only) with seeded random
values, so tests can exercise parsing, storage and scoring without ever
touching a real export.
"""

from datetime import datetime, timedelta, timezone

import numpy as np


def hr_csv_bytes(start_utc, end_utc, spacing=3, seed=0, source="Fitbit Air",
                 base=62, spread=10, marker=None):
    """Takeout-style heart-rate CSV text between two aware UTC datetimes.

    `marker`, if given, makes every bpm equal to it, so a test can recognise
    one person's numbers anywhere in a page.
    """
    rng = np.random.default_rng(seed)
    lines = ["timestamp,beats per minute,data source"]
    t = start_utc
    while t < end_utc:
        value = float(marker) if marker is not None else float(
            base + rng.integers(-spread // 2, spread // 2 + 1))
        lines.append("{},{},{}".format(t.strftime("%Y-%m-%dT%H:%M:%SZ"), value, source))
        t += timedelta(seconds=spacing)
    return ("\n".join(lines) + "\n").encode()


def utc_day_file(day, **kwargs):
    """(filename, bytes) for one full UTC day, named like a Takeout file."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return ("heart_rate_{}.csv".format(day),
            hr_csv_bytes(start, start + timedelta(days=1), **kwargs))


def make_user(email, staff=False, linked_to=None):
    """A synthetic account for tests. Returns (user, password).

    The password is random per run and never written anywhere: tests that need to
    log in through the form use it, others use `client.force_login`.
    """
    import secrets

    from django.contrib.auth import get_user_model

    password = secrets.token_urlsafe(18)
    user = get_user_model().objects.create_user(
        username=email.lower(), email=email.lower(), password=password,
        is_staff=staff)
    if linked_to is not None:
        linked_to.owner = user
        linked_to.save(update_fields=["owner"])
    return user, password


def _staff_test_case():
    from django.test import TestCase

    class StaffTestCase(TestCase):
        """A TestCase whose client is signed in as a synthetic staff account.

        Every page is behind a login, so the many tests about what a page shows
        run as an administrator, who may open any profile. Tests about WHO may
        open what use their own accounts instead.
        """

        def setUp(self):
            super().setUp()
            self.staff_user, _ = make_user("staff@example.com", staff=True)
            self.client.force_login(self.staff_user)

    return StaffTestCase


StaffTestCase = _staff_test_case()
