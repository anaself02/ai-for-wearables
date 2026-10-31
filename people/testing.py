"""Synthetic data for tests. Contains no real personal data.

Files use the real Takeout column names (structure only) with seeded random
values, so tests can exercise parsing, storage and scoring without ever
touching a real export.
"""

from datetime import datetime, timedelta, timezone

import numpy as np


def hr_csv_bytes(start_utc, end_utc, spacing=3, seed=0, source="Synthetic Band",
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
