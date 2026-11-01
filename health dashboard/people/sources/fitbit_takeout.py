"""Fitbit / Google Takeout heart-rate files.

A Takeout export holds hundreds of files; this adapter reads exactly one kind,
`heart_rate_YYYY-MM-DD.csv`, from `Physical Activity_GoogleData/Heart Rate/`.
The profile, device, location and security files in the same export are never
accepted, so they are never parsed.

Observed format (one real export, structure only):

    timestamp,beats per minute,data source
    2026-07-15T22:35:50Z,158.0,Fitbit Air

Timestamps are UTC with a trailing "Z"; the file's date is the UTC start date,
so a local evening lands in the next file. Spacing is about 2-3 s.
"""

import csv
import io
import re
from datetime import datetime, timezone

import numpy as np

from .base import DeviceAdapter, FileReport, ParsedFile, SourceError

NAME_RE = re.compile(r"^heart_rate_(\d{4}-\d{2}-\d{2})\.csv$")
COLUMNS = ["timestamp", "beats per minute", "data source"]
HAS_ZONE_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")

#: Held exactly by the packed storage; anything beyond is not a heart rate.
MAX_STORABLE_BPM = 6553.0


def check_header(name, data):
    """Read ONLY the first line and require exactly the three heart-rate columns.

    This runs before anything else is parsed, so a file with a different layout
    (a profile, a device list, a location file) is refused on its header alone and
    none of its other columns is ever read, stored or shown. The refusal says how
    many unexpected columns there were, never what they were called.
    """
    first_line = data.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    first_line = first_line.rstrip("\r")
    try:
        header = [c.strip() for c in next(csv.reader([first_line]), [])]
    except csv.Error:
        header = []
    missing = [c for c in COLUMNS if c not in header]
    extra = len([c for c in header if c not in COLUMNS])
    if missing or not header:
        raise SourceError(
            "{} is not a Fitbit Takeout heart-rate file: it must have the columns {}."
            .format(name, ", ".join(COLUMNS)))
    if extra or len(header) != len(COLUMNS):
        raise SourceError(
            "{} has {} column(s) beyond the three heart-rate columns ({}). Only that "
            "layout is accepted, and the other columns are never read. No other layout "
            "is guessed.".format(name, max(extra, len(header) - len(COLUMNS)),
                                 ", ".join(COLUMNS)))


class FitbitTakeoutAdapter(DeviceAdapter):
    key = "fitbit-takeout"
    device_label = "Fitbit"

    def accepts(self, name):
        return bool(NAME_RE.match(name))

    def parse_file(self, source_file):
        import pandas as pd

        name = source_file.name
        check_header(name, source_file.data)
        try:
            table = pd.read_csv(io.BytesIO(source_file.data), usecols=COLUMNS)
        except ValueError:
            raise SourceError(
                "{} is not a Fitbit Takeout heart-rate file: it must have the "
                "columns {}.".format(name, ", ".join(COLUMNS)))
        except Exception as exc:  # unreadable bytes, bad quoting, empty file
            raise SourceError("{} could not be read as a CSV file ({}).".format(
                name, type(exc).__name__))

        if table.empty:
            raise SourceError("{} has a header but no readings.".format(name))

        raw_times = table["timestamp"].dropna()
        sample = str(raw_times.iloc[0]) if len(raw_times) else ""
        if not HAS_ZONE_RE.search(sample):
            raise SourceError(
                "{} has timestamps without a timezone (for example {!r}). "
                "Refusing to guess; Takeout times end in 'Z'.".format(name, sample))

        stamps = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
        values = pd.to_numeric(table["beats per minute"], errors="coerce")
        good = (stamps.notna() & values.notna()
                & np.isfinite(values.fillna(np.nan).astype(float)))
        dropped_bad = int((~good).sum())
        if not good.any():
            raise SourceError("{} has no readable heart-rate rows.".format(name))

        epochs = (stamps[good].dt.tz_convert("UTC").dt.tz_localize(None)
                  .values.astype("datetime64[s]").astype("int64"))
        bpm = values[good].astype(float).values
        sources = table.loc[good, "data source"].astype(str).values

        order = np.argsort(epochs, kind="stable")
        epochs, bpm, sources = epochs[order], bpm[order], sources[order]
        keep = np.ones(len(epochs), dtype=bool)
        keep[1:] = epochs[1:] != epochs[:-1]
        dropped_dup = int((~keep).sum())
        epochs, bpm, sources = epochs[keep], bpm[keep], sources[keep]

        if bpm.min() < 0 or bpm.max() > MAX_STORABLE_BPM:
            raise SourceError(
                "{} has heart-rate values outside 0-{:.0f} (min {:.1f}, max "
                "{:.1f}); this does not look like bpm.".format(
                    name, MAX_STORABLE_BPM, bpm.min(), bpm.max()))

        names, codes = np.unique(sources, return_inverse=True)
        report = FileReport(
            name=name, status="parsed", rows=int(len(epochs)),
            first_utc=datetime.fromtimestamp(int(epochs[0]), timezone.utc),
            last_utc=datetime.fromtimestamp(int(epochs[-1]), timezone.utc),
            dropped_unparseable=dropped_bad, dropped_duplicate=dropped_dup)
        return ParsedFile(epochs=epochs, bpm=bpm,
                          source_names=[str(n) for n in names],
                          source_codes=codes.astype("uint8"), report=report)
