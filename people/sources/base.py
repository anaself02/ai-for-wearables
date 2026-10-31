"""The boundary between device file formats and the rest of the app.

Each device exports data in its own format. An adapter's only job is to turn
one uploaded file into a `ParsedFile`: a sorted, de-duplicated heart-rate
series in UTC epoch seconds, plus a plain-language report of what was found.
Everything downstream (storage, views, the arousal model) sees only that.

The contract for heart rate is deliberately strict, because the arousal model
was fitted on one kind of signal and must not be handed another by accident:

  * `epochs`  int64 UTC seconds, strictly increasing
  * `bpm`     float, the value the DEVICE reported, never one derived here
  * `sources` {name as written in the file: reading count}

An adapter must never derive heart rate from a raw waveform without that being
a separate, declared decision: a derived series has a different resolution
from a device-reported one and the arousal caveats would change.
"""

from dataclasses import dataclass, field

import numpy as np


class SourceError(Exception):
    """A file cannot be used; the message is shown to the user as written."""


class SourceNotImplemented(SourceError):
    """The adapter for this device exists as a placeholder only."""


@dataclass(frozen=True)
class SourceFile:
    """One uploaded file: its name (no directories) and its bytes."""

    name: str
    data: bytes


@dataclass
class FileReport:
    """What happened to one file, for the validation page."""

    name: str
    status: str            # "parsed", "skipped" or "error"
    reason: str = ""
    rows: int = 0
    first_utc: object = None
    last_utc: object = None
    dropped_unparseable: int = 0
    dropped_duplicate: int = 0


@dataclass
class ParsedFile:
    epochs: np.ndarray
    bpm: np.ndarray
    #: Per-row source as a small integer code into `source_names`, so a merge can
    #: keep exact per-source counts without holding millions of strings.
    source_names: list = field(default_factory=list)
    source_codes: np.ndarray = None
    report: FileReport = None

    @property
    def sources(self):
        """{source name: reading count}"""
        counts = np.bincount(self.source_codes, minlength=len(self.source_names))
        return {name: int(c) for name, c in zip(self.source_names, counts)}


class DeviceAdapter:
    """Interface every device format implements."""

    key = ""
    device_label = ""
    #: False for a placeholder adapter, so the UI can say so up front.
    implemented = True
    not_implemented_message = ""

    def accepts(self, name):
        """True if a file with this name is one this adapter reads.

        Decided from the name alone, so a file that is not wanted is never
        opened or parsed.
        """
        raise NotImplementedError

    def parse_file(self, source_file):
        """SourceFile -> ParsedFile. Raise SourceError with a plain message."""
        raise NotImplementedError
