"""Who the data belongs to, and the heart-rate readings loaded for them.

This is a two-person course project. Each person collects data only on
themselves, with their own device, and the two are never pooled: the devices
differ, and that difference is a confound the app keeps visible. Every reading
therefore hangs off exactly one `Person`, and nothing in the app reads a
reading without naming whose it is.

There is no login. The app is local-only, so a `Person` is a label for a
dataset, not an account.
"""

import numpy as np
from django.db import models
from django.db.models import Max, Min

#: Readings are stored one UTC day at a time, the way the source files are cut.
#: Re-importing a day is then a plain replace-or-merge, never a duplicate, and a
#: local day is assembled at read time from the one or two UTC days it touches.
SECONDS_PER_DAY = 86400

#: bpm is stored as an unsigned 16-bit integer in tenths of a bpm, so a device
#: that reports 71.5 and one that reports 71 are both held exactly.
BPM_SCALE = 10


class Person(models.Model):
    """One participant and the device they collect with."""

    class Adapter(models.TextChoices):
        FITBIT_TAKEOUT = "fitbit-takeout", "Fitbit / Google Takeout heart_rate CSV"
        EMOTIBIT = "emotibit", "EmotiBit export (parser not implemented)"

    slug = models.SlugField(unique=True)
    display_name = models.CharField(max_length=60)
    device_label = models.CharField(
        max_length=80,
        help_text="The device this person collects with, shown on every page.",
    )
    adapter_key = models.CharField(max_length=32, choices=Adapter.choices)
    timezone = models.CharField(
        max_length=64,
        default="America/Toronto",
        help_text="IANA zone used to cut local days. Files are in UTC.",
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["display_name"]
        verbose_name_plural = "people"

    def __str__(self):
        return "{} ({})".format(self.display_name, self.device_label)

    @property
    def has_data(self):
        return self.heart_rate_days.exists()

    @property
    def status(self):
        return "data loaded" if self.has_data else "awaiting data"

    def utc_range(self):
        """(first, last) reading time as aware UTC datetimes, or None."""
        agg = self.heart_rate_days.aggregate(first=Min("first_utc"),
                                             last=Max("last_utc"))
        if agg["first"] is None:
            return None
        return agg["first"], agg["last"]

    def local_range(self):
        """(first, last) as dates in this person's own timezone, or None."""
        from zoneinfo import ZoneInfo

        span = self.utc_range()
        if span is None:
            return None
        tz = ZoneInfo(self.timezone)
        return span[0].astimezone(tz).date(), span[1].astimezone(tz).date()


class Recording(models.Model):
    """One uploaded file, kept as a record of what was loaded and when.

    The file itself is not kept: its readings live in `HeartRateDay`, and the
    original stays with its owner. The hash is what makes re-uploading the same
    file a no-op.
    """

    person = models.ForeignKey(Person, on_delete=models.CASCADE,
                               related_name="recordings")
    original_filename = models.CharField(max_length=255)
    sha256 = models.CharField(max_length=64)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    row_count = models.PositiveIntegerField()
    first_utc = models.DateTimeField()
    last_utc = models.DateTimeField()
    device_seen = models.CharField(
        max_length=255, blank=True,
        help_text="Source names found in the file itself, e.g. 'Fitbit Air'.",
    )
    timezone_assumed = models.CharField(max_length=64)

    class Meta:
        ordering = ["-uploaded_at"]
        constraints = [
            models.UniqueConstraint(fields=["person", "sha256"],
                                    name="unique_recording_per_person"),
        ]

    def __str__(self):
        return "{} for {}".format(self.original_filename, self.person.display_name)


class HeartRateDay(models.Model):
    """One person's heart-rate readings for one UTC day, packed.

    A day of readings at ~3 s spacing is about 30,000 rows. Storing each as its
    own database row would mean millions of rows for a few months of data;
    packed, a day is a couple of hundred kilobytes and is read in one query.
    """

    person = models.ForeignKey(Person, on_delete=models.CASCADE,
                               related_name="heart_rate_days")
    utc_date = models.DateField()

    n = models.PositiveIntegerField()
    first_utc = models.DateTimeField()
    last_utc = models.DateTimeField()
    bpm_min = models.FloatField()
    bpm_max = models.FloatField()
    bpm_mean = models.FloatField()
    gap_median = models.FloatField(null=True)
    gap_p05 = models.FloatField(null=True)
    gap_p95 = models.FloatField(null=True)
    sources = models.JSONField(default=dict, help_text="{source name: reading count}")

    offsets = models.BinaryField(help_text="uint32 seconds since 00:00 UTC, sorted")
    bpm_x10 = models.BinaryField(help_text="uint16 bpm * 10, same order")

    class Meta:
        ordering = ["utc_date"]
        constraints = [
            models.UniqueConstraint(fields=["person", "utc_date"],
                                    name="unique_heart_rate_day"),
        ]

    def __str__(self):
        return "{} {} ({} readings)".format(self.person.display_name,
                                            self.utc_date, self.n)

    # -- packing -------------------------------------------------------------

    @staticmethod
    def pack(epochs, bpm):
        """(epoch seconds, bpm) arrays for ONE UTC day -> (offsets, bpm_x10) bytes."""
        epochs = np.asarray(epochs, dtype="int64")
        bpm = np.asarray(bpm, dtype=float)
        midnight = (epochs[0] // SECONDS_PER_DAY) * SECONDS_PER_DAY
        offsets = (epochs - midnight).astype("<u4")
        if offsets.max() >= SECONDS_PER_DAY:
            raise ValueError("readings span more than one UTC day")
        scaled = np.rint(bpm * BPM_SCALE)
        if scaled.min() < 0 or scaled.max() > np.iinfo("<u2").max:
            raise ValueError("bpm outside the storable range")
        return offsets.tobytes(), scaled.astype("<u2").tobytes()

    def unpack(self):
        """-> (epoch seconds int64, bpm float64), sorted by time."""
        midnight = int(np.datetime64(self.utc_date, "s").astype("int64"))
        offsets = np.frombuffer(bytes(self.offsets), dtype="<u4").astype("int64")
        bpm = np.frombuffer(bytes(self.bpm_x10), dtype="<u2").astype(float) / BPM_SCALE
        return midnight + offsets, bpm
