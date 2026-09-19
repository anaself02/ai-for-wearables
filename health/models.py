"""Storage for metrics that only exist in a historical export.

Some Fitbit-era metrics did not survive the migration to Google Health. The
passive VO2 max estimate (`demographic_vo2_max`) is the clearest case: the live
API only derives VO2 max from GPS-tracked outdoor runs, so for anyone who has
not recorded one the live cards are permanently empty even though years of
estimates exist in their Takeout export.

These rows are imported once by hand and never synced. They are always labelled
as historical in the UI so they cannot be mistaken for live data.
"""

from django.conf import settings
from django.db import models


class HistoricalMetric(models.Model):
    """One legacy reading: a user, a metric, a date, a value, and its provenance."""

    class MetricType(models.TextChoices):
        VO2_MAX = "vo2-max", "VO2 max"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="historical_metrics",
    )
    metric_type = models.CharField(max_length=64, choices=MetricType.choices)
    date = models.DateField()
    value = models.FloatField()
    unit = models.CharField(max_length=32, blank=True)
    source = models.CharField(
        max_length=128,
        help_text='Where the reading came from, e.g. "Fitbit demographic estimate".',
    )
    #: Anything else the export carried that is worth keeping but not charting.
    extra = models.JSONField(default=dict, blank=True)
    imported_at = models.DateTimeField(auto_now=True)

    class Meta:
        # One reading per metric per day, so re-running an import is idempotent.
        constraints = [
            models.UniqueConstraint(
                fields=["user", "metric_type", "date"],
                name="unique_historical_reading",
            )
        ]
        indexes = [models.Index(fields=["user", "metric_type", "-date"])]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.metric_type} {self.value} on {self.date}"

    @classmethod
    def latest_for(cls, user, metric_type):
        if not user or not user.is_authenticated:
            return None
        return cls.objects.filter(user=user, metric_type=metric_type).first()

    @classmethod
    def series_for(cls, user, metric_type):
        """Oldest-first, ready for charting."""
        if not user or not user.is_authenticated:
            return []
        return list(
            cls.objects.filter(user=user, metric_type=metric_type).order_by("date")
        )
